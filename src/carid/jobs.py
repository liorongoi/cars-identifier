"""Persistent background jobs (collection, re-indexing, benchmarks) run by one worker thread."""

from __future__ import annotations

import logging
import threading
import traceback
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select, update

from carid.db import Database
from carid.models import Job, JobStatus

log = logging.getLogger(__name__)

MAX_LOG_CHARS = 200_000


class JobContext:
    def __init__(self, manager: JobManager, job_id: int, params: dict[str, Any]) -> None:
        self.manager = manager
        self.job_id = job_id
        self.params = params
        self._lines: list[str] = []

    def progress(self, fraction: float, message: str = "") -> None:
        with self.manager.db.session() as s:
            s.execute(update(Job).where(Job.id == self.job_id).values(
                progress=max(0.0, min(1.0, fraction)), message=message[:500]))

    def log(self, line: str) -> None:
        stamp = datetime.now(UTC).strftime("%H:%M:%S")
        self._lines.append(f"[{stamp}] {line}")
        log.info("job %s: %s", self.job_id, line)
        with self.manager.db.session() as s:
            job = s.get(Job, self.job_id)
            if job is not None:
                job.log = ("\n".join(self._lines))[-MAX_LOG_CHARS:]

    def set_counters(self, counters: dict[str, Any]) -> None:
        with self.manager.db.session() as s:
            s.execute(update(Job).where(Job.id == self.job_id).values(counters=counters))

    def should_stop(self) -> bool:
        if self.manager.stopping.is_set():
            return True
        with self.manager.db.session() as s:
            return bool(s.scalar(select(Job.cancel_requested).where(Job.id == self.job_id)))


Handler = Callable[[JobContext], dict[str, Any] | None]


class JobManager:
    def __init__(self, db: Database) -> None:
        self.db = db
        self.handlers: dict[str, Handler] = {}
        self.stopping = threading.Event()
        self._wake = threading.Event()
        self._thread: threading.Thread | None = None

    def register(self, kind: str, handler: Handler) -> None:
        self.handlers[kind] = handler

    def submit(self, kind: str, params: dict[str, Any], requested_by: str | None = None) -> int:
        if kind not in self.handlers:
            raise ValueError(f"unknown job kind: {kind}")
        with self.db.session() as s:
            job = Job(kind=kind, params=params, requested_by=requested_by)
            s.add(job)
            s.flush()
            job_id = job.id
        self._wake.set()
        return job_id

    def cancel(self, job_id: int) -> bool:
        with self.db.session() as s:
            job = s.get(Job, job_id)
            if job is None or job.status not in (JobStatus.queued, JobStatus.running):
                return False
            if job.status == JobStatus.queued:
                job.status = JobStatus.cancelled
                job.finished_at = datetime.now(UTC)
            else:
                job.cancel_requested = True
            return True

    def start(self) -> None:
        # Jobs interrupted by a restart are re-queued (they are resumable by design).
        with self.db.session() as s:
            s.execute(update(Job).where(Job.status == JobStatus.running).values(
                status=JobStatus.queued, message="re-queued after restart"))
        self.stopping.clear()
        self._thread = threading.Thread(target=self._loop, name="carid-jobs", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 10.0) -> None:
        self.stopping.set()
        self._wake.set()
        if self._thread:
            self._thread.join(timeout)

    def _next(self) -> tuple[int, str, dict[str, Any]] | None:
        with self.db.session() as s:
            job = s.scalars(select(Job).where(Job.status == JobStatus.queued)
                            .order_by(Job.id).limit(1)).first()
            if job is None:
                return None
            job.status = JobStatus.running
            job.started_at = datetime.now(UTC)
            job.cancel_requested = False
            return job.id, job.kind, dict(job.params)

    def run_pending(self) -> int:
        """Run queued jobs synchronously until none remain (used by the CLI and tests)."""
        n = 0
        while not self.stopping.is_set():
            nxt = self._next()
            if nxt is None:
                break
            self._execute(*nxt)
            n += 1
        return n

    def _loop(self) -> None:
        while not self.stopping.is_set():
            self.run_pending()
            self._wake.wait(timeout=2.0)
            self._wake.clear()

    def _execute(self, job_id: int, kind: str, params: dict[str, Any]) -> None:
        ctx = JobContext(self, job_id, params)
        status, message = JobStatus.succeeded, "done"
        result: dict[str, Any] | None = None
        try:
            result = self.handlers[kind](ctx)
            if ctx.should_stop():
                status, message = JobStatus.cancelled, "cancelled"
        except Exception as exc:
            status, message = JobStatus.failed, f"{type(exc).__name__}: {exc}"
            ctx.log(traceback.format_exc())
        with self.db.session() as s:
            job = s.get(Job, job_id)
            if job is None:
                return
            if self.stopping.is_set() and status == JobStatus.cancelled \
                    and not job.cancel_requested:
                job.status, job.message = JobStatus.queued, "paused for shutdown"
                return
            job.status = status
            job.message = message[:500]
            job.finished_at = datetime.now(UTC)
            if status == JobStatus.succeeded:
                job.progress = 1.0
            if result:
                job.counters = {**(job.counters or {}), **result}

"""Collector building blocks: targets, candidates, the polite HTTP client, connector API."""

from __future__ import annotations

import logging
import threading
import time
import urllib.robotparser
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, field
from email.utils import parsedate_to_datetime
from typing import Any
from urllib.parse import urlsplit

import httpx

log = logging.getLogger(__name__)


@dataclass
class Target:
    """What to collect: one model of one make (any year)."""

    make_id: int
    make: str
    make_aliases: list[str]
    model_id: int
    model: str
    model_aliases: list[str]
    sibling_models: list[str] = field(default_factory=list)  # other models of the make

    @property
    def display(self) -> str:
        return f"{self.make} {self.model}"


@dataclass
class Candidate:
    url: str  # direct image URL to download
    page_url: str | None = None
    title: str = ""
    description: str = ""
    license: str | None = None
    credit: str | None = None
    extra_text: str = ""  # categories, captions, alt text — used for labelling
    local_path: str | None = None  # for local sources

    @property
    def text(self) -> str:
        return " ".join(t for t in (self.title, self.description, self.extra_text) if t)


class FetchError(RuntimeError):
    pass


class HostPaused(FetchError):
    """The host asked for a back-off longer than we are willing to wait."""


class SourceExhausted(RuntimeError):
    """The source's quota is used up for now; skip it for the rest of the job."""


class PoliteClient:
    """HTTP client with per-host throttling, retry/back-off and robots.txt checks."""

    def __init__(self, user_agent: str, timeout: float = 30.0, min_interval: float = 1.0,
                 max_retries: int = 5, max_bytes: int = 25 * 1024 * 1024,
                 max_wait: float = 60.0,
                 transport: httpx.BaseTransport | None = None,
                 sleep: Any = time.sleep) -> None:
        self.client = httpx.Client(
            headers={"User-Agent": user_agent}, timeout=timeout, follow_redirects=True,
            transport=transport,
        )
        self.user_agent = user_agent
        self.min_interval = min_interval
        self.max_retries = max_retries
        self.max_bytes = max_bytes
        self._last: dict[str, float] = {}
        self.host_intervals: dict[str, float] = {}  # per-host minimum spacing (API limits)
        self.max_wait = max_wait  # longer Retry-After => pause that host instead of sleeping
        self._cooldown: dict[str, float] = {}
        self.host_max_wait: dict[str, float] = {}  # per-host override of max_wait
        self._robots: dict[str, urllib.robotparser.RobotFileParser | None] = {}
        self._lock = threading.Lock()
        self._sleep = sleep

    def close(self) -> None:
        self.client.close()

    def _throttle(self, host: str) -> None:
        with self._lock:
            interval = self.host_intervals.get(host, self.min_interval)
            wait = self._last.get(host, 0.0) + interval - time.monotonic()
            if wait > 0:
                self._sleep(wait)
            self._last[host] = time.monotonic()

    def allowed(self, url: str) -> bool:
        parts = urlsplit(url)
        base = f"{parts.scheme}://{parts.netloc}"
        if base not in self._robots:
            rp: urllib.robotparser.RobotFileParser | None = urllib.robotparser.RobotFileParser()
            assert rp is not None
            try:
                r = self.client.get(f"{base}/robots.txt")
                if r.status_code >= 400:
                    rp = None  # no robots.txt => allowed
                else:
                    rp.parse(r.text.splitlines())
            except httpx.HTTPError:
                rp = None
            self._robots[base] = rp
        rp = self._robots[base]
        return True if rp is None else rp.can_fetch(self.user_agent, url)

    def get(self, url: str, *, params: dict[str, Any] | None = None,
            check_robots: bool = True, **kwargs: Any) -> httpx.Response:
        if check_robots and not self.allowed(url):
            raise FetchError(f"robots.txt disallows {url}")
        host = urlsplit(url).netloc
        until = self._cooldown.get(host, 0.0)
        if until > time.monotonic():
            raise HostPaused(f"{host} asked us to back off for {until - time.monotonic():.0f}s")
        delay = 2.0
        for attempt in range(self.max_retries + 1):
            self._throttle(host)
            try:
                r = self.client.get(url, params=params, **kwargs)
            except httpx.TransportError as exc:
                if attempt == self.max_retries:
                    raise FetchError(f"network error for {url}: {exc}") from exc
                self._sleep(delay)
                delay = min(delay * 2, 120)
                continue
            if r.status_code == 429 or r.status_code >= 500:
                wait = _retry_after(r)
                if wait > self.host_max_wait.get(host, self.max_wait):
                    # Don't block the whole job on one throttled host: skip it for now,
                    # other hosts keep going; URLs are retried on the next run.
                    self._cooldown[host] = time.monotonic() + wait
                    raise HostPaused(f"HTTP {r.status_code} for {url}: {host} asks to wait "
                                     f"{wait:.0f}s; pausing that host")
                if attempt == self.max_retries:
                    raise FetchError(f"HTTP {r.status_code} for {url} after retries")
                self._sleep(max(delay, wait))
                delay = min(delay * 2, 120)
                continue
            if r.status_code >= 400:
                raise FetchError(f"HTTP {r.status_code} for {url}")
            return r
        raise FetchError(f"giving up on {url}")  # pragma: no cover

    def download_image(self, url: str, check_robots: bool = True) -> bytes:
        r = self.get(url, check_robots=check_robots)
        ctype = r.headers.get("content-type", "").split(";")[0].strip().lower()
        if ctype and not ctype.startswith("image/") and ctype != "application/octet-stream":
            raise FetchError(f"not an image ({ctype}): {url}")
        if len(r.content) > self.max_bytes:
            raise FetchError(f"file too large ({len(r.content)} bytes): {url}")
        return r.content


def _retry_after(r: httpx.Response) -> float:
    value = r.headers.get("retry-after")
    if not value:
        return 0.0
    try:
        return max(0.0, float(value))
    except ValueError:
        try:
            return max(0.0, parsedate_to_datetime(value).timestamp() - time.time())
        except (TypeError, ValueError):
            return 0.0


class SourceConnector(ABC):
    """A place images can be collected from."""

    #: short machine name, also stored on each image
    name: str = ""
    #: human description shown in the UI
    description: str = ""
    #: connectors touching social networks or copyrighted media need explicit legal approval
    requires_legal_approval: bool = False

    def __init__(self, config: dict[str, Any], http: PoliteClient) -> None:
        self.config = config
        self.http = http

    @property
    def enabled(self) -> bool:
        return bool(self.config.get("enabled", False))

    @property
    def legal_approved(self) -> bool:
        return bool(self.config.get("legal_approved", False))

    @property
    def runnable(self) -> bool:
        return self.enabled and (self.legal_approved or not self.requires_legal_approval)

    def status(self) -> dict[str, Any]:
        reason = ""
        if not self.enabled:
            reason = "disabled in configuration"
        elif self.requires_legal_approval and not self.legal_approved:
            reason = "legal approval not recorded (set legal_approved: true)"
        return {"name": self.name, "description": self.description, "enabled": self.enabled,
                "requires_legal_approval": self.requires_legal_approval,
                "legal_approved": self.legal_approved, "runnable": self.runnable,
                "reason": reason}

    @abstractmethod
    def search(self, target: Target, limit: int) -> Iterator[Candidate]:
        """Yield up to ``limit`` candidate images for the target."""

    def fetch(self, candidate: Candidate) -> bytes:
        return self.http.download_image(candidate.url)

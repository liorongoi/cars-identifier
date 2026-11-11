"""Runs collection: targets × enabled sources → labelled, filtered, de-duplicated images."""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml
from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from carid.collector.base import (
    FetchError,
    PoliteClient,
    SourceConnector,
    SourceExhausted,
    Target,
)
from carid.collector.labeler import infer_label
from carid.collector.sources import CONNECTOR_CLASSES
from carid.config import Settings
from carid.engine import Engine, ReferenceLabel, SourceInfo
from carid.models import Make, SeenUrl
from carid.vision.imageio import ImageDecodeError

log = logging.getLogger(__name__)

# Sources whose search results are uncurated; without a deep interior classifier we only
# accept their images when the text explicitly says "interior".
UNCURATED_SOURCES = {"wikimedia", "flickr"}


def load_sources_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"wikimedia": {"enabled": True}, "folder": {"enabled": True, "path": "imports"}}
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a mapping of source name -> settings")
    return data


def build_connectors(config: dict[str, Any], http: PoliteClient) -> dict[str, SourceConnector]:
    out: dict[str, SourceConnector] = {}
    for name, cls in CONNECTOR_CLASSES.items():
        out[name] = cls(dict(config.get(name) or {}), http)
    return out


def make_http(settings: Settings) -> PoliteClient:
    return PoliteClient(settings.user_agent, timeout=settings.request_timeout_s,
                        min_interval=settings.min_request_interval_s,
                        max_retries=settings.max_retries,
                        max_bytes=settings.max_download_mb * 1024 * 1024)


def build_targets(session: Session, make_ids: list[int] | None = None,
                  model_ids: list[int] | None = None) -> list[Target]:
    stmt = select(Make).options(
        selectinload(Make.models)).order_by(Make.name)
    if make_ids:
        stmt = stmt.where(Make.id.in_(make_ids))
    targets: list[Target] = []
    for make in session.scalars(stmt):
        siblings = [m.name for m in make.models]
        for model in make.models:
            if model_ids and model.id not in model_ids:
                continue
            targets.append(Target(
                make_id=make.id, make=make.name, make_aliases=list(make.aliases),
                model_id=model.id, model=model.name, model_aliases=list(model.aliases),
                sibling_models=siblings,
            ))
    return targets


ProgressFn = Callable[[float, str], None]


class Collector:
    def __init__(self, engine: Engine, connectors: dict[str, SourceConnector]) -> None:
        self.engine = engine
        self.connectors = connectors
        self.has_interior_classifier = callable(getattr(engine.embedder, "interior_score", None))

    def run(self, targets: list[Target], per_model_limit: int = 150,
            sources: list[str] | None = None, progress: ProgressFn | None = None,
            should_stop: Callable[[], bool] | None = None,
            log_fn: Callable[[str], None] | None = None) -> dict[str, int]:
        counters: Counter[str] = Counter()
        active = [c for n, c in self.connectors.items()
                  if (sources is None or n in sources)]
        say = log_fn or log.info
        for c in active:
            if not c.runnable:
                say(f"source '{c.name}' skipped: {c.status()['reason']}")
        active = [c for c in active if c.runnable]
        if not active:
            say("no runnable sources")
            return dict(counters)
        added_since_save = 0
        exhausted: set[str] = set()
        for t_no, target in enumerate(targets):
            if should_stop and should_stop():
                say("cancelled")
                break
            added_for_target = 0
            for connector in active:
                if should_stop and should_stop():
                    break
                if connector.name in exhausted:
                    continue
                remaining = per_model_limit - added_for_target
                if remaining <= 0:
                    break
                try:
                    added = self._collect_one(connector, target, remaining * 4, remaining,
                                              counters, say, should_stop)
                except SourceExhausted as exc:
                    exhausted.add(connector.name)
                    counters["sources_exhausted"] += 1
                    say(f"{connector.name}: {exc}; skipping it for the rest of this job")
                    added = 0
                except Exception as exc:  # one broken source must not stop the job
                    counters["source_errors"] += 1
                    say(f"{connector.name} failed for {target.display}: {exc}")
                    added = 0
                added_for_target += added
                added_since_save += added
                if added_since_save >= 25:
                    self.engine.save_index()
                    added_since_save = 0
            say(f"{target.display}: +{added_for_target} images")
            if progress:
                progress((t_no + 1) / len(targets), f"{target.display} done "
                         f"({t_no + 1}/{len(targets)})")
        self.engine.save_index()
        return dict(counters)

    def _collect_one(self, connector: SourceConnector, target: Target, scan_limit: int,
                     add_limit: int, counters: Counter[str], say: Callable[[str], None],
                     should_stop: Callable[[], bool] | None) -> int:
        added = 0
        for cand in connector.search(target, scan_limit):
            if should_stop and should_stop():
                break
            counters["candidates"] += 1
            key = cand.url[:1000]  # SeenUrl primary key is limited to 1000 chars
            with self.engine.db.session() as s:
                if s.get(SeenUrl, key) is not None:
                    counters["already_seen"] += 1
                    continue
            outcome, image_id = self._process(connector, target, cand, counters)
            if outcome != "fetch_error":
                with self.engine.db.session() as s:
                    s.merge(SeenUrl(url=key, outcome=outcome, image_id=image_id))
            counters[outcome] += 1
            if outcome == "added":
                added += 1
                if added >= add_limit:
                    break
        return added

    def _process(self, connector: SourceConnector, target: Target, cand: Any,
                 counters: Counter[str]) -> tuple[str, int | None]:
        guess = infer_label(cand.text, target, title=cand.title or None)
        if guess is None:
            return "label_mismatch", None
        if guess.interior_hint is False:
            return "not_interior_text", None
        if (not self.has_interior_classifier and guess.interior_hint is None
                and connector.name in UNCURATED_SOURCES):
            return "uncertain_interior", None
        try:
            data = connector.fetch(cand)
        except (FetchError, OSError) as exc:
            log.debug("fetch failed %s: %s", cand.url, exc)
            return "fetch_error", None
        label = ReferenceLabel(make_id=target.make_id, model_id=target.model_id,
                               confidence=guess.confidence)
        info = SourceInfo(source=connector.name, source_url=cand.url[:1000],
                          page_url=(cand.page_url or "")[:1000] or None, license=cand.license,
                          credit=cand.credit, title=cand.title)
        try:
            result = self.engine.ingest(data, label, info)
        except ImageDecodeError:
            return "invalid_image", None
        return result.status, result.image_id


def recheck_titles(engine: Engine, actor: str = "recheck") -> list[tuple[int, str]]:
    """Reject stored images whose title names the make but a different model."""
    from carid.collector.labeler import title_verdict
    from carid.models import Image, ReviewStatus

    with engine.db.session() as s:
        targets = {t.model_id: t for t in build_targets(s)}
        rows = s.execute(select(Image.id, Image.model_id, Image.title).where(
            Image.review_status != ReviewStatus.rejected, Image.title.is_not(None))).all()
    bad = [(iid, title) for iid, model_id, title in rows
           if title and model_id in targets
           and title_verdict(title, targets[model_id]) == "other"]
    for iid, _ in bad:
        engine.set_review_status(iid, ReviewStatus.rejected, actor)
    engine.save_index()
    return bad

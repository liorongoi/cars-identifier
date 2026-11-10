"""Wires the engine, collector and job handlers together."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from carid.bundle import import_bundle
from carid.collector.runner import (
    Collector,
    build_connectors,
    build_targets,
    load_sources_config,
    make_http,
)
from carid.config import Settings
from carid.engine import Engine
from carid.evaluation import run_benchmark
from carid.jobs import JobContext, JobManager
from carid.models import AuditEvent
from carid.taxonomy import upsert_make, wikidata_make_spec


class Services:
    def __init__(self, settings: Settings, engine: Engine | None = None) -> None:
        self.settings = settings
        self.engine = engine or Engine(settings)
        self.jobs = JobManager(self.engine.db)
        self.jobs.register("collect", self._job_collect)
        self.jobs.register("reindex", self._job_reindex)
        self.jobs.register("benchmark", self._job_benchmark)
        self.jobs.register("expand_make", self._job_expand_make)
        self.jobs.register("import_bundle", self._job_import_bundle)

    def sources_status(self) -> list[dict[str, Any]]:
        http = make_http(self.settings)
        try:
            config = load_sources_config(self.settings.sources_file)
            return [c.status() for c in build_connectors(config, http).values()]
        finally:
            http.close()

    # ---------------------------------------------------------------- job handlers

    def _job_collect(self, ctx: JobContext) -> dict[str, Any]:
        p = ctx.params
        with self.engine.db.session() as s:
            targets = build_targets(s, p.get("make_ids"), p.get("model_ids"))
        ctx.log(f"collecting for {len(targets)} model(s)")
        http = make_http(self.settings)
        try:
            config = load_sources_config(self.settings.sources_file)
            collector = Collector(self.engine, build_connectors(config, http))
            counters = collector.run(
                targets, per_model_limit=int(p.get("per_model_limit", 150)),
                sources=p.get("sources"), progress=ctx.progress, should_stop=ctx.should_stop,
                log_fn=ctx.log)
        finally:
            http.close()
        ctx.set_counters(counters)
        return counters

    def _job_reindex(self, ctx: JobContext) -> dict[str, Any]:
        n = self.engine.rebuild_index(progress=ctx.progress, should_stop=ctx.should_stop)
        ctx.log(f"re-indexed {n} images with {self.engine.embedder.name}")
        return {"indexed": n}

    def _job_benchmark(self, ctx: JobContext) -> dict[str, Any]:
        p = ctx.params
        report = run_benchmark(
            self.engine, test_fraction=float(p.get("test_fraction", 0.2)),
            max_test=int(p.get("max_test", 200)), degradations=p.get("degradations"),
            modes=p.get("modes"), seed=int(p.get("seed", 0)), progress=ctx.progress,
            should_stop=ctx.should_stop, job_id=ctx.job_id)
        ctx.log(f"best mode: {report['best_mode']} {report['summary']}")
        return {"best_mode": report["best_mode"], "test_images": report["test_images"]}

    def _job_expand_make(self, ctx: JobContext) -> dict[str, Any]:
        name = str(ctx.params["make"])
        http = make_http(self.settings)
        try:
            spec = wikidata_make_spec(name, http.client)
        finally:
            http.close()
        with self.engine.db.session() as s:
            make, models, gens = upsert_make(s, spec, added_by=ctx.params.get("added_by")
                                             or "wikidata")
            make_id = make.id
            s.add(AuditEvent(actor=ctx.params.get("added_by"), action="make.expand",
                             target=f"make:{make_id}",
                             details={"models_added": models, "generations_added": gens}))
        ctx.log(f"{name}: +{models} models, +{gens} generations from Wikidata")
        if ctx.params.get("collect"):
            job = self.jobs.submit("collect", {
                "make_ids": [make_id],
                "per_model_limit": ctx.params.get("per_model_limit", 150)},
                requested_by=ctx.params.get("added_by"))
            ctx.log(f"queued collection job #{job}")
        return {"models_added": models, "generations_added": gens}

    def _job_import_bundle(self, ctx: JobContext) -> dict[str, Any]:
        return import_bundle(self.engine, Path(ctx.params["path"]), progress=ctx.progress)

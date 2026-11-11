from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from carid.bundle import export_bundle, import_bundle
from carid.config import Settings
from carid.engine import Engine
from carid.evaluation import run_benchmark, split_holdout
from carid.jobs import JobContext, JobManager
from carid.matcher import Label
from carid.models import Benchmark, Job, JobStatus
from carid.services import Services


def _status(mgr: JobManager, job_id: int) -> Job:
    with mgr.db.session() as s:
        job = s.get(Job, job_id)
        assert job is not None
        return job


def test_job_lifecycle(engine: Engine) -> None:
    mgr = JobManager(engine.db)
    seen: list[dict[str, Any]] = []

    def ok(ctx: JobContext) -> dict[str, Any]:
        ctx.progress(0.5, "half")
        ctx.log("hello")
        seen.append(ctx.params)
        return {"x": 1}

    def boom(ctx: JobContext) -> None:
        raise RuntimeError("kaput")

    mgr.register("ok", ok)
    mgr.register("boom", boom)
    a = mgr.submit("ok", {"p": 1})
    b = mgr.submit("boom", {})
    c = mgr.submit("ok", {"p": 3})
    assert mgr.cancel(c)
    assert mgr.run_pending() == 2
    ja, jb, jc = _status(mgr, a), _status(mgr, b), _status(mgr, c)
    assert ja.status == JobStatus.succeeded and ja.progress == 1.0 and ja.counters == {"x": 1}
    assert "hello" in ja.log
    assert jb.status == JobStatus.failed and "kaput" in jb.message and "Traceback" in jb.log
    assert jc.status == JobStatus.cancelled and seen == [{"p": 1}]
    assert not mgr.cancel(a)
    with pytest.raises(ValueError):
        mgr.submit("nope", {})


def test_running_jobs_requeued_on_start(engine: Engine) -> None:
    mgr = JobManager(engine.db)
    mgr.register("ok", lambda ctx: None)
    job_id = mgr.submit("ok", {})
    with engine.db.session() as s:
        s.get(Job, job_id).status = JobStatus.running  # type: ignore[union-attr]
    mgr.start()
    try:
        for _ in range(100):
            if _status(mgr, job_id).status == JobStatus.succeeded:
                break
            import time

            time.sleep(0.05)
    finally:
        mgr.stop()
    assert _status(mgr, job_id).status == JobStatus.succeeded


def test_split_holdout_keeps_training_images() -> None:
    labels = {i: Label(i, 1, i % 3) for i in range(30)}
    labels[99] = Label(99, 1, 7)  # singleton class never tested
    test = split_holdout(labels, 0.5, 100, seed=1)
    assert 99 not in test
    for model in range(3):
        members = [i for i, lab in labels.items() if lab.model_id == model]
        assert 0 < len([i for i in members if i in test]) < len(members)
    assert split_holdout(labels, 0.5, 100, seed=1) == test
    assert len(split_holdout(labels, 0.5, 4, seed=1)) == 4


def test_benchmark_report(populated: Engine) -> None:
    report = run_benchmark(populated, test_fraction=0.34, max_test=8,
                           degradations=["low_light", "occlusion"], seed=0)
    assert report["test_images"] == 8
    assert report["conditions"] == ["clean", "low_light", "occlusion"]
    assert report["best_mode"] in ("fusion", "global", "parts")
    clean = report["results"]["clean"]["fusion"]
    assert clean["model"]["n"] == 8 and clean["model"]["top5"] >= 0.75
    assert clean["make"]["top1"] >= 0.6
    with populated.db.session() as s:
        assert s.query(Benchmark).count() == 1
    with pytest.raises(ValueError):
        run_benchmark(populated, degradations=["nope"])


def test_bundle_roundtrip(populated: Engine, tmp_path: Path) -> None:
    dest = tmp_path / "bundle.tar.gz"
    out = export_bundle(populated, dest)
    assert out["images"] == 24 and dest.stat().st_size > 0
    fresh_settings = Settings(
        data_dir=tmp_path / "offline", embedder="handcrafted", detector="layout",
        models_cache=tmp_path / "m", taxonomy_file=populated.settings.taxonomy_file,
        min_image_side=200, near_duplicate_distance=1)
    fresh = Engine(fresh_settings)
    counts = import_bundle(fresh, dest)
    assert counts["added"] + counts["skipped"] == 24 and counts["missing"] == 0
    assert counts["added"] >= 20
    assert fresh.index.sizes()["whole"] == counts["added"]
    again = import_bundle(fresh, dest)
    assert again["added"] == 0


def test_services_collect_job_uses_folder(settings: Settings, engine: Engine) -> None:
    svc = Services(settings, engine)
    statuses = {s["name"]: s for s in svc.sources_status()}
    assert statuses["folder"]["runnable"] and not statuses["wikimedia"]["runnable"]
    job = svc.jobs.submit("collect", {"per_model_limit": 5})
    svc.jobs.run_pending()
    j = _status(svc.jobs, job)
    assert j.status == JobStatus.succeeded, j.log
    reindex = svc.jobs.submit("reindex", {})
    svc.jobs.run_pending()
    assert _status(svc.jobs, reindex).status == JobStatus.succeeded


def test_split_holdout_never_splits_a_group() -> None:
    labels = {i: Label(i, 1, 1) for i in range(12)}
    groups = {i: f"photographer{i // 4}" for i in range(12)}  # 3 groups of 4
    for seed in range(5):
        test = set(split_holdout(labels, 0.4, 100, seed, groups=groups))
        for g in range(3):
            members = {i for i in range(12) if i // 4 == g}
            assert members <= test or not (members & test)
        assert len(test) in (4, 8)
    single = {i: f"same{i * 0}" for i in range(12)}
    assert split_holdout(labels, 0.5, 100, 0, groups=single) == []

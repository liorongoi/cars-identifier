"""Benchmark harness: hold-out evaluation of matcher modes under degradations (FR-EV)."""

from __future__ import annotations

import itertools
import math
import random
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import replace
from typing import Any

import numpy as np
from sqlalchemy import select

from carid.adapter import Adapter
from carid.engine import Engine
from carid.matcher import ClassScore, Label, MatchParams, Mode, rank
from carid.models import Benchmark
from carid.vision.degrade import DEGRADATIONS
from carid.vision.imageio import decode_image

ALL_MODES: tuple[Mode, ...] = ("fusion", "global", "parts")


def split_holdout(labels: dict[int, Label], test_fraction: float, max_test: int,
                  seed: int, groups: dict[int, str] | None = None) -> list[int]:
    """Pick test images per class, leaving at least one training group in every class.

    ``groups`` maps image id -> group (e.g. photographer). A group is never split
    between train and test: several shots of the same car by the same person would
    otherwise make the test look easier than real casework (FR-EV-1).
    """
    by_class: dict[tuple[int, int | None], dict[str, list[int]]] = defaultdict(
        lambda: defaultdict(list))
    for iid, lab in labels.items():
        g = (groups or {}).get(iid) or f"image:{iid}"
        by_class[(lab.model_id, lab.generation_id)][g].append(iid)
    rnd = random.Random(seed)
    chosen: list[list[int]] = []
    for class_groups in by_class.values():
        names = sorted(class_groups)
        if len(names) < 2:
            continue
        rnd.shuffle(names)
        total = sum(len(v) for v in class_groups.values())
        target = max(1, math.floor(total * test_fraction))
        taken = 0
        for name in names[:-1]:  # the last group always stays in training
            if taken >= target:
                break
            chosen.append(sorted(class_groups[name]))
            taken += len(class_groups[name])
    rnd.shuffle(chosen)
    test: list[int] = []
    for group_ids in chosen:  # cap by whole groups so no group is split
        if len(test) + len(group_ids) > max_test:
            continue
        test.extend(group_ids)
    return sorted(test)


def _rank_of(ranked: list[ClassScore], pred: Callable[[ClassScore], bool]) -> int | None:
    for i, c in enumerate(ranked, start=1):
        if pred(c):
            return i
    return None


def _make_rank(ranked: list[ClassScore], make_id: int) -> int | None:
    seen: list[int] = []
    for c in ranked:
        if c.make_id not in seen:
            seen.append(c.make_id)
        if c.make_id == make_id:
            return len(seen)
    return None


def _metrics(ranks: list[int | None]) -> dict[str, float]:
    n = len(ranks)
    if n == 0:
        return {"n": 0}
    def top(k: int) -> float:
        return round(sum(1 for r in ranks if r is not None and r <= k) / n, 4)
    mrr = sum(1.0 / r for r in ranks if r is not None) / n
    return {"n": n, "top1": top(1), "top5": top(5), "top10": top(10), "mrr": round(mrr, 4)}


def run_benchmark(engine: Engine, *, test_fraction: float = 0.2, max_test: int = 200,
                  degradations: list[str] | None = None, modes: list[Mode] | None = None,
                  seed: int = 0, compare_adapter: bool = True,
                  progress: Callable[[float, str], None] | None = None,
                  should_stop: Callable[[], bool] | None = None,
                  job_id: int | None = None) -> dict[str, Any]:
    labels = engine.labels()
    from carid.models import Image as _Image

    with engine.db.session() as s:
        groups = {iid: f"{credit}" for iid, credit in s.execute(
            select(_Image.id, _Image.credit).where(_Image.credit.is_not(None))).all()}
    test_ids = split_holdout(labels, test_fraction, max_test, seed, groups=groups)
    conditions = ["clean", *(degradations if degradations is not None else list(DEGRADATIONS))]
    unknown = [c for c in conditions if c != "clean" and c not in DEGRADATIONS]
    if unknown:
        raise ValueError(f"unknown degradations: {unknown}")
    modes = list(modes or ALL_MODES)
    train_labels = {k: v for k, v in labels.items() if k not in set(test_ids)}
    base = MatchParams(
        neighbors_per_part=engine.settings.neighbors_per_part,
        votes_per_class=engine.settings.votes_per_class,
        similarity_temperature=engine.settings.similarity_temperature,
        softmax_temperature=engine.settings.softmax_temperature,
        part_weights=engine.settings.part_weights,
    )
    # Variants: raw vectors, and (if enough data) an LDA adapter fitted on the
    # training split only, so test images never influence the adapter.
    variants: dict[str, dict[str, Adapter]] = {"": {}}
    if compare_adapter:
        fitted = engine.fit_adapters(exclude_image_ids=set(test_ids))
        if fitted:
            variants["+lda"] = fitted
    keys = [f"{m}{v}" for m in modes for v in variants]
    # ranks[condition][mode key][level] -> list of ranks
    ranks: dict[str, dict[str, dict[str, list[int | None]]]] = {
        c: {k: defaultdict(list) for k in keys} for c in conditions}
    latency: list[float] = []
    rng = np.random.default_rng(seed)
    total = max(len(test_ids) * len(conditions), 1)
    step = 0
    from carid.models import Image  # local import keeps module import light

    for iid in test_ids:
        if should_stop and should_stop():
            break
        with engine.db.session() as s:
            image = s.get(Image, iid)
            if image is None:
                continue
            path = engine.settings.images_dir / image.path
        if not path.exists():
            continue
        img = decode_image(path.read_bytes(), engine.settings.max_image_pixels)
        truth = labels[iid]
        for cond in conditions:
            step += 1
            q = img if cond == "clean" else DEGRADATIONS[cond](img, rng)
            t0 = time.perf_counter()
            analysis = engine.analyze(q)
            for mode, (suffix, adapters) in itertools.product(modes, variants.items()):
                params = replace(base, mode=mode, adapters=adapters)
                with engine.index.lock:
                    ranked = rank(analysis.vectors, engine.index, train_labels, params)
                r = ranks[cond][f"{mode}{suffix}"]
                r["make"].append(_make_rank(ranked, truth.make_id))
                r["model"].append(_rank_of(ranked, _model_pred(truth)))
                r["model_year"].append(_rank_of(ranked, _year_pred(truth)))
            latency.append(time.perf_counter() - t0)
            if progress:
                progress(step / total, f"image {iid} [{cond}]")

    results = {
        cond: {mode: {level: _metrics(v) for level, v in levels.items()}
               for mode, levels in by_mode.items()}
        for cond, by_mode in ranks.items()
    }
    summary: dict[str, dict[str, float]] = {}
    for key in keys:
        vals = [results[c][key]["model"].get("top5", 0.0) for c in conditions
                if results[c][key].get("model")]
        summary[key] = {"mean_model_top5": round(float(np.mean(vals)) if vals else 0.0, 4)}
    best = max(summary, key=lambda m: summary[m]["mean_model_top5"]) if summary else None
    report: dict[str, Any] = {
        "embedder": engine.embedder.name,
        "detector": engine.detector.name,
        "reference_images": len(labels),
        "test_images": len(test_ids),
        "classes": len({(lab.model_id, lab.generation_id) for lab in labels.values()}),
        "conditions": conditions,
        "modes": keys,
        "results": results,
        "summary": summary,
        "best_mode": best,
        "mean_latency_ms": round(1000 * float(np.mean(latency)), 1) if latency else None,
        "params": {"test_fraction": test_fraction, "max_test": max_test, "seed": seed},
    }
    with engine.db.session() as s:
        s.add(Benchmark(job_id=job_id, embedder=engine.embedder.name, report=report))
    return report


def _model_pred(truth: Label) -> Callable[[ClassScore], bool]:
    return lambda c: c.key[0] == truth.model_id


def _year_pred(truth: Label) -> Callable[[ClassScore], bool]:
    return lambda c: _year_ok(c, truth)


def _year_ok(c: ClassScore, truth: Label) -> bool:
    if c.key[0] != truth.model_id:
        return False
    if truth.generation_id is not None and c.key[1] is not None:
        return c.key[1] == truth.generation_id
    if truth.year_start is None or c.year_min is None or c.year_max is None:
        return True  # no year information to contradict
    return c.year_min <= (truth.year_end or truth.year_start) and truth.year_start <= c.year_max

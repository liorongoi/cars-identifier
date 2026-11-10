"""Fuse per-part nearest-neighbour evidence into ranked, calibrated candidates.

See docs/ALGORITHMS.md. Every reference neighbour votes for its label with
weight ``w_part * decay^i * exp((sim - best_sim) / tau)``; only the best
``votes_per_class`` neighbours of a class count per part, so classes with many
images do not win on volume alone.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from carid.adapter import Adapter
from carid.index import IndexSet
from carid.vision.embed import Vectors

Mode = Literal["fusion", "global", "parts"]
ClassKey = tuple[int, int | None]  # (model_id, generation_id)


@dataclass(frozen=True)
class Label:
    image_id: int
    make_id: int
    model_id: int
    generation_id: int | None
    year_start: int | None
    year_end: int | None


@dataclass
class Evidence:
    part: str
    image_id: int
    part_id: int
    similarity: float


@dataclass
class ClassScore:
    key: ClassKey
    make_id: int
    score: float
    probability: float = 0.0
    part_scores: dict[str, float] = field(default_factory=dict)
    evidence: list[Evidence] = field(default_factory=list)  # best hit per part
    year_min: int | None = None
    year_max: int | None = None


@dataclass
class MatchParams:
    mode: Mode = "fusion"
    neighbors_per_part: int = 50
    votes_per_class: int = 3
    similarity_temperature: float = 0.05
    softmax_temperature: float = 1.0
    part_weights: dict[str, float] = field(default_factory=dict)
    make_ids: set[int] | None = None
    year_min: int | None = None
    year_max: int | None = None
    # None: use the index's own adapters; a dict overrides them ({} = raw vectors).
    adapters: dict[str, Adapter] | None = None


def parts_for_mode(mode: Mode, available: list[str]) -> list[str]:
    if mode == "global":
        return [p for p in available if p == "whole"]
    if mode == "parts":
        return [p for p in available if p != "whole"]
    return list(available)


def _label_allowed(label: Label, params: MatchParams) -> bool:
    if params.make_ids is not None and label.make_id not in params.make_ids:
        return False
    if params.year_min is not None and label.year_end is not None \
            and label.year_end < params.year_min:
        return False
    return not (params.year_max is not None and label.year_start is not None
                and label.year_start > params.year_max)


def rank(
    query: dict[str, Vectors],
    index: IndexSet,
    labels: dict[int, Label],
    params: MatchParams,
) -> list[ClassScore]:
    """Rank all classes with any evidence; probabilities sum to 1 over returned classes."""
    scores: dict[ClassKey, ClassScore] = {}
    years: dict[ClassKey, list[tuple[int, int]]] = defaultdict(list)
    decay = [0.5**i for i in range(params.votes_per_class)]

    for part in parts_for_mode(params.mode, list(query)):
        idx = index.get(part)
        if idx is None:
            continue
        weight = params.part_weights.get(part, 1.0)
        if weight <= 0:
            continue
        override: Adapter | str | None = (
            "default" if params.adapters is None else params.adapters.get(part))
        part_ids, image_ids, sims = idx.search(query[part], params.neighbors_per_part, override)
        hits_by_class: dict[ClassKey, list[tuple[float, int, int, Label]]] = defaultdict(list)
        for pid, iid, sim in zip(part_ids.tolist(), image_ids.tolist(), sims.tolist(),
                                 strict=True):
            label = labels.get(iid)
            if label is None or not _label_allowed(label, params):
                continue
            key = (label.model_id, label.generation_id)
            if len(hits_by_class[key]) < params.votes_per_class:
                hits_by_class[key].append((float(sim), iid, pid, label))
        if not hits_by_class:
            continue
        best_sim = max(h[0][0] for h in hits_by_class.values())
        for key, hits in hits_by_class.items():
            part_score = weight * sum(
                d * math.exp((sim - best_sim) / params.similarity_temperature)
                for d, (sim, _, _, _) in zip(decay, hits, strict=False)
            )
            cs = scores.get(key)
            if cs is None:
                cs = scores[key] = ClassScore(key=key, make_id=hits[0][3].make_id, score=0.0)
            cs.score += part_score
            cs.part_scores[part] = round(part_score, 6)
            sim, iid, pid, _ = hits[0]
            cs.evidence.append(Evidence(part, iid, pid, round(sim, 4)))
            for _, _, _, lab in hits:
                if lab.year_start is not None:
                    years[key].append((lab.year_start, lab.year_end or lab.year_start))

    ranked = sorted(scores.values(), key=lambda c: -c.score)
    if not ranked:
        return []
    raw = np.array([c.score for c in ranked], dtype=np.float64)
    logits = np.log(np.maximum(raw, 1e-12)) / params.softmax_temperature
    probs = np.exp(logits - float(np.max(logits)))
    probs /= probs.sum()
    for cs, p in zip(ranked, probs.tolist(), strict=True):
        cs.probability = float(p)
        if years[cs.key]:
            cs.year_min = min(y[0] for y in years[cs.key])
            cs.year_max = max(y[1] for y in years[cs.key])
    return ranked


def make_probabilities(ranked: list[ClassScore]) -> dict[int, float]:
    out: dict[int, float] = defaultdict(float)
    for cs in ranked:
        out[cs.make_id] += cs.probability
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))

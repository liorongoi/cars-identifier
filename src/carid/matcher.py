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
ClassKey = int  # model_id: the system identifies make + model (not years)


@dataclass(frozen=True)
class Label:
    image_id: int
    make_id: int
    model_id: int


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


@dataclass
class MatchParams:
    mode: Mode = "fusion"
    neighbors_per_part: int = 50
    votes_per_class: int = 3
    similarity_temperature: float = 0.05
    softmax_temperature: float = 1.0
    part_weights: dict[str, float] = field(default_factory=dict)
    make_ids: set[int] | None = None
    # None: use the index's own adapters; a dict overrides them ({} = raw vectors).
    adapters: dict[str, Adapter] | None = None


def parts_for_mode(mode: Mode, available: list[str]) -> list[str]:
    if mode == "global":
        return [p for p in available if p == "whole"]
    if mode == "parts":
        return [p for p in available if p != "whole"]
    return list(available)


def _label_allowed(label: Label, params: MatchParams) -> bool:
    return params.make_ids is None or label.make_id in params.make_ids


def rank(
    query: dict[str, Vectors],
    index: IndexSet,
    labels: dict[int, Label],
    params: MatchParams,
) -> list[ClassScore]:
    """Rank all classes with any evidence; probabilities sum to 1 over returned classes."""
    scores: dict[ClassKey, ClassScore] = {}
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
            key = label.model_id
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

    ranked = sorted(scores.values(), key=lambda c: -c.score)
    if not ranked:
        return []
    raw = np.array([c.score for c in ranked], dtype=np.float64)
    logits = np.log(np.maximum(raw, 1e-12)) / params.softmax_temperature
    probs = np.exp(logits - float(np.max(logits)))
    probs /= probs.sum()
    for cs, p in zip(ranked, probs.tolist(), strict=True):
        cs.probability = float(p)
    return ranked


def make_probabilities(ranked: list[ClassScore]) -> dict[int, float]:
    out: dict[int, float] = defaultdict(float)
    for cs in ranked:
        out[cs.make_id] += cs.probability
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))

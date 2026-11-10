from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from carid.index import IndexSet, VectorIndex
from carid.matcher import Label, MatchParams, make_probabilities, parts_for_mode, rank
from carid.vision.embed import l2_normalize


def _vecs(n: int, dim: int, seed: int) -> np.ndarray:
    return l2_normalize(np.random.default_rng(seed).standard_normal((n, dim)))


def test_vector_index_search_remove_compact_and_persist(tmp_path: Path) -> None:
    idx = VectorIndex(8)
    v = _vecs(20, 8, 0)
    idx.add(np.arange(20), np.arange(100, 120), v)
    pids, iids, sims = idx.search(v[3], 5)
    assert pids[0] == 3 and iids[0] == 103 and sims[0] == pytest.approx(1.0, abs=1e-5)
    assert list(sims) == sorted(sims, reverse=True)
    idx.remove_images({103})
    pids, iids, _ = idx.search(v[3], 5)
    assert 103 not in iids and len(pids) == 5
    idx.save(tmp_path / "p.npz")
    loaded = VectorIndex.load(tmp_path / "p.npz")
    assert len(loaded) == 19
    with pytest.raises(ValueError):
        idx.add(np.arange(1), np.arange(1), _vecs(1, 4, 1))
    with pytest.raises(ValueError):
        idx.add(np.arange(2), np.arange(1), _vecs(2, 8, 1))
    assert VectorIndex(8).search(v[0], 3)[0].size == 0


def test_re_adding_removed_image_makes_it_visible_again() -> None:
    idx = VectorIndex(4)
    v = _vecs(2, 4, 3)
    idx.add(np.array([1]), np.array([10]), v[:1])
    idx.remove_images({10})
    assert idx.search(v[0], 1)[0].size == 0
    idx.add(np.array([2]), np.array([10]), v[:1])
    assert idx.search(v[0], 1)[1].tolist() == [10]


def test_index_set_roundtrip_and_dim_check(tmp_path: Path) -> None:
    s = IndexSet(tmp_path, "emb/x", 8)
    s.add("whole", np.arange(3), np.arange(3), _vecs(3, 8, 2))
    s.save()
    s2 = IndexSet(tmp_path, "emb/x", 8)
    s2.load()
    assert s2.sizes() == {"whole": 3}
    with pytest.raises(ValueError, match="rebuild"):
        IndexSet(tmp_path, "emb/x", 16).load()


def _setup() -> tuple[IndexSet, dict[int, Label], np.ndarray]:
    """Two classes; class A (model 1) has 10 near-identical images, B (model 2) has 2 exact."""
    dim = 16
    base = _vecs(2, dim, 5)
    idx = IndexSet(Path("/nonexistent"), "t", dim)
    labels: dict[int, Label] = {}
    vecs, iids = [], []
    for i in range(10):
        vecs.append(l2_normalize(base[0] + 0.3 * _vecs(1, dim, 10 + i)[0]))
        iids.append(i)
        labels[i] = Label(i, 1, 1, None, 2010, 2012)
    for j in range(2):
        vecs.append(base[1])
        iids.append(100 + j)
        labels[100 + j] = Label(100 + j, 2, 2, 7, 2015, 2015)
    arr = np.stack(vecs)
    for part in ("whole", "steering_wheel"):
        idx.add(part, np.arange(len(iids)) + (0 if part == "whole" else 1000), np.array(iids), arr)
    return idx, labels, base


def test_rank_prefers_best_match_not_volume() -> None:
    idx, labels, base = _setup()
    q = {"whole": base[1], "steering_wheel": base[1]}
    ranked = rank(q, idx, labels, MatchParams(votes_per_class=3))
    assert ranked[0].key == (2, 7)
    assert sum(c.probability for c in ranked) == pytest.approx(1.0)
    assert ranked[0].year_min == 2015 and ranked[0].year_max == 2015
    assert {e.part for e in ranked[0].evidence} == {"whole", "steering_wheel"}
    probs = make_probabilities(ranked)
    assert next(iter(probs)) == 2


def test_rank_filters_and_modes() -> None:
    idx, labels, base = _setup()
    q = {"whole": base[1], "steering_wheel": base[1]}
    only_make1 = rank(q, idx, labels, MatchParams(make_ids={1}))
    assert {c.make_id for c in only_make1} == {1}
    years = rank(q, idx, labels, MatchParams(year_min=2016))
    assert all(c.key[0] != 2 for c in years)
    years_max = rank(q, idx, labels, MatchParams(year_max=2009))
    assert years_max == []
    glob = rank(q, idx, labels, MatchParams(mode="global"))
    assert all(set(c.part_scores) == {"whole"} for c in glob)
    parts = rank(q, idx, labels, MatchParams(mode="parts"))
    assert all("whole" not in c.part_scores for c in parts)
    zero_w = rank(q, idx, labels, MatchParams(part_weights={"whole": 0, "steering_wheel": 0}))
    assert zero_w == []
    assert parts_for_mode("fusion", ["whole", "x"]) == ["whole", "x"]


def test_rank_ignores_unlabelled_hits() -> None:
    idx, labels, base = _setup()
    labels = {k: v for k, v in labels.items() if v.model_id == 1}
    ranked = rank({"whole": base[1]}, idx, labels, MatchParams())
    assert [c.key[0] for c in ranked] == [1]


def test_softmax_temperature_sharpens() -> None:
    idx, labels, base = _setup()
    q = {"whole": l2_normalize(base[0] + base[1])}
    soft = rank(q, idx, labels, MatchParams(softmax_temperature=4.0))
    sharp = rank(q, idx, labels, MatchParams(softmax_temperature=0.25))
    assert sharp[0].probability >= soft[0].probability

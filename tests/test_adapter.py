from __future__ import annotations

from pathlib import Path

import numpy as np

from carid.adapter import fit_lda, load_adapters, save_adapters
from carid.engine import Engine
from carid.evaluation import run_benchmark
from carid.vision.embed import l2_normalize


def _data(seed: int = 0) -> tuple[np.ndarray, list[object]]:
    """Classes that differ in 2 dims, with large shared 'lighting' noise in others."""
    rng = np.random.default_rng(seed)
    d = 32
    centres = rng.standard_normal((6, d)) * 0.0
    centres[:, :2] = rng.standard_normal((6, 2)) * 0.6
    x, y = [], []
    for c in range(6):
        for _ in range(8):
            v = centres[c].copy()
            v[2:] += rng.standard_normal(d - 2) * 1.0  # nuisance variation
            v[:2] += rng.standard_normal(2) * 0.05
            x.append(v)
            y.append(c)
    return l2_normalize(np.array(x)), y


def _nn_accuracy(x: np.ndarray, y: list[object]) -> float:
    sims = x @ x.T
    np.fill_diagonal(sims, -9)
    nn = sims.argmax(axis=1)
    return float(np.mean([y[i] == y[j] for i, j in enumerate(nn)]))


def test_lda_improves_nearest_neighbour_on_held_out_data() -> None:
    x_train, y_train = _data(0)
    x_test, y_test = _data(0)  # same class centres, fresh noise below
    rng = np.random.default_rng(99)
    x_test = l2_normalize(x_test + rng.standard_normal(x_test.shape) * 0.02)
    adapter = fit_lda(x_train, y_train, shrinkage=0.3)
    assert adapter is not None
    raw = _nn_accuracy(x_test, y_test)
    adapted = _nn_accuracy(adapter.transform(x_test), y_test)
    assert adapted > raw + 0.2, (raw, adapted)
    out = adapter.transform(x_test)
    np.testing.assert_allclose(np.linalg.norm(out, axis=1), 1.0, rtol=1e-4)


def test_lda_needs_two_classes_with_two_samples() -> None:
    x = l2_normalize(np.random.default_rng(1).standard_normal((3, 8)))
    assert fit_lda(x, ["a", "b", "c"]) is None
    assert fit_lda(x, ["a", "a", "b"]) is None


def test_adapter_save_load(tmp_path: Path) -> None:
    x, y = _data(2)
    a = fit_lda(x, y)
    assert a is not None
    save_adapters(tmp_path / "_adapters.npz", {"whole": a})
    loaded = load_adapters(tmp_path / "_adapters.npz")
    np.testing.assert_allclose(loaded["whole"].transform(x), a.transform(x), atol=1e-5)
    assert load_adapters(tmp_path / "missing.npz") == {}


def test_engine_train_adapters_persist_and_apply(populated: Engine) -> None:
    result = populated.train_adapters()
    assert "whole" in result["parts"]
    assert populated.describe()["adapter"] == result["parts"]
    assert populated.index.adapters_path.exists()
    reopened = Engine(populated.settings, db=populated.db)
    assert sorted(reopened.index.adapters) == result["parts"]
    assert reopened.index.sizes() == populated.index.sizes()  # adapter file is not a part


def test_benchmark_compares_raw_and_adapted(populated: Engine) -> None:
    report = run_benchmark(populated, test_fraction=0.34, max_test=8, degradations=[], seed=1)
    assert set(report["modes"]) == {"fusion", "fusion+lda", "global", "global+lda",
                                    "parts", "parts+lda"}
    assert report["results"]["clean"]["fusion+lda"]["model"]["n"] == 8

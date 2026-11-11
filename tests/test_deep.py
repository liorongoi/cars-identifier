"""Deep-model smoke tests; skipped unless the `ml` extra and cached weights are present."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

CACHE = Path(__file__).resolve().parents[1] / "models_cache"
pytest.importorskip("open_clip")
pytest.importorskip("transformers")
if not (CACHE / "open_clip").exists():
    pytest.skip("deep model weights not cached (run `carid fetch-models`)", allow_module_level=True)

from carid.vision.deep import ClipEmbedder, OwlPartDetector  # noqa: E402
from tests.conftest import CLASS_STYLES, draw_interior  # noqa: E402


def test_clip_embedder_offline() -> None:
    emb = ClipEmbedder("ViT-B-32", "laion2b_s34b_b79k", str(CACHE), allow_download=False)
    img = draw_interior(next(iter(CLASS_STYLES.values())), np.random.default_rng(0))
    v = emb.embed([img, img])
    assert v.shape == (2, emb.dim)
    np.testing.assert_allclose(np.linalg.norm(v, axis=1), 1.0, rtol=1e-4)
    assert emb.interior_score(img) > 0.5
    assert emb.interior_score(np.full((224, 224, 3), 255, np.uint8)) < 0.9


def test_owl_detector_offline_returns_all_parts() -> None:
    det = OwlPartDetector("google/owlv2-base-patch16-ensemble", str(CACHE), allow_download=False)
    img = draw_interior(next(iter(CLASS_STYLES.values())), np.random.default_rng(0))
    parts = {p.part for p in det.detect(img)}
    assert {"whole", "steering_wheel", "infotainment", "dashboard"} <= parts

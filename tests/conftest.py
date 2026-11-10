from __future__ import annotations

import io
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from carid.config import Settings
from carid.engine import Engine, ReferenceLabel, SourceInfo
from carid.models import ReviewStatus
from carid.taxonomy import find_make, find_model, seed

SMALL_TAXONOMY = """
makes:
  - name: Alpha
    country: Testland
    aliases: [ALF]
    models:
      - {name: Roadster, generations: [[R1, 2001, 2008], [R2, 2008, 2016], [R3, 2016, null]]}
      - {name: Wagon, aliases: [Estate], generations: [[W1, 2005, 2014], [W2, 2014, null]]}
  - name: Beta
    country: Testland
    models:
      - {name: City, generations: [[C1, 1999, 2010], [C2, 2010, null]]}
      - {name: Cruiser, generations: [[X1, 2012, null]]}
"""

# Visual "design language" per class: distinct dashboards the synthetic generator draws.
CLASS_STYLES: dict[str, dict[str, object]] = {
    "Alpha/Roadster/2010": {"dash": (60, 60, 70), "spokes": 3, "screen": (0.55, 0.2, 0.75, 0.4),
                            "vent": "round", "accent": (200, 30, 30)},
    "Alpha/Wagon/2016": {"dash": (150, 130, 100), "spokes": 2, "screen": (0.45, 0.15, 0.85, 0.45),
                         "vent": "slot", "accent": (30, 30, 200)},
    "Beta/City/2012": {"dash": (30, 30, 30), "spokes": 4, "screen": (0.5, 0.3, 0.65, 0.42),
                       "vent": "square", "accent": (240, 200, 0)},
    "Beta/Cruiser/2018": {"dash": (200, 200, 205), "spokes": 3, "screen": (0.4, 0.1, 0.9, 0.3),
                          "vent": "slot", "accent": (0, 160, 80)},
}


def draw_interior(style: dict[str, object], rng: np.random.Generator,
                  size: tuple[int, int] = (800, 600)) -> np.ndarray:
    w, h = size
    dx, dy = (rng.uniform(-0.03, 0.03, 2) * [w, h]).astype(int)
    img = np.zeros((h, w, 3), np.uint8)
    img[:] = (90, 110, 140)  # windscreen / background
    dash = tuple(int(c) for c in style["dash"])  # type: ignore[attr-defined]
    accent = tuple(int(c) for c in style["accent"])  # type: ignore[attr-defined]
    cv2.rectangle(img, (0, int(h * 0.3) + dy), (w, h), dash, -1)
    # instrument cluster hood
    cv2.ellipse(img, (int(w * 0.3) + dx, int(h * 0.38) + dy), (int(w * 0.14), int(h * 0.09)),
                0, 180, 360, tuple(max(0, c - 25) for c in dash), -1)
    for gx in (0.25, 0.35):
        cv2.circle(img, (int(w * gx) + dx, int(h * 0.4) + dy), int(h * 0.05), accent, 3)
    # steering wheel with spokes
    cx, cy, r = int(w * 0.3) + dx, int(h * 0.62) + dy, int(h * 0.22)
    cv2.circle(img, (cx, cy), r, (20, 20, 20), 14)
    spokes = int(style["spokes"])  # type: ignore[call-overload]
    for k in range(spokes):
        ang = np.pi / 2 + 2 * np.pi * k / spokes
        cv2.line(img, (cx, cy), (int(cx + r * np.cos(ang)), int(cy + r * np.sin(ang))),
                 (25, 25, 25), 16)
    cv2.circle(img, (cx, cy), int(r * 0.25), accent, -1)
    # infotainment screen
    sx0, sy0, sx1, sy1 = style["screen"]  # type: ignore[misc]
    cv2.rectangle(img, (int(w * sx0) + dx, int(h * sy0) + int(h * 0.25) + dy),
                  (int(w * sx1) + dx, int(h * sy1) + int(h * 0.25) + dy), (10, 15, 25), -1)
    cv2.rectangle(img, (int(w * sx0) + dx + 6, int(h * sy0) + int(h * 0.25) + dy + 6),
                  (int(w * (sx0 + sx1) / 2) + dx, int(h * sy1) + int(h * 0.25) + dy - 6),
                  accent, -1)
    # vents
    for vx in (0.5, 0.88):
        c = (int(w * vx) + dx, int(h * 0.38) + dy)
        if style["vent"] == "round":
            cv2.circle(img, c, int(h * 0.045), (15, 15, 15), -1)
        elif style["vent"] == "square":
            cv2.rectangle(img, (c[0] - 25, c[1] - 25), (c[0] + 25, c[1] + 25), (15, 15, 15), -1)
        else:
            for k in range(4):
                cv2.line(img, (c[0] - 45, c[1] - 12 + 8 * k), (c[0] + 45, c[1] - 12 + 8 * k),
                         (15, 15, 15), 3)
    # console + gear knob
    cv2.rectangle(img, (int(w * 0.55) + dx, int(h * 0.75) + dy), (int(w * 0.75) + dx, h),
                  tuple(max(0, c - 15) for c in dash), -1)
    cv2.circle(img, (int(w * 0.65) + dx, int(h * 0.85) + dy), 18, accent, -1)
    # camera variation: viewpoint (perspective) and zoom
    jitter = rng.uniform(-0.09, 0.09, (4, 2)) * [w, h]
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32(src + jitter)
    img = cv2.warpPerspective(img, cv2.getPerspectiveTransform(src, dst), (w, h),
                              borderMode=cv2.BORDER_REPLICATE)
    # photographic variation: brightness, texture noise
    gain = rng.uniform(0.85, 1.15)
    noisy = img.astype(np.float32) * gain + rng.normal(0, 4, img.shape)
    return np.clip(noisy, 0, 255).astype(np.uint8)


def to_jpeg(img: np.ndarray, quality: int = 92) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    tax = tmp_path / "taxonomy.yaml"
    tax.write_text(SMALL_TAXONOMY, encoding="utf-8")
    sources = tmp_path / "sources.yaml"
    sources.write_text("folder:\n  enabled: true\n  path: " + str(tmp_path / "imports") + "\n",
                       encoding="utf-8")
    return Settings(
        data_dir=tmp_path / "var", embedder="handcrafted", detector="layout",
        models_cache=tmp_path / "models", taxonomy_file=tax, sources_file=sources,
        min_request_interval_s=0.0, min_image_side=200, ui_dir=tmp_path / "no-ui",
        near_duplicate_distance=1,
    )


@pytest.fixture
def engine(settings: Settings) -> Iterator[Engine]:
    eng = Engine(settings)
    with eng.db.session() as s:
        seed(s, settings.taxonomy_file)
    yield eng
    eng.db.engine.dispose()


def label_for(engine: Engine, key: str) -> ReferenceLabel:
    make_name, model_name, year = key.split("/")
    with engine.db.session() as s:
        make = find_make(s, make_name)
        assert make is not None
        model = find_model(make, model_name)
        assert model is not None
        return ReferenceLabel(make.id, model.id, int(year), int(year), None, 0.9)


@pytest.fixture
def populated(engine: Engine) -> Engine:
    """Engine with 6 synthetic reference images per class."""
    rng = np.random.default_rng(7)
    for key, style in CLASS_STYLES.items():
        label = label_for(engine, key)
        for i in range(6):
            img = draw_interior(style, rng)
            res = engine.ingest(to_jpeg(img), label,
                                SourceInfo("test", source_url=f"mem://{key}/{i}"),
                                interior_score=0.9, review_status=ReviewStatus.approved)
            assert res.status in ("added", "near_duplicate"), res
    engine.save_index()
    return engine

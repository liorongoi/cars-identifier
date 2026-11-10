from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

from carid.vision.degrade import DEGRADATIONS
from carid.vision.embed import HandcraftedEmbedder
from carid.vision.enhance import clahe_low_light, enhance
from carid.vision.faces import FaceDetector, mask_regions
from carid.vision.imageio import (
    ImageDecodeError,
    crop,
    decode_image,
    perceptual_hash,
    phash_distance,
    resize_max_side,
)
from carid.vision.parts import LayoutPartDetector
from carid.vision.quality import assess_quality
from tests.conftest import CLASS_STYLES, draw_interior, to_jpeg

STYLE = next(iter(CLASS_STYLES.values()))


def test_decode_roundtrip_and_validation() -> None:
    img = draw_interior(STYLE, np.random.default_rng(0))
    out = decode_image(to_jpeg(img))
    assert out.shape == img.shape and out.dtype == np.uint8
    with pytest.raises(ImageDecodeError):
        decode_image(b"")
    with pytest.raises(ImageDecodeError):
        decode_image(b"not an image at all")
    with pytest.raises(ImageDecodeError, match="too large"):
        decode_image(to_jpeg(img), max_pixels=1000)


def test_decode_applies_exif_orientation() -> None:
    im = Image.new("RGB", (200, 100), (255, 0, 0))
    exif = im.getexif()
    exif[0x0112] = 6  # rotate 90° CW
    buf = io.BytesIO()
    im.save(buf, format="JPEG", exif=exif)
    assert decode_image(buf.getvalue()).shape[:2] == (200, 100)


def test_crop_and_resize() -> None:
    img = np.zeros((100, 200, 3), np.uint8)
    assert crop(img, (10, 10, 50, 40)).shape == (30, 40, 3)
    assert crop(img, (-5, -5, 500, 500)).shape == (100, 200, 3)
    with pytest.raises(ValueError):
        crop(img, (50, 50, 50, 60))
    assert resize_max_side(img, 100).shape[:2] == (50, 100)
    assert resize_max_side(img, 1000) is img


def test_perceptual_hash_distance() -> None:
    rng = np.random.default_rng(1)
    a = draw_interior(STYLE, rng)
    b = (a.astype(np.int16) + 3).clip(0, 255).astype(np.uint8)
    other = draw_interior(list(CLASS_STYLES.values())[2], rng)
    assert phash_distance(perceptual_hash(a), perceptual_hash(b)) <= 4
    assert phash_distance(perceptual_hash(a), perceptual_hash(other)) > 6


def test_quality_flags_dark_and_blurry() -> None:
    rng = np.random.default_rng(2)
    img = draw_interior(STYLE, rng)
    good = assess_quality(img)
    dark = assess_quality(DEGRADATIONS["low_light"](img, rng))
    blurry = assess_quality(DEGRADATIONS["blur"](img, rng))
    assert dark.is_low_light
    assert {"low light", "very dark", "nearly black"} & set(dark.issues)
    assert dark.score < good.score
    assert blurry.sharpness < good.sharpness
    assert good.verdict in ("good", "fair")


def test_enhance_brightens_dark_images() -> None:
    rng = np.random.default_rng(3)
    dark = DEGRADATIONS["low_light"](draw_interior(STYLE, rng), rng)
    out, steps = enhance(dark, assess_quality(dark))
    assert "low-light" in steps
    assert out.mean() > dark.mean() * 1.5
    assert clahe_low_light(dark).shape == dark.shape


def test_mask_regions_fill_and_blur() -> None:
    img = np.full((100, 100, 3), 200, np.uint8)
    img[10:30, 10:30] = 0
    filled = mask_regions(img, [(10, 10, 30, 30)], mode="fill")
    assert filled[20, 20].min() > 100
    blurred = mask_regions(img, [(10, 10, 30, 30)], mode="blur")
    assert blurred.shape == img.shape
    assert mask_regions(img, []) is img


def test_face_detector_runs_without_false_positives_on_dashboard() -> None:
    det = FaceDetector()
    assert det.backend == "haar"
    assert det.detect(draw_interior(STYLE, np.random.default_rng(4))) == []


def test_layout_detector_returns_parts_within_image() -> None:
    img = draw_interior(STYLE, np.random.default_rng(5))
    parts = LayoutPartDetector().detect(img)
    names = {p.part for p in parts}
    assert {"whole", "steering_wheel", "infotainment", "dashboard"} <= names
    for p in parts:
        x0, y0, x1, y1 = p.bbox
        assert 0 <= x0 < x1 <= img.shape[1] and 0 <= y0 < y1 <= img.shape[0]


def test_handcrafted_embedder_is_normalised_and_discriminative() -> None:
    emb = HandcraftedEmbedder()
    rng = np.random.default_rng(6)
    styles = list(CLASS_STYLES.values())
    a1, a2 = draw_interior(styles[0], rng), draw_interior(styles[0], rng)
    b = draw_interior(styles[2], rng)
    v = emb.embed([a1, a2, b])
    assert v.shape == (3, emb.dim)
    np.testing.assert_allclose(np.linalg.norm(v, axis=1), 1.0, rtol=1e-5)
    assert v[0] @ v[1] > v[0] @ v[2]
    assert emb.embed([]).shape == (0, emb.dim)
    np.testing.assert_allclose(HandcraftedEmbedder().embed([a1]), v[:1], atol=1e-6)


@pytest.mark.parametrize("name", sorted(DEGRADATIONS))
def test_degradations_keep_shape(name: str) -> None:
    img = draw_interior(STYLE, np.random.default_rng(8), size=(320, 240))
    out = DEGRADATIONS[name](img, np.random.default_rng(9))
    assert out.shape == img.shape and out.dtype == np.uint8

"""Synthetic degradations used by the benchmark to measure robustness (FR-EV-2)."""

from __future__ import annotations

from collections.abc import Callable

import cv2
import numpy as np

from carid.vision.imageio import RGBImage


def gaussian_blur(img: RGBImage, rng: np.random.Generator) -> RGBImage:
    sigma = max(img.shape[:2]) / 300 * rng.uniform(1.5, 3.0)
    return np.asarray(cv2.GaussianBlur(img, (0, 0), sigma), dtype=np.uint8)


def motion_blur(img: RGBImage, rng: np.random.Generator) -> RGBImage:
    k = max(5, int(max(img.shape[:2]) / 60 * rng.uniform(0.8, 1.6)))
    kernel = np.zeros((k, k), np.float32)
    kernel[k // 2, :] = 1.0 / k
    rot = cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), float(rng.uniform(0, 180)), 1.0)
    kernel = np.asarray(cv2.warpAffine(kernel, rot, (k, k)), dtype=np.float32)
    kernel /= max(float(kernel.sum()), 1e-6)
    return np.asarray(cv2.filter2D(img, -1, kernel), dtype=np.uint8)


def low_light(img: RGBImage, rng: np.random.Generator) -> RGBImage:
    gain = rng.uniform(0.15, 0.3)
    dark = (img.astype(np.float32) / 255.0) ** 1.6 * gain * 255.0
    noise = rng.normal(0, 4.0, img.shape)
    return np.clip(dark + noise, 0, 255).astype(np.uint8)


def sensor_noise(img: RGBImage, rng: np.random.Generator) -> RGBImage:
    noise = rng.normal(0, rng.uniform(15, 25), img.shape)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def jpeg_artifacts(img: RGBImage, rng: np.random.Generator) -> RGBImage:
    q = int(rng.integers(8, 18))
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                           [cv2.IMWRITE_JPEG_QUALITY, q])
    assert ok
    decoded = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    assert decoded is not None
    out = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
    return np.asarray(out, dtype=np.uint8)


def occlusion(img: RGBImage, rng: np.random.Generator) -> RGBImage:
    """Skin-toned ellipses + a dark blob, imitating people/hands in front of the dash."""
    out = img.copy()
    h, w = img.shape[:2]
    for _ in range(int(rng.integers(1, 3))):
        center = (int(rng.uniform(0.1, 0.9) * w), int(rng.uniform(0.2, 0.9) * h))
        axes = (int(w * rng.uniform(0.07, 0.14)), int(h * rng.uniform(0.12, 0.22)))
        tone = tuple(int(c) for c in rng.integers([150, 100, 80], [230, 180, 150]))
        cv2.ellipse(out, center, axes, 0, 0, 360, tone, -1)
    x0, y0 = int(rng.uniform(0, 0.7) * w), int(rng.uniform(0.4, 0.8) * h)
    cv2.rectangle(out, (x0, y0), (x0 + w // 4, h), (25, 25, 30), -1)
    return out


def closeup(img: RGBImage, rng: np.random.Generator) -> RGBImage:
    """A photo of a small item: a random 25-45% (of each side) region, enlarged."""
    h, w = img.shape[:2]
    frac = rng.uniform(0.25, 0.45)
    ch, cw = int(h * frac), int(w * frac)
    y = int(rng.integers(0, h - ch + 1))
    x = int(rng.integers(0, w - cw + 1))
    out = cv2.resize(img[y:y + ch, x:x + cw], (w, h), interpolation=cv2.INTER_CUBIC)
    return np.asarray(out, dtype=np.uint8)


DEGRADATIONS: dict[str, Callable[[RGBImage, np.random.Generator], RGBImage]] = {
    "blur": gaussian_blur,
    "motion_blur": motion_blur,
    "low_light": low_light,
    "noise": sensor_noise,
    "jpeg": jpeg_artifacts,
    "occlusion": occlusion,
    "closeup": closeup,
}

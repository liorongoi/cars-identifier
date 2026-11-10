"""Enhancement for poor queries: low light, noise, mild blur."""

from __future__ import annotations

import cv2
import numpy as np

from carid.vision.imageio import RGBImage
from carid.vision.quality import QualityReport


def clahe_low_light(img: RGBImage, clip_limit: float = 2.5) -> RGBImage:
    """Brighten shadows with CLAHE on the L channel plus an adaptive gamma."""
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l_chan, a_chan, b_chan = cv2.split(lab)
    mean = float(l_chan.mean())
    if mean < 110:
        gamma = float(np.clip(np.log(0.45) / np.log(max(mean, 5.0) / 255.0), 0.35, 1.0))
        lut = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)], dtype=np.uint8)
        l_chan = cv2.LUT(l_chan, lut)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    l_chan = clahe.apply(l_chan)
    out = cv2.cvtColor(cv2.merge((l_chan, a_chan, b_chan)), cv2.COLOR_LAB2RGB)
    return np.asarray(out, dtype=np.uint8)


def denoise(img: RGBImage, strength: float) -> RGBImage:
    h = float(np.clip(strength, 3, 15))
    out = cv2.fastNlMeansDenoisingColored(img, None, h, h, 7, 21)
    return np.asarray(out, dtype=np.uint8)


def unsharp(img: RGBImage, amount: float = 0.8, sigma: float = 1.5) -> RGBImage:
    blurred = cv2.GaussianBlur(img, (0, 0), sigma)
    out = cv2.addWeighted(img, 1 + amount, blurred, -amount, 0)
    return np.asarray(out, dtype=np.uint8)


def enhance(img: RGBImage, quality: QualityReport) -> tuple[RGBImage, list[str]]:
    """Apply only the corrections the quality report calls for. Returns steps applied."""
    steps: list[str] = []
    out = img
    if quality.is_noisy:
        out = denoise(out, quality.noise * 1.2)
        steps.append("denoise")
    if quality.is_low_light or quality.contrast < 25:
        out = clahe_low_light(out)
        steps.append("low-light")
    if 20 <= quality.sharpness < 80:
        out = unsharp(out)
        steps.append("sharpen")
    return out, steps

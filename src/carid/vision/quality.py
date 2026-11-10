"""Image quality assessment (sharpness, exposure, contrast, noise)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import cv2
import numpy as np

from carid.vision.imageio import RGBImage, resize_max_side


@dataclass
class QualityReport:
    width: int
    height: int
    sharpness: float  # variance of Laplacian at 1024 px; < 60 is blurry
    brightness: float  # mean luminance 0-255
    contrast: float  # std of luminance
    noise: float  # robust noise sigma estimate
    dark_fraction: float  # share of pixels with luminance < 30
    score: float  # 0-100 overall
    verdict: str  # good | fair | poor
    issues: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def is_low_light(self) -> bool:
        return self.brightness < 70 or self.dark_fraction > 0.45

    @property
    def is_noisy(self) -> bool:
        return self.noise > 8.0


def _noise_sigma(gray: np.ndarray) -> float:
    """Immerkær's fast noise variance estimate."""
    kernel = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float32)
    conv = cv2.filter2D(gray.astype(np.float32), -1, kernel)
    h, w = gray.shape
    return float(np.sum(np.abs(conv)) * np.sqrt(0.5 * np.pi) / (6 * (w - 2) * (h - 2)))


def assess_quality(img: RGBImage) -> QualityReport:
    h, w = img.shape[:2]
    small = resize_max_side(img, 1024)
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    # Light median filter first so sensor noise is not mistaken for detail.
    sharpness = float(cv2.Laplacian(cv2.medianBlur(gray, 3), cv2.CV_64F).var())
    brightness = float(gray.mean())
    contrast = float(gray.std())
    noise = _noise_sigma(gray)
    dark_fraction = float((gray < 30).mean())

    issues: list[str] = []
    score = 100.0
    if sharpness < 30:
        issues.append("very blurry")
        score -= 35
    elif sharpness < 80:
        issues.append("slightly blurry")
        score -= 15
    if brightness < 25:
        issues.append("nearly black")
        score -= 55
    elif brightness < 50 or dark_fraction > 0.6:
        issues.append("very dark")
        score -= 35
    elif brightness < 80 or dark_fraction > 0.4:
        issues.append("low light")
        score -= 12
    if brightness > 215:
        issues.append("overexposed")
        score -= 20
    if contrast < 25:
        issues.append("low contrast")
        score -= 12
    if noise > 12:
        issues.append("very noisy")
        score -= 20
    elif noise > 7:
        issues.append("noisy")
        score -= 8
    if min(h, w) < 400:
        issues.append("low resolution")
        score -= 15

    score = float(max(0.0, min(100.0, score)))
    verdict = "good" if score >= 75 else "fair" if score >= 45 else "poor"
    return QualityReport(
        width=w,
        height=h,
        sharpness=round(sharpness, 1),
        brightness=round(brightness, 1),
        contrast=round(contrast, 1),
        noise=round(noise, 2),
        dark_fraction=round(dark_fraction, 3),
        score=round(score, 1),
        verdict=verdict,
        issues=issues,
    )

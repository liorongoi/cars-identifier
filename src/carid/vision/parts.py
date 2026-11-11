"""Interior part detection.

Two detectors share one interface:

* ``OwlPartDetector`` (in ``deep.py``): open-vocabulary detector prompted with part
  names — accurate, needs the ``ml`` extra.
* ``LayoutPartDetector``: geometric prior of where parts sit in a typical
  interior photo, refined with a circle search for the steering wheel. Always
  available; reference and query images go through the same detector, so its
  crops act as consistent regional descriptors.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import cv2
import numpy as np

from carid.vision.imageio import RGBImage, resize_max_side


@dataclass(frozen=True)
class PartBox:
    part: str
    bbox: tuple[int, int, int, int]  # x0, y0, x1, y1
    score: float
    method: str

    def to_dict(self) -> dict[str, object]:
        return {"part": self.part, "bbox": list(self.bbox), "score": round(self.score, 3),
                "method": self.method}


class PartDetector(Protocol):
    name: str

    def detect(self, img: RGBImage) -> list[PartBox]: ...


# Relative boxes (x0, y0, x1, y1) for a typical front-cabin photo.
LAYOUT_PRIOR: dict[str, tuple[float, float, float, float]] = {
    "dashboard": (0.0, 0.05, 1.0, 0.70),
    "steering_wheel": (0.08, 0.22, 0.55, 0.88),
    "instrument_cluster": (0.15, 0.15, 0.50, 0.50),
    "infotainment": (0.38, 0.12, 0.78, 0.55),
    "air_vent": (0.36, 0.08, 0.74, 0.38),
    "climate_control": (0.38, 0.45, 0.76, 0.75),
    "center_console": (0.36, 0.52, 0.82, 1.0),
}


def whole_box(img: RGBImage) -> PartBox:
    h, w = img.shape[:2]
    return PartBox("whole", (0, 0, w, h), 1.0, "full-frame")


def _rel_to_abs(rel: tuple[float, float, float, float], w: int,
                h: int) -> tuple[int, int, int, int]:
    return (int(rel[0] * w), int(rel[1] * h), int(rel[2] * w), int(rel[3] * h))


class LayoutPartDetector:
    name = "layout"

    def detect(self, img: RGBImage) -> list[PartBox]:
        h, w = img.shape[:2]
        boxes = [whole_box(img)]
        wheel = self._find_wheel(img)
        for part, rel in LAYOUT_PRIOR.items():
            if part == "steering_wheel" and wheel is not None:
                boxes.append(wheel)
                continue
            boxes.append(PartBox(part, _rel_to_abs(rel, w, h), 0.2, "layout-prior"))
        return boxes

    @staticmethod
    def _find_wheel(img: RGBImage) -> PartBox | None:
        """Look for a large circle in the lower-left/centre area (steering wheel rim)."""
        small = resize_max_side(img, 640)
        scale = img.shape[1] / small.shape[1]
        gray = cv2.medianBlur(cv2.cvtColor(small, cv2.COLOR_RGB2GRAY), 5)
        sh, sw = gray.shape
        m = min(sh, sw)
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, dp=1.5, minDist=m / 2, param1=120, param2=60,
            minRadius=int(m * 0.18), maxRadius=int(m * 0.48),
        )
        if circles is None:
            return None
        for x, y, r in np.asarray(circles)[0]:
            if 0.05 * sw < x < 0.75 * sw and 0.3 * sh < y < 1.1 * sh:
                x0, y0 = max(0, int((x - r) * scale)), max(0, int((y - r) * scale))
                x1 = min(img.shape[1], int((x + r) * scale))
                y1 = min(img.shape[0], int((y + r) * scale))
                if x1 - x0 > 16 and y1 - y0 > 16:
                    return PartBox("steering_wheel", (x0, y0, x1, y1), 0.45, "circle")
        return None


def get_part_detector(kind: str, owl_model: str, models_cache: str,
                      allow_download: bool) -> PartDetector:
    """Build the configured detector; ``auto`` falls back to layout if deep libs are missing."""
    if kind in ("auto", "owl"):
        try:
            from carid.vision.deep import OwlPartDetector

            return OwlPartDetector(owl_model, models_cache, allow_download)
        except Exception:
            if kind == "owl":
                raise
    return LayoutPartDetector()

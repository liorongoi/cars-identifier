"""Face/person detection and masking, so people in the frame do not drive matching."""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from carid.vision.imageio import RGBImage, resize_max_side

log = logging.getLogger(__name__)

Box = tuple[int, int, int, int]


class FaceDetector:
    """YuNet when its model file is available (more accurate), else OpenCV Haar cascades."""

    def __init__(self, models_cache: Path | None = None) -> None:
        self._yunet_path: Path | None = None
        if models_cache is not None:
            p = models_cache / "face_detection_yunet_2023mar.onnx"
            if p.exists():
                self._yunet_path = p
        cascade_dir = Path(cv2.data.haarcascades)  # type: ignore[attr-defined]
        self._cascades = [
            cv2.CascadeClassifier(str(cascade_dir / name))
            for name in ("haarcascade_frontalface_default.xml", "haarcascade_profileface.xml")
        ]
        self._cascades = [c for c in self._cascades if not c.empty()]

    @property
    def backend(self) -> str:
        return "yunet" if self._yunet_path else "haar"

    def detect(self, img: RGBImage) -> list[Box]:
        # Faces in cabin photos are large; 960 px is plenty and keeps Haar fast.
        small = resize_max_side(img, 960)
        scale = img.shape[1] / small.shape[1]
        boxes = self._detect_yunet(small) if self._yunet_path else self._detect_haar(small)
        out: list[Box] = []
        for x, y, w, h in boxes:
            # Enlarge to cover hair/neck, then map to original resolution.
            pad_w, pad_h = int(w * 0.25), int(h * 0.35)
            x0 = max(0, int((x - pad_w) * scale))
            y0 = max(0, int((y - pad_h) * scale))
            x1 = min(img.shape[1], int((x + w + pad_w) * scale))
            y1 = min(img.shape[0], int((y + h + pad_h * 2) * scale))
            out.append((x0, y0, x1, y1))
        return out

    def _detect_haar(self, img: RGBImage) -> list[tuple[int, int, int, int]]:
        gray = cv2.equalizeHist(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY))
        min_side = max(24, min(gray.shape) // 20)
        found: list[tuple[int, int, int, int]] = []
        for cascade in self._cascades:
            rects = cascade.detectMultiScale(
                gray, scaleFactor=1.15, minNeighbors=6, minSize=(min_side, min_side)
            )
            found.extend((int(x), int(y), int(w), int(h)) for x, y, w, h in rects)
        return _dedupe(found)

    def _detect_yunet(self, img: RGBImage) -> list[tuple[int, int, int, int]]:
        assert self._yunet_path is not None
        h, w = img.shape[:2]
        det = cv2.FaceDetectorYN.create(str(self._yunet_path), "", (w, h), 0.7)
        _, faces = det.detect(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        if faces is None:
            return []
        return [(int(f[0]), int(f[1]), int(f[2]), int(f[3])) for f in faces]


def _dedupe(rects: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
    kept: list[tuple[int, int, int, int]] = []
    for r in sorted(rects, key=lambda r: -r[2] * r[3]):
        if all(_iou(r, k) < 0.3 for k in kept):
            kept.append(r)
    return kept


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, bx1, by1 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    iw = max(0, min(ax1, bx1) - max(a[0], b[0]))
    ih = max(0, min(ay1, by1) - max(a[1], b[1]))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union else 0.0


def mask_regions(img: RGBImage, boxes: list[Box], mode: str = "fill") -> RGBImage:
    """Neutralise regions: 'fill' (mean colour, best for matching) or 'blur' (for storage)."""
    if not boxes:
        return img
    out = img.copy()
    for x0, y0, x1, y1 in boxes:
        if x1 <= x0 or y1 <= y0:
            continue
        region = out[y0:y1, x0:x1]
        if mode == "blur":
            k = max(15, (min(region.shape[:2]) // 3) | 1)
            out[y0:y1, x0:x1] = cv2.GaussianBlur(region, (k, k), 0)
        else:
            out[y0:y1, x0:x1] = np.asarray(img.mean(axis=(0, 1)), dtype=np.uint8)
    return out

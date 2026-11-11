"""Image embedders: map an image (or part crop) to an L2-normalised vector."""

from __future__ import annotations

import logging
from typing import Protocol

import cv2
import numpy as np
from numpy.typing import NDArray

from carid.vision.imageio import RGBImage

log = logging.getLogger(__name__)

Vectors = NDArray[np.float32]


class Embedder(Protocol):
    name: str
    dim: int

    def embed(self, images: list[RGBImage]) -> Vectors: ...


def l2_normalize(v: NDArray[np.floating]) -> Vectors:
    v = np.asarray(v, dtype=np.float32)
    norms = np.linalg.norm(v, axis=-1, keepdims=True)
    return np.asarray(v / np.maximum(norms, 1e-12), dtype=np.float32)


class HandcraftedEmbedder:
    """Shape + colour + layout descriptor. No downloads, deterministic, CPU-fast.

    Accuracy is well below a deep embedder; it exists so the system always works
    offline and so tests run without ML dependencies.
    """

    name = "handcrafted-v1"
    dim = 512
    _size = 128

    def __init__(self) -> None:
        self._hog = cv2.HOGDescriptor((128, 128), (32, 32), (16, 16), (16, 16), 9)
        raw_dim = 1764 + 128 + 256
        rng = np.random.default_rng(20261110)
        self._proj = (rng.standard_normal((raw_dim, self.dim)) / np.sqrt(self.dim)).astype(
            np.float32
        )

    def _describe(self, img: RGBImage) -> NDArray[np.float32]:
        im = cv2.resize(img, (self._size, self._size), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
        gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(gray)
        hog = np.sqrt(np.abs(np.asarray(self._hog.compute(gray), dtype=np.float32).ravel()))
        hsv = cv2.cvtColor(im, cv2.COLOR_RGB2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 4, 4], [0, 180, 0, 256, 0, 256]).ravel()
        hist = np.sqrt(hist / max(hist.sum(), 1.0))
        thumb = cv2.resize(gray, (16, 16), interpolation=cv2.INTER_AREA).astype(np.float32).ravel()
        thumb = (thumb - thumb.mean()) / (thumb.std() + 1e-6)
        parts = [
            l2_normalize(hog) * 1.0,
            l2_normalize(hist) * 0.6,
            l2_normalize(thumb) * 0.8,
        ]
        return np.concatenate(parts).astype(np.float32)

    def embed(self, images: list[RGBImage]) -> Vectors:
        if not images:
            return np.zeros((0, self.dim), dtype=np.float32)
        raw = np.stack([self._describe(im) for im in images])
        return l2_normalize(raw @ self._proj)


def get_embedder(kind: str, clip_model: str, clip_pretrained: str, models_cache: str,
                 allow_download: bool, dino_model: str = "facebook/dinov2-base",
                 dino_size: int = 224) -> Embedder:
    if kind == "dinov2":
        from carid.vision.deep import DinoV2Embedder

        return DinoV2Embedder(dino_model, models_cache, allow_download, dino_size)
    if kind in ("auto", "clip"):
        try:
            from carid.vision.deep import ClipEmbedder

            return ClipEmbedder(clip_model, clip_pretrained, models_cache, allow_download)
        except Exception as exc:
            if kind == "clip":
                raise
            log.warning("Deep embedder unavailable (%s); using handcrafted fallback", exc)
    return HandcraftedEmbedder()

"""Learned adapter: makes generic embeddings specific to car-interior classes.

Regularised linear discriminant analysis (LDA) fitted per part type on the
labelled reference vectors. It finds the directions in which different
generations differ most relative to how much photos of the same generation
vary, then whitens them. Closed-form, CPU-cheap, deterministic, and it keeps the
nearest-neighbour design (new makes still need no retraining; refit after big
collections).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from carid.vision.embed import Vectors, l2_normalize


@dataclass
class Adapter:
    mean: NDArray[np.float32]  # (d,)
    proj: NDArray[np.float32]  # (d, k)

    @property
    def out_dim(self) -> int:
        return int(self.proj.shape[1])

    def transform(self, v: NDArray[np.floating]) -> Vectors:
        x = np.asarray(v, dtype=np.float32)
        return l2_normalize((x - self.mean) @ self.proj)


def fit_lda(vectors: NDArray[np.floating], labels: list[object], *, shrinkage: float = 0.5,
            max_dim: int = 256, min_per_class: int = 2) -> Adapter | None:
    """Fit regularised LDA. Returns None when there is too little labelled data.

    ``shrinkage`` blends the within-class scatter with the identity (0 = plain LDA,
    1 = ignore within-class structure); with few images per class it must be high.
    The output keeps the ``max_dim`` most discriminative directions plus a residual
    of the original space, so classes never seen in training still separate.
    """
    x = np.asarray(vectors, dtype=np.float64)
    counts = Counter(labels)
    keep_classes = {c for c, n in counts.items() if n >= min_per_class}
    if len(keep_classes) < 2:
        return None
    mask = np.array([lab in keep_classes for lab in labels])
    x_fit = x[mask]
    y_fit = [lab for lab, m in zip(labels, mask, strict=True) if m]
    d = x.shape[1]
    mean = x.mean(axis=0)
    sw = np.zeros((d, d))
    sb = np.zeros((d, d))
    for c in keep_classes:
        xc = x_fit[[i for i, lab in enumerate(y_fit) if lab == c]]
        mc = xc.mean(axis=0)
        diff = xc - mc
        sw += diff.T @ diff
        dm = (mc - mean)[:, None]
        sb += len(xc) * (dm @ dm.T)
    sw /= max(len(x_fit), 1)
    sb /= max(len(x_fit), 1)
    scale = np.trace(sw) / d if np.trace(sw) > 0 else 1.0
    sw_reg = (1 - shrinkage) * sw + shrinkage * scale * np.eye(d)
    # Whitening by the regularised within-class scatter: directions where the same
    # interior varies a lot (lighting, angle) are shrunk, stable ones amplified.
    evals, evecs = np.linalg.eigh(sw_reg)
    evals = np.maximum(evals, 1e-9)
    whiten = evecs @ np.diag(evals ** -0.5)
    sb_w = whiten.T @ sb @ whiten
    b_evals, b_evecs = np.linalg.eigh(sb_w)
    order = np.argsort(b_evals)[::-1]
    k = min(max_dim, len(keep_classes) - 1, d)
    # Weight discriminative directions by their separation, keep the rest at unit weight.
    weights = np.ones(d)
    weights[order[:k]] = 1.0 + np.sqrt(np.maximum(b_evals[order[:k]], 0) /
                                       max(float(b_evals[order[0]]), 1e-12)) * 2.0
    proj = whiten @ b_evecs[:, order] @ np.diag(weights[order])
    return Adapter(mean=mean.astype(np.float32), proj=proj.astype(np.float32))


def save_adapters(path: Path, adapters: dict[str, Adapter]) -> None:
    arrays: dict[str, NDArray[np.float32]] = {}
    for part, a in adapters.items():
        arrays[f"{part}__mean"] = a.mean
        arrays[f"{part}__proj"] = a.proj
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, **arrays)  # type: ignore[arg-type]
    tmp.replace(path)


def load_adapters(path: Path) -> dict[str, Adapter]:
    if not path.exists():
        return {}
    out: dict[str, Adapter] = {}
    with np.load(path) as data:
        parts = {k.split("__")[0] for k in data.files}
        for part in parts:
            out[part] = Adapter(mean=data[f"{part}__mean"].astype(np.float32),
                                proj=data[f"{part}__proj"].astype(np.float32))
    return out

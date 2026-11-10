"""Per-part vector indexes (exact cosine search, NumPy; FAISS used when installed)."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

IntArray = NDArray[np.int64]
FloatArray = NDArray[np.float32]

try:  # optional acceleration
    import faiss

    HAVE_FAISS = True
except ImportError:  # pragma: no cover - depends on environment
    faiss = None
    HAVE_FAISS = False


class VectorIndex:
    """Exact inner-product index over L2-normalised vectors for one part type."""

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self._part_ids: IntArray = np.zeros(0, dtype=np.int64)
        self._image_ids: IntArray = np.zeros(0, dtype=np.int64)
        self._vecs: FloatArray = np.zeros((0, dim), dtype=np.float32)
        self._pending: list[tuple[IntArray, IntArray, FloatArray]] = []
        self._removed: set[int] = set()
        self._faiss: Any = None

    def __len__(self) -> int:
        self._consolidate()
        return int(self._part_ids.size)

    def add(self, part_ids: IntArray, image_ids: IntArray, vecs: FloatArray) -> None:
        vecs = np.asarray(vecs, dtype=np.float32)
        if vecs.ndim != 2 or vecs.shape[1] != self.dim:
            raise ValueError(f"expected vectors of shape (n, {self.dim}), got {vecs.shape}")
        if not (len(part_ids) == len(image_ids) == len(vecs)):
            raise ValueError("ids and vectors must have the same length")
        self._pending.append((np.asarray(part_ids, np.int64), np.asarray(image_ids, np.int64),
                              vecs))
        self._removed.difference_update(int(i) for i in image_ids)

    def remove_images(self, image_ids: set[int]) -> None:
        """Hide images from search results (physically dropped on ``compact``)."""
        self._removed.update(image_ids)

    def compact(self) -> None:
        self._consolidate()
        if self._removed:
            keep = ~np.isin(self._image_ids, np.fromiter(self._removed, np.int64))
            self._part_ids, self._image_ids = self._part_ids[keep], self._image_ids[keep]
            self._vecs = self._vecs[keep]
            self._removed.clear()
            self._faiss = None

    def _consolidate(self) -> None:
        if not self._pending:
            return
        self._part_ids = np.concatenate([self._part_ids, *(p[0] for p in self._pending)])
        self._image_ids = np.concatenate([self._image_ids, *(p[1] for p in self._pending)])
        self._vecs = np.concatenate([self._vecs, *(p[2] for p in self._pending)])
        self._pending.clear()
        self._faiss = None

    def search(self, query: FloatArray, k: int) -> tuple[IntArray, IntArray, FloatArray]:
        """Return (part_ids, image_ids, similarities) of the k nearest vectors, best first."""
        self._consolidate()
        n = self._part_ids.size
        empty = (np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0, np.float32))
        if n == 0 or k <= 0:
            return empty
        q = np.asarray(query, dtype=np.float32).reshape(-1)
        fetch = min(n, k + len(self._removed))
        if HAVE_FAISS and n > 20_000:
            if self._faiss is None:
                self._faiss = faiss.IndexFlatIP(self.dim)
                self._faiss.add(self._vecs)
            sims, idx = self._faiss.search(q[None, :], fetch)
            order, sims_sel = idx[0], sims[0].astype(np.float32)
        else:
            sims_all = self._vecs @ q
            if fetch < n:
                part = np.argpartition(-sims_all, fetch - 1)[:fetch]
                order = part[np.argsort(-sims_all[part], kind="stable")]
            else:
                order = np.argsort(-sims_all, kind="stable")
            sims_sel = sims_all[order]
        img_ids = self._image_ids[order]
        if self._removed:
            keep = ~np.isin(img_ids, np.fromiter(self._removed, np.int64))
            order, img_ids, sims_sel = order[keep], img_ids[keep], sims_sel[keep]
        return self._part_ids[order][:k], img_ids[:k], sims_sel[:k]

    def save(self, path: Path) -> None:
        self.compact()
        tmp = path.with_suffix(".tmp.npz")
        np.savez(tmp, part_ids=self._part_ids, image_ids=self._image_ids, vecs=self._vecs)
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path) -> VectorIndex:
        with np.load(path) as data:
            vecs = data["vecs"].astype(np.float32)
            idx = cls(int(vecs.shape[1]))
            idx._part_ids = data["part_ids"].astype(np.int64)
            idx._image_ids = data["image_ids"].astype(np.int64)
            idx._vecs = vecs
        return idx


class IndexSet:
    """All part indexes for one embedder, stored under ``root/<embedder name>/``."""

    def __init__(self, root: Path, embedder_name: str, dim: int) -> None:
        self.dir = root / embedder_name.replace("/", "_")
        self.dim = dim
        self.embedder_name = embedder_name
        self.parts: dict[str, VectorIndex] = {}
        self.lock = threading.RLock()
        self._dirty = False

    def get(self, part: str) -> VectorIndex | None:
        return self.parts.get(part)

    def add(self, part: str, part_ids: IntArray, image_ids: IntArray, vecs: FloatArray) -> None:
        with self.lock:
            if part not in self.parts:
                self.parts[part] = VectorIndex(self.dim)
            self.parts[part].add(part_ids, image_ids, vecs)
            self._dirty = True

    def remove_images(self, image_ids: set[int]) -> None:
        with self.lock:
            for idx in self.parts.values():
                idx.remove_images(image_ids)
            self._dirty = True

    def clear(self) -> None:
        with self.lock:
            self.parts.clear()
            self._dirty = True

    def sizes(self) -> dict[str, int]:
        with self.lock:
            return {p: len(i) for p, i in sorted(self.parts.items())}

    def save(self) -> None:
        with self.lock:
            if not self._dirty:
                return
            self.dir.mkdir(parents=True, exist_ok=True)
            for part, idx in self.parts.items():
                idx.save(self.dir / f"{part}.npz")
            for stale in self.dir.glob("*.npz"):
                if stale.stem not in self.parts:
                    stale.unlink()
            meta = {"embedder": self.embedder_name, "dim": self.dim}
            (self.dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
            self._dirty = False

    def load(self) -> None:
        with self.lock:
            self.parts.clear()
            meta_path = self.dir / "meta.json"
            if not meta_path.exists():
                return
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("dim") != self.dim:
                raise ValueError(
                    f"index at {self.dir} has dim {meta.get('dim')}, embedder has {self.dim}; "
                    "rebuild the index"
                )
            for f in sorted(self.dir.glob("*.npz")):
                if f.name.endswith(".tmp.npz"):
                    continue
                self.parts[f.stem] = VectorIndex.load(f)

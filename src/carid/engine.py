"""The identification engine: analysis, reference ingestion, indexing and ranking."""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
from sqlalchemy import delete, func, select, true, update
from sqlalchemy.orm import Session

from carid.adapter import Adapter, fit_lda, load_adapters, save_adapters
from carid.config import Settings
from carid.db import Database
from carid.index import IndexSet
from carid.matcher import ClassScore, Label, MatchParams, Mode, make_probabilities, rank
from carid.models import (
    AuditEvent,
    Benchmark,
    Identification,
    Image,
    ImagePart,
    Make,
    ReviewStatus,
    VehicleModel,
)
from carid.taxonomy import slugify
from carid.vision.embed import Embedder, Vectors, get_embedder
from carid.vision.enhance import enhance
from carid.vision.faces import FaceDetector, mask_regions
from carid.vision.imageio import (
    RGBImage,
    crop,
    decode_image,
    encode_jpeg,
    perceptual_hash,
    resize_max_side,
    sha256_bytes,
)
from carid.vision.parts import PartBox, PartDetector, detail_tiles, get_part_detector
from carid.vision.quality import QualityReport, assess_quality

log = logging.getLogger(__name__)

ANALYSIS_MAX_SIDE = 1600
STORE_MAX_SIDE = 1600


@dataclass
class Analysis:
    image: RGBImage  # processed image actually used for matching
    quality: QualityReport
    enhancement: list[str]
    faces: list[tuple[int, int, int, int]]
    parts: list[PartBox]
    vectors: dict[str, Vectors]


@dataclass
class ReferenceLabel:
    """What a reference image shows: make and model (years are not identified)."""

    make_id: int
    model_id: int
    confidence: float = 0.5


@dataclass
class SourceInfo:
    source: str
    source_url: str | None = None
    page_url: str | None = None
    license: str | None = None
    credit: str | None = None
    title: str | None = None


@dataclass
class IngestResult:
    status: str  # added | duplicate | near_duplicate | rejected_quality | rejected_not_interior
    image_id: int | None = None
    detail: str = ""


@dataclass
class IdentifyOptions:
    top_k: int = 10
    enhance: bool = True
    mask_faces: bool = True
    mode: Mode | None = None
    make_ids: list[int] | None = None
    manual_parts: list[dict[str, Any]] = field(default_factory=list)
    case_ref: str | None = None
    analyst: str | None = None
    store_query: bool = True


class Engine:
    """Long-lived service object; thread-safe for concurrent identify calls."""

    def __init__(self, settings: Settings, db: Database | None = None,
                 embedder: Embedder | None = None, detector: PartDetector | None = None) -> None:
        self.settings = settings
        settings.ensure_dirs()
        self.db = db or Database(settings)
        cache = str(settings.models_cache)
        self.embedder = embedder or get_embedder(
            settings.embedder, settings.clip_model, settings.clip_pretrained, cache,
            settings.allow_model_download, settings.dino_model, settings.dino_image_size)
        self.detector = detector or get_part_detector(
            settings.detector, settings.owl_model, cache, settings.allow_model_download)
        self.faces = FaceDetector(settings.models_cache)
        self.index = IndexSet(settings.index_dir, self.embedder.name, self.embedder.dim)
        if settings.use_adapter:
            self.index.adapters = load_adapters(self.index.adapters_path)
        self.index.load()
        self._ingest_lock = threading.Lock()
        self._labels: dict[int, Label] = {}
        self._labels_version = -1
        self._version = 0
        self._labels_lock = threading.Lock()
        self._auto_mode: tuple[Mode, bool] | None = None
        self._ph_ids = np.zeros(0, dtype=np.int64)
        self._ph_vals = np.zeros(0, dtype=np.uint64)
        self._load_phashes()

    # ------------------------------------------------------------------ status

    def describe(self) -> dict[str, Any]:
        return {
            "embedder": self.embedder.name,
            "embedder_dim": self.embedder.dim,
            "detector": self.detector.name,
            "face_detector": self.faces.backend,
            "deep_models": not self.embedder.name.startswith("handcrafted"),
            "index_sizes": self.index.sizes(),
            "matcher_mode": self._mode_label(),
            "adapter": sorted(self.index.adapters),
        }

    def index_warnings(self) -> list[str]:
        """Detect an index that does not match the database (e.g. after an embedder change)."""
        with self.db.session() as s:
            n_images = s.scalar(select(func.count()).select_from(Image).where(
                Image.review_status != ReviewStatus.rejected)) or 0
        whole = self.index.get("whole")
        n_vecs = len(whole) if whole is not None else 0
        if n_images and n_vecs < n_images * 0.98:
            return [f"Index out of date: {n_images} reference images but {n_vecs} indexed for "
                    f"embedder '{self.embedder.name}'. Run 'Rebuild index' (Jobs page)."]
        return []

    def invalidate_labels(self) -> None:
        self._version += 1

    def labels(self) -> dict[int, Label]:
        with self._labels_lock:
            if self._labels_version != self._version:
                with self.db.session() as s:
                    rows = s.execute(
                        select(Image.id, Image.make_id, Image.model_id)
                        .where(Image.review_status != ReviewStatus.rejected)
                    ).all()
                self._labels = {r[0]: Label(*r) for r in rows}
                self._labels_version = self._version
            return self._labels

    def _load_phashes(self) -> None:
        with self.db.session() as s:
            rows = s.execute(select(Image.id, Image.phash)).all()
        self._ph_ids = np.array([r[0] for r in rows], dtype=np.int64)
        self._ph_vals = np.array([int(r[1], 16) for r in rows], dtype=np.uint64)

    def _add_phash(self, image_id: int, phash: str) -> None:
        self._ph_ids = np.append(self._ph_ids, np.int64(image_id))
        self._ph_vals = np.append(self._ph_vals, np.uint64(int(phash, 16)))

    def _near_duplicate(self, phash: str) -> int | None:
        """Closest stored image by perceptual-hash Hamming distance (vectorised popcount)."""
        if self._ph_vals.size == 0:
            return None
        x = np.bitwise_xor(self._ph_vals, np.uint64(int(phash, 16)))
        dist = np.unpackbits(x.view(np.uint8)).reshape(-1, 64).sum(axis=1)
        j = int(np.argmin(dist))
        if int(dist[j]) <= self.settings.near_duplicate_distance:
            return int(self._ph_ids[j])
        return None

    # ---------------------------------------------------------------- analysis

    def analyze(self, img: RGBImage, *, do_enhance: bool = True, do_mask_faces: bool = True,
                manual_parts: list[dict[str, Any]] | None = None) -> Analysis:
        img = resize_max_side(img, ANALYSIS_MAX_SIDE)
        quality = assess_quality(img)
        steps: list[str] = []
        work = img
        if do_enhance:
            work, steps = enhance(img, quality)
        faces = self.faces.detect(work) if do_mask_faces else []
        if faces:
            work = mask_regions(work, faces, mode="fill")
            steps.append(f"masked {len(faces)} face(s)")
        parts = self.detector.detect(work)
        for mp in manual_parts or []:
            x0, y0, x1, y1 = (int(v) for v in mp["bbox"])
            parts = [p for p in parts if p.part != mp["part"]]
            parts.append(PartBox(str(mp["part"]), (x0, y0, x1, y1), 1.0, "manual"))
        vectors = self.embed_parts(work, parts)
        return Analysis(work, quality, steps, faces, parts, vectors)

    def embed_parts(self, img: RGBImage, parts: list[PartBox]) -> dict[str, Vectors]:
        crops: list[RGBImage] = []
        names: list[str] = []
        for p in parts:
            try:
                crops.append(crop(img, p.bbox))
                names.append(p.part)
            except ValueError:
                log.debug("skipping empty crop %s", p)
        vecs = self.embedder.embed(crops)
        return {name: vecs[i] for i, name in enumerate(names)}

    # ---------------------------------------------------------------- identify

    def resolved_mode(self) -> tuple[Mode, bool]:
        """(mode, use adapter) — from settings, or from the latest benchmark when 'auto'."""
        if self.settings.matcher_mode != "auto":
            return self.settings.matcher_mode, True
        if self._auto_mode is None:
            best = None
            with self.db.session() as s:
                b = s.scalars(select(Benchmark).where(Benchmark.embedder == self.embedder.name)
                              .order_by(Benchmark.id.desc()).limit(1)).first()
                if b is not None:
                    best = b.report.get("best_mode")
            mode, _, variant = str(best or "fusion").partition("+")
            if mode not in ("fusion", "global", "parts"):
                mode = "fusion"
            # Adapter stays on unless the benchmark showed the raw variant winning.
            self._auto_mode = (mode, variant == "lda" or best is None)  # type: ignore[assignment]
        assert self._auto_mode is not None
        return self._auto_mode

    def _mode_label(self) -> str:
        mode, use_adapter = self.resolved_mode()
        source = "auto: " if self.settings.matcher_mode == "auto" else ""
        return f"{source}{mode}{'' if use_adapter else ' (adapter off)'}"

    def forget_auto_mode(self) -> None:
        self._auto_mode = None

    def match(self, analysis: Analysis, opts: IdentifyOptions) -> list[ClassScore]:
        mode, use_adapter = self.resolved_mode()
        params = MatchParams(
            mode=opts.mode or mode,
            adapters=None if use_adapter else {},
            neighbors_per_part=self.settings.neighbors_per_part,
            votes_per_class=self.settings.votes_per_class,
            similarity_temperature=self.settings.similarity_temperature,
            softmax_temperature=self.settings.softmax_temperature,
            part_weights=self.settings.part_weights,
            make_ids=set(opts.make_ids) if opts.make_ids else None,
        )
        with self.index.lock:
            return rank(analysis.vectors, self.index, self.labels(), params)

    def identify(self, data: bytes, opts: IdentifyOptions) -> dict[str, Any]:
        t0 = time.perf_counter()
        img = decode_image(data, self.settings.max_image_pixels)
        analysis = self.analyze(img, do_enhance=opts.enhance, do_mask_faces=opts.mask_faces,
                                manual_parts=opts.manual_parts)
        ranked = self.match(analysis, opts)
        duration_ms = int((time.perf_counter() - t0) * 1000)
        with self.db.session() as s:
            result = self._build_result(s, analysis, ranked, opts.top_k)
            result["duration_ms"] = duration_ms
            sha = sha256_bytes(data)
            query_path = None
            if opts.store_query and self.settings.query_retention_days > 0:
                rel = Path(f"{sha[:2]}/{sha}.jpg")
                dest = self.settings.queries_dir / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(encode_jpeg(resize_max_side(img, 2400)))
                query_path = str(rel)
            ident = Identification(
                case_ref=opts.case_ref, analyst=opts.analyst, query_sha256=sha,
                query_path=query_path, duration_ms=duration_ms, result=result,
                options={"top_k": opts.top_k, "enhance": opts.enhance,
                         "mask_faces": opts.mask_faces, "mode": opts.mode,
                         "make_ids": opts.make_ids,
                         "manual_parts": opts.manual_parts},
            )
            s.add(ident)
            s.flush()
            result["id"] = ident.id
            ident.result = dict(result)
        return result

    def _build_result(self, s: Session, analysis: Analysis, ranked: list[ClassScore],
                      top_k: int) -> dict[str, Any]:
        top = ranked[:top_k]
        model_ids = {c.key for c in top}
        models = {m.id: m for m in s.scalars(select(VehicleModel).where(
            VehicleModel.id.in_(model_ids)))} if model_ids else {}
        makes = {m.id: m for m in s.scalars(select(Make))}
        counts = self.class_image_counts(s, [c.key for c in top])  # per model
        evidence_ids = {e.image_id for c in top for e in c.evidence}
        ev_images = {i.id: i for i in s.scalars(select(Image).where(
            Image.id.in_(evidence_ids)))} if evidence_ids else {}
        ev_parts = {p.id: p for p in s.scalars(select(ImagePart).where(
            ImagePart.id.in_({e.part_id for c in top for e in c.evidence})))} if top else {}

        candidates = []
        for i, c in enumerate(top, start=1):
            model = models.get(c.key)
            evidence = []
            for e in sorted(c.evidence, key=lambda e: -c.part_scores.get(e.part, 0)):
                im = ev_images.get(e.image_id)
                part = ev_parts.get(e.part_id)
                evidence.append({
                    "part": e.part, "similarity": e.similarity, "image_id": e.image_id,
                    "bbox": part.bbox if part else None,
                    "source": im.source if im else None,
                    "page_url": im.page_url if im else None,
                    "credit": im.credit if im else None,
                    "license": im.license if im else None,
                })
            candidates.append({
                "rank": i,
                "make_id": c.make_id,
                "make": makes[c.make_id].name if c.make_id in makes else "?",
                "model_id": c.key,
                "model": model.name if model else "?",
                "probability": round(c.probability, 4),
                "confidence": _confidence_band(c.probability),
                "score": round(c.score, 5),
                "part_scores": c.part_scores,
                "reference_images": counts.get(c.key, 0),
                "evidence": evidence,
            })
        make_probs = make_probabilities(ranked)
        top_prob = ranked[0].probability if ranked else 0.0
        warnings: list[str] = []
        if not ranked:
            warnings.append("No matches: the reference database is empty or nothing matched "
                            "the filters.")
        elif top_prob < self.settings.low_confidence_threshold:
            warnings.append("Low confidence: no candidate stands out. Treat the list as leads, "
                            "not an identification.")
        if analysis.quality.verdict == "poor":
            warnings.append("Poor image quality (" + ", ".join(analysis.quality.issues)
                            + "); results are less reliable.")
        return {
            "candidates": candidates,
            "makes": [{"make_id": k, "make": makes[k].name if k in makes else "?",
                       "probability": round(v, 4)} for k, v in list(make_probs.items())[:10]],
            "quality": analysis.quality.to_dict(),
            "enhancement": analysis.enhancement,
            "faces": [list(b) for b in analysis.faces],
            "parts": [p.to_dict() for p in analysis.parts],
            "image_size": [int(analysis.image.shape[1]), int(analysis.image.shape[0])],
            "warnings": warnings,
            "engine": self.describe(),
            "created_at": datetime.now(UTC).isoformat(),
        }

    def class_image_counts(self, s: Session, model_ids: list[int]) -> dict[int, int]:
        """Usable reference images per model."""
        if not model_ids:
            return {}
        rows = s.execute(
            select(Image.model_id, func.count())
            .where(Image.model_id.in_(set(model_ids)),
                   Image.review_status != ReviewStatus.rejected)
            .group_by(Image.model_id)
        ).all()
        return {m: int(n) for m, n in rows}

    # ---------------------------------------------------------------- ingest

    def ingest(self, data: bytes, label: ReferenceLabel, source: SourceInfo, *,
               interior_score: float | None = None, review_status: ReviewStatus | None = None,
               original_sha256: str | None = None) -> IngestResult:
        """Validate, de-duplicate, anonymise, store and index one reference image.

        ``original_sha256`` identifies the originally downloaded file when ``data`` is a
        re-encoded copy (bundle import), so the same source file is never stored twice.
        """
        sha = original_sha256 or sha256_bytes(data)
        with self.db.session() as s:
            existing = s.scalar(select(Image.id).where(Image.sha256 == sha))
        if existing is not None:
            return IngestResult("duplicate", existing, "identical file already stored")
        img = decode_image(data, self.settings.max_image_pixels)
        if min(img.shape[:2]) < self.settings.min_image_side:
            return IngestResult("rejected_quality", None,
                                f"too small ({img.shape[1]}x{img.shape[0]})")
        img = resize_max_side(img, STORE_MAX_SIDE)
        quality = assess_quality(img)
        # Only reject the hopeless: sharpness alone depends on scene content.
        if (quality.sharpness < 5 and quality.contrast < 20) or quality.brightness < 15:
            return IngestResult("rejected_quality", None, ", ".join(quality.issues))
        if interior_score is None:
            scorer = getattr(self.embedder, "interior_score", None)
            if callable(scorer):
                interior_score = float(scorer(img))
        if interior_score is not None and interior_score < self.settings.interior_threshold:
            return IngestResult("rejected_not_interior", None,
                                f"interior score {interior_score:.2f}")

        faces = self.faces.detect(img)
        stored = mask_regions(img, faces, mode="blur") if faces else img
        phash = perceptual_hash(stored)

        with self._ingest_lock:
            dup = self._near_duplicate(phash)
            if dup is not None:
                return IngestResult("near_duplicate", dup, "visually identical image stored")
            with self.db.session() as s:
                model = s.get(VehicleModel, label.model_id)
                if model is None or model.make_id != label.make_id:
                    raise ValueError("label model does not belong to label make")
                rel = Path(slugify(model.make.name)) / slugify(model.name) / f"{sha[:20]}.jpg"
                dest = self.settings.images_dir / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                tmp = dest.with_suffix(".part")
                tmp.write_bytes(encode_jpeg(stored, quality=90))
                tmp.replace(dest)
                image = Image(
                    sha256=sha, phash=phash, path=rel.as_posix(),
                    width=int(stored.shape[1]), height=int(stored.shape[0]),
                    make_id=label.make_id, model_id=label.model_id,
                    label_confidence=label.confidence, source=source.source,
                    source_url=source.source_url, page_url=source.page_url,
                    license=source.license, credit=source.credit,
                    title=(source.title or "")[:500] or None,
                    quality=quality.to_dict(), interior_score=interior_score,
                    faces_blurred=len(faces),
                    review_status=review_status or ReviewStatus.pending,
                )
                s.add(image)
                s.flush()
                self._index_image(s, image, stored)
                image_id = image.id
            self._add_phash(image_id, phash)
        self.invalidate_labels()
        return IngestResult("added", image_id)

    def _index_image(self, s: Session, image: Image, img: RGBImage) -> None:
        # References use the same enhancement as queries so both sides look alike.
        q = assess_quality(img)
        work, _ = enhance(img, q)
        parts = self.detector.detect(work)
        vectors = self.embed_parts(work, parts)
        s.execute(delete(ImagePart).where(ImagePart.image_id == image.id))
        for p in parts:
            if p.part not in vectors:
                continue
            row = ImagePart(image_id=image.id, part=p.part, bbox=list(p.bbox), score=p.score)
            s.add(row)
            s.flush()
            self.index.add(p.part, np.array([row.id]), np.array([image.id]),
                           vectors[p.part][None, :])
        if self.settings.detail_tiles:
            self._index_tiles(s, image, work)
        image.indexed = True

    def _embed_stored_parts(self, image: Image, img: RGBImage) -> None:
        """Embed an image's stored part/tile boxes with the current embedder."""
        work, _ = enhance(img, assess_quality(img))
        by_part: dict[str, list[ImagePart]] = {}
        for p in image.parts:
            by_part.setdefault(p.part, []).append(p)
        rows = [r for rs in by_part.values() for r in rs]
        crops, keep = [], []
        for r in rows:
            try:
                crops.append(crop(work, tuple(int(v) for v in r.bbox)))  # type: ignore[arg-type]
                keep.append(r)
            except ValueError:
                continue
        vecs = self.embedder.embed(crops)
        for part in {r.part for r in keep}:
            idx = [i for i, r in enumerate(keep) if r.part == part]
            self.index.add(part, np.array([keep[i].id for i in idx]),
                           np.full(len(idx), image.id, dtype=np.int64), vecs[idx])

    def _index_tiles(self, s: Session, image: Image, work: RGBImage) -> int:
        """Index multi-scale detail tiles so close-ups of small items can be matched."""
        tiles = detail_tiles(work)
        if not tiles:
            return 0
        vecs = self.embedder.embed([crop(work, t.bbox) for t in tiles])
        rows = [ImagePart(image_id=image.id, part="tile", bbox=list(t.bbox), score=1.0)
                for t in tiles]
        s.add_all(rows)
        s.flush()
        self.index.add("tile", np.array([r.id for r in rows]),
                       np.full(len(rows), image.id, dtype=np.int64), vecs)
        return len(rows)

    def add_missing_tiles(self, progress: Any = None, should_stop: Any = None) -> int:
        """Upgrade path: add detail tiles to images indexed before tiles existed."""
        with self.db.session() as s:
            have = set(s.scalars(select(ImagePart.image_id).where(ImagePart.part == "tile")))
            ids = [i for i in s.scalars(select(Image.id).where(
                Image.review_status != ReviewStatus.rejected).order_by(Image.id)) if i not in have]
        done = 0
        for n, image_id in enumerate(ids, start=1):
            if should_stop and should_stop():
                break
            with self.db.session() as s:
                image = s.get(Image, image_id)
                if image is None:
                    continue
                path = self.settings.images_dir / image.path
                if not path.exists():
                    continue
                img = decode_image(path.read_bytes(), self.settings.max_image_pixels)
                work, _ = enhance(img, assess_quality(img))
                self._index_tiles(s, image, work)
                done += 1
            if progress:
                progress(n / max(len(ids), 1), f"tiles {n}/{len(ids)}")
            if done % 50 == 0:
                self.index.save()
        self.index.save()
        return done

    def fit_adapters(self, exclude_image_ids: set[int] | None = None) -> dict[str, Adapter]:
        """Fit one LDA adapter per part on the labelled reference vectors."""
        labels = self.labels()
        exclude = exclude_image_ids or set()
        out: dict[str, Adapter] = {}
        with self.index.lock:
            parts = dict(self.index.parts)
        for part, idx in parts.items():
            _, image_ids, vecs = idx.raw_vectors()
            rows = [i for i, iid in enumerate(image_ids.tolist())
                    if iid in labels and iid not in exclude]
            if len(rows) < 4:
                continue
            classes: list[object] = [labels[int(image_ids[i])].model_id for i in rows]
            adapter = fit_lda(vecs[rows], classes, shrinkage=self.settings.adapter_shrinkage)
            if adapter is not None:
                out[part] = adapter
        return out

    def train_adapters(self) -> dict[str, Any]:
        """Fit adapters on all references, save them and use them for searching."""
        adapters = self.fit_adapters()
        self.index.dir.mkdir(parents=True, exist_ok=True)
        save_adapters(self.index.adapters_path, adapters)
        if self.settings.use_adapter:
            self.index.set_adapters(adapters)
        return {"parts": sorted(adapters), "images": len(self.labels())}

    def save_index(self) -> None:
        self.index.save()

    def rebuild_index(self, progress: Any = None, should_stop: Any = None,
                      redetect: bool = False) -> int:
        """Re-embed every non-rejected image (e.g. after switching embedder).

        By default the stored part and tile boxes are reused: no re-detection, and the
        part records stay shared by every embedder's index. ``redetect`` re-runs part
        detection (and replaces the boxes) instead.
        """
        with self.db.session() as s:
            ids = list(s.scalars(select(Image.id).where(
                Image.review_status != ReviewStatus.rejected).order_by(Image.id)))
            s.execute(update(Image).values(indexed=False))
        self.index.clear()
        done = 0
        for n, image_id in enumerate(ids, start=1):
            if should_stop and should_stop():
                break
            with self.db.session() as s:
                image = s.get(Image, image_id)
                if image is None:
                    continue
                path = self.settings.images_dir / image.path
                if not path.exists():
                    log.warning("missing file for image %s: %s", image_id, path)
                    continue
                img = decode_image(path.read_bytes(), self.settings.max_image_pixels)
                if redetect or not image.parts:
                    self._index_image(s, image, img)
                else:
                    self._embed_stored_parts(image, img)
                    image.indexed = True
                done += 1
            if progress:
                progress(n / max(len(ids), 1), f"indexed {n}/{len(ids)}")
            if n % 100 == 0:
                self.index.save()
        self.index.save()
        self.invalidate_labels()
        return done

    def set_review_status(self, image_id: int, status: ReviewStatus, actor: str | None) -> None:
        with self.db.session() as s:
            image = s.get(Image, image_id)
            if image is None:
                raise KeyError(image_id)
            old = image.review_status
            image.review_status = status
            s.add(AuditEvent(actor=actor, action="image.review", target=f"image:{image_id}",
                             details={"from": old.value, "to": status.value}))
        if status == ReviewStatus.rejected:
            self.index.remove_images({image_id})
        elif old == ReviewStatus.rejected:
            self._reindex_one(image_id)
        self.invalidate_labels()

    def relabel(self, image_id: int, label: ReferenceLabel, actor: str | None) -> None:
        """Change an image's label; the file moves to its new Make/Model folder."""
        with self.db.session() as s:
            image = s.get(Image, image_id)
            model = s.get(VehicleModel, label.model_id)
            if image is None:
                raise KeyError(image_id)
            if model is None or model.make_id != label.make_id:
                raise ValueError("label model does not belong to label make")
            old = {"make_id": image.make_id, "model_id": image.model_id}
            rel = Path(slugify(model.make.name)) / slugify(model.name) / Path(image.path).name
            if rel.as_posix() != image.path:
                src = self.settings.images_dir / image.path
                dest = self.settings.images_dir / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                if src.exists():
                    src.replace(dest)
                image.path = rel.as_posix()
            image.make_id, image.model_id = label.make_id, label.model_id
            image.label_confidence = label.confidence
            s.add(AuditEvent(actor=actor, action="image.relabel", target=f"image:{image_id}",
                             details={"from": old, "to": {
                                 "make_id": label.make_id, "model_id": label.model_id}}))
        self.invalidate_labels()

    def repair_index(self, progress: Any = None) -> int:
        """Index images that are in the database but missing from the index (e.g. after
        a job was interrupted before the index was saved)."""
        whole = self.index.get("whole")
        have = set(whole.raw_vectors()[1].tolist()) if whole is not None else set()
        with self.db.session() as s:
            ids = [i for i in s.scalars(select(Image.id).where(
                Image.review_status != ReviewStatus.rejected).order_by(Image.id))
                if i not in have]
        for n, image_id in enumerate(ids, start=1):
            self._reindex_one(image_id)
            if progress:
                progress(n / len(ids), f"repaired {n}/{len(ids)}")
        if ids:
            self.index.save()
            self.invalidate_labels()
        return len(ids)

    def _reindex_one(self, image_id: int) -> None:
        with self.db.session() as s:
            image = s.get(Image, image_id)
            if image is None:
                return
            img = decode_image((self.settings.images_dir / image.path).read_bytes(),
                               self.settings.max_image_pixels)
            # Old vectors reference part rows _index_image is about to replace.
            self.index.drop_images({image_id})
            self._index_image(s, image, img)

    def purge_old_queries(self) -> int:
        """Delete stored query images older than the retention period (NFR-6)."""
        days = self.settings.query_retention_days
        cutoff = datetime.now(UTC) - timedelta(days=days)
        removed = 0
        with self.db.session() as s:
            rows = s.scalars(select(Identification).where(
                Identification.query_path.is_not(None),
                Identification.created_at < cutoff if days > 0 else true(),
            ))
            for ident in rows:
                assert ident.query_path is not None
                (self.settings.queries_dir / ident.query_path).unlink(missing_ok=True)
                ident.query_path = None
                removed += 1
        return removed


def _confidence_band(p: float) -> str:
    if p >= 0.6:
        return "high"
    if p >= 0.3:
        return "medium"
    return "low"

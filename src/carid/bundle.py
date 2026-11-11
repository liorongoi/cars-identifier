"""Portable database bundles: move collected data from the online host to the offline server."""

from __future__ import annotations

import json
import tarfile
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import selectinload

from carid.engine import Engine, ReferenceLabel, SourceInfo
from carid.models import Image, Make, ReviewStatus
from carid.taxonomy import MakeSpec, ModelSpec, find_make, find_model, upsert_make

FORMAT_VERSION = 1


def export_bundle(engine: Engine, dest: Path) -> dict[str, int]:
    with engine.db.session() as s:
        makes = list(s.scalars(select(Make).options(
            selectinload(Make.models))))
        taxonomy = [{
            "name": m.name, "country": m.country, "aliases": m.aliases,
            "models": [{"name": mod.name, "aliases": mod.aliases} for mod in m.models],
        } for m in makes]
        images: list[dict[str, Any]] = []
        for im in s.scalars(select(Image).where(Image.review_status != ReviewStatus.rejected)):
            images.append({
                "file": im.path, "sha256": im.sha256, "make": im.make.name,
                "model": im.model.name,
                "label_confidence": im.label_confidence, "source": im.source,
                "source_url": im.source_url, "page_url": im.page_url, "license": im.license,
                "credit": im.credit, "title": im.title, "interior_score": im.interior_score,
                "review_status": im.review_status.value,
            })
    manifest = {"format": FORMAT_VERSION, "taxonomy": taxonomy, "images": images}
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(dest, "w:gz") as tar:
        data = json.dumps(manifest, ensure_ascii=False).encode()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as f:
            f.write(data)
            tmp = Path(f.name)
        try:
            tar.add(tmp, arcname="manifest.json")
        finally:
            tmp.unlink()
        for rec in images:
            path = engine.settings.images_dir / rec["file"]
            if path.exists():
                tar.add(path, arcname=f"images/{rec['file']}")
    return {"makes": len(taxonomy), "images": len(images)}


def import_bundle(engine: Engine, src: Path,
                  progress: Callable[[float, str], None] | None = None) -> dict[str, int]:
    counts = {"added": 0, "skipped": 0, "missing": 0}
    with tempfile.TemporaryDirectory() as tmpdir, tarfile.open(src, "r:gz") as tar:
        tar.extractall(tmpdir, filter="data")  # safe extraction: no links/absolute paths
        root = Path(tmpdir)
        manifest: dict[str, Any] = json.loads((root / "manifest.json").read_text("utf-8"))
        if manifest.get("format") != FORMAT_VERSION:
            raise ValueError(f"unsupported bundle format {manifest.get('format')}")
        with engine.db.session() as s:
            for m in manifest["taxonomy"]:
                upsert_make(s, MakeSpec(
                    name=m["name"], country=m.get("country"), aliases=m.get("aliases", []),
                    models=[ModelSpec(name=mod["name"], aliases=mod.get("aliases", []))
                            for mod in m.get("models", [])]), added_by="bundle")
        records = manifest["images"]
        for n, rec in enumerate(records, start=1):
            with engine.db.session() as s:
                if s.scalar(select(Image.id).where(Image.sha256 == rec["sha256"])) is not None:
                    counts["skipped"] += 1
                    continue
            file = root / "images" / rec["file"]
            if not file.resolve().is_relative_to(root.resolve()) or not file.exists():
                counts["missing"] += 1
                continue
            with engine.db.session() as s:
                make = find_make(s, rec["make"])
                model = find_model(make, rec["model"]) if make else None
                if make is None or model is None:
                    counts["missing"] += 1
                    continue
                label = ReferenceLabel(make.id, model.id,
                                       float(rec.get("label_confidence", 0.5)))
            res = engine.ingest(
                file.read_bytes(), label,
                SourceInfo(rec["source"], rec.get("source_url"), rec.get("page_url"),
                           rec.get("license"), rec.get("credit"), rec.get("title")),
                interior_score=rec.get("interior_score") or 1.0,
                review_status=ReviewStatus(rec.get("review_status", "pending")),
                original_sha256=rec["sha256"],
            )
            counts["added" if res.status == "added" else "skipped"] += 1
            if progress:
                progress(n / len(records), f"imported {n}/{len(records)}")
    engine.save_index()
    return counts


def export_manifest(engine: Engine, dest: Path) -> int:
    """Write links + labels + licences of collected web images (no image files) as CSV."""
    import csv

    from carid.collector.sources import MANIFEST_FIELDS

    dest.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with engine.db.session() as s, dest.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        w.writeheader()
        stmt = (select(Image).where(Image.review_status != ReviewStatus.rejected,
                                    Image.source_url.like("http%"))
                .order_by(Image.make_id, Image.model_id, Image.id))
        for im in s.scalars(stmt):
            w.writerow({"url": im.source_url, "make": im.make.name, "model": im.model.name,
                        "license": im.license or "", "credit": im.credit or "",
                        "page_url": im.page_url or "", "title": im.title or "",
                        "source": im.source})
            n += 1
    return n

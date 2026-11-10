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
from carid.models import Image, Make, ReviewStatus, VehicleModel
from carid.taxonomy import GenerationSpec, MakeSpec, ModelSpec, find_make, find_model, upsert_make

FORMAT_VERSION = 1


def export_bundle(engine: Engine, dest: Path) -> dict[str, int]:
    with engine.db.session() as s:
        makes = list(s.scalars(select(Make).options(
            selectinload(Make.models).selectinload(VehicleModel.generations))))
        taxonomy = [{
            "name": m.name, "country": m.country, "aliases": m.aliases,
            "models": [{"name": mod.name, "aliases": mod.aliases,
                        "year_start": mod.year_start, "year_end": mod.year_end,
                        "generations": [[g.name, g.year_start, g.year_end]
                                        for g in mod.generations]}
                       for mod in m.models],
        } for m in makes]
        images: list[dict[str, Any]] = []
        for im in s.scalars(select(Image).where(Image.review_status != ReviewStatus.rejected)):
            images.append({
                "file": im.path, "sha256": im.sha256, "make": im.make.name,
                "model": im.model.name,
                "generation": im.generation.name if im.generation else None,
                "year_start": im.year_start, "year_end": im.year_end,
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
                    models=[ModelSpec(name=mod["name"], aliases=mod.get("aliases", []),
                                      year_start=mod.get("year_start"),
                                      year_end=mod.get("year_end"),
                                      generations=[GenerationSpec(*g) for g in
                                                   mod.get("generations", [])])
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
                gen_id = next((g.id for g in model.generations
                               if g.name == rec.get("generation")), None)
                label = ReferenceLabel(make.id, model.id, rec.get("year_start"),
                                       rec.get("year_end"), gen_id,
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

"""FastAPI application: REST API + the analyst UI (static build)."""

from __future__ import annotations

import csv
import io
import json
import logging
from collections import defaultdict
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any, TypeVar

from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from sqlalchemy import func, select

from carid import __version__
from carid.api.schemas import AddMakeIn, AddModelIn, FeedbackIn, ImagePatch, JobIn
from carid.config import Settings, get_settings
from carid.engine import IdentifyOptions, ReferenceLabel, SourceInfo
from carid.models import (
    AuditEvent,
    Benchmark,
    Generation,
    Identification,
    Image,
    Job,
    Make,
    ReviewStatus,
    VehicleModel,
)
from carid.services import Services
from carid.taxonomy import (
    GenerationSpec,
    MakeSpec,
    ModelSpec,
    catalog_tree,
    find_make,
    load_taxonomy_file,
    upsert_make,
)
from carid.vision.imageio import ImageDecodeError, decode_image, encode_jpeg, resize_max_side

log = logging.getLogger(__name__)


def create_app(settings: Settings | None = None, services: Services | None = None,
               start_jobs: bool = True) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        svc = services or Services(settings)
        app.state.svc = svc
        if start_jobs:
            svc.jobs.start()
        svc.engine.purge_old_queries()
        yield
        svc.jobs.stop()
        svc.engine.save_index()

    app = FastAPI(title="CarID-Interior API", version=__version__, lifespan=lifespan,
                  docs_url="/api/docs", openapi_url="/api/openapi.json")

    def svc(request: Request) -> Services:
        s: Services = request.app.state.svc
        return s

    @app.exception_handler(ImageDecodeError)
    async def _bad_image(_: Request, exc: ImageDecodeError) -> JSONResponse:
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    # ------------------------------------------------------------------ system

    @app.get("/api/health")
    def health(request: Request) -> dict[str, Any]:
        engine = svc(request).engine
        warnings = []
        if not engine.describe()["deep_models"]:
            warnings.append("Deep embedder not installed: using the lightweight fallback. "
                            "Install the 'ml' extra for production accuracy.")
        return {"status": "ok", "version": __version__, "engine": engine.describe(),
                "warnings": warnings}

    @app.get("/api/stats")
    def stats(request: Request) -> dict[str, Any]:
        engine = svc(request).engine
        with engine.db.session() as s:
            by_status = dict(s.execute(select(Image.review_status, func.count())
                                       .group_by(Image.review_status)).all())
            by_source = dict(s.execute(select(Image.source, func.count())
                                       .group_by(Image.source)).all())
            n_makes = s.scalar(select(func.count()).select_from(Make)) or 0
            n_models = s.scalar(select(func.count()).select_from(VehicleModel)) or 0
            models_with_images = s.scalar(select(func.count(func.distinct(Image.model_id)))) or 0
            n_ident = s.scalar(select(func.count()).select_from(Identification)) or 0
        return {
            "makes": n_makes, "models": n_models, "models_with_images": models_with_images,
            "images": sum(by_status.values()),
            "images_by_status": {k.value: v for k, v in by_status.items()},
            "images_by_source": by_source, "identifications": n_ident,
            "index": engine.index.sizes(),
        }

    @app.get("/api/sources")
    def sources(request: Request) -> list[dict[str, Any]]:
        return svc(request).sources_status()

    @app.get("/api/audit")
    def audit(request: Request, limit: int = Query(100, le=1000)) -> list[dict[str, Any]]:
        with svc(request).engine.db.session() as s:
            rows = s.scalars(select(AuditEvent).order_by(AuditEvent.id.desc()).limit(limit))
            return [{"id": e.id, "actor": e.actor, "action": e.action, "target": e.target,
                     "details": e.details, "created_at": e.created_at.isoformat()} for e in rows]

    # ---------------------------------------------------------------- identify

    @app.post("/api/identify")
    async def identify(
        request: Request,
        file: Annotated[UploadFile, File()],
        top_k: Annotated[int, Form(ge=1, le=50)] = 10,
        enhance: Annotated[bool, Form()] = True,
        mask_faces: Annotated[bool, Form()] = True,
        mode: Annotated[str | None, Form()] = None,
        make_ids: Annotated[str | None, Form()] = None,
        year_min: Annotated[int | None, Form()] = None,
        year_max: Annotated[int | None, Form()] = None,
        manual_parts: Annotated[str | None, Form()] = None,
        case_ref: Annotated[str | None, Form(max_length=120)] = None,
        analyst: Annotated[str | None, Form(max_length=80)] = None,
    ) -> dict[str, Any]:
        engine = svc(request).engine
        data = await _read_limited(file, settings.max_upload_mb)
        if mode not in (None, "", "fusion", "global", "parts"):
            raise HTTPException(422, "mode must be fusion, global or parts")
        opts = IdentifyOptions(
            top_k=top_k, enhance=enhance, mask_faces=mask_faces,
            mode=mode or None,  # type: ignore[arg-type]
            make_ids=_parse_ids(make_ids), year_min=year_min, year_max=year_max,
            manual_parts=_parse_manual_parts(manual_parts), case_ref=case_ref or None,
            analyst=analyst or None,
        )
        return engine.identify(data, opts)

    @app.get("/api/identifications")
    def list_identifications(request: Request, limit: int = Query(50, le=500),
                             offset: int = 0, q: str | None = None) -> dict[str, Any]:
        with svc(request).engine.db.session() as s:
            stmt = select(Identification)
            if q:
                stmt = stmt.where((Identification.case_ref.contains(q))
                                  | (Identification.analyst.contains(q)))
            total = s.scalar(select(func.count()).select_from(stmt.subquery())) or 0
            rows = s.scalars(stmt.order_by(Identification.id.desc()).limit(limit).offset(offset))
            items = []
            for r in rows:
                top = (r.result.get("candidates") or [None])[0]
                items.append({
                    "id": r.id, "case_ref": r.case_ref, "analyst": r.analyst,
                    "created_at": r.created_at.isoformat(), "duration_ms": r.duration_ms,
                    "has_query_image": r.query_path is not None,
                    "top": top and {k: top[k] for k in ("make", "model", "year_start",
                                                        "year_end", "probability")},
                    "quality": (r.result.get("quality") or {}).get("verdict"),
                    "feedback_model_id": r.feedback_model_id,
                })
        return {"total": total, "items": items}

    @app.get("/api/identifications/{ident_id}")
    def get_identification(request: Request, ident_id: int) -> dict[str, Any]:
        with svc(request).engine.db.session() as s:
            r = _get_or_404(s.get(Identification, ident_id), "identification")
            return {"id": r.id, "case_ref": r.case_ref, "analyst": r.analyst,
                    "created_at": r.created_at.isoformat(), "options": r.options,
                    "has_query_image": r.query_path is not None,
                    "feedback_model_id": r.feedback_model_id, "feedback_note": r.feedback_note,
                    **r.result}

    @app.get("/api/identifications/{ident_id}/query")
    def get_query_image(request: Request, ident_id: int) -> FileResponse:
        with svc(request).engine.db.session() as s:
            r = _get_or_404(s.get(Identification, ident_id), "identification")
            if r.query_path is None:
                raise HTTPException(404, "query image not retained")
            path = _safe_path(settings.queries_dir, r.query_path)
        return FileResponse(path, media_type="image/jpeg")

    @app.post("/api/identifications/{ident_id}/feedback")
    def feedback(request: Request, ident_id: int, body: FeedbackIn) -> dict[str, Any]:
        with svc(request).engine.db.session() as s:
            r = _get_or_404(s.get(Identification, ident_id), "identification")
            if body.model_id is not None:
                _get_or_404(s.get(VehicleModel, body.model_id), "model")
            r.feedback_model_id = body.model_id
            r.feedback_note = body.note
            s.add(AuditEvent(actor=body.analyst, action="identification.feedback",
                             target=f"identification:{ident_id}",
                             details={"model_id": body.model_id, "note": body.note}))
        return {"ok": True}

    @app.get("/api/identifications/{ident_id}/export")
    def export_identification(request: Request, ident_id: int,
                              format: str = Query("json", pattern="^(json|csv)$")) -> Response:
        with svc(request).engine.db.session() as s:
            r = _get_or_404(s.get(Identification, ident_id), "identification")
            payload = {"id": r.id, "case_ref": r.case_ref, "analyst": r.analyst,
                       "created_at": r.created_at.isoformat(), "query_sha256": r.query_sha256,
                       "options": r.options, **r.result}
        name = f"carid-case-{ident_id}"
        if format == "json":
            return Response(json.dumps(payload, indent=2, default=str),
                            media_type="application/json",
                            headers={"Content-Disposition": f'attachment; filename="{name}.json"'})
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(["rank", "make", "model", "generation", "year_start", "year_end",
                    "probability", "confidence", "reference_images", "parts"])
        for c in payload.get("candidates", []):
            w.writerow([c["rank"], c["make"], c["model"], c.get("generation") or "",
                        c.get("year_start") or "", c.get("year_end") or "", c["probability"],
                        c["confidence"], c["reference_images"],
                        " ".join(sorted(c.get("part_scores", {})))])
        return Response(buf.getvalue(), media_type="text/csv",
                        headers={"Content-Disposition": f'attachment; filename="{name}.csv"'})

    # ---------------------------------------------------------------- catalog

    @app.get("/api/catalog")
    def catalog(request: Request) -> dict[str, Any]:
        engine = svc(request).engine
        with engine.db.session() as s:
            makes = catalog_tree(s)
            rows = s.execute(
                select(Image.model_id, Image.generation_id, Image.year_start, Image.year_end,
                       Image.review_status, func.count())
                .group_by(Image.model_id, Image.generation_id, Image.year_start,
                          Image.year_end, Image.review_status)).all()
            return _catalog_payload(makes, list(rows))

    @app.post("/api/catalog/makes", status_code=201)
    def add_make(request: Request, body: AddMakeIn) -> dict[str, Any]:
        services_ = svc(request)
        engine = services_.engine
        actor = f"analyst:{body.requested_by}" if body.requested_by else "analyst"
        spec = MakeSpec(name=body.name, country=body.country, aliases=body.aliases,
                        models=[_model_spec(m) for m in body.models])
        if not spec.models and settings.taxonomy_file.exists():
            seeded = next((m for m in load_taxonomy_file(settings.taxonomy_file)
                           if m.name.lower() == body.name.lower()), None)
            if seeded:
                spec = seeded
        jobs: list[int] = []
        with engine.db.session() as s:
            existed = find_make(s, body.name) is not None
            make, models_added, gens_added = upsert_make(s, spec, added_by=actor)
            make_id = make.id
            s.add(AuditEvent(actor=body.requested_by, action="make.add",
                             target=f"make:{make_id}",
                             details={"name": make.name, "models_added": models_added,
                                      "existed": existed}))
            has_models = bool(make.models)
        if body.use_wikidata:
            jobs.append(services_.jobs.submit("expand_make", {
                "make": body.name, "collect": body.collect,
                "per_model_limit": body.per_model_limit, "added_by": body.requested_by},
                requested_by=body.requested_by))
        elif body.collect and has_models:
            jobs.append(services_.jobs.submit("collect", {
                "make_ids": [make_id], "per_model_limit": body.per_model_limit},
                requested_by=body.requested_by))
        return {"make_id": make_id, "existed": existed, "models_added": models_added,
                "generations_added": gens_added, "jobs": jobs}

    @app.post("/api/catalog/models", status_code=201)
    def add_model(request: Request, body: AddModelIn) -> dict[str, Any]:
        services_ = svc(request)
        with services_.engine.db.session() as s:
            make = _get_or_404(s.get(Make, body.make_id), "make")
            _, models_added, gens_added = upsert_make(
                s, MakeSpec(name=make.name, models=[_model_spec(body)]),
                added_by=f"analyst:{body.requested_by or ''}")
            model_id = next(m.id for m in make.models if m.name.lower() == body.name.lower()
                            or body.name.lower() in [a.lower() for a in m.aliases])
            s.add(AuditEvent(actor=body.requested_by, action="model.add",
                             target=f"model:{model_id}", details={"name": body.name}))
        jobs = []
        if body.collect:
            jobs.append(services_.jobs.submit("collect", {
                "model_ids": [model_id], "make_ids": [body.make_id],
                "per_model_limit": body.per_model_limit}, requested_by=body.requested_by))
        return {"model_id": model_id, "models_added": models_added,
                "generations_added": gens_added, "jobs": jobs}

    # ---------------------------------------------------------------- images

    @app.get("/api/images")
    def list_images(request: Request, make_id: int | None = None, model_id: int | None = None,
                    generation_id: int | None = None, year: int | None = None,
                    status: str | None = None, source: str | None = None,
                    limit: int = Query(60, le=500), offset: int = 0) -> dict[str, Any]:
        with svc(request).engine.db.session() as s:
            stmt = select(Image)
            if make_id:
                stmt = stmt.where(Image.make_id == make_id)
            if model_id:
                stmt = stmt.where(Image.model_id == model_id)
            if generation_id:
                stmt = stmt.where(Image.generation_id == generation_id)
            if year:
                stmt = stmt.where(Image.year_start <= year, Image.year_end >= year)
            if status:
                stmt = stmt.where(Image.review_status == ReviewStatus(status))
            if source:
                stmt = stmt.where(Image.source == source)
            total = s.scalar(select(func.count()).select_from(stmt.subquery())) or 0
            rows = s.scalars(stmt.order_by(Image.id.desc()).limit(limit).offset(offset))
            return {"total": total, "items": [_image_dict(i) for i in rows]}

    @app.get("/api/images/{image_id}")
    def get_image(request: Request, image_id: int) -> dict[str, Any]:
        with svc(request).engine.db.session() as s:
            im = _get_or_404(s.get(Image, image_id), "image")
            d = _image_dict(im)
            d["parts"] = [{"part": p.part, "bbox": p.bbox, "score": p.score} for p in im.parts]
            return d

    @app.get("/api/images/{image_id}/file")
    def image_file(request: Request, image_id: int, size: int = Query(0, ge=0, le=2400),
                   bbox: str | None = None) -> Response:
        with svc(request).engine.db.session() as s:
            im = _get_or_404(s.get(Image, image_id), "image")
            path = _safe_path(settings.images_dir, im.path)
        if not size and not bbox:
            return FileResponse(path, media_type="image/jpeg",
                                headers={"Cache-Control": "max-age=3600"})
        img = decode_image(path.read_bytes())
        if bbox:
            x0, y0, x1, y1 = _parse_bbox(bbox)
            h, w = img.shape[:2]
            img = img[max(0, y0):min(h, y1), max(0, x0):min(w, x1)]
            if img.size == 0:
                raise HTTPException(422, "empty crop")
        if size:
            img = resize_max_side(img, size)
        return Response(encode_jpeg(img, quality=85), media_type="image/jpeg",
                        headers={"Cache-Control": "max-age=3600"})

    @app.patch("/api/images/{image_id}")
    def patch_image(request: Request, image_id: int, body: ImagePatch) -> dict[str, Any]:
        engine = svc(request).engine
        with engine.db.session() as s:
            im = _get_or_404(s.get(Image, image_id), "image")
            current = (im.make_id, im.model_id, im.year_start, im.year_end)
        if body.model_id is not None or body.year_start is not None or body.make_id is not None:
            model_id = body.model_id or current[1]
            with engine.db.session() as s:
                model = _get_or_404(s.get(VehicleModel, model_id), "model")
                make_id = model.make_id
            if body.make_id is not None and body.make_id != make_id:
                raise HTTPException(422, "model does not belong to make")
            ys = body.year_start if body.year_start is not None else current[2]
            ye = body.year_end if body.year_end is not None else (
                body.year_start if body.year_start is not None else current[3])
            if ys is not None and ye is not None and ye < ys:
                raise HTTPException(422, "year_end before year_start")
            engine.relabel(image_id, ReferenceLabel(make_id, model_id, ys, ye, None, 1.0),
                           body.actor)
        if body.review_status is not None:
            engine.set_review_status(image_id, ReviewStatus(body.review_status), body.actor)
        engine.save_index()
        return get_image(request, image_id)

    @app.post("/api/images/upload", status_code=201)
    async def upload_reference(
        request: Request,
        file: Annotated[UploadFile, File()],
        model_id: Annotated[int, Form()],
        year_start: Annotated[int | None, Form(ge=1900, le=2100)] = None,
        year_end: Annotated[int | None, Form(ge=1900, le=2100)] = None,
        license: Annotated[str | None, Form(max_length=200)] = None,
        credit: Annotated[str | None, Form(max_length=400)] = None,
        actor: Annotated[str | None, Form(max_length=80)] = None,
    ) -> dict[str, Any]:
        engine = svc(request).engine
        data = await _read_limited(file, settings.max_upload_mb)
        with engine.db.session() as s:
            model = _get_or_404(s.get(VehicleModel, model_id), "model")
            make_id = model.make_id
        res = engine.ingest(
            data, ReferenceLabel(make_id, model_id, year_start, year_end or year_start,
                                 None, 0.95),
            SourceInfo("upload", license=license, credit=credit or actor,
                       title=file.filename),
            interior_score=1.0, review_status=ReviewStatus.approved)
        engine.save_index()
        return {"status": res.status, "image_id": res.image_id, "detail": res.detail}

    # ---------------------------------------------------------------- jobs

    @app.get("/api/jobs")
    def list_jobs(request: Request, limit: int = Query(50, le=500)) -> list[dict[str, Any]]:
        with svc(request).engine.db.session() as s:
            rows = s.scalars(select(Job).order_by(Job.id.desc()).limit(limit))
            return [_job_dict(j, with_log=False) for j in rows]

    @app.get("/api/jobs/{job_id}")
    def get_job(request: Request, job_id: int) -> dict[str, Any]:
        with svc(request).engine.db.session() as s:
            return _job_dict(_get_or_404(s.get(Job, job_id), "job"), with_log=True)

    @app.post("/api/jobs", status_code=201)
    def create_job(request: Request, body: JobIn) -> dict[str, Any]:
        if body.kind == "import_bundle":
            p = Path(str(body.params.get("path", "")))
            if not p.is_file():
                raise HTTPException(422, "bundle path not found on server")
        job_id = svc(request).jobs.submit(body.kind, body.params, body.requested_by)
        return {"job_id": job_id}

    @app.post("/api/jobs/{job_id}/cancel")
    def cancel_job(request: Request, job_id: int) -> dict[str, Any]:
        return {"cancelled": svc(request).jobs.cancel(job_id)}

    # ---------------------------------------------------------------- benchmarks

    @app.get("/api/benchmarks")
    def list_benchmarks(request: Request) -> list[dict[str, Any]]:
        with svc(request).engine.db.session() as s:
            rows = s.scalars(select(Benchmark).order_by(Benchmark.id.desc()).limit(50))
            return [{"id": b.id, "embedder": b.embedder, "created_at": b.created_at.isoformat(),
                     "best_mode": b.report.get("best_mode"), "summary": b.report.get("summary"),
                     "test_images": b.report.get("test_images")} for b in rows]

    @app.get("/api/benchmarks/{bench_id}")
    def get_benchmark(request: Request, bench_id: int) -> dict[str, Any]:
        with svc(request).engine.db.session() as s:
            b = _get_or_404(s.get(Benchmark, bench_id), "benchmark")
            return {"id": b.id, "created_at": b.created_at.isoformat(), **b.report}

    # ---------------------------------------------------------------- UI

    ui_dir = settings.ui_dir or Path(__file__).resolve().parents[3] / "frontend" / "dist"
    if ui_dir.is_dir():
        app.mount("/assets", StaticFiles(directory=ui_dir / "assets"), name="assets")

        @app.get("/{path:path}", include_in_schema=False)
        def spa(path: str) -> FileResponse:
            if path.startswith("api/"):
                raise HTTPException(404)
            candidate = (ui_dir / path).resolve()
            if path and candidate.is_file() and candidate.is_relative_to(ui_dir):
                return FileResponse(candidate)
            return FileResponse(ui_dir / "index.html")

    return app


# ---------------------------------------------------------------------- helpers


async def _read_limited(file: UploadFile, max_mb: int) -> bytes:
    limit = max_mb * 1024 * 1024
    data = await file.read(limit + 1)
    if len(data) > limit:
        raise HTTPException(413, f"file larger than {max_mb} MB")
    if not data:
        raise HTTPException(422, "empty file")
    return data


def _parse_ids(raw: str | None) -> list[int] | None:
    if not raw:
        return None
    try:
        return [int(x) for x in raw.split(",") if x.strip()]
    except ValueError as exc:
        raise HTTPException(422, "make_ids must be comma-separated integers") from exc


def _parse_bbox(raw: str) -> tuple[int, int, int, int]:
    try:
        x0, y0, x1, y1 = (int(float(v)) for v in raw.split(","))
    except ValueError as exc:
        raise HTTPException(422, "bbox must be x0,y0,x1,y1") from exc
    if x1 <= x0 or y1 <= y0:
        raise HTTPException(422, "bbox must have positive size")
    return x0, y0, x1, y1


def _parse_manual_parts(raw: str | None) -> list[dict[str, Any]]:
    from carid.config import PART_NAMES

    if not raw:
        return []
    try:
        items = json.loads(raw)
        out = []
        for it in items:
            if it["part"] not in PART_NAMES:
                raise ValueError(f"unknown part {it['part']}")
            x0, y0, x1, y1 = (int(v) for v in it["bbox"])
            if x1 <= x0 or y1 <= y0:
                raise ValueError("bbox must have positive size")
            out.append({"part": it["part"], "bbox": [x0, y0, x1, y1]})
        return out
    except (ValueError, KeyError, TypeError) as exc:
        raise HTTPException(422, f"invalid manual_parts: {exc}") from exc


T = TypeVar("T")


def _get_or_404(obj: T | None, what: str) -> T:
    if obj is None:
        raise HTTPException(404, f"{what} not found")
    return obj


def _safe_path(root: Path, rel: str) -> Path:
    path = (root / rel).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise HTTPException(404, "file not found")
    return path


def _model_spec(m: Any) -> ModelSpec:
    return ModelSpec(name=m.name, aliases=m.aliases, year_start=m.year_start,
                     year_end=m.year_end,
                     generations=[GenerationSpec(g.name, g.year_start, g.year_end)
                                  for g in m.generations])


def _image_dict(im: Image) -> dict[str, Any]:
    return {
        "id": im.id, "make_id": im.make_id, "model_id": im.model_id,
        "make": im.make.name, "model": im.model.name,
        "generation_id": im.generation_id,
        "generation": im.generation.name if im.generation else None,
        "year_start": im.year_start, "year_end": im.year_end,
        "label_confidence": im.label_confidence, "source": im.source,
        "source_url": im.source_url, "page_url": im.page_url, "license": im.license,
        "credit": im.credit, "title": im.title, "width": im.width, "height": im.height,
        "quality": im.quality, "interior_score": im.interior_score,
        "faces_blurred": im.faces_blurred, "review_status": im.review_status.value,
        "created_at": im.created_at.isoformat(),
    }


def _job_dict(j: Job, with_log: bool) -> dict[str, Any]:
    d = {"id": j.id, "kind": j.kind, "params": j.params, "status": j.status.value,
         "progress": j.progress, "message": j.message, "counters": j.counters,
         "requested_by": j.requested_by, "created_at": j.created_at.isoformat(),
         "started_at": j.started_at.isoformat() if j.started_at else None,
         "finished_at": j.finished_at.isoformat() if j.finished_at else None}
    if with_log:
        d["log"] = j.log
    return d


def _catalog_payload(makes: list[Make], rows: list[Any]) -> dict[str, Any]:
    """Build Make › Model › Generation/Year tree with image counts (FR-DB-3)."""
    per_model: dict[int, int] = defaultdict(int)
    per_model_approved: dict[int, int] = defaultdict(int)
    per_gen: dict[int, int] = defaultdict(int)
    per_year: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    unknown_year: dict[int, int] = defaultdict(int)
    for model_id, gen_id, ys, ye, status, n in rows:
        if status == ReviewStatus.rejected:
            continue
        per_model[model_id] += n
        if status == ReviewStatus.approved:
            per_model_approved[model_id] += n
        if gen_id is not None:
            per_gen[gen_id] += n
        if ys is None:
            unknown_year[model_id] += n
        else:
            for y in range(ys, (ye or ys) + 1):
                per_year[model_id][y] += n
    out_makes = []
    total = 0
    for make in makes:
        models = []
        make_total = 0
        for m in make.models:
            years = dict(sorted(per_year[m.id].items()))
            models.append({
                "id": m.id, "name": m.name, "aliases": m.aliases,
                "year_start": m.year_start, "year_end": m.year_end,
                "images": per_model[m.id], "approved": per_model_approved[m.id],
                "unknown_year": unknown_year[m.id], "years": years,
                "generations": [_gen_dict(g, per_gen[g.id]) for g in m.generations],
            })
            make_total += per_model[m.id]
        total += make_total
        out_makes.append({"id": make.id, "name": make.name, "country": make.country,
                          "aliases": make.aliases, "added_by": make.added_by,
                          "images": make_total, "models": models})
    return {"total_images": total, "makes": out_makes}


def _gen_dict(g: Generation, n: int) -> dict[str, Any]:
    return {"id": g.id, "name": g.name, "year_start": g.year_start, "year_end": g.year_end,
            "images": n}


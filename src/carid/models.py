"""Database schema: Make › Model › (Generation) › Year, reference images, jobs, cases."""

from __future__ import annotations

import enum
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utcnow() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    type_annotation_map = {dict[str, Any]: JSON, list[Any]: JSON}


class Make(Base):
    __tablename__ = "makes"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(80), unique=True, index=True)
    country: Mapped[str | None] = mapped_column(String(60))
    aliases: Mapped[list[Any]] = mapped_column(default=list)
    wikidata_id: Mapped[str | None] = mapped_column(String(20))
    added_by: Mapped[str | None] = mapped_column(String(80))  # "seed", "analyst:<name>", ...
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    models: Mapped[list[VehicleModel]] = relationship(
        back_populates="make", cascade="all, delete-orphan", order_by="VehicleModel.name"
    )


class VehicleModel(Base):
    __tablename__ = "vehicle_models"
    __table_args__ = (UniqueConstraint("make_id", "name"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    make_id: Mapped[int] = mapped_column(ForeignKey("makes.id", ondelete="CASCADE"), index=True)
    name: Mapped[str] = mapped_column(String(80))
    aliases: Mapped[list[Any]] = mapped_column(default=list)
    year_start: Mapped[int | None] = mapped_column(Integer)
    year_end: Mapped[int | None] = mapped_column(Integer)  # None = still in production

    make: Mapped[Make] = relationship(back_populates="models")
    generations: Mapped[list[Generation]] = relationship(
        back_populates="model", cascade="all, delete-orphan", order_by="Generation.year_start"
    )


class Generation(Base):
    """A run of model years with (essentially) the same interior."""

    __tablename__ = "generations"
    __table_args__ = (UniqueConstraint("model_id", "name"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    model_id: Mapped[int] = mapped_column(
        ForeignKey("vehicle_models.id", ondelete="CASCADE"), index=True
    )
    name: Mapped[str] = mapped_column(String(80))
    year_start: Mapped[int] = mapped_column(Integer)
    year_end: Mapped[int | None] = mapped_column(Integer)

    model: Mapped[VehicleModel] = relationship(back_populates="generations")


class ReviewStatus(enum.StrEnum):
    pending = "pending"
    approved = "approved"
    rejected = "rejected"


class Image(Base):
    __tablename__ = "images"
    __table_args__ = (Index("ix_images_label", "model_id", "generation_id"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    sha256: Mapped[str] = mapped_column(String(64), unique=True)
    phash: Mapped[str] = mapped_column(String(16), index=True)
    path: Mapped[str] = mapped_column(String(400))  # relative to settings.images_dir
    width: Mapped[int] = mapped_column(Integer)
    height: Mapped[int] = mapped_column(Integer)

    make_id: Mapped[int] = mapped_column(ForeignKey("makes.id", ondelete="CASCADE"), index=True)
    model_id: Mapped[int] = mapped_column(ForeignKey("vehicle_models.id", ondelete="CASCADE"))
    generation_id: Mapped[int | None] = mapped_column(
        ForeignKey("generations.id", ondelete="SET NULL")
    )
    year_start: Mapped[int | None] = mapped_column(Integer)
    year_end: Mapped[int | None] = mapped_column(Integer)
    label_confidence: Mapped[float] = mapped_column(Float, default=0.5)

    source: Mapped[str] = mapped_column(String(40))
    source_url: Mapped[str | None] = mapped_column(String(1000))
    page_url: Mapped[str | None] = mapped_column(String(1000))
    license: Mapped[str | None] = mapped_column(String(200))
    credit: Mapped[str | None] = mapped_column(String(400))
    title: Mapped[str | None] = mapped_column(String(500))

    quality: Mapped[dict[str, Any]] = mapped_column(default=dict)
    interior_score: Mapped[float | None] = mapped_column(Float)
    faces_blurred: Mapped[int] = mapped_column(Integer, default=0)
    review_status: Mapped[ReviewStatus] = mapped_column(
        Enum(ReviewStatus), default=ReviewStatus.pending
    )
    indexed: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    make: Mapped[Make] = relationship()
    model: Mapped[VehicleModel] = relationship()
    generation: Mapped[Generation | None] = relationship()
    parts: Mapped[list[ImagePart]] = relationship(
        back_populates="image", cascade="all, delete-orphan"
    )


class ImagePart(Base):
    """A detected part crop of a reference image; its vector lives in the part's index."""

    __tablename__ = "image_parts"

    id: Mapped[int] = mapped_column(primary_key=True)
    image_id: Mapped[int] = mapped_column(ForeignKey("images.id", ondelete="CASCADE"), index=True)
    part: Mapped[str] = mapped_column(String(40), index=True)
    bbox: Mapped[list[Any]] = mapped_column()  # [x0, y0, x1, y1] in pixels
    score: Mapped[float] = mapped_column(Float)

    image: Mapped[Image] = relationship(back_populates="parts")


class SeenUrl(Base):
    """URLs already examined by the collector (kept or rejected) so jobs can resume."""

    __tablename__ = "seen_urls"

    url: Mapped[str] = mapped_column(String(1000), primary_key=True)
    outcome: Mapped[str] = mapped_column(String(40))
    image_id: Mapped[int | None] = mapped_column(Integer)
    seen_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class JobStatus(enum.StrEnum):
    queued = "queued"
    running = "running"
    succeeded = "succeeded"
    failed = "failed"
    cancelled = "cancelled"


class Job(Base):
    __tablename__ = "jobs"

    id: Mapped[int] = mapped_column(primary_key=True)
    kind: Mapped[str] = mapped_column(String(40))
    params: Mapped[dict[str, Any]] = mapped_column(default=dict)
    status: Mapped[JobStatus] = mapped_column(Enum(JobStatus), default=JobStatus.queued, index=True)
    progress: Mapped[float] = mapped_column(Float, default=0.0)
    message: Mapped[str] = mapped_column(String(500), default="")
    counters: Mapped[dict[str, Any]] = mapped_column(default=dict)
    log: Mapped[str] = mapped_column(Text, default="")
    cancel_requested: Mapped[bool] = mapped_column(Boolean, default=False)
    requested_by: Mapped[str | None] = mapped_column(String(80))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class Identification(Base):
    """One analyst query (a 'case'): input, settings and result. Also the audit trail."""

    __tablename__ = "identifications"

    id: Mapped[int] = mapped_column(primary_key=True)
    case_ref: Mapped[str | None] = mapped_column(String(120), index=True)
    analyst: Mapped[str | None] = mapped_column(String(80))
    query_sha256: Mapped[str] = mapped_column(String(64))
    query_path: Mapped[str | None] = mapped_column(String(400))  # None once purged
    options: Mapped[dict[str, Any]] = mapped_column(default=dict)
    result: Mapped[dict[str, Any]] = mapped_column(default=dict)
    feedback_model_id: Mapped[int | None] = mapped_column(Integer)
    feedback_note: Mapped[str | None] = mapped_column(String(1000))
    duration_ms: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow,
                                                 index=True)


class Benchmark(Base):
    __tablename__ = "benchmarks"

    id: Mapped[int] = mapped_column(primary_key=True)
    job_id: Mapped[int | None] = mapped_column(Integer)
    embedder: Mapped[str] = mapped_column(String(80))
    report: Mapped[dict[str, Any]] = mapped_column(default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class AuditEvent(Base):
    __tablename__ = "audit_events"

    id: Mapped[int] = mapped_column(primary_key=True)
    actor: Mapped[str | None] = mapped_column(String(80))
    action: Mapped[str] = mapped_column(String(60))
    target: Mapped[str] = mapped_column(String(120))
    details: Mapped[dict[str, Any]] = mapped_column(default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

"""Request bodies for the REST API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class GenerationIn(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    year_start: int = Field(ge=1900, le=2100)
    year_end: int | None = Field(default=None, ge=1900, le=2100)


class ModelIn(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    aliases: list[str] = Field(default_factory=list)
    year_start: int | None = Field(default=None, ge=1900, le=2100)
    year_end: int | None = Field(default=None, ge=1900, le=2100)
    generations: list[GenerationIn] = Field(default_factory=list)


class AddMakeIn(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    country: str | None = Field(default=None, max_length=60)
    aliases: list[str] = Field(default_factory=list)
    models: list[ModelIn] = Field(default_factory=list)
    use_wikidata: bool = False
    collect: bool = True
    per_model_limit: int = Field(default=150, ge=1, le=5000)
    requested_by: str | None = Field(default=None, max_length=80)

    @field_validator("name")
    @classmethod
    def strip_name(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("name must not be blank")
        return v


class AddModelIn(ModelIn):
    make_id: int
    collect: bool = True
    per_model_limit: int = Field(default=150, ge=1, le=5000)
    requested_by: str | None = Field(default=None, max_length=80)


class JobIn(BaseModel):
    kind: Literal["collect", "reindex", "benchmark", "expand_make", "import_bundle",
                  "train_adapter"]
    params: dict[str, Any] = Field(default_factory=dict)
    requested_by: str | None = Field(default=None, max_length=80)


class ImagePatch(BaseModel):
    review_status: Literal["pending", "approved", "rejected"] | None = None
    make_id: int | None = None
    model_id: int | None = None
    year_start: int | None = Field(default=None, ge=1900, le=2100)
    year_end: int | None = Field(default=None, ge=1900, le=2100)
    actor: str | None = Field(default=None, max_length=80)


class FeedbackIn(BaseModel):
    model_id: int | None = None
    note: str | None = Field(default=None, max_length=1000)
    analyst: str | None = Field(default=None, max_length=80)

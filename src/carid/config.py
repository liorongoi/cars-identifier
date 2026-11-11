"""Application settings.

Values come from (highest priority first): environment variables prefixed
``CARID_``, a ``.env`` file, then the defaults below.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PartName = Literal[
    "whole",
    "dashboard",
    "steering_wheel",
    "instrument_cluster",
    "infotainment",
    "air_vent",
    "center_console",
    "gear_selector",
    "climate_control",
    "door_panel",
]

PART_NAMES: tuple[str, ...] = (
    "whole",
    "dashboard",
    "steering_wheel",
    "instrument_cluster",
    "infotainment",
    "air_vent",
    "center_console",
    "gear_selector",
    "climate_control",
    "door_panel",
)

# How much each part's evidence counts in the fused score. Tuned by the benchmark.
DEFAULT_PART_WEIGHTS: dict[str, float] = {
    "whole": 1.0,
    "dashboard": 0.9,
    "steering_wheel": 1.2,
    "instrument_cluster": 1.0,
    "infotainment": 0.9,
    "air_vent": 0.7,
    "center_console": 0.6,
    "gear_selector": 0.6,
    "climate_control": 0.7,
    "door_panel": 0.5,
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CARID_", env_file=".env", extra="ignore")

    data_dir: Path = Path("var")
    database_url: str | None = None  # defaults to SQLite inside data_dir

    # Models
    embedder: Literal["auto", "clip", "handcrafted"] = "auto"
    clip_model: str = "ViT-B-32"
    clip_pretrained: str = "laion2b_s34b_b79k"
    detector: Literal["auto", "owl", "layout"] = "auto"
    owl_model: str = "google/owlv2-base-patch16-ensemble"
    models_cache: Path = Path("models_cache")
    allow_model_download: bool = True  # set False on the offline server

    # Matching
    # "auto": use the best mode (and adapter on/off) of the latest benchmark; before any
    # benchmark exists it falls back to "fusion".
    matcher_mode: Literal["auto", "fusion", "global", "parts"] = "auto"
    use_adapter: bool = True  # apply the learned LDA adapter when one has been trained
    adapter_shrinkage: float = Field(0.5, ge=0.0, le=1.0)
    top_k: int = Field(10, ge=1, le=50)
    neighbors_per_part: int = Field(50, ge=5, le=500)
    votes_per_class: int = Field(3, ge=1, le=20)
    similarity_temperature: float = Field(0.05, gt=0)
    softmax_temperature: float = Field(1.0, gt=0)
    low_confidence_threshold: float = 0.25
    part_weights: dict[str, float] = Field(default_factory=lambda: dict(DEFAULT_PART_WEIGHTS))

    # Pre-processing
    enhance_low_light: bool = True
    mask_faces: bool = True
    max_upload_mb: int = 40
    max_image_pixels: int = 60_000_000

    # Privacy
    query_retention_days: int = Field(30, ge=0)

    # Collection
    user_agent: str = (
        "CarID-Interior/0.1 (vehicle interior research; "
        "+https://github.com/liorongoi/cars-identifier)"
    )
    request_timeout_s: float = 30.0
    min_request_interval_s: float = 1.0
    max_retries: int = 5
    max_download_mb: int = 25
    min_image_side: int = 320
    near_duplicate_distance: int = 4  # pHash bits; re-encodes/resizes score 0-2
    interior_threshold: float = 0.55
    sources_file: Path = Path("config/sources.yaml")
    taxonomy_file: Path = Path("data/taxonomy/pilot.yaml")
    ui_dir: Path | None = None  # built front end; defaults to frontend/dist in the repo

    @property
    def db_url(self) -> str:
        if self.database_url:
            return self.database_url
        return f"sqlite:///{(self.data_dir / 'carid.db').resolve()}"

    @property
    def images_dir(self) -> Path:
        return self.data_dir / "images"

    @property
    def queries_dir(self) -> Path:
        return self.data_dir / "queries"

    @property
    def index_dir(self) -> Path:
        return self.data_dir / "index"

    @property
    def reports_dir(self) -> Path:
        return self.data_dir / "reports"

    def ensure_dirs(self) -> None:
        for d in (self.data_dir, self.images_dir, self.queries_dir, self.index_dir,
                  self.reports_dir):
            d.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()

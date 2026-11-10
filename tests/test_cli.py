from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from carid.cli import app
from carid.config import Settings, get_settings
from tests.conftest import CLASS_STYLES, draw_interior, to_jpeg


@pytest.fixture
def env(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> Settings:
    for key, value in {
        "CARID_DATA_DIR": settings.data_dir, "CARID_EMBEDDER": "handcrafted",
        "CARID_DETECTOR": "layout", "CARID_TAXONOMY_FILE": settings.taxonomy_file,
        "CARID_SOURCES_FILE": settings.sources_file, "CARID_MIN_IMAGE_SIDE": 200,
        "CARID_NEAR_DUPLICATE_DISTANCE": 1, "CARID_MODELS_CACHE": settings.models_cache,
    }.items():
        monkeypatch.setenv(key, str(value))
    get_settings.cache_clear()
    yield settings
    get_settings.cache_clear()


def test_cli_end_to_end(env: Settings, tmp_path: Path) -> None:
    runner = CliRunner()
    r = runner.invoke(app, ["init"])
    assert r.exit_code == 0, r.output
    assert "'makes': 2" in r.output
    # Import folder layout and collect from it.
    rng = np.random.default_rng(0)
    for key, style in CLASS_STYLES.items():
        make, model, year = key.split("/")
        d = tmp_path / "imports" / make / model / year
        d.mkdir(parents=True)
        for i in range(3):
            (d / f"{i}.jpg").write_bytes(to_jpeg(draw_interior(style, rng)))
    r = runner.invoke(app, ["collect", "--per-model-limit", "10"])
    assert r.exit_code == 0, r.output
    assert "succeeded" in r.output and '"added": 12' in r.output
    r = runner.invoke(app, ["collect", "--make", "Nope"])
    assert r.exit_code != 0
    q = tmp_path / "q.jpg"
    q.write_bytes(to_jpeg(draw_interior(CLASS_STYLES["Beta/City/2012"], rng)))
    r = runner.invoke(app, ["identify", str(q), "--top-k", "3"])
    assert r.exit_code == 0, r.output
    assert "Beta" in r.output and "City" in r.output.splitlines()[2]
    r = runner.invoke(app, ["index"])
    assert r.exit_code == 0 and '"whole": 12' in r.output
    r = runner.invoke(app, ["benchmark", "--max-test", "4", "--degradations", "none"])
    assert r.exit_code == 0 and "best_mode" in r.output
    bundle = tmp_path / "b.tar.gz"
    r = runner.invoke(app, ["export-bundle", str(bundle)])
    assert r.exit_code == 0 and bundle.exists()
    r = runner.invoke(app, ["import-bundle", str(bundle)])
    assert r.exit_code == 0 and "'added': 0" in r.output
    r = runner.invoke(app, ["add-make", "Alpha", "--no-wikidata"])
    assert r.exit_code == 0 and "added from seed file" in r.output
    r = runner.invoke(app, ["add-make", "Gamma", "--no-wikidata"])
    assert r.exit_code != 0  # not in seed file and Wikidata disabled

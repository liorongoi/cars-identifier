"""Command-line interface: `carid --help`."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer

from carid.config import get_settings

if TYPE_CHECKING:
    from carid.services import Services

app = typer.Typer(add_completion=False, no_args_is_help=True,
                  help="CarID-Interior — vehicle identification from interior images.")


def _services() -> Services:
    from carid.services import Services

    return Services(get_settings())


@app.callback()
def main(verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False) -> None:
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")


@app.command()
def init(seed_file: Annotated[Path | None, typer.Option(help="taxonomy YAML")] = None) -> None:
    """Create the database and load the seed taxonomy."""
    from carid.db import Database
    from carid.taxonomy import seed

    settings = get_settings()
    db = Database(settings)
    with db.session() as s:
        totals = seed(s, seed_file or settings.taxonomy_file)
    typer.echo(f"database: {settings.db_url}")
    typer.echo(f"seeded: {totals}")


@app.command()
def collect(
    make: Annotated[list[str] | None, typer.Option(help="make name (repeatable); default all")]
    = None,
    model: Annotated[str | None, typer.Option(help="limit to one model of --make")] = None,
    source: Annotated[list[str] | None, typer.Option(help="source name (repeatable)")] = None,
    per_model_limit: int = 150,
) -> None:
    """Collect reference images from the enabled sources (online)."""
    from carid.taxonomy import find_make, find_model

    svc = _services()
    with svc.engine.db.session() as s:
        make_ids: list[int] = []
        model_ids: list[int] = []
        for name in make or []:
            mk = find_make(s, name)
            if mk is None:
                raise typer.BadParameter(f"unknown make: {name} (add it first: carid add-make)")
            make_ids.append(mk.id)
            if model:
                md = find_model(mk, model)
                if md is None:
                    raise typer.BadParameter(f"unknown model {model} for {name}")
                model_ids.append(md.id)
    job = svc.jobs.submit("collect", {"make_ids": make_ids or None,
                                      "model_ids": model_ids or None,
                                      "sources": source or None,
                                      "per_model_limit": per_model_limit}, "cli")
    svc.jobs.run_pending()
    _print_job(svc, job)


@app.command("add-make")
def add_make(name: str, wikidata: bool = True, collect: bool = False) -> None:
    """Add a make (from the seed file, or Wikidata when online)."""
    from carid.taxonomy import load_taxonomy_file, upsert_make

    svc = _services()
    settings = get_settings()
    seeded = next((m for m in load_taxonomy_file(settings.taxonomy_file)
                   if m.name.lower() == name.lower()), None)
    if seeded:
        with svc.engine.db.session() as s:
            _, models, gens = upsert_make(s, seeded, added_by="cli")
        typer.echo(f"added from seed file: +{models} models, +{gens} generations")
    elif wikidata:
        job = svc.jobs.submit("expand_make", {"make": name, "collect": collect,
                                              "added_by": "cli"}, "cli")
        svc.jobs.run_pending()
        _print_job(svc, job)
    else:
        raise typer.BadParameter("make not in seed file; enable --wikidata or use the UI")


@app.command()
def index(rebuild: bool = False) -> None:
    """Show index sizes, or rebuild the index (after changing the embedder)."""
    svc = _services()
    if rebuild:
        n = svc.engine.rebuild_index(progress=lambda f, m: typer.echo(f"{f:6.1%} {m}"))
        typer.echo(f"re-indexed {n} images")
    typer.echo(json.dumps(svc.engine.describe(), indent=2))


@app.command()
def identify(image: Path, top_k: int = 10, mode: str | None = None) -> None:
    """Identify one image and print the ranked table."""
    from carid.engine import IdentifyOptions

    svc = _services()
    result = svc.engine.identify(image.read_bytes(),
                                 IdentifyOptions(top_k=top_k, mode=mode,  # type: ignore[arg-type]
                                                 analyst="cli", case_ref=image.name))
    q = result["quality"]
    typer.echo(f"quality: {q['verdict']} ({q['score']}) {', '.join(q['issues'])}")
    typer.echo(f"{'#':>2}  {'make':<15} {'model':<20} {'years':<11} {'prob':>6}  refs")
    for c in result["candidates"]:
        years = f"{c['year_start'] or '?'}-{c['year_end'] or 'now'}"
        typer.echo(f"{c['rank']:>2}  {c['make']:<15} {c['model']:<20} {years:<11} "
                   f"{c['probability']:>6.1%}  {c['reference_images']}")
    for w in result["warnings"]:
        typer.echo(f"! {w}")


@app.command()
def benchmark(max_test: int = 200, test_fraction: float = 0.2, seed: int = 0,
              degradations: Annotated[str | None, typer.Option(
                  help="comma list; default all, 'none' for clean only")] = None) -> None:
    """Evaluate matcher modes on a hold-out split (clean + degraded)."""
    from carid.evaluation import run_benchmark

    svc = _services()
    degs = None if degradations is None else (
        [] if degradations == "none" else degradations.split(","))
    report = run_benchmark(svc.engine, max_test=max_test, test_fraction=test_fraction,
                           seed=seed, degradations=degs)
    typer.echo(json.dumps({k: report[k] for k in (
        "embedder", "reference_images", "test_images", "classes", "summary", "best_mode",
        "mean_latency_ms")}, indent=2))


@app.command("export-bundle")
def export_bundle_cmd(dest: Path) -> None:
    """Export images + metadata to a .tar.gz for the offline server."""
    from carid.bundle import export_bundle

    typer.echo(export_bundle(_services().engine, dest))


@app.command("import-bundle")
def import_bundle_cmd(src: Path) -> None:
    """Import a bundle produced by export-bundle (re-indexes locally)."""
    from carid.bundle import import_bundle

    typer.echo(import_bundle(_services().engine, src))


@app.command("fetch-models")
def fetch_models() -> None:
    """Download and cache deep-model weights for offline use (needs the 'ml' extra)."""
    import urllib.request

    settings = get_settings()
    settings.models_cache.mkdir(parents=True, exist_ok=True)
    from carid.vision.deep import ClipEmbedder, OwlPartDetector

    ClipEmbedder(settings.clip_model, settings.clip_pretrained, str(settings.models_cache), True)
    OwlPartDetector(settings.owl_model, str(settings.models_cache), True)
    yunet = settings.models_cache / "face_detection_yunet_2023mar.onnx"
    if not yunet.exists():
        url = ("https://github.com/opencv/opencv_zoo/raw/main/models/"
               "face_detection_yunet/face_detection_yunet_2023mar.onnx")
        urllib.request.urlretrieve(url, yunet)
    typer.echo(f"models cached in {settings.models_cache}")


@app.command()
def serve(host: str = "127.0.0.1", port: int = 8000) -> None:
    """Run the API and analyst UI."""
    import uvicorn

    from carid.api.app import create_app

    uvicorn.run(create_app(), host=host, port=port)


def _print_job(svc: Services, job_id: int) -> None:
    from carid.models import Job

    with svc.engine.db.session() as s:
        j = s.get(Job, job_id)
        assert j is not None
        typer.echo(f"job #{job_id}: {j.status.value} — {j.message}")
        typer.echo(json.dumps(j.counters, indent=2))

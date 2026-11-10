"""Make › Model › Generation › Year taxonomy: seeding, lookups, Wikidata expansion."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import yaml
from sqlalchemy import func, select
from sqlalchemy.orm import Session, selectinload

from carid.models import Generation, Make, VehicleModel

FIRST_YEAR = 1996


def slugify(text: str) -> str:
    norm = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "-", norm.lower()).strip("-") or "unnamed"


def normalize_name(text: str) -> str:
    norm = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+", "", norm)


@dataclass
class GenerationSpec:
    name: str
    year_start: int
    year_end: int | None


@dataclass
class ModelSpec:
    name: str
    aliases: list[str] = field(default_factory=list)
    generations: list[GenerationSpec] = field(default_factory=list)
    year_start: int | None = None
    year_end: int | None = None


@dataclass
class MakeSpec:
    name: str
    country: str | None = None
    aliases: list[str] = field(default_factory=list)
    models: list[ModelSpec] = field(default_factory=list)


def _validate_years(start: int, end: int | None, where: str) -> None:
    if not 1900 <= start <= 2100:
        raise ValueError(f"{where}: invalid start year {start}")
    if end is not None and (end < start or end > 2100):
        raise ValueError(f"{where}: invalid end year {end}")


def parse_taxonomy(data: dict[str, Any]) -> list[MakeSpec]:
    makes: list[MakeSpec] = []
    for m in data.get("makes", []):
        make = MakeSpec(name=str(m["name"]), country=m.get("country"),
                        aliases=[str(a) for a in m.get("aliases", [])])
        for mod in m.get("models", []):
            gens: list[GenerationSpec] = []
            for g in mod.get("generations", []):
                name, start, end = str(g[0]), int(g[1]), (None if g[2] is None else int(g[2]))
                _validate_years(start, end, f"{make.name} {mod['name']} {name}")
                gens.append(GenerationSpec(name, start, end))
            make.models.append(ModelSpec(
                name=str(mod["name"]), aliases=[str(a) for a in mod.get("aliases", [])],
                generations=gens, year_start=mod.get("year_start"), year_end=mod.get("year_end"),
            ))
        makes.append(make)
    return makes


def load_taxonomy_file(path: Path) -> list[MakeSpec]:
    with path.open(encoding="utf-8") as f:
        return parse_taxonomy(yaml.safe_load(f) or {})


def find_make(session: Session, name: str) -> Make | None:
    key = normalize_name(name)
    for make in session.scalars(select(Make)):
        if normalize_name(make.name) == key or key in {normalize_name(a) for a in make.aliases}:
            return make
    return None


def find_model(make: Make, name: str) -> VehicleModel | None:
    key = normalize_name(name)
    for model in make.models:
        if normalize_name(model.name) == key or key in {normalize_name(a) for a in model.aliases}:
            return model
    return None


def upsert_make(session: Session, spec: MakeSpec, added_by: str) -> tuple[Make, int, int]:
    """Create or extend a make. Returns (make, models_added, generations_added)."""
    make = find_make(session, spec.name)
    if make is None:
        make = Make(name=spec.name.strip(), country=spec.country, aliases=spec.aliases,
                    added_by=added_by)
        session.add(make)
        session.flush()
    else:
        make.aliases = sorted(set(make.aliases) | set(spec.aliases))
        make.country = make.country or spec.country
    models_added = gens_added = 0
    for mspec in spec.models:
        model = find_model(make, mspec.name)
        if model is None:
            model = VehicleModel(make=make, name=mspec.name.strip(), aliases=mspec.aliases)
            session.add(model)
            make.models.append(model)
            models_added += 1
        else:
            model.aliases = sorted(set(model.aliases) | set(mspec.aliases))
        existing = {normalize_name(g.name) for g in model.generations}
        for g in mspec.generations:
            if g.year_end is not None and g.year_end < FIRST_YEAR:
                continue
            if normalize_name(g.name) in existing:
                continue
            model.generations.append(Generation(name=g.name, year_start=g.year_start,
                                                year_end=g.year_end))
            existing.add(normalize_name(g.name))
            gens_added += 1
        _refresh_model_years(model, mspec)
    session.flush()
    return make, models_added, gens_added


def _refresh_model_years(model: VehicleModel, spec: ModelSpec | None = None) -> None:
    starts = [g.year_start for g in model.generations]
    if spec and spec.year_start:
        starts.append(spec.year_start)
    if starts:
        model.year_start = max(FIRST_YEAR, min(starts))
    ends = [g.year_end for g in model.generations]
    if spec and spec.generations == [] and spec.year_end:
        ends.append(spec.year_end)
    if ends:
        model.year_end = None if any(e is None for e in ends) else max(e for e in ends if e)


def seed(session: Session, path: Path) -> dict[str, int]:
    totals = {"makes": 0, "models": 0, "generations": 0}
    for spec in load_taxonomy_file(path):
        before = session.scalar(select(func.count()).select_from(Make)) or 0
        _, models, gens = upsert_make(session, spec, added_by="seed")
        after = session.scalar(select(func.count()).select_from(Make)) or 0
        totals["makes"] += after - before
        totals["models"] += models
        totals["generations"] += gens
    return totals


def resolve_generation(model: VehicleModel, year_start: int | None,
                       year_end: int | None) -> Generation | None:
    """Pick the generation that best contains the given year range (None if ambiguous)."""
    if year_start is None:
        return None
    lo, hi = year_start, year_end or year_start
    best: tuple[float, Generation] | None = None
    for g in model.generations:
        g_end = g.year_end or 9999
        overlap = min(hi, g_end) - max(lo, g.year_start) + 1
        if overlap <= 0:
            continue
        coverage = overlap / (hi - lo + 1)
        if coverage < 0.99:
            continue
        # Prefer the generation whose core (not overlap years) contains the range.
        tightness = -(g_end - g.year_start) if g.year_end else -50
        score = coverage * 1000 + tightness
        if best is None or score > best[0]:
            best = (score, g)
    if best is None:
        return None
    # If two generations both fully contain a single year (changeover year), it's ambiguous.
    contenders = [g for g in model.generations
                  if g.year_start <= lo and hi <= (g.year_end or 9999)]
    if len(contenders) > 1 and lo == hi:
        return None
    return best[1]


def catalog_tree(session: Session) -> list[Make]:
    stmt = (select(Make).options(selectinload(Make.models).selectinload(VehicleModel.generations))
            .order_by(Make.name))
    return list(session.scalars(stmt))


# --------------------------------------------------------------------------- Wikidata

WIKIDATA_API = "https://www.wikidata.org/w/api.php"
WIKIDATA_SPARQL = "https://query.wikidata.org/sparql"

_SPARQL_MODELS = """
SELECT DISTINCT ?model ?modelLabel ?start ?end ?inception WHERE {{
  ?model wdt:P31/wdt:P279* wd:Q3231690 .
  {{ ?model wdt:P1716 wd:{qid} }} UNION {{ ?model wdt:P176 wd:{qid} }}
  OPTIONAL {{ ?model wdt:P2031 ?start }}
  OPTIONAL {{ ?model wdt:P2032 ?end }}
  OPTIONAL {{ ?model wdt:P571 ?inception }}
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
}}
"""

_GEN_RE = re.compile(r"^(?P<model>.+?)\s*\((?P<gen>[^)]+)\)\s*$")


def _year(value: str | None) -> int | None:
    if not value:
        return None
    m = re.match(r"^\+?(\d{4})", value)
    return int(m.group(1)) if m else None


def wikidata_make_spec(make_name: str, client: httpx.Client) -> MakeSpec:
    """Fetch models/generations of a make from Wikidata (online only)."""
    r = client.get(WIKIDATA_API, params={
        "action": "wbsearchentities", "search": make_name, "language": "en",
        "type": "item", "format": "json", "limit": 5,
    })
    r.raise_for_status()
    hits = r.json().get("search", [])
    if not hits:
        raise LookupError(f"make '{make_name}' not found on Wikidata")
    qid = hits[0]["id"]
    r = client.get(WIKIDATA_SPARQL, params={"query": _SPARQL_MODELS.format(qid=qid)},
                   headers={"Accept": "application/sparql-results+json"})
    r.raise_for_status()
    spec = MakeSpec(name=make_name)
    models: dict[str, ModelSpec] = {}
    prefix = re.compile(rf"^{re.escape(make_name)}\s+", re.IGNORECASE)
    for row in r.json()["results"]["bindings"]:
        label = row.get("modelLabel", {}).get("value", "")
        if not label or re.fullmatch(r"Q\d+", label):
            continue
        start = _year(row.get("start", {}).get("value")) or _year(
            row.get("inception", {}).get("value"))
        end = _year(row.get("end", {}).get("value"))
        if start is None or (end is not None and end < FIRST_YEAR):
            continue
        label = prefix.sub("", label).strip()
        m = _GEN_RE.match(label)
        model_name, gen_name = (m.group("model"), m.group("gen")) if m else (label, None)
        mspec = models.setdefault(model_name, ModelSpec(name=model_name))
        if gen_name:
            if all(g.name != gen_name for g in mspec.generations):
                mspec.generations.append(GenerationSpec(gen_name, start, end))
        else:
            mspec.year_start = min(filter(None, [mspec.year_start, start]))
            mspec.year_end = end
    spec.models = sorted(models.values(), key=lambda s: s.name)
    return spec

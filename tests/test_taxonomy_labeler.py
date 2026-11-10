from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
from sqlalchemy import func, select

from carid.collector.base import GenerationRef, Target
from carid.collector.labeler import infer_label
from carid.config import Settings
from carid.db import Database
from carid.models import Generation, Make, VehicleModel
from carid.taxonomy import (
    find_make,
    find_model,
    load_taxonomy_file,
    parse_taxonomy,
    resolve_generation,
    seed,
    slugify,
    wikidata_make_spec,
)

PILOT = Path(__file__).resolve().parents[1] / "data" / "taxonomy" / "pilot.yaml"


def test_pilot_taxonomy_is_valid() -> None:
    makes = load_taxonomy_file(PILOT)
    assert len(makes) == 20
    names = {m.name for m in makes}
    assert {"Toyota", "Volkswagen", "Hyundai", "Kia", "Tesla", "BMW"} <= names
    for make in makes:
        assert make.models, make.name
        for model in make.models:
            assert model.generations, (make.name, model.name)
            for g in model.generations:
                assert 1985 <= g.year_start <= 2026
                assert g.year_end is None or g.year_end >= g.year_start


def test_seed_is_idempotent(settings: Settings) -> None:
    db = Database(settings)
    with db.session() as s:
        first = seed(s, PILOT)
    with db.session() as s:
        second = seed(s, PILOT)
        n_makes = s.scalar(select(func.count()).select_from(Make))
        # Generations ending before 1996 are skipped.
        old = s.scalar(select(func.count()).select_from(Generation)
                       .where(Generation.year_end < 1996))
    assert first["makes"] == 20 and first["models"] > 100
    assert second == {"makes": 0, "models": 0, "generations": 0}
    assert n_makes == 20 and old == 0


def test_find_by_alias(engine: object) -> None:
    from carid.engine import Engine

    assert isinstance(engine, Engine)
    with engine.db.session() as s:
        make = find_make(s, "alf")
        assert make is not None and make.name == "Alpha"
        model = find_model(make, "estate")
        assert model is not None and model.name == "Wagon"
        assert find_make(s, "Gamma") is None


def test_resolve_generation(engine: object) -> None:
    from carid.engine import Engine

    assert isinstance(engine, Engine)
    with engine.db.session() as s:
        make = find_make(s, "Alpha")
        assert make is not None
        model = find_model(make, "Roadster")
        assert isinstance(model, VehicleModel)
        assert resolve_generation(model, 2005, 2005).name == "R1"  # type: ignore[union-attr]
        assert resolve_generation(model, 2020, 2022).name == "R3"  # type: ignore[union-attr]
        assert resolve_generation(model, 2008, 2008) is None  # changeover year: ambiguous
        assert resolve_generation(model, 2005, 2012) is None  # spans two generations
        assert resolve_generation(model, None, None) is None


def test_parse_taxonomy_rejects_bad_years() -> None:
    with pytest.raises(ValueError):
        parse_taxonomy({"makes": [{"name": "X", "models": [
            {"name": "Y", "generations": [["G", 2010, 2001]]}]}]})


def test_slugify() -> None:
    assert slugify("Mercedes-Benz") == "mercedes-benz"
    assert slugify("Škoda Octavia") == "skoda-octavia"
    assert slugify("!!!") == "unnamed"


def test_wikidata_spec_parsing() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if "wbsearchentities" in str(request.url):
            return httpx.Response(200, json={"search": [{"id": "Q999"}]})
        rows = [
            {"modelLabel": {"value": "Gamma Sprint (G1)"}, "start": {"value": "2001-01-01"},
             "end": {"value": "2009-01-01"}},
            {"modelLabel": {"value": "Gamma Sprint (G2)"}, "start": {"value": "2009-01-01"}},
            {"modelLabel": {"value": "Gamma Old"}, "start": {"value": "1980-01-01"},
             "end": {"value": "1990-01-01"}},
            {"modelLabel": {"value": "Gamma Hauler"}, "inception": {"value": "2015-05-01"}},
            {"modelLabel": {"value": "Q123456"}, "start": {"value": "2015-01-01"}},
        ]
        return httpx.Response(200, content=json.dumps({"results": {"bindings": rows}}))

    client = httpx.Client(transport=httpx.MockTransport(handler))
    spec = wikidata_make_spec("Gamma", client)
    by_name = {m.name: m for m in spec.models}
    assert set(by_name) == {"Sprint", "Hauler"}
    assert [g.name for g in by_name["Sprint"].generations] == ["G1", "G2"]
    assert by_name["Hauler"].year_start == 2015


def _target() -> Target:
    return Target(
        make_id=1, make="Toyota", make_aliases=[], model_id=2, model="Corolla",
        model_aliases=[], year_start=1996, year_end=None,
        generations=[GenerationRef(10, "E170", 2013, 2019), GenerationRef(11, "E210", 2018, None)],
        sibling_models=["Corolla", "Camry", "RAV4", "Yaris"],
    )


@pytest.mark.parametrize(("text", "expect"), [
    ("2017 Toyota Corolla interior", {"year_start": 2017, "interior_hint": True}),
    ("Toyota Corolla E210 dashboard", {"generation_id": 11, "year_start": 2018}),
    ("Toyota Corolla 2014-2016 cockpit", {"year_start": 2014, "year_end": 2016}),
    ("Toyota Corolla 2014–16 cockpit", {"year_start": 2014, "year_end": 2016}),
    ("Toyota Corolla seen from the side, exterior", {"interior_hint": False}),
    ("toyota corolla", {"year_start": None, "interior_hint": None}),
])
def test_infer_label(text: str, expect: dict[str, object]) -> None:
    guess = infer_label(text, _target())
    assert guess is not None
    for k, v in expect.items():
        assert getattr(guess, k) == v, (k, guess)


def test_infer_label_rejects_and_penalises() -> None:
    t = _target()
    assert infer_label("Toyota Camry interior 2015", t) is None
    clean = infer_label("2015 Toyota Corolla interior", t)
    mixed = infer_label("2015 Toyota Corolla vs Camry interior", t)
    assert clean is not None and mixed is not None
    assert mixed.confidence < clean.confidence
    assert infer_label("Toyota Corolla interior 1950", t).year_start is None  # type: ignore[union-attr]
    preset = infer_label("Toyota Corolla", t, preset_year=2020)
    assert preset is not None and preset.year_start == 2020


def test_short_model_names_need_word_boundaries() -> None:
    t = Target(make_id=1, make="Audi", make_aliases=[], model_id=1, model="A3",
               model_aliases=[], year_start=1996, year_end=None)
    assert infer_label("Audi A3 interior", t) is not None
    assert infer_label("Audi A35 interior", t) is None


def test_plain_word_model_names_need_the_make() -> None:
    t = Target(make_id=1, make="Ford", make_aliases=[], model_id=1, model="Focus",
               model_aliases=[], year_start=1998, year_end=None,
               sibling_models=["Focus", "Fiesta"])
    assert infer_label("2015 Ford Focus interior", t) is not None
    assert infer_label("Ford Fiesta interior, shallow focus", t) is None
    assert infer_label("refocused dashboard photo", t) is None
    assert infer_label("Interior of a Focus ST (Ford), 2016", t) is not None
    assert infer_label("Interior of a Focus ST, 2016", t) is None  # make never named
    fiesta = Target(make_id=1, make="Ford", make_aliases=[], model_id=2, model="Fiesta",
                    model_aliases=[], year_start=1995, year_end=None,
                    sibling_models=["Focus", "Fiesta"])
    clean = infer_label("2012 Ford Fiesta interior, shallow focus", fiesta)
    assert clean is not None and "also mentions" not in clean.reason
    # Code-like names stand alone and tolerate punctuation.
    crv = Target(make_id=1, make="Honda", make_aliases=[], model_id=3, model="CR-V",
                 model_aliases=[], year_start=1995, year_end=None)
    assert infer_label("CRV dashboard 2018", crv) is not None
    assert infer_label("CR V dashboard 2018", crv) is not None

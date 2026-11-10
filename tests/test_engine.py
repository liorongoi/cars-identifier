from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest
from sqlalchemy import select

from carid.engine import Engine, IdentifyOptions, SourceInfo
from carid.models import Identification, Image, ReviewStatus
from carid.vision.degrade import DEGRADATIONS
from carid.vision.imageio import ImageDecodeError
from tests.conftest import CLASS_STYLES, draw_interior, label_for, to_jpeg


def test_ingest_dedup_and_rejections(engine: Engine) -> None:
    rng = np.random.default_rng(1)
    key = "Alpha/Roadster/2010"
    label = label_for(engine, key)
    img = draw_interior(CLASS_STYLES[key], rng)
    data = to_jpeg(img)
    src = SourceInfo("test")
    first = engine.ingest(data, label, src, interior_score=0.9)
    assert first.status == "added" and first.image_id
    assert engine.ingest(data, label, src, interior_score=0.9).status == "duplicate"
    # Re-encoded copy of the same picture: near duplicate.
    assert engine.ingest(to_jpeg(img, 70), label, src, interior_score=0.9).status == \
        "near_duplicate"
    small = to_jpeg(draw_interior(CLASS_STYLES[key], rng, size=(150, 100)))
    assert engine.ingest(small, label, src, interior_score=0.9).status == "rejected_quality"
    black = to_jpeg(np.zeros((400, 600, 3), np.uint8))
    assert engine.ingest(black, label, src, interior_score=0.9).status == "rejected_quality"
    other = to_jpeg(draw_interior(CLASS_STYLES["Beta/City/2012"], rng))
    assert engine.ingest(other, label, src, interior_score=0.1).status == "rejected_not_interior"
    with pytest.raises(ImageDecodeError):
        engine.ingest(b"garbage", label, src)


def test_ingest_stores_by_make_model_year_and_resolves_generation(engine: Engine) -> None:
    key = "Alpha/Roadster/2010"
    res = engine.ingest(to_jpeg(draw_interior(CLASS_STYLES[key], np.random.default_rng(2))),
                        label_for(engine, key), SourceInfo("test"), interior_score=0.9)
    with engine.db.session() as s:
        im = s.get(Image, res.image_id)
        assert im is not None
        assert im.path.startswith("alpha/roadster/2010/")
        assert im.generation is not None and im.generation.name == "R2"
        assert im.indexed and {p.part for p in im.parts} >= {"whole", "steering_wheel"}
    assert (engine.settings.images_dir / im.path).is_file()


def test_ingest_rejects_label_mismatch(engine: Engine) -> None:
    label = label_for(engine, "Alpha/Roadster/2010")
    other = label_for(engine, "Beta/City/2012")
    label.make_id = other.make_id
    with pytest.raises(ValueError):
        engine.ingest(to_jpeg(draw_interior(CLASS_STYLES["Alpha/Roadster/2010"],
                                            np.random.default_rng(3))),
                      label, SourceInfo("test"), interior_score=0.9)


@pytest.mark.parametrize("condition", ["clean", "low_light", "noise", "occlusion"])
def test_identify_finds_the_right_car(populated: Engine, condition: str) -> None:
    rng = np.random.default_rng(42)
    hits = 0
    for key, style in CLASS_STYLES.items():
        img = draw_interior(style, rng)
        if condition != "clean":
            img = DEGRADATIONS[condition](img, rng)
        result = populated.identify(to_jpeg(img), IdentifyOptions(top_k=5, analyst="t"))
        make, model, _year = key.split("/")
        names = [(c["make"], c["model"]) for c in result["candidates"]]
        assert (make, model) in names, (condition, key, names)  # always in top-5
        hits += names[0] == (make, model)
        top = result["candidates"][0]
        assert top["reference_images"] >= 1 and top["evidence"]
    assert hits >= 3, f"top-1 hits {hits}/4 under {condition}"


def test_identify_result_contents_and_audit(populated: Engine) -> None:
    key = "Beta/City/2012"
    img = draw_interior(CLASS_STYLES[key], np.random.default_rng(5))
    res = populated.identify(to_jpeg(img), IdentifyOptions(case_ref="CASE-1", analyst="ana"))
    assert res["id"] > 0 and res["duration_ms"] >= 0
    c = res["candidates"][0]
    assert c["rank"] == 1 and c["make"] == "Beta" and c["model"] == "City"
    assert c["generation"] == "C2" and c["year_start"] == 2010 and c["year_end"] is None
    assert abs(sum(x["probability"] for x in res["candidates"]) - 1) < 0.05 or \
        len(res["candidates"]) == 10
    assert {p["part"] for p in res["parts"]} >= {"whole", "steering_wheel"}
    assert res["makes"][0]["make"] == "Beta"
    assert res["quality"]["verdict"] in ("good", "fair", "poor")
    with populated.db.session() as s:
        ident = s.get(Identification, res["id"])
        assert ident is not None and ident.case_ref == "CASE-1" and ident.query_path
        assert ident.result["candidates"][0]["model"] == "City"
    assert (populated.settings.queries_dir / ident.query_path).is_file()


def test_identify_filters_and_manual_parts(populated: Engine) -> None:
    img = to_jpeg(draw_interior(CLASS_STYLES["Beta/City/2012"], np.random.default_rng(6)))
    with populated.db.session():
        alpha_id = label_for(populated, "Alpha/Roadster/2010").make_id
    res = populated.identify(img, IdentifyOptions(make_ids=[alpha_id]))
    assert {c["make"] for c in res["candidates"]} == {"Alpha"}
    res = populated.identify(img, IdentifyOptions(
        mode="parts", manual_parts=[{"part": "steering_wheel", "bbox": [100, 200, 400, 560]}]))
    assert any(p["method"] == "manual" for p in res["parts"])
    assert all("whole" not in c["part_scores"] for c in res["candidates"])


def test_identify_with_empty_database_warns(engine: Engine) -> None:
    img = to_jpeg(draw_interior(CLASS_STYLES["Beta/City/2012"], np.random.default_rng(6)))
    res = engine.identify(img, IdentifyOptions())
    assert res["candidates"] == [] and res["warnings"]


def test_reject_and_restore_reference(populated: Engine) -> None:
    with populated.db.session() as s:
        ids = list(s.scalars(select(Image.id).where(Image.model_id == label_for(
            populated, "Beta/City/2012").model_id)))
    for i in ids:
        populated.set_review_status(i, ReviewStatus.rejected, "tester")
    img = to_jpeg(draw_interior(CLASS_STYLES["Beta/City/2012"], np.random.default_rng(9)))
    res = populated.identify(img, IdentifyOptions())
    assert all(c["model"] != "City" for c in res["candidates"])
    populated.set_review_status(ids[0], ReviewStatus.approved, "tester")
    res = populated.identify(img, IdentifyOptions())
    assert any(c["model"] == "City" for c in res["candidates"])
    with pytest.raises(KeyError):
        populated.set_review_status(99999, ReviewStatus.approved, None)


def test_relabel_moves_file(populated: Engine) -> None:
    with populated.db.session() as s:
        im = s.scalars(select(Image)).first()
        assert im is not None
        image_id, old_path = im.id, im.path
    new = label_for(populated, "Beta/Cruiser/2018")
    populated.relabel(image_id, new, "curator")
    with populated.db.session() as s:
        im = s.get(Image, image_id)
        assert im is not None and im.path.startswith("beta/cruiser/2018/")
        assert im.generation is not None and im.generation.name == "X1"
    assert (populated.settings.images_dir / im.path).is_file()
    assert not (populated.settings.images_dir / old_path).exists()


def test_rebuild_index_and_persistence(populated: Engine) -> None:
    sizes = populated.index.sizes()
    assert sizes["whole"] == 24 or sizes["whole"] >= 20
    n = populated.rebuild_index()
    assert populated.index.sizes() == sizes and n == sizes["whole"]
    reopened = Engine(populated.settings, db=populated.db)
    assert reopened.index.sizes() == sizes


def test_query_retention_purge(populated: Engine) -> None:
    img = to_jpeg(draw_interior(CLASS_STYLES["Beta/City/2012"], np.random.default_rng(6)))
    res = populated.identify(img, IdentifyOptions())
    assert populated.purge_old_queries() == 0  # still within retention
    with populated.db.session() as s:
        ident = s.get(Identification, res["id"])
        assert ident is not None
        ident.created_at = datetime.now(UTC) - timedelta(days=60)
        path = populated.settings.queries_dir / str(ident.query_path)
    assert populated.purge_old_queries() == 1
    assert not path.exists()


def test_no_query_storage_when_retention_zero(populated: Engine) -> None:
    populated.settings.query_retention_days = 0
    img = to_jpeg(draw_interior(CLASS_STYLES["Beta/City/2012"], np.random.default_rng(6)))
    res = populated.identify(img, IdentifyOptions())
    with populated.db.session() as s:
        assert s.get(Identification, res["id"]).query_path is None  # type: ignore[union-attr]

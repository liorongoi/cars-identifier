from __future__ import annotations

import json
from collections.abc import Iterator

import numpy as np
import pytest
from fastapi.testclient import TestClient

from carid.api.app import create_app
from carid.engine import Engine
from carid.services import Services
from tests.conftest import CLASS_STYLES, draw_interior, to_jpeg


@pytest.fixture
def client(populated: Engine) -> Iterator[TestClient]:
    svc = Services(populated.settings, populated)
    app = create_app(populated.settings, svc, start_jobs=False)
    with TestClient(app) as c:
        c.svc = svc  # type: ignore[attr-defined]
        yield c


def _img(key: str = "Beta/City/2012", seed: int = 11) -> bytes:
    return to_jpeg(draw_interior(CLASS_STYLES[key], np.random.default_rng(seed)))


def test_health_and_stats(client: TestClient) -> None:
    h = client.get("/api/health").json()
    assert h["status"] == "ok" and h["engine"]["embedder"] == "handcrafted-v1"
    assert h["warnings"]  # fallback embedder warning
    s = client.get("/api/stats").json()
    assert s["makes"] == 2 and s["images"] == 24 and s["index"]["whole"] == 24
    assert {x["name"] for x in client.get("/api/sources").json()} >= {"wikimedia", "flickr"}


def test_catalog_counts(client: TestClient) -> None:
    cat = client.get("/api/catalog").json()
    assert cat["total_images"] == 24
    beta = next(m for m in cat["makes"] if m["name"] == "Beta")
    city = next(m for m in beta["models"] if m["name"] == "City")
    assert city["images"] == 6 and city["approved"] == 6
    assert "years" not in city and "generations" not in city


def test_identify_flow(client: TestClient) -> None:
    r = client.post("/api/identify", files={"file": ("q.jpg", _img(), "image/jpeg")},
                    data={"top_k": "5", "case_ref": "CASE-9", "analyst": "dana"})
    assert r.status_code == 200, r.text
    res = r.json()
    assert res["candidates"][0]["model"] == "City" and len(res["candidates"]) <= 5
    ident_id = res["id"]
    hist = client.get("/api/identifications", params={"q": "CASE-9"}).json()
    assert hist["total"] == 1 and hist["items"][0]["top"]["model"] == "City"
    one = client.get(f"/api/identifications/{ident_id}").json()
    assert one["case_ref"] == "CASE-9" and one["id"] == ident_id
    assert client.get(f"/api/identifications/{ident_id}/query").status_code == 200
    csv_r = client.get(f"/api/identifications/{ident_id}/export", params={"format": "csv"})
    assert csv_r.status_code == 200 and csv_r.text.startswith("rank,make,model")
    js = client.get(f"/api/identifications/{ident_id}/export").json()
    assert js["query_sha256"] and js["candidates"]
    model_id = res["candidates"][0]["model_id"]
    fb = client.post(f"/api/identifications/{ident_id}/feedback",
                     json={"model_id": model_id, "note": "confirmed", "analyst": "dana"})
    assert fb.status_code == 200
    assert client.get("/api/audit").json()[0]["action"] == "identification.feedback"
    ev = res["candidates"][0]["evidence"][0]
    thumb = client.get(f"/api/images/{ev['image_id']}/file",
                       params={"size": 200, "bbox": ",".join(map(str, ev["bbox"]))})
    assert thumb.status_code == 200 and thumb.headers["content-type"] == "image/jpeg"


def test_identify_validation(client: TestClient) -> None:
    bad = client.post("/api/identify", files={"file": ("x.jpg", b"nope", "image/jpeg")})
    assert bad.status_code == 422
    empty = client.post("/api/identify", files={"file": ("x.jpg", b"", "image/jpeg")})
    assert empty.status_code == 422
    mode = client.post("/api/identify", files={"file": ("q.jpg", _img(), "image/jpeg")},
                       data={"mode": "magic"})
    assert mode.status_code == 422
    bogus = json.dumps([{"part": "bogus", "bbox": [0, 0, 1, 1]}])
    parts = client.post("/api/identify", files={"file": ("q.jpg", _img(), "image/jpeg")},
                        data={"manual_parts": bogus})
    assert parts.status_code == 422
    ids = client.post("/api/identify", files={"file": ("q.jpg", _img(), "image/jpeg")},
                      data={"make_ids": "1,x"})
    assert ids.status_code == 422
    client.app.state.svc.settings.max_upload_mb = 0  # type: ignore[attr-defined]
    big = client.post("/api/identify", files={"file": ("q.jpg", _img(), "image/jpeg")})
    assert big.status_code == 413


def test_identify_with_filters(client: TestClient) -> None:
    cat = client.get("/api/catalog").json()
    alpha = next(m for m in cat["makes"] if m["name"] == "Alpha")
    r = client.post("/api/identify", files={"file": ("q.jpg", _img(), "image/jpeg")},
                    data={"make_ids": str(alpha["id"]),
                          "enhance": "false", "mask_faces": "false", "mode": "global"})
    assert r.status_code == 200
    assert {c["make"] for c in r.json()["candidates"]} == {"Alpha"}


def test_add_make_manual_and_seeded(client: TestClient) -> None:
    r = client.post("/api/catalog/makes", json={
        "name": "Gamma", "country": "Nowhere", "requested_by": "dana",
        "models": [{"name": "Hauler"}]})
    assert r.status_code == 201, r.text
    body = r.json()
    assert not body["existed"] and body["models_added"] == 1 and len(body["jobs"]) == 1
    again = client.post("/api/catalog/makes", json={"name": "gamma", "collect": False}).json()
    assert again["existed"] and again["jobs"] == []
    wiki = client.post("/api/catalog/makes", json={"name": "Delta", "use_wikidata": True}).json()
    assert len(wiki["jobs"]) == 1
    jobs = client.get("/api/jobs").json()
    assert {j["kind"] for j in jobs} == {"collect", "expand_make"}
    m = client.post("/api/catalog/models", json={
        "make_id": body["make_id"], "name": "Van", "collect": False})
    assert m.status_code == 201 and m.json()["models_added"] == 1
    assert client.post("/api/catalog/makes", json={"name": "  "}).status_code == 422
    assert client.post("/api/catalog/models", json={"make_id": 999, "name": "X"}).status_code \
        == 404


def test_images_browse_patch_upload(client: TestClient) -> None:
    city_id = next(m["id"] for mk in client.get("/api/catalog").json()["makes"]
                   for m in mk["models"] if m["name"] == "City")
    lst = client.get("/api/images", params={"model_id": city_id, "limit": 3}).json()
    assert lst["total"] == 6 and len(lst["items"]) == 3
    img = lst["items"][0]
    detail = client.get(f"/api/images/{img['id']}").json()
    assert detail["parts"]
    assert client.get(f"/api/images/{img['id']}/file").status_code == 200
    rej = client.patch(f"/api/images/{img['id']}", json={"review_status": "rejected",
                                                         "actor": "cur"})
    assert rej.json()["review_status"] == "rejected"
    assert client.get("/api/images", params={"status": "rejected"}).json()["total"] == 1
    cat = client.get("/api/catalog").json()
    alpha = next(m for m in cat["makes"] if m["name"] == "Alpha")
    roadster = next(m for m in alpha["models"] if m["name"] == "Roadster")
    moved = client.patch(f"/api/images/{img['id']}", json={
        "model_id": roadster["id"]}).json()
    assert moved["model"] == "Roadster" and "year_start" not in moved
    assert client.patch(f"/api/images/{img['id']}", json={
        "model_id": roadster["id"], "make_id": 9999}).status_code == 422
    assert client.get("/api/images/99999").status_code == 404
    up = client.post("/api/images/upload",
                     files={"file": ("ref.jpg", _img("Alpha/Wagon/2016", 77), "image/jpeg")},
                     data={"model_id": str(roadster["id"])})
    assert up.status_code == 201 and up.json()["status"] == "added"
    bad_bbox = client.get(f"/api/images/{img['id']}/file", params={"bbox": "5,5,1,1"})
    assert bad_bbox.status_code == 422


def test_jobs_and_benchmarks_api(client: TestClient) -> None:
    r = client.post("/api/jobs", json={"kind": "benchmark", "params": {
        "max_test": 4, "degradations": []}})
    assert r.status_code == 201
    job_id = r.json()["job_id"]
    client.svc.jobs.run_pending()  # type: ignore[attr-defined]
    job = client.get(f"/api/jobs/{job_id}").json()
    assert job["status"] == "succeeded", job["log"]
    benches = client.get("/api/benchmarks").json()
    assert len(benches) == 1
    full = client.get(f"/api/benchmarks/{benches[0]['id']}").json()
    assert full["results"]["clean"]["fusion"]["model"]["n"] == 4
    assert client.post("/api/jobs", json={"kind": "import_bundle",
                                          "params": {"path": "/nope"}}).status_code == 422
    assert client.post("/api/jobs", json={"kind": "rm -rf"}).status_code == 422
    q = client.post("/api/jobs", json={"kind": "reindex"}).json()["job_id"]
    assert client.post(f"/api/jobs/{q}/cancel").json() == {"cancelled": True}
    assert client.get("/api/jobs/9999").status_code == 404


def test_path_traversal_is_blocked(client: TestClient, populated: Engine) -> None:
    from carid.models import Image

    with populated.db.session() as s:
        im = s.query(Image).first()
        assert im is not None
        im.path = "../../../etc/passwd"
        image_id = im.id
    assert client.get(f"/api/images/{image_id}/file").status_code == 404

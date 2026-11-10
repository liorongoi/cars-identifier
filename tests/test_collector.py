from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pytest
from sqlalchemy import func, select

from carid.collector.base import Candidate, FetchError, PoliteClient, SourceConnector, Target
from carid.collector.runner import Collector, build_connectors, build_targets
from carid.collector.sources import (
    FlickrSource,
    FolderSource,
    VideoFramesSource,
    WebPagesSource,
    WikimediaCommonsSource,
    search_queries,
)
from carid.engine import Engine
from carid.models import Image, SeenUrl
from tests.conftest import CLASS_STYLES, draw_interior, to_jpeg


def _client(handler: Any) -> tuple[PoliteClient, list[float]]:
    sleeps: list[float] = []
    c = PoliteClient("test-agent", min_interval=0, max_retries=3,
                     transport=httpx.MockTransport(handler), sleep=sleeps.append)
    return c, sleeps


def test_polite_client_retries_on_429_and_honours_retry_after() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return httpx.Response(404)
        calls["n"] += 1
        if calls["n"] < 3:
            return httpx.Response(429, headers={"Retry-After": "7"})
        return httpx.Response(200, content=b"ok")

    client, sleeps = _client(handler)
    assert client.get("https://x.test/a").content == b"ok"
    assert 7.0 in sleeps


def test_polite_client_gives_up_and_maps_errors() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return httpx.Response(404)
        if request.url.path == "/missing":
            return httpx.Response(404)
        return httpx.Response(503)

    client, _ = _client(handler)
    with pytest.raises(FetchError, match="404"):
        client.get("https://x.test/missing")
    with pytest.raises(FetchError, match="after retries"):
        client.get("https://x.test/down")


def test_polite_client_obeys_robots_and_checks_content() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return httpx.Response(200, text="User-agent: *\nDisallow: /private\n")
        if request.url.path == "/page.html":
            return httpx.Response(200, text="<html>", headers={"content-type": "text/html"})
        return httpx.Response(200, content=b"\xff\xd8", headers={"content-type": "image/jpeg"})

    client, _ = _client(handler)
    assert not client.allowed("https://x.test/private/a.jpg")
    with pytest.raises(FetchError, match="robots"):
        client.get("https://x.test/private/a.jpg")
    assert client.download_image("https://x.test/ok.jpg") == b"\xff\xd8"
    with pytest.raises(FetchError, match="not an image"):
        client.download_image("https://x.test/page.html")


def _target(engine: Engine, make: str = "Alpha", model: str = "Roadster") -> Target:
    with engine.db.session() as s:
        return next(t for t in build_targets(s) if t.make == make and t.model == model)


def test_build_targets_and_queries(engine: Engine) -> None:
    with engine.db.session() as s:
        all_t = build_targets(s)
    assert len(all_t) == 4
    t = _target(engine, "Alpha", "Wagon")
    assert t.model_aliases == ["Estate"] and [g.name for g in t.generations] == ["W1", "W2"]
    assert "Roadster" in t.sibling_models
    qs = search_queries(t)
    assert "Alpha Wagon interior" in qs and "Alpha Wagon W2 interior" in qs
    assert "Alpha Estate interior" in qs


def test_wikimedia_source_parses_api(engine: Engine) -> None:
    page = {
        "title": "File:2012 Alpha Roadster R2 interior.jpg",
        "imageinfo": [{"url": "https://upload.test/full.jpg", "thumburl": "https://upload.test/t.jpg",
                       "descriptionurl": "https://commons.test/File:x", "mime": "image/jpeg",
                       "extmetadata": {"LicenseShortName": {"value": "CC BY-SA 4.0"},
                                       "Artist": {"value": "<a href='u'>Jane</a>"},
                                       "ImageDescription": {"value": "Dashboard"}}}],
        "categories": [{"title": "Category:Alpha Roadster interiors"}],
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"query": {"pages": [page, {"title": "File:no.jpg"}]}})

    client, _ = _client(handler)
    src = WikimediaCommonsSource({"enabled": True}, client)
    cands = list(src.search(_target(engine), 10))
    assert len(cands) == 1
    c = cands[0]
    assert c.url == "https://upload.test/t.jpg" and c.license == "CC BY-SA 4.0"
    assert c.credit == "Jane" and "interiors" in c.extra_text
    assert src.runnable


def test_webpages_source_extracts_images(engine: Engine) -> None:
    html = """<html><head><title>Alpha Roadster 2012 press</title></head><body>
      <img src="/img/dash.jpg" alt="Cockpit">
      <img srcset="/s.jpg 300w, /big.jpg 1200w" alt="Interior">
      <img src="/logo.png"><a href="/gallery/2.jpeg">photo</a></body></html>"""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return httpx.Response(404)
        return httpx.Response(200, text=html, headers={"content-type": "text/html"})

    client, _ = _client(handler)
    src = WebPagesSource({"enabled": True, "sites": [
        {"make": "Alpha", "model": "Roadster", "year": 2012, "urls": ["https://p.test/x"]},
        {"make": "Beta", "urls": ["https://p.test/y"]},
    ]}, client)
    cands = list(src.search(_target(engine), 10))
    urls = [c.url for c in cands]
    assert urls == ["https://p.test/img/dash.jpg", "https://p.test/big.jpg",
                    "https://p.test/gallery/2.jpeg"]
    assert all(c.preset_year == 2012 and "Alpha Roadster" in c.extra_text for c in cands)


def test_opt_in_sources_are_gated(engine: Engine) -> None:
    client, _ = _client(lambda r: httpx.Response(500))
    for cls in (FlickrSource, VideoFramesSource):
        off = cls({"enabled": True}, client)
        assert not off.runnable and "legal" in off.status()["reason"]
        on = cls({"enabled": True, "legal_approved": True}, client)
        assert on.runnable
    assert FolderSource({}, client).status()["reason"] == "disabled in configuration"


class FakeSource(SourceConnector):
    name = "fake"
    description = "test"

    def __init__(self, items: list[tuple[str, bytes]]) -> None:
        super().__init__({"enabled": True}, None)  # type: ignore[arg-type]
        self.items = items

    def search(self, target: Target, limit: int) -> Iterator[Candidate]:
        for i, (text, _) in enumerate(self.items[:limit]):
            yield Candidate(url=f"mem://{i}", title=text)

    def fetch(self, candidate: Candidate) -> bytes:
        return self.items[int(candidate.url.split("//")[1])][1]


def test_collector_end_to_end(engine: Engine) -> None:
    rng = np.random.default_rng(3)
    style = CLASS_STYLES["Alpha/Roadster/2010"]
    good = [(f"2010 Alpha Roadster interior {i}", to_jpeg(draw_interior(style, rng)))
            for i in range(3)]
    items = [
        *good,
        ("Alpha Wagon interior", to_jpeg(draw_interior(style, rng))),  # wrong model
        ("Alpha Roadster exterior", to_jpeg(draw_interior(style, rng))),  # exterior text
        ("Alpha Roadster", to_jpeg(draw_interior(style, rng))),  # unknown => uncurated? no
        ("2011 Alpha Roadster dashboard", b"broken"),
        good[0],  # duplicate content
    ]
    collector = Collector(engine, {"fake": FakeSource(items)})
    target = _target(engine)
    counters = collector.run([target], per_model_limit=50)
    assert counters["added"] == 4  # 3 good + the unlabelled-year one (curated fake source)
    assert counters["label_mismatch"] == 1
    assert counters["not_interior_text"] == 1
    assert counters["invalid_image"] == 1
    assert counters["duplicate"] == 1
    with engine.db.session() as s:
        n_seen = s.scalar(select(func.count()).select_from(SeenUrl))
        years = sorted(y for y in s.scalars(select(Image.year_start)) if y)
    assert n_seen == len(items) and years == [2010, 2010, 2010]
    # Second run: everything already seen.
    again = collector.run([target], per_model_limit=50)
    assert again.get("added", 0) == 0 and again["already_seen"] == len(items)


def test_collector_without_classifier_skips_uncertain_web_images(engine: Engine) -> None:
    rng = np.random.default_rng(4)
    src = FakeSource([("Alpha Roadster", to_jpeg(draw_interior(
        CLASS_STYLES["Alpha/Roadster/2010"], rng)))])
    src.name = "wikimedia"
    counters = Collector(engine, {"wikimedia": src}).run([_target(engine)])
    assert counters["uncertain_interior"] == 1


def test_collector_skips_disabled_sources(engine: Engine) -> None:
    logs: list[str] = []
    client, _ = _client(lambda r: httpx.Response(500))
    conns = build_connectors({"flickr": {"enabled": True}}, client)
    counters = Collector(engine, conns).run([_target(engine)], log_fn=logs.append)
    assert counters == {} and any("legal approval" in line for line in logs)


def test_folder_source_imports_layout(engine: Engine, tmp_path: Path) -> None:
    root = tmp_path / "imports"
    rng = np.random.default_rng(5)
    for year in (2003, 2012):
        d = root / "Alpha" / "Roadster" / str(year)
        d.mkdir(parents=True)
        (d / "a.jpg").write_bytes(to_jpeg(draw_interior(CLASS_STYLES["Alpha/Roadster/2010"],
                                                        rng)))
    (root / "Alpha" / "Roadster" / "notes.txt").write_text("x")
    (root / "Beta" / "City").mkdir(parents=True)
    client, _ = _client(lambda r: httpx.Response(500))
    src = FolderSource({"enabled": True, "path": str(root)}, client)
    cands = list(src.search(_target(engine), 10))
    assert sorted(c.preset_year for c in cands) == [2003, 2012]
    counters = Collector(engine, {"folder": src}).run([_target(engine)])
    assert counters["added"] == 2
    with engine.db.session() as s:
        gens = sorted(im.generation.name for im in s.scalars(select(Image))
                      if im.generation is not None)
    assert gens == ["R1", "R2"]


def _ov_result(i: int, title: str, provider: str = "flickr") -> dict[str, Any]:
    return {"url": f"https://img.test/{i}.jpg", "title": title, "license": "by-sa",
            "license_version": "2.0", "creator": "Ann", "source": provider,
            "provider": provider, "foreign_landing_url": f"https://p.test/{i}",
            "tags": [{"name": "dashboard"}]}


def test_openverse_source_parses_and_filters(engine: Engine) -> None:
    from carid.collector.sources import OpenverseSource

    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, json={"page_count": 1, "results": [
            _ov_result(1, "2012 Alpha Roadster interior"),
            _ov_result(2, "Alpha Roadster cockpit", provider="museum"),
            _ov_result(1, "duplicate url"),
        ]})

    client, _ = _client(handler)
    src = OpenverseSource({"enabled": True, "exclude_providers": ["museum"],
                           "max_queries_per_model": 1}, client)
    assert client.host_intervals["api.openverse.org"] > 3
    cands = list(src.search(_target(engine), 10))
    assert [c.url for c in cands] == ["https://img.test/1.jpg"]
    c = cands[0]
    assert c.license == "CC BY-SA 2.0" and c.credit == "Ann via flickr"
    assert "dashboard" in c.extra_text and c.page_url == "https://p.test/1"
    assert calls[0].url.params["mature"] == "false"
    assert len(calls) == 1


def test_openverse_quota_stops_source_for_the_job(engine: Engine) -> None:
    from carid.collector.sources import OpenverseSource

    n = {"calls": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        n["calls"] += 1
        return httpx.Response(429, headers={"Retry-After": "1"})

    client, _ = _client(handler)
    src = OpenverseSource({"enabled": True}, client)
    logs: list[str] = []
    with engine.db.session() as s:
        targets = build_targets(s)
    counters = Collector(engine, {"openverse": src}).run(targets, log_fn=logs.append)
    assert counters["sources_exhausted"] == 1
    assert n["calls"] == 4  # one request + 3 retries, then no more targets tried
    assert any("skipping it for the rest" in line for line in logs)


def test_openverse_sustained_header_raises_before_next_call(engine: Engine) -> None:
    from carid.collector.base import SourceExhausted
    from carid.collector.sources import OpenverseSource

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"x-ratelimit-available-anon_sustained": "0"},
                              json={"page_count": 1, "results": [_ov_result(1, "Alpha Roadster")]})

    client, _ = _client(handler)
    src = OpenverseSource({"enabled": True, "max_queries_per_model": 3}, client)
    gen = src.search(_target(engine), 50)
    assert next(gen).url == "https://img.test/1.jpg"
    with pytest.raises(SourceExhausted):
        next(gen)


def test_manifest_roundtrip(populated_web: Engine, tmp_path: Path) -> None:
    from carid.bundle import export_manifest
    from carid.collector.sources import ManifestSource

    dest = tmp_path / "m" / "pilot.csv"
    assert export_manifest(populated_web, dest) == 2
    client, _ = _client(lambda r: httpx.Response(500))
    src = ManifestSource({"enabled": True, "paths": [str(tmp_path / "m" / "*.csv")]}, client)
    cands = list(src.search(_target(populated_web), 10))
    assert len(cands) == 2 and {c.preset_year for c in cands} == {2010}
    assert all(c.license == "CC BY 4.0" for c in cands)
    guess_text = cands[0].text
    assert "Alpha Roadster interior" in guess_text
    assert list(src.search(_target(populated_web, "Beta", "City"), 10)) == []


@pytest.fixture
def populated_web(engine: Engine) -> Engine:
    rng = np.random.default_rng(12)
    from carid.engine import SourceInfo
    from tests.conftest import label_for

    for i in range(2):
        engine.ingest(to_jpeg(draw_interior(CLASS_STYLES["Alpha/Roadster/2010"], rng)),
                      label_for(engine, "Alpha/Roadster/2010"),
                      SourceInfo("openverse", source_url=f"https://img.test/{i}.jpg",
                                 license="CC BY 4.0"), interior_score=0.9)
    engine.ingest(to_jpeg(draw_interior(CLASS_STYLES["Beta/City/2012"], rng)),
                  label_for(engine, "Beta/City/2012"), SourceInfo("folder", source_url="file:///x"),
                  interior_score=0.9)
    return engine


def test_polite_client_pauses_host_with_long_retry_after() -> None:
    calls: dict[str, int] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        calls[request.url.host] = calls.get(request.url.host, 0) + 1
        if request.url.host == "slow.test":
            return httpx.Response(429, headers={"Retry-After": "600"})
        return httpx.Response(200, content=b"ok")

    client, sleeps = _client(handler)
    with pytest.raises(FetchError, match="pausing that host"):
        client.get("https://slow.test/a.jpg", check_robots=False)
    with pytest.raises(FetchError, match="back off"):
        client.get("https://slow.test/b.jpg", check_robots=False)
    assert calls["slow.test"] == 1 and 600.0 not in sleeps  # no long sleep, no 2nd request
    assert client.get("https://fast.test/c.jpg", check_robots=False).content == b"ok"

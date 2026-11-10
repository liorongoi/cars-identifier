"""Source connectors.

Enabled by default (owner decision D2): Wikimedia Commons, brochure/press web
pages listed in the sources file, local folder import (images and brochure PDFs).
Opt-in, gated by ``legal_approved``: Flickr (social network, official API) and
video frames (movies/clips the owner supplies).
"""

from __future__ import annotations

import html
import logging
import os
import re
from collections.abc import Iterator
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

from carid.collector.base import (
    Candidate,
    FetchError,
    HostPaused,
    SourceConnector,
    SourceExhausted,
    Target,
)
from carid.collector.labeler import _phrase_in

log = logging.getLogger(__name__)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".heic"}


def search_queries(target: Target) -> list[str]:
    base = f"{target.make} {target.model}"
    queries = [f"{base} interior", f"{base} dashboard", f"{base} cockpit"]
    for g in target.generations:
        if not g.name.lower().endswith(" gen") and g.name.lower() != "original":
            queries.append(f"{base} {g.name} interior")
    for alias in target.model_aliases[:2]:
        queries.append(f"{target.make} {alias} interior")
    return list(dict.fromkeys(queries))


# --------------------------------------------------------------------------- Wikimedia


def _strip_html(text: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", " ", text or "")).strip()


class WikimediaCommonsSource(SourceConnector):
    name = "wikimedia"
    description = "Wikimedia Commons (openly licensed photos, with licence and author)"
    API = "https://commons.wikimedia.org/w/api.php"

    def search(self, target: Target, limit: int) -> Iterator[Candidate]:
        seen: set[str] = set()
        thumb_width = int(self.config.get("thumb_width", 1600))
        for query in search_queries(target):
            offset: int | None = 0
            while offset is not None and len(seen) < limit:
                params: dict[str, Any] = {
                    "action": "query", "format": "json", "formatversion": "2",
                    "generator": "search", "gsrsearch": f"{query} filetype:bitmap",
                    "gsrnamespace": "6", "gsrlimit": "50", "gsroffset": str(offset),
                    "prop": "imageinfo|categories", "iiprop": "url|extmetadata|size|mime",
                    "iiurlwidth": str(thumb_width), "cllimit": "max", "clshow": "!hidden",
                }
                try:
                    data = self.http.get(self.API, params=params, check_robots=False).json()
                except FetchError as exc:
                    log.warning("wikimedia search failed for %r: %s", query, exc)
                    break
                pages = data.get("query", {}).get("pages", [])
                for page in pages:
                    title = page.get("title", "")
                    if title in seen or not page.get("imageinfo"):
                        continue
                    info = page["imageinfo"][0]
                    if not str(info.get("mime", "")).startswith("image/"):
                        continue
                    meta = info.get("extmetadata", {})
                    cats = " ".join(c.get("title", "").removeprefix("Category:")
                                    for c in page.get("categories", []))
                    seen.add(title)
                    yield Candidate(
                        url=info.get("thumburl") or info["url"],
                        page_url=info.get("descriptionurl"),
                        title=title.removeprefix("File:").rsplit(".", 1)[0].replace("_", " "),
                        description=_strip_html(meta.get("ImageDescription", {}).get("value", "")),
                        license=_strip_html(meta.get("LicenseShortName", {}).get("value", "")),
                        credit=_strip_html(meta.get("Artist", {}).get("value", ""))[:400],
                        extra_text=cats,
                    )
                    if len(seen) >= limit:
                        return
                cont = data.get("continue", {})
                offset = int(cont["gsroffset"]) if "gsroffset" in cont else None


# --------------------------------------------------------------------------- brochure / press pages


class _ImgParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.images: list[dict[str, str]] = []
        self.title = ""
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        a = {k: v or "" for k, v in attrs}
        if tag == "img":
            src = a.get("data-src") or a.get("src") or ""
            srcset = a.get("srcset") or a.get("data-srcset") or ""
            if srcset:  # take the largest candidate
                src = srcset.split(",")[-1].strip().split(" ")[0] or src
            if src:
                self.images.append({"src": src, "alt": a.get("alt", ""),
                                    "title": a.get("title", "")})
        elif tag == "a" and re.search(r"\.(jpe?g|png|webp)(\?|$)", a.get("href", ""), re.I):
            self.images.append({"src": a["href"], "alt": a.get("title", ""), "title": ""})
        elif tag == "title":
            self._in_title = True

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self._in_title = False

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title += data


class WebPagesSource(SourceConnector):
    """Images from manufacturer brochure / press / review pages listed in the sources file.

    Config::

        brochure_sites:
          enabled: true
          sites:
            - {make: Toyota, model: Corolla, year: 2020, urls: ["https://..."], license: "..."}
    """

    name = "brochure_sites"
    description = "Manufacturer brochure / press pages you list in config (robots.txt obeyed)"

    def search(self, target: Target, limit: int) -> Iterator[Candidate]:
        count = 0
        for site in self.config.get("sites", []):
            if not _phrase_in(str(site.get("make", "")), target.make):
                continue
            site_model = site.get("model")
            if site_model and not any(_phrase_in(str(site_model), n)
                                      for n in [target.model, *target.model_aliases]):
                continue
            for page_url in site.get("urls", []):
                try:
                    r = self.http.get(page_url)
                except FetchError as exc:
                    log.warning("page fetch failed %s: %s", page_url, exc)
                    continue
                parser = _ImgParser()
                parser.feed(r.text)
                for img in parser.images:
                    url = urljoin(str(r.url), img["src"])
                    if not url.startswith(("http://", "https://")):
                        continue
                    if re.search(r"(logo|icon|sprite|avatar|\.svg)", url, re.I):
                        continue
                    # When the site entry names the model, page context is enough for labelling.
                    context = " ".join(filter(None, [
                        f"{target.make} {site_model}" if site_model else "",
                        str(site.get("year", "")), img["alt"], img["title"], parser.title,
                    ]))
                    yield Candidate(url=url, page_url=page_url, title=img["alt"] or parser.title,
                                    extra_text=context, license=site.get("license"),
                                    credit=site.get("credit"),
                                    preset_year=site.get("year"))
                    count += 1
                    if count >= limit:
                        return


# --------------------------------------------------------------------------- local folder


class FolderSource(SourceConnector):
    """Import from a local folder laid out as ``<Make>/<Model>/<Year or Generation>/*``.

    PDFs (brochures) are expanded into their embedded images when PyMuPDF is installed.
    """

    name = "folder"
    description = "Local folder import: <Make>/<Model>/<Year>/ images and brochure PDFs"

    def _root(self) -> Path:
        return Path(str(self.config.get("path", "imports"))).expanduser()

    def search(self, target: Target, limit: int) -> Iterator[Candidate]:
        root = self._root()
        if not root.is_dir():
            return
        count = 0
        for make_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            if not any(_phrase_in(make_dir.name, n) and len(make_dir.name) <= len(n) + 3
                       for n in [target.make, *target.make_aliases]):
                continue
            for model_dir in sorted(p for p in make_dir.iterdir() if p.is_dir()):
                if not any(_phrase_in(model_dir.name, n) and len(model_dir.name) <= len(n) + 3
                           for n in [target.model, *target.model_aliases]):
                    continue
                for f in sorted(model_dir.rglob("*")):
                    if not f.is_file():
                        continue
                    sub = f.parent.relative_to(model_dir).as_posix()
                    m = re.match(r"^(19|20)\d{2}$", sub.split("/")[0]) if sub != "." else None
                    year = int(sub.split("/")[0]) if m else None
                    text = f"{target.make} {target.model} {sub if sub != '.' else ''} {f.stem}"
                    suffix = f.suffix.lower()
                    if suffix in IMAGE_SUFFIXES:
                        yield Candidate(url=f.resolve().as_uri(), local_path=str(f), title=f.stem,
                                        extra_text=text, preset_year=year, license="local import",
                                        credit=self.config.get("credit"))
                        count += 1
                    elif suffix == ".pdf":
                        for cand in self._pdf_candidates(f, text, year):
                            yield cand
                            count += 1
                            if count >= limit:
                                return
                    if count >= limit:
                        return

    def _pdf_candidates(self, pdf: Path, text: str, year: int | None) -> Iterator[Candidate]:
        try:
            import fitz  # PyMuPDF
        except ImportError:
            log.warning("PyMuPDF not installed; skipping brochure %s", pdf)
            return
        with fitz.open(pdf) as doc:
            for page_no, page in enumerate(doc, start=1):
                for img_no, info in enumerate(page.get_images(full=True)):
                    yield Candidate(url=f"{pdf.resolve().as_uri()}#p{page_no}i{img_no}",
                                    local_path=f"{pdf}::{info[0]}", title=f"{pdf.stem} p{page_no}",
                                    extra_text=text, preset_year=year,
                                    license="local brochure import")

    def fetch(self, candidate: Candidate) -> bytes:
        assert candidate.local_path is not None
        if "::" in candidate.local_path:
            import fitz

            pdf, xref = candidate.local_path.rsplit("::", 1)
            with fitz.open(pdf) as doc:
                return bytes(doc.extract_image(int(xref))["image"])
        return Path(candidate.local_path).read_bytes()


# --------------------------------------------------------------------------- opt-in: Flickr


class FlickrSource(SourceConnector):
    """Flickr photo search (official API, Creative-Commons licences only). Opt-in."""

    name = "flickr"
    description = "Flickr social network via official API, CC licences only (opt-in)"
    requires_legal_approval = True
    API = "https://api.flickr.com/services/rest/"
    LICENSES = {"1": "CC BY-NC-SA 2.0", "2": "CC BY-NC 2.0", "3": "CC BY-NC-ND 2.0",
                "4": "CC BY 2.0", "5": "CC BY-SA 2.0", "6": "CC BY-ND 2.0",
                "9": "CC0 1.0", "10": "Public Domain Mark"}

    def search(self, target: Target, limit: int) -> Iterator[Candidate]:
        key = os.environ.get(str(self.config.get("api_key_env", "FLICKR_API_KEY")), "")
        if not key:
            log.warning("flickr: API key environment variable not set; skipping")
            return
        count = 0
        for query in search_queries(target):
            params = {
                "method": "flickr.photos.search", "api_key": key, "text": query,
                "license": self.config.get("licenses", "1,2,3,4,5,6,9,10"),
                "content_type": "1", "media": "photos", "per_page": "100",
                "extras": "url_l,url_c,license,owner_name,description,tags",
                "format": "json", "nojsoncallback": "1", "sort": "relevance",
            }
            try:
                data = self.http.get(self.API, params=params, check_robots=False).json()
            except FetchError as exc:
                log.warning("flickr search failed: %s", exc)
                continue
            for p in data.get("photos", {}).get("photo", []):
                url = p.get("url_l") or p.get("url_c")
                if not url:
                    continue
                yield Candidate(
                    url=url, page_url=f"https://www.flickr.com/photos/{p['owner']}/{p['id']}",
                    title=p.get("title", ""),
                    description=_strip_html(p.get("description", {}).get("_content", "")),
                    extra_text=p.get("tags", ""),
                    license=self.LICENSES.get(str(p.get("license")), "unknown"),
                    credit=p.get("ownername"),
                )
                count += 1
                if count >= limit:
                    return

    def fetch(self, candidate: Candidate) -> bytes:
        return self.http.download_image(candidate.url, check_robots=False)


# --------------------------------------------------------------------------- opt-in: video frames


class VideoFramesSource(SourceConnector):
    """Frames sampled from video files the owner supplies (movies, clips). Opt-in.

    Layout: ``<path>/<Make>/<Model>/<Year>/*.mp4``. Every ``every_seconds`` a frame
    is extracted; the interior filter discards non-interior frames during ingest.
    """

    name = "video_frames"
    description = "Frames from movie/video files you supply, <Make>/<Model>/<Year>/ (opt-in)"
    requires_legal_approval = True
    VIDEO_SUFFIXES = {".mp4", ".mkv", ".avi", ".mov", ".m4v", ".webm"}

    def search(self, target: Target, limit: int) -> Iterator[Candidate]:
        folder = FolderSource({"path": self.config.get("path", "imports/videos")}, self.http)
        root = folder._root()
        if not root.is_dir():
            return
        every = float(self.config.get("every_seconds", 2.0))
        count = 0
        for make_dir in (p for p in root.iterdir() if p.is_dir()):
            if not _phrase_in(make_dir.name, target.make):
                continue
            for model_dir in (p for p in make_dir.iterdir() if p.is_dir()):
                if not _phrase_in(model_dir.name, target.model):
                    continue
                for video in sorted(model_dir.rglob("*")):
                    if video.suffix.lower() not in self.VIDEO_SUFFIXES:
                        continue
                    sub = video.parent.relative_to(model_dir).as_posix().split("/")[0]
                    year = int(sub) if re.fullmatch(r"(19|20)\d{2}", sub) else None
                    for t in self._timestamps(video, every):
                        yield Candidate(url=f"{video.resolve().as_uri()}#t={t:.1f}",
                                        local_path=f"{video}::{t:.3f}",
                                        title=f"{video.stem} @ {t:.1f}s",
                                        extra_text=f"{target.make} {target.model} {sub}",
                                        preset_year=year, license="owner-supplied video")
                        count += 1
                        if count >= limit:
                            return

    @staticmethod
    def _timestamps(video: Path, every: float) -> Iterator[float]:
        import cv2

        cap = cv2.VideoCapture(str(video))
        try:
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        finally:
            cap.release()
        duration = frames / fps if fps else 0
        t = every / 2
        while t < duration:
            yield t
            t += every

    def fetch(self, candidate: Candidate) -> bytes:
        import cv2

        assert candidate.local_path is not None
        path, ts = candidate.local_path.rsplit("::", 1)
        cap = cv2.VideoCapture(path)
        try:
            cap.set(cv2.CAP_PROP_POS_MSEC, float(ts) * 1000)
            ok, frame = cap.read()
        finally:
            cap.release()
        if not ok:
            raise FetchError(f"cannot read frame at {ts}s from {path}")
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if not ok:
            raise FetchError("frame encode failed")
        return bytes(buf.tobytes())


CONNECTOR_CLASSES: dict[str, type[SourceConnector]] = {
    c.name: c for c in (WikimediaCommonsSource, WebPagesSource, FolderSource, FlickrSource,
                        VideoFramesSource)
}


# --------------------------------------------------------------------------- Openverse


class OpenverseSource(SourceConnector):
    """Openverse (openverse.org): openly licensed images from Wikimedia, Flickr and others,
    searched through its official API. Every result carries its licence and creator.

    Anonymous use allows 200 searches/day and 20/minute; set ``client_id_env`` /
    ``client_secret_env`` to a registered (free) application for higher limits.
    ``exclude_providers`` (e.g. ``[flickr]``) drops results originating from a provider.
    """

    name = "openverse"
    description = "Openverse open-licence image catalogue (official API, CC licences)"
    API = "https://api.openverse.org/v1/images/"
    TOKEN_URL = "https://api.openverse.org/v1/auth_tokens/token/"  # noqa: S105 - a URL

    def __init__(self, config: dict[str, Any], http: Any) -> None:
        super().__init__(config, http)
        self._token: str | None = None
        self._exhausted_after_this = False
        if http is not None:
            # Burst limit is 20/min for anonymous use: keep ~3.2 s between API calls.
            http.host_intervals.setdefault("api.openverse.org",
                                           float(config.get("min_interval_s", 3.5)))
            # It is the only search source: a short burst back-off is worth waiting for;
            # longer ones mean the daily quota is gone (SourceExhausted below).
            http.host_max_wait.setdefault("api.openverse.org",
                                          float(config.get("max_backoff_s", 900)))

    def _auth_headers(self) -> dict[str, str]:
        cid = os.environ.get(str(self.config.get("client_id_env", "OPENVERSE_CLIENT_ID")), "")
        secret = os.environ.get(
            str(self.config.get("client_secret_env", "OPENVERSE_CLIENT_SECRET")), "")
        if not (cid and secret):
            return {}
        if self._token is None:
            r = self.http.client.post(self.TOKEN_URL, data={
                "grant_type": "client_credentials", "client_id": cid, "client_secret": secret})
            r.raise_for_status()
            self._token = str(r.json()["access_token"])
        return {"Authorization": f"Bearer {self._token}"}

    def _query(self, params: dict[str, Any]) -> dict[str, Any]:
        try:
            r = self.http.get(self.API, params=params, check_robots=False,
                              headers=self._auth_headers())
        except HostPaused as exc:
            raise SourceExhausted(f"Openverse rate limit reached ({exc})") from exc
        except FetchError as exc:
            if "429" in str(exc):
                raise SourceExhausted("Openverse rate limit reached") from exc
            raise
        for key, value in r.headers.items():
            if key.lower().startswith("x-ratelimit-available") and "sustained" in key.lower() \
                    and value.strip() == "0":
                # This response is still valid; the next call would be refused.
                self._exhausted_after_this = True
        data: dict[str, Any] = r.json()
        return data

    def search(self, target: Target, limit: int) -> Iterator[Candidate]:
        self._exhausted_after_this = False
        excluded = {str(p).lower() for p in self.config.get("exclude_providers", [])}
        page_size = int(self.config.get("page_size", 20))
        max_queries = int(self.config.get("max_queries_per_model", 2))
        max_pages = int(self.config.get("max_pages", 1))
        seen: set[str] = set()
        for query in search_queries(target)[:max_queries]:
            for page in range(1, max_pages + 1):
                if self._exhausted_after_this:
                    raise SourceExhausted("Openverse daily quota used up")
                params: dict[str, Any] = {"q": query, "page_size": page_size, "page": page,
                                          "mature": "false"}
                if self.config.get("license_type"):
                    params["license_type"] = self.config["license_type"]
                data = self._query(params)
                results = data.get("results", [])
                for r in results:
                    url = r.get("url")
                    if not url or url in seen:
                        continue
                    if str(r.get("provider", "")).lower() in excluded \
                            or str(r.get("source", "")).lower() in excluded:
                        continue
                    seen.add(url)
                    lic = str(r.get("license", "")).upper()
                    licence = (lic if lic in ("CC0", "PDM")
                               else f"CC {lic} {r.get('license_version') or ''}".strip())
                    yield Candidate(
                        url=url,
                        page_url=r.get("foreign_landing_url"),
                        title=str(r.get("title") or ""),
                        extra_text=" ".join(t.get("name", "") for t in r.get("tags") or []),
                        license=licence,
                        credit=f"{r.get('creator') or 'unknown'} via {r.get('source')}"[:400],
                    )
                    if len(seen) >= limit:
                        return
                if page >= int(data.get("page_count", 1)):
                    break


# --------------------------------------------------------------------------- manifest


MANIFEST_FIELDS = ["url", "make", "model", "year_start", "year_end", "license", "credit",
                   "page_url", "title", "source"]


class ManifestSource(SourceConnector):
    """Download images listed in CSV manifests (``url, make, model, year_start, ...``).

    Manifests hold links, labels and licences only, so a collection made on one machine
    (``carid export-manifest``) can be reproduced on another without searching again.
    """

    name = "manifest"
    description = "CSV manifests of image links + labels (reproduce another host's collection)"

    def _rows(self) -> Iterator[dict[str, str]]:
        import csv
        import glob

        for pattern in self.config.get("paths", ["data/manifests/*.csv"]):
            for name in sorted(glob.glob(str(pattern))):  # noqa: PTH207 - abs or rel patterns
                with Path(name).open(encoding="utf-8", newline="") as f:
                    yield from csv.DictReader(f)

    def search(self, target: Target, limit: int) -> Iterator[Candidate]:
        count = 0
        names = [target.model, *target.model_aliases]
        for row in self._rows():
            if not (_phrase_in(row.get("make", ""), target.make)
                    and any(_phrase_in(row.get("model", ""), n) for n in names)):
                continue
            year = row.get("year_start") or ""
            yield Candidate(
                url=row["url"], page_url=row.get("page_url") or None,
                title=row.get("title") or "",
                # Labels in a manifest are authoritative: state them explicitly for the labeler.
                extra_text=f"{target.make} {target.model} interior {year}",
                license=row.get("license") or None, credit=row.get("credit") or None,
                preset_year=int(year) if year.isdigit() else None,
            )
            count += 1
            if count >= limit:
                return


CONNECTOR_CLASSES[OpenverseSource.name] = OpenverseSource
CONNECTOR_CLASSES[ManifestSource.name] = ManifestSource

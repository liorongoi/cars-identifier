"""Source connectors.

Enabled by default (owner decision D2): Wikimedia Commons, brochure/press web
pages listed in the sources file, local folder import (images and brochure PDFs).
Opt-in, gated by ``legal_approved``: Flickr (social network, official API), Brave
web image search (official API, unknown licences) and video frames (movies/clips
the owner supplies).
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
from urllib.parse import quote_plus, urljoin, urlparse

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
        self.links: list[tuple[str, str]] = []  # (href, link text)
        self.title = ""
        self._in_title = False
        self._link: list[str] | None = None  # [href, text so far] while inside <a>

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
        elif tag == "a" and a.get("href"):
            self._link = [a["href"], a.get("title", "")]
        elif tag == "title":
            self._in_title = True

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self._in_title = False
        elif tag == "a" and self._link is not None:
            self.links.append((self._link[0], " ".join(self._link[1].split())))
            self._link = None

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title += data
        if self._link is not None and len(self._link[1]) < 300:
            self._link[1] += " " + data


class WebPagesSource(SourceConnector):
    """Images from manufacturer brochure / press / review pages listed in the sources file.

    Config::

        brochure_sites:
          enabled: true
          sites:
            - {make: Toyota, model: Corolla, urls: ["https://..."], license: "..."}
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
                        img["alt"], img["title"], parser.title,
                    ]))
                    yield Candidate(url=url, page_url=page_url, title=img["alt"] or parser.title,
                                    extra_text=context, license=site.get("license"),
                                    credit=site.get("credit"))
                    count += 1
                    if count >= limit:
                        return


# --------------------------------------------------------------------------- local folder


class FolderSource(SourceConnector):
    """Import from a local folder laid out as ``<Make>/<Model>/**`` (any sub-folders).

    PDFs (brochures) are expanded into their embedded images when PyMuPDF is installed.
    """

    name = "folder"
    description = "Local folder import: <Make>/<Model>/ images and brochure PDFs"

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
                    text = f"{target.make} {target.model} {f.stem}"
                    suffix = f.suffix.lower()
                    if suffix in IMAGE_SUFFIXES:
                        yield Candidate(url=f.resolve().as_uri(), local_path=str(f), title=f.stem,
                                        extra_text=text, license="local import",
                                        credit=self.config.get("credit"))
                        count += 1
                    elif suffix == ".pdf":
                        for cand in self._pdf_candidates(f, text):
                            yield cand
                            count += 1
                            if count >= limit:
                                return
                    if count >= limit:
                        return

    def _pdf_candidates(self, pdf: Path, text: str) -> Iterator[Candidate]:
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
                                    extra_text=text, license="local brochure import")

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

    Layout: ``<path>/<Make>/<Model>/**/*.mp4``. Every ``every_seconds`` a frame
    is extracted; the interior filter discards non-interior frames during ingest.
    """

    name = "video_frames"
    description = "Frames from movie/video files you supply, <Make>/<Model>/ (opt-in)"
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
                    for t in self._timestamps(video, every):
                        yield Candidate(url=f"{video.resolve().as_uri()}#t={t:.1f}",
                                        local_path=f"{video}::{t:.3f}",
                                        title=f"{video.stem} @ {t:.1f}s",
                                        extra_text=f"{target.make} {target.model}",
                                        license="owner-supplied video")
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
            try:
                r = self.http.get(self.API, params=params, check_robots=False,
                                  headers=self._auth_headers())
            except FetchError as exc:
                if "HTTP 401" not in str(exc) or self._token is None:
                    raise
                self._token = None  # access token expired: get a fresh one, retry once
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


MANIFEST_FIELDS = ["url", "make", "model", "license", "credit", "page_url", "title", "source"]


class ManifestSource(SourceConnector):
    """Download images listed in CSV manifests (``url, make, model, license, ...``).

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
            yield Candidate(
                url=row["url"], page_url=row.get("page_url") or None,
                title=row.get("title") or "",
                # Labels in a manifest are authoritative: state them explicitly for the labeler.
                extra_text=f"{target.make} {target.model} interior",
                license=row.get("license") or None, credit=row.get("credit") or None,
            )
            count += 1
            if count >= limit:
                return


CONNECTOR_CLASSES[OpenverseSource.name] = OpenverseSource
CONNECTOR_CLASSES[ManifestSource.name] = ManifestSource


# --------------------------------------------------------------------------- opt-in: Brave


class BraveImageSource(SourceConnector):
    """Web image search through the Brave Search API (its own index of the open web). Opt-in.

    Results come from arbitrary websites with unknown licences, so this connector is
    gated by ``legal_approved``. Brave's standard plans do not grant storage rights
    for search results; check the plan's terms before approving. Search calls are
    billed per request, so ``max_queries_per_model`` keeps the cost per model low.
    Images are downloaded from the original site, obeying its robots.txt.
    """

    name = "brave_images"
    description = "Open-web image search via Brave Search API, unknown licences (opt-in)"
    requires_legal_approval = True
    API = "https://api.search.brave.com/res/v1/images/search"

    def __init__(self, config: dict[str, Any], http: Any) -> None:
        super().__init__(config, http)
        if http is not None:
            http.host_intervals.setdefault("api.search.brave.com",
                                           float(config.get("min_interval_s", 1.1)))

    def _query(self, key: str, params: dict[str, Any]) -> dict[str, Any]:
        try:
            r = self.http.get(self.API, params=params, check_robots=False,
                              headers={"X-Subscription-Token": key,
                                       "Accept": "application/json"})
        except HostPaused as exc:
            raise SourceExhausted(f"Brave rate limit reached ({exc})") from exc
        except FetchError as exc:
            msg = str(exc)
            if "HTTP 401" in msg or "HTTP 403" in msg:
                raise SourceExhausted("Brave API key rejected") from exc
            if "HTTP 402" in msg or "429" in msg:
                raise SourceExhausted("Brave quota or credit used up") from exc
            raise
        data: dict[str, Any] = r.json()
        return data

    def search(self, target: Target, limit: int) -> Iterator[Candidate]:
        key = os.environ.get(str(self.config.get("api_key_env", "BRAVE_API_KEY")), "")
        if not key:
            raise SourceExhausted("API key environment variable not set")
        count = max(1, min(int(self.config.get("count", 50)), 200))
        max_queries = int(self.config.get("max_queries_per_model", 2))
        excluded = {str(d).lower() for d in self.config.get("exclude_domains", [])}
        seen: set[str] = set()
        for query in search_queries(target)[:max_queries]:
            params: dict[str, Any] = {"q": query, "count": count,
                                      "safesearch": self.config.get("safesearch", "strict"),
                                      # Model names are often not dictionary words.
                                      "spellcheck": "false"}
            for opt in ("country", "search_lang"):
                if self.config.get(opt):
                    params[opt] = self.config[opt]
            for r in self._query(key, params).get("results", []):
                url = (r.get("properties") or {}).get("url")
                host = str(r.get("source") or "").lower()
                if not url or url in seen or not str(url).startswith(("http://", "https://")):
                    continue
                if any(host == d or host.endswith("." + d) for d in excluded):
                    continue
                seen.add(url)
                path = str((r.get("meta_url") or {}).get("path") or "")
                yield Candidate(
                    url=url, page_url=r.get("url"),
                    title=" ".join(_strip_html(str(r.get("title") or "")).split()),
                    # Page paths often name the car ("/2020-toyota-corolla-interior").
                    extra_text=re.sub(r"[/_\-.›]+", " ", path).strip(),
                    license="unknown (web search result; rights not verified)",
                    credit=host or None,
                )
                if len(seen) >= limit:
                    return


CONNECTOR_CLASSES[BraveImageSource.name] = BraveImageSource


# --------------------------------------------------------------------------- press-site search

_SIZE_SUFFIX = re.compile(r"-(\d{2,4})x(\d{2,4})(?=\.\w{3,4}$)")
_SKIP_IMAGE = re.compile(r"(logo|icon|sprite|avatar|placeholder|banner-ad|\.svg|\.gif)", re.I)


def _url_words(url: str) -> str:
    """Words from an image or page URL path: '/2021_Corolla-Apex_Interior_001.jpg' ->
    '2021 Corolla Apex Interior 001'."""
    path = re.sub(r"^https?://[^/]+", "", url).split("?")[0]
    path = _SIZE_SUFFIX.sub("", path)
    path = re.sub(r"\.\w{3,4}$", "", path.rsplit("/", 1)[-1])
    return " ".join(re.sub(r"[_\-+.%]+|(?<=[a-z])(?=[A-Z])", " ", path).split())


def _largest_variants(urls: list[str], max_side: int = 2400) -> list[str]:
    """Keep one URL per picture: CMSs publish '<name>-768x470.jpg', '<name>-1500x900.jpg'...
    Choose the largest sized variant up to ``max_side``, else the unsized original."""
    best: dict[str, tuple[int, str]] = {}
    for url in urls:
        m = _SIZE_SUFFIX.search(url.split("?")[0])
        key = _SIZE_SUFFIX.sub("", url.split("?")[0])
        if m:
            side = max(int(m.group(1)), int(m.group(2)))
            score = side if side <= max_side else 0
        else:
            score = max_side // 2  # original of unknown size: below a large sized copy
        if key not in best or score > best[key][0]:
            best[key] = (score, url)
    return [url for _, url in best.values()]


_THUMB_MARK = re.compile(r"(?:__?thumb(?:nail)?|-thumbnail|_small|-small)(?=\.\w{3,4}$)", re.I)


def _full_size_guess(url: str, min_side: int = 1000) -> str | None:
    """Full-size address for a listed thumbnail, by the common CMS conventions
    ('name-272x182.jpg' -> 'name.jpg', 'name__thumb.jpg' -> 'name.jpg'); None when the
    listed image is already large or follows no known convention."""
    base = url.split("?")[0]
    m = _SIZE_SUFFIX.search(base)
    if m and max(int(m.group(1)), int(m.group(2))) < min_side:
        return _SIZE_SUFFIX.sub("", base)
    if _THUMB_MARK.search(base):
        return _THUMB_MARK.sub("", base)
    return None


class PressSearchSource(SourceConnector):
    """Manufacturer press rooms searched per model, through the site's own search page.

    For each configured site of the target's make, the search URL is filled with the
    model name; images on the results page are collected, and with ``follow_links``
    so are images on result pages (press releases, press kits) whose address or link
    text names the model. Every image's file name and caption are used for labelling,
    so a photo of another model on the same page is rejected by the labeller. Pages
    are fetched politely and robots.txt is obeyed. Config::

        press_sites:
          enabled: true
          sites:
            - make: Toyota
              search_url: "https://pressroom.toyota.com/?s={query}"
              query: "{model} interior"
              follow_links: 0
              license: "Manufacturer press image (editorial use)"
    """

    name = "press_sites"
    description = "Manufacturer press rooms searched per model (robots.txt obeyed)"

    def __init__(self, config: dict[str, Any], http: Any) -> None:
        super().__init__(config, http)
        self._fallback: dict[str, str] = {}  # full-size guess -> listed image

    def fetch(self, candidate: Candidate) -> bytes:
        try:
            return self.http.download_image(candidate.url)
        except FetchError:
            listed = self._fallback.get(candidate.url)
            if listed is None:
                raise
            return self.http.download_image(listed)

    def _page(self, url: str) -> tuple[str, _ImgParser] | None:
        try:
            r = self.http.get(url)
        except FetchError as exc:
            log.info("press page fetch failed %s: %s", url, exc)
            return None
        parser = _ImgParser()
        parser.feed(r.text)
        return str(r.url), parser

    def _names(self, target: Target) -> list[str]:
        return [target.model, *target.model_aliases]

    def _names_model(self, text: str, target: Target) -> bool:
        return any(_phrase_in(text, n) for n in self._names(target))

    def search(self, target: Target, limit: int) -> Iterator[Candidate]:
        seen: set[str] = set()
        default_follow = int(self.config.get("follow_links", 4))
        for site in self.config.get("sites", []):
            makes = [str(site.get("make", "")), *[str(a) for a in site.get("make_aliases", [])]]
            if not any(_phrase_in(m, n) for m in makes
                       for n in [target.make, *target.make_aliases] if m):
                continue
            query = str(site.get("query", "{model}")).format(model=target.model,
                                                              make=target.make)
            search_url = str(site["search_url"]).replace("{query}", quote_plus(query))
            first = self._page(search_url)
            if first is None:
                continue
            pages: list[tuple[str, _ImgParser, bool]] = [(first[0], first[1], False)]
            follow = int(site.get("follow_links", default_follow))
            host = urlparse(first[0]).netloc
            hosts = {host, *[str(h) for h in site.get("link_hosts", [])]}
            followed: set[str] = set()
            for href, text in first[1].links:
                if len(followed) >= follow:
                    break
                url = urljoin(first[0], href).split("#")[0]
                if (urlparse(url).netloc not in hosts or url in followed or url == first[0]
                        or "search" in url.lower()
                        or not self._names_model(f"{text} {_url_words(url)}", target)):
                    continue
                followed.add(url)
                page = self._page(url)
                if page is not None:
                    pages.append((page[0], page[1], True))
            for page_url, parser, is_article in pages:
                urls = []
                captions: dict[str, str] = {}
                for img in parser.images:
                    url = urljoin(page_url, img["src"])
                    if not url.startswith(("http://", "https://")) or _SKIP_IMAGE.search(url):
                        continue
                    urls.append(url)
                    captions.setdefault(_SIZE_SUFFIX.sub("", url.split("?")[0]),
                                        " ".join(filter(None, [img["alt"], img["title"]])))
                for url in _largest_variants(urls):
                    if url in seen:
                        continue
                    own = " ".join(filter(None, [
                        captions.get(_SIZE_SUFFIX.sub("", url.split("?")[0]), ""),
                        _url_words(url)]))
                    # An image whose own name/caption names no model of this make takes
                    # the article's title (a press release is about one model); search
                    # pages give none.
                    page_title = " ".join(parser.title.split())
                    if is_article and not any(
                            _phrase_in(own, n)
                            for n in [*self._names(target), *target.sibling_models]):
                        own = f"{own} {page_title}"
                    seen.add(url)
                    full = _full_size_guess(url)
                    if full is not None and full not in seen:
                        self._fallback[full] = url
                        seen.add(full)
                        url = full
                    yield Candidate(
                        url=url, page_url=page_url, title=own[:300],
                        extra_text=f"{target.make} {own}",
                        license=site.get("license", "Manufacturer press image (editorial use)"),
                        credit=site.get("credit") or f"{target.make} press room")
                    if len(seen) >= limit:
                        return


CONNECTOR_CLASSES[PressSearchSource.name] = PressSearchSource

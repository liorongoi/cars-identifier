# Software Requirements Specification
## CarID-Interior — Vehicle Identification from Interior Images

| Field | Value |
|---|---|
| Document | SRS, version 1.0 |
| Status | Baseline for implementation (Phase 1) |
| Date | 2026-11-10 |
| Audience | Image analysts, project owner, developers, reviewers |

---

## 1. Introduction

### 1.1 Purpose
This document specifies the requirements for **CarID-Interior**, a system that
identifies a vehicle's **make, model and model year (or year range)** from a
photograph of the vehicle's **interior**. It is the reference for design,
implementation, testing and acceptance.

### 1.2 Scope
CarID-Interior consists of:

1. **Reference database** — interior images of passenger vehicles, organised
   `Make › Model › Year`, collected from the web by an automated collector and
   from analyst uploads.
2. **Identification engine** — finds distinctive interior parts (steering
   wheel, instrument cluster, infotainment screen, air vents, gear selector,
   climate controls, door panel, center console…), compares them with the
   reference database and returns the **top 5–10 candidates ranked by
   probability**.
3. **Analyst workstation** — a web UI for running identifications, viewing the
   evidence, browsing database coverage, and requesting new makes/models to be
   collected.
4. **Evaluation harness** — measures and compares identification algorithms
   so the best one is chosen on evidence, not assumption.

Out of scope for v1: exterior identification, licence-plate reading, video
ingestion (single frames may be uploaded), trim-level identification.

### 1.3 Decisions taken with the project owner (interview, 2026-11-10)

| # | Topic | Decision |
|---|---|---|
| D1 | Deployment | **Local server, offline.** Analysis must run with no internet. Data collection is a separate, online activity whose output is imported. |
| D2 | Data sources | **Open sources + manufacturer brochures/press sites enabled by default.** Social-network and movie-frame connectors are implemented but **disabled** until the owner records legal approval in configuration. |
| D3 | UI language | **English.** |
| D4 | First scope | **Pilot of ~20 makes**, full system; further makes added from the UI. |
| D5 | Time span | Model years **1996 – current** (30 years). |

### 1.4 Definitions

| Term | Meaning |
|---|---|
| Model-year (MY) | One `(make, model, year)` triple — the atomic label. |
| Generation | A run of model years with essentially the same interior. Results are reported as year ranges because adjacent years are often visually identical. |
| Part | A localised interior component (see §3.3, Table P). |
| Query | An image submitted by an analyst for identification. |
| Candidate | A ranked result: make, model, year range, probability, evidence. |
| Top-k accuracy | Fraction of queries where the true model appears in the first k candidates. |
| Degradation | Blur, low light, noise, compression, occlusion (faces, hands, people). |

### 1.5 References
- IEEE 830 / ISO/IEC/IEEE 29148 (structure of this SRS)
- `docs/PLAN.md` — implementation plan and milestones
- `docs/ARCHITECTURE.md` — component design
- `docs/ALGORITHMS.md` — algorithm candidates and selection method

---

## 2. Overall description

### 2.1 Product perspective
A self-contained system: Python back end (REST API, ML pipeline, collector,
job runner), SQLite database (upgradeable to PostgreSQL), file-system image
store, vector indexes, and a browser front end served by the same server.
No external service is called during identification.

```
 ┌──────────── online (collection host) ─────────────┐     ┌─────────── offline analysis server ────────────┐
 │ Source connectors → Downloader → De-dup → Interior│ ==> │ Image store ─ Part detector ─ Embedder ─ Index │
 │ filter → Labeler → Image store + metadata         │     │      ▲                                   │     │
 └───────────────────────────────────────────────────┘     │ REST API ◄── Analyst UI (browser)      Matcher  │
                                                           └────────────────────────────────────────────────┘
```
Both halves are the same software; collection simply needs network access.

### 2.2 Users

| Role | Needs |
|---|---|
| **Image analyst** (primary) | Fast, trustworthy answers with visible evidence; tolerates poor images; exports results for reports. |
| **Data curator** | Adds makes/models, launches collection, reviews and corrects image labels, sees coverage gaps. |
| **Administrator** | Installs, configures sources and legal flags, backs up, monitors jobs. |

### 2.3 Operating environment
- Linux server (Ubuntu 22.04+ or any Docker host), 4+ CPU cores, 16 GB RAM.
  An NVIDIA GPU is optional; it speeds up indexing and identification.
- Modern browser (Chrome, Edge, Firefox) on the analysts' network.
- Python 3.11+, Node 20+ (build time only).

### 2.4 Constraints
- C1 Identification must work fully offline (D1).
- C2 The collector must obey robots.txt, site rate limits and source licences;
  disabled-by-default connectors may only run when `legal_approved=true` is set
  for that connector (D2).
- C3 Query images are sensitive: they are never sent outside the server and
  are kept only as long as the retention setting allows.
- C4 Only open-licence ML models with weights that can be stored locally.

### 2.5 Assumptions
- A1 The interior of most cars stays the same across a generation (4–8 years);
  a facelift often changes the screen, vents or steering wheel. The system
  therefore reports year **ranges** and the true year is "in range".
- A2 Web coverage is uneven; rare models will have few images. The system must
  show image counts so analysts can judge reliability.

---

## 3. Functional requirements

Priority: **M** = must (v1), **S** = should (v1 if time allows), **C** = could (later).

### 3.1 Reference database & taxonomy

| ID | Requirement | Pri |
|---|---|---|
| FR-DB-1 | Store a taxonomy `Make › Model › Year`, with optional generation name and year range per model. | M |
| FR-DB-2 | Store every reference image with: file hash, perceptual hash, source, source URL, licence, author/credit, collection date, label (make/model/year or year range), label confidence, quality metrics, review status. | M |
| FR-DB-3 | Report, for every make, model and model-year, **how many images** are in the database (total, reviewed, and per part). | M |
| FR-DB-4 | Ship a seed taxonomy for the ~20 pilot makes (1996 – current). | M |
| FR-DB-5 | Expand the taxonomy for any make from Wikidata (online) or by manual entry. | M |
| FR-DB-6 | Prevent duplicates: identical files (SHA-256) and near-duplicates (perceptual hash distance ≤ threshold) are stored once. | M |
| FR-DB-7 | Import/export the database (images + metadata) as a portable bundle, to move data from the collection host to the offline server. | M |
| FR-DB-8 | Curators can relabel, approve, or reject an image; changes are audited. | S |

### 3.2 Data collection

| ID | Requirement | Pri |
|---|---|---|
| FR-COL-1 | Pluggable **source connectors** with a common interface (search → candidates → download). | M |
| FR-COL-2 | Enabled-by-default connectors: **Wikimedia Commons**, **Openverse** (open-licence catalogue via its official API; Flickr-origin results can be excluded by config), **manufacturer brochure / press sites** (configured URL list, robots.txt respected), **local folder import** (brochure PDFs and images the owner already holds), **CSV manifests** (links + labels exported from another host). | M |
| FR-COL-3 | Opt-in connectors (disabled unless `legal_approved`): **social networks** (via official APIs and credentials supplied by the owner), **movie frames** (from video files the owner supplies; frames sampled and filtered). | M |
| FR-COL-4 | Query generation per model-year (e.g. "2018 Toyota Corolla interior", "Corolla E210 dashboard"), multilingual variants for non-English markets. | M |
| FR-COL-5 | **Interior filter**: discard images that are not car interiors (zero-shot classifier score below threshold). | M |
| FR-COL-6 | **Label inference**: derive make/model/year from source metadata (title, categories, captions); keep a label-confidence value. | M |
| FR-COL-7 | Throttling, retry with exponential back-off on HTTP 429/5xx, resumable jobs, per-source quotas. | M |
| FR-COL-8 | Collection runs as **background jobs** with progress, counts, errors and logs visible in the UI. | M |
| FR-COL-9 | Analyst can **add a missing make** (or model) from the UI; this creates the taxonomy entry and queues a collection job for it. | M |
| FR-COL-10 | Brochure PDF import: extract embedded images and pages, then run the interior filter. | S |

### 3.3 Identification

Table P — part types detected (v1):

| Part | Why it is discriminative |
|---|---|
| steering_wheel | Spoke layout, badge, buttons — strongly brand/generation specific |
| instrument_cluster | Gauge shapes, digital vs analog, hood shape |
| infotainment | Screen size/shape, floating vs integrated, bezel buttons |
| air_vent | Vent shapes are a common facelift change |
| center_console | Layout, storage, materials |
| gear_selector | Lever / rotary / column / buttons |
| climate_control | Knob vs button vs touch; layout |
| door_panel | Handle, speaker grille, window switches |
| dashboard (whole) | Overall architecture and symmetry |

| ID | Requirement | Pri |
|---|---|---|
| FR-ID-1 | Accept JPEG, PNG, WebP, BMP, TIFF and HEIC* query images up to 40 MB; drag-and-drop, file picker and clipboard paste. (*HEIC when the optional decoder is installed.) | M |
| FR-ID-2 | **Quality assessment** of the query: sharpness, brightness, contrast, noise, resolution; shown to the analyst with a plain-language verdict. | M |
| FR-ID-3 | **Robust pre-processing**: low-light enhancement (CLAHE / gamma), denoising, optional deblurring; the analyst can toggle it. | M |
| FR-ID-4 | **Distraction handling**: detect faces/people and mask them so they do not affect matching; masked regions shown in the UI. | M |
| FR-ID-5 | **Part detection**: locate the parts in Table P with bounding boxes and confidence; analyst can draw/adjust a box manually and re-run. | M (draw: S) |
| FR-ID-6 | **Matching**: compare the whole image and each detected part with the same part type in the reference database. | M |
| FR-ID-7 | **Ranking**: fuse the per-part evidence into one probability per model and generation/year range; return **top 5–10** (configurable), highest first. | M |
| FR-ID-8 | **Result table** columns: rank, make, model, year range, probability, confidence band, number of reference images for that candidate, contributing parts, best matching reference thumbnails. | M |
| FR-ID-9 | **Evidence view**: for a candidate, show the query part next to the best matching reference part, with similarity score. | M |
| FR-ID-10 | Flag **low-confidence / unknown** results (e.g. top probability below threshold or poor quality), never present a guess as certain. | M |
| FR-ID-11 | Let the analyst filter candidates by make or year range (prior knowledge). | S |
| FR-ID-12 | Export a case report (PDF/JSON/CSV) with the query, quality report, candidates and evidence. | M (JSON/CSV), S (PDF) |
| FR-ID-13 | Keep a history of identifications with case number, analyst name, time and result (audit log). | M |
| FR-ID-14 | Analyst feedback ("correct answer was X") stored and usable to improve the model. | S |

### 3.4 Algorithm evaluation

| ID | Requirement | Pri |
|---|---|---|
| FR-EV-1 | Hold-out evaluation split, stratified by model, with no near-duplicates across splits. | M |
| FR-EV-2 | **Degradation suite**: each test image is also evaluated blurred, darkened, noised, JPEG-compressed and partly occluded, to measure robustness. | M |
| FR-EV-3 | Metrics: top-1/top-5/top-10 accuracy at make, model and model+year-range level; mean reciprocal rank; latency. | M |
| FR-EV-4 | Compare at least: (a) whole-image embedding, (b) part-based matching, (c) fused part+whole; report which is best; the server uses the configured winner. | M |
| FR-EV-5 | Benchmark reports stored and viewable in the UI. | S |

### 3.5 User interface

| ID | Requirement | Pri |
|---|---|---|
| FR-UI-1 | Dark, high-contrast "analyst console" design; keyboard friendly; works at 1366×768 and up. | M |
| FR-UI-2 | Screens: **Identify**, **Results/Evidence**, **Database** (tree with counts and coverage heat-map), **Collection jobs**, **Add make/model**, **History**, **Benchmarks**, **Settings/System health**. | M |
| FR-UI-3 | Every number shown (probabilities, counts) is traceable to data (tooltips / drill-down). | M |
| FR-UI-4 | Clear error messages; long operations show progress and never freeze the UI. | M |

---

## 4. Non-functional requirements

| ID | Category | Requirement |
|---|---|---|
| NFR-1 | Performance | Identification ≤ 5 s on CPU (≤ 1.5 s with GPU) for a 12 MP image with a 500k-vector index. UI first paint ≤ 2 s. |
| NFR-2 | Accuracy (targets, measured by FR-EV) | Pilot set, clean images: model top-5 ≥ 85 %, top-10 ≥ 92 %. Degraded images: top-5 ≥ 70 %. Make top-1 ≥ 85 %. Targets are revised after the first benchmark on real collected data. |
| NFR-3 | Scalability | 150+ makes, 10 000+ model-years, 2 M reference images on a single server (vector index sharded per part type). |
| NFR-4 | Reliability | Jobs resumable after restart; database writes transactional; no partial images in the store. |
| NFR-5 | Security | Runs on internal network; optional login (roles: analyst, curator, admin); upload size and type validation; no execution of downloaded content; path-traversal-safe storage; audit log. |
| NFR-6 | Privacy | Query images stored only for the retention period (default 30 days, configurable, 0 = do not store). Faces in reference images are blurred before storage. |
| NFR-7 | Legal | Every reference image keeps source URL, licence and credit. Disabled connectors cannot run without the legal flag. |
| NFR-8 | Maintainability | Typed Python (mypy strict on core), TypeScript strict, lint clean (ruff, eslint), unit + integration tests ≥ 80 % line coverage on core modules, CI on every push, mandatory code review before merge. |
| NFR-9 | Portability | Docker image + docker-compose; also installable with pip. Model weights downloadable once and stored locally for offline use. |
| NFR-10 | Observability | Structured logs, `/api/health` endpoint, job and index statistics in the UI. |

---

## 5. External interfaces

### 5.1 REST API (summary — full schema at `/api/docs`)

| Method & path | Purpose |
|---|---|
| `POST /api/identify` | Multipart image + options → quality, parts, candidates. |
| `GET /api/identifications` / `{id}` | History / one case. |
| `GET /api/identifications/{id}/export?format=json|csv` | Case export. |
| `GET /api/catalog` | Make › Model › Year tree with image counts. |
| `POST /api/catalog/makes` | Add a make (optionally with models) and queue collection. |
| `POST /api/catalog/models` | Add a model with year range. |
| `GET /api/images` / `{id}/file` | Browse/serve reference images. |
| `PATCH /api/images/{id}` | Relabel / approve / reject. |
| `GET /api/jobs`, `POST /api/jobs`, `POST /api/jobs/{id}/cancel` | Collection & indexing jobs. |
| `GET /api/sources` | Connector list with enabled/legal status. |
| `GET /api/benchmarks` / `POST /api/benchmarks` | Evaluation reports. |
| `GET /api/health`, `GET /api/stats` | System status. |

### 5.2 Command line (`carid`)
`init`, `seed`, `collect`, `import-folder`, `index`, `identify`, `benchmark`,
`export-bundle`, `import-bundle`, `serve`, `fetch-models`.

---

## 6. Verification & acceptance

| Area | Method |
|---|---|
| Each FR | Mapped to automated tests (unit/integration) or a manual acceptance script in `docs/ACCEPTANCE.md`. |
| Accuracy (NFR-2) | `carid benchmark` report on the hold-out set, reviewed with the owner. |
| Code quality (NFR-8) | CI: ruff, mypy, pytest + coverage, eslint, tsc, vitest, front-end build. Every change reviewed (automated review + human). |
| Usability | Walk-through with 2+ analysts on real case images; findings logged as issues. |

---

## 7. Risks

| Risk | Impact | Mitigation |
|---|---|---|
| Sparse web data for rare models | Low accuracy on those models | Show image counts; curator can import brochures; generation-level labels pool images across years. |
| Wrong labels in web data | Noisy matching | Label-confidence, interior filter, curator review, outlier detection on embeddings. |
| Adjacent years look identical | Year error | Report year ranges; probability mass merged per generation. |
| Source rate limits / ToS | Slow or blocked collection | Back-off, quotas, official APIs, local import paths. |
| Heavily degraded queries | Wrong answer | Quality verdict, low-confidence flag, analyst crop tool. |
| Model weights unavailable offline | System fails offline | `carid fetch-models` bundles weights; fallback lightweight embedder always available. |

---

## 8. Open questions for the owner (next interview)

1. Which ~20 makes for the pilot? (Default list in `data/taxonomy/pilot.yaml` — please review.)
2. Do analysts need user accounts/login in v1, or is the internal network enough?
3. Which manufacturer/brochure sites do you already have rights to use?
4. Retention period for query images (default 30 days)?
5. Do you have labelled real case images we can use (offline) as a test set?

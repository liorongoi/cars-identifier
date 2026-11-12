# Architecture

## Components

```
src/carid/
├── config.py            Settings (env CARID_*), part list and weights
├── models.py / db.py    SQLAlchemy schema: Make › VehicleModel › Generation, Image, ImagePart,
│                        Job, Identification (cases), Benchmark, AuditEvent, SeenUrl
├── taxonomy.py          Seed loading, alias lookup, generation resolution, Wikidata expansion
├── vision/
│   ├── imageio.py       Safe decode (format allow-list, pixel cap, EXIF), hashes, crops
│   ├── quality.py       Sharpness, brightness, contrast, noise → score + verdict
│   ├── enhance.py       Denoise, CLAHE/gamma low-light, unsharp — only when needed
│   ├── faces.py         YuNet (if cached) or Haar face detection; fill/blur masking
│   ├── parts.py         Part detector interface; layout-prior + steering-wheel circle fallback
│   ├── embed.py         Embedder interface; hand-crafted fallback descriptor
│   ├── deep.py          OpenCLIP embedder + zero-shot interior classifier; OWLv2 part detector
│   └── degrade.py       Synthetic degradations for the benchmark
├── index.py             Per-part exact cosine index (NumPy; FAISS when installed), atomic saves
├── matcher.py           Vote fusion across parts → calibrated probabilities, year ranges, evidence
├── engine.py            Orchestrates analyse / identify / ingest / re-index / relabel / retention
├── collector/
│   ├── base.py          Target, Candidate, PoliteClient (throttle, retry, robots.txt), connector ABC
│   ├── labeler.py       Text → (year range, generation, confidence, interior hint)
│   ├── sources.py       Wikimedia, brochure pages, press-room search, folder/PDF, Openverse, manifests, Flickr / Brave web search / video frames (opt-in)
│   └── runner.py        Targets × sources → label → fetch → ingest; resumable via SeenUrl
├── jobs.py              Persistent job queue + worker thread; cancel; re-queue after restart
├── evaluation.py        Hold-out split, degradations, top-k/MRR per mode, report
├── bundle.py            Export/import tar.gz to move data to the offline server
├── services.py          Wires engine + job handlers
├── api/                 FastAPI app (REST + static UI)
└── cli.py               `carid` command
frontend/                React + TypeScript analyst console (Vite build served by the API)
```

## Key decisions

| Decision | Reason |
|---|---|
| Nearest-neighbour matching instead of a closed classifier | New makes need images, not retraining (FR-COL-9); evidence is explainable (FR-ID-9). |
| Per-part indexes + fusion | Robust to occlusion and partial views; mirrors how analysts reason. |
| Generation as the label unit | Interiors change per generation/facelift, not per year (SRS A1). |
| Same enhancement on references and queries | Keeps both sides in the same visual domain. |
| Deep models optional, fallback always present | Offline robustness (C1, NFR-9); tests run without GPUs or downloads. |
| SQLite + files by default | Single-server deployment; schema is PostgreSQL-compatible. |
| Background jobs in-process | No extra infrastructure; jobs are persisted and resumable. |

## Data layout

```
var/
├── carid.db
├── images/<make>/<model>/<year|unknown-year>/<sha>.jpg   reference images (faces blurred)
├── queries/<sha[:2]>/<sha>.jpg                           query images (retention-limited)
├── index/<embedder>/<part>.npz                           vectors per part
└── reports/
```

## Security & privacy

- Upload limits, format allow-list, decompression-bomb guard, no execution of downloaded content.
- File serving resolves paths inside the data root only.
- Tar import uses the `data` extraction filter (no links, no absolute paths).
- Faces blurred in stored reference images; query images purged after the retention period.
- All curator and analyst actions are written to the audit log.
- The API has no authentication in v1 and binds to 127.0.0.1 by default — see SRS §8 Q2.

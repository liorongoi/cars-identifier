# CarID-Interior

Identify a vehicle's **make, model and year range** from a photo of its **interior**,
for image analysts. Works offline on a local server; tolerates people in the frame,
blur and poor light; shows its evidence.

- **Requirements:** [docs/SRS.md](docs/SRS.md)
- **Plan & phases:** [docs/PLAN.md](docs/PLAN.md)
- **Design:** [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) · **Algorithm study:** [docs/ALGORITHMS.md](docs/ALGORITHMS.md)
- **Acceptance checklist:** [docs/ACCEPTANCE.md](docs/ACCEPTANCE.md)

## What it does

1. **Collects** interior images per `Make › Model` from Wikimedia Commons,
   Openverse (open-licence catalogue, official API), manufacturer press rooms
   (searched per model: Toyota, Škoda, Nissan, Honda, Subaru, Seat), brochure/press
   pages you list,
   a local import folder (images + brochure PDFs), and CSV manifests of links.
   Social-network (Flickr API), open-web image search (Brave Search API) and
   movie-frame connectors exist but stay **off** until legal approval is recorded
   in `config/sources.yaml`.
2. **Cleans** them: de-duplication, interior check, faces blurred, label confidence.
3. **Identifies** a query image: quality check → enhancement → face masking →
   part detection (steering wheel, cluster, screen, vents, console…) → per-part
   nearest-neighbour search → fused, calibrated ranking → **top 5–10 candidates**
   with probability, year range, reference-image counts and visual evidence.
4. **Benchmarks** matching modes on clean and degraded images to pick the best.

## Quick start (development)

```bash
make install            # Python venv + front-end packages
.venv/bin/carid init    # create DB, load the 20-make pilot taxonomy
make serve              # builds the UI, serves UI + API on http://127.0.0.1:8000
```

For production accuracy install the deep models (CLIP embedder, OWLv2 part detector):

```bash
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install -e ".[ml]"
.venv/bin/carid fetch-models      # caches weights in models_cache/ for offline use
```

Without them the system runs on a lightweight built-in descriptor and says so in the UI.

## Typical workflow

| Where | Step | Command / screen |
|---|---|---|
| Online host | Collect images | `carid collect` (all makes) or `carid collect --make Toyota --model Corolla`, or **Database › Collect** |
| Online host | Add a missing make | **Add make / model** screen, or `carid add-make Chery` |
| Online host | Export | `carid export-bundle bundle.tar.gz` (images) or `carid export-manifest data/manifests/x.csv` (links + labels only) |
| Offline server | Import | `carid import-bundle bundle.tar.gz` |
| Offline server | Measure accuracy | `carid benchmark` or **Jobs › Run benchmark** |
| Analyst | Identify | **Identify** screen (drag-drop, paste, or file picker) |

Local brochures and images: put them under `imports/<Make>/<Model>/<Year>/` and run a
collection job; the `folder` source picks them up (PDFs need the `pdf` extra).

## API keys for collection sources

Keys are read from environment variables on the server (never commit them):

| Source | Variables | Where to get one |
|---|---|---|
| Openverse (optional, higher limits) | `OPENVERSE_CLIENT_ID`, `OPENVERSE_CLIENT_SECRET` | `POST https://api.openverse.org/v1/auth_tokens/register/` |
| Brave web image search (opt-in) | `BRAVE_API_KEY` | https://brave.com/search/api/ |
| Flickr (opt-in) | `FLICKR_API_KEY` | https://www.flickr.com/services/apps/create/ |

To turn on an opt-in source, set `enabled: true` and, once someone authorised has
approved it, `legal_approved: true` for that source in `config/sources.yaml`.

## Docker

```bash
docker compose up --build    # UI + API on http://127.0.0.1:8000, data in named volumes
```

## Quality gates

`make check` runs ruff, mypy (strict), pytest with coverage, eslint, tsc and vitest.
CI runs the same on every push (`.github/workflows/ci.yml`).

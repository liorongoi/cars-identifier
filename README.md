# CarID-Interior

Identify a vehicle's **make, model and year range** from a photo of its **interior**,
for image analysts. Works offline on a local server; tolerates people in the frame,
blur and poor light; shows its evidence.

- **Requirements:** [docs/SRS.md](docs/SRS.md)
- **Plan & phases:** [docs/PLAN.md](docs/PLAN.md)
- **Design:** [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) · **Algorithm study:** [docs/ALGORITHMS.md](docs/ALGORITHMS.md)
- **Acceptance checklist:** [docs/ACCEPTANCE.md](docs/ACCEPTANCE.md)

## What it does

1. **Collects** interior images per `Make › Model › Year` from Wikimedia Commons,
   brochure/press pages you list, and a local import folder (images + brochure PDFs).
   Social-network (Flickr API) and movie-frame connectors exist but stay **off**
   until legal approval is recorded in `config/sources.yaml`.
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
| Online host | Export | `carid export-bundle bundle.tar.gz` |
| Offline server | Import | `carid import-bundle bundle.tar.gz` |
| Offline server | Measure accuracy | `carid benchmark` or **Jobs › Run benchmark** |
| Analyst | Identify | **Identify** screen (drag-drop, paste, or file picker) |

Local brochures and images: put them under `imports/<Make>/<Model>/<Year>/` and run a
collection job; the `folder` source picks them up (PDFs need the `pdf` extra).

## Docker

```bash
docker compose up --build    # UI + API on http://127.0.0.1:8000, data in named volumes
```

## Quality gates

`make check` runs ruff, mypy (strict), pytest with coverage, eslint, tsc and vitest.
CI runs the same on every push (`.github/workflows/ci.yml`).

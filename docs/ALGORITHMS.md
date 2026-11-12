# Identification algorithm — study and selection

## Problem shape
Fine-grained visual recognition with **thousands of classes**, **few and
noisy examples per class**, an **open-ended class list** (makes are added
later) and **degraded queries**. That rules out a plain closed-set classifier
as the only approach: it would need retraining for every new make and needs
many images per class.

## Candidates

| # | Approach | Pros | Cons |
|---|---|---|---|
| A | Closed-set CNN/ViT classifier (fine-tuned) | Strong when data is plentiful | Retrain per new make; weak with few images; no evidence to show |
| B | **Whole-image embedding + nearest neighbours** (CLIP / DINOv2) | New classes = just add images; works with few examples; robust pretrained features | Distracted by people, clutter, camera angle |
| C | **Part-based embedding + nearest neighbours** | Matches what an analyst looks at (steering wheel, vents…); robust to partial views and occlusion; gives visual evidence | Depends on part detection quality |
| D | Local feature matching (SIFT/SuperPoint + RANSAC) | Precise on near-identical views | Slow at scale, weak across lighting/angles |
| E | **Fusion of B + C** with learned/tuned weights | Combines global layout and local detail | More parameters to tune |

## Chosen design: E (fusion), implemented so B and C can be benchmarked alone

1. **Pre-process**: quality metrics → CLAHE + gamma for low light → non-local-
   means denoise when noisy → faces detected and masked (filled with neutral
   colour so they do not dominate embeddings).
2. **Detect parts**: open-vocabulary detector (OWL-ViT/OWLv2) prompted with
   part names; when unavailable or below confidence, a *layout prior* proposes
   standard crops (cluster above the wheel, screen at center-top, etc.).
3. **Embed**: whole image + each part crop → L2-normalised vector.
   Default backbone: OpenCLIP ViT-B/32; fallback: hand-crafted descriptor
   (HOG + colour + edge-orientation histograms) that needs no download.
4. **Search**: per-part cosine k-NN in that part's index (FAISS when
   installed, NumPy otherwise).
5. **Vote**: each neighbour votes for its model-year with weight
   `w_part · exp(sim / τ)`; per class only the best few neighbours count, so
   classes with many images do not win just by volume.
6. **Fuse & calibrate**: sum votes per model-year, then pool into
   **generation / year range** (adjacent years whose evidence is within a
   margin are merged). Softmax with temperature gives probabilities.
7. **Explain**: keep the best neighbour per part per candidate as evidence.

## Selection method
`carid benchmark` runs B, C and E (and later the fine-tuned variants)
on the same hold-out split, clean and degraded, and reports top-1/5/10 and
MRR. The configured `matcher.mode` is set to the winner. Results are stored
and shown in the Benchmarks screen.

## Next step after data collection
Fine-tune the backbone with metric learning on the collected images (Phase 3).
This keeps the nearest-neighbour design (new makes still need no retraining)
while making embeddings specific to car interiors.

## First real benchmark — 2026-11-11

Data: 811 interior photos from Openverse (20 pilot makes, 116 models, 182
model/generation classes, ~7 images per model). Embedder CLIP ViT-B/32, part
detector OWLv2, CPU. 120 test photos; **split grouped by photographer** so a
series of shots of the same car is never on both sides (a random split gave
inflated top-5 of ~80% because the system recognised sibling shots).

Model top-5 / make top-1 (honest grouped split):

| Mode | Clean | Low light | Blur | Occlusion |
|---|---|---|---|---|
| fusion | 23% / 50% | 16% / 37% | 7% / 12% | 22% / 42% |
| fusion + LDA adapter | 29% / 50% | 23% / 45% | 13% / 27% | 26% / 46% |
| global (whole image) | 34% / 43% | 23% / 30% | 14% / 21% | 27% / 33% |
| **global + LDA adapter** | **34% / 46%** | **29% / 40%** | **21% / 28%** | **28% / 35%** |
| parts | 19% / 42% | 15% / 34% | 7% / 8% | 17% / 35% |
| parts + LDA adapter | 25% / 51% | 20% / 42% | 7% / 19% | 23% / 42% |

Findings:
1. The LDA adapter improves every mode (+4 to +10 points mean model top-5).
2. Whole-image matching beats part-based matching on real photos; OWLv2 parts
   on uncurated photos are noisy. Parts help make-level accuracy (parts+LDA has
   the best clean make top-1, 51%).
3. With ~7 images per model, identifying the model of an unseen car is far
   below the SRS target (NFR-2: 85% top-5). The main lever is **more and
   cleaner data per generation**; second, a stronger backbone (ViT-L/14 or
   DINOv2) and metric-learning fine-tuning once data allows.

The server now runs in `matcher_mode: auto`, which uses the latest benchmark's
winner (currently `global` with the adapter).

## Make + model only, and small items — 2026-11-11

Classes are now models (owner decision D6); 839 photos, 116 models, same
photographer-grouped split (120 test photos). "Close-up" = a random 25-45% crop
of the photo, enlarged (a photo of a small item).

Model top-5, best mode (global + LDA adapter):

| Condition | Detail tiles off | Detail tiles on | Tiles on, no adapter on tiles |
|---|---|---|---|
| Full photo | 23% | **28%** | 24% |
| Low light | 21% | **25%** | — |
| Person in frame | 26% | **29%** | 21% |
| Close-up (small item) | **18%** | 14% | 12% |

Findings:
1. Detail tiles help full photos consistently (+3 to +8 points across conditions
   and modes); kept on.
2. Close-ups of small items stay at ~12-18% model top-5 whatever the setting
   (differences of 4 points = ~5 photos, within noise). The bottleneck is the
   embedder: CLIP ViT-B/32 sees 224 px images in 32 px patches, too coarse for
   fine detail on knobs, vents and switches.
3. Next experiment for small items: a finer-grained backbone (DINOv2 ViT-B/14 or
   CLIP ViT-L/14, 14 px patches), plus more photos per model.

## Backbone comparison: DINOv2-small vs CLIP ViT-B/32 — 2026-11-11

Same 839 photos, same photographer-grouped split (120 test photos), detail tiles
on, adapter retrained for each backbone. Model top-5 (model top-10):

| Condition | CLIP global+LDA | CLIP fusion+LDA | DINOv2-small best mode |
|---|---|---|---|
| Full photo | **28% (38%)** | 31% (36%) | 15% (18%) |
| Close-up (small item) | **14% (25%)** | 11% (17%) | 12% (22%) |
| Low light | **25% (33%)** | 22% (32%) | 13% (22%) |
| Occlusion | **29% (39%)** | 22% (35%) | 17% (22%) |

Make top-5: CLIP 50-69%, DINOv2-small 41-53%.

Findings:
1. DINOv2-small is worse than CLIP everywhere, including close-ups, so CLIP
   ViT-B/32 stays the default (`CARID_EMBEDDER=dinov2` remains available).
   DINOv2's self-supervised features capture texture and lighting more than
   the semantic cues (shapes of controls, layout) that separate models.
2. Close-ups score about the same with both backbones (12-14% top-5), so the
   backbone is not the main limit for small items. The limit is the data: with
   about 7 photos per model, the specific knob or vent in a close-up is often
   not visible in any reference photo of that model.
3. Next steps, in order of expected gain: more photos per model (the 574
   Wikimedia images that the sandbox could not download, plus brochures and
   the user's own folders); then CLIP ViT-L/14 on a GPU server; then
   metric-learning fine-tuning once there are about 30+ photos per model.

## Free image sources reviewed — 2026-11-12

| Source | Decision | Reason |
|---|---|---|
| Wikimedia Commons, Openverse | In use | Open licences, official APIs |
| Manufacturer press rooms | **Added** (`press_sites`) | Sharp, model-named interior photos; editorial use. Six of the 20 pilot makes have a search page that works without JavaScript or a bot check (Toyota, Škoda, Nissan US, Honda US, Subaru US, Seat); the others need their pages listed in `brochure_sites` by hand |
| Flickr, Brave web search | Built, gated | Social network / unknown licences: need `legal_approved` |
| Pexels, Pixabay | **Not built** | Their terms forbid scraping and systematic bulk copying, explicitly including for machine-learning purposes |
| Unsplash | Not built | API terms require showing images from Unsplash's own links (hotlinking), which conflicts with keeping a local reference copy |
| CompCars research dataset | Owner action | 1,716 models with part images; non-commercial research only; release agreement signed by full-time staff |

Dry run of `press_sites` on two random pilot models per make: Toyota Corolla 10,
Škoda Fabia 29 / Karoq 10, Honda HR-V 24, Seat Ibiza 12 labelled candidates;
Nissan and Honda press rooms are US sites, so models not sold in the US (Juke,
Micra, Jazz) find nothing there.

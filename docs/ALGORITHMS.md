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

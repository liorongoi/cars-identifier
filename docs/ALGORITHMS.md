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

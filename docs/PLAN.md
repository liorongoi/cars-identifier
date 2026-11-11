# Implementation Plan

Companion to [SRS.md](SRS.md). Each phase ends with a working, tested,
reviewed system — never a half-built one.

## Phase 0 — Foundations ✅ (this delivery)
- SRS, architecture, algorithm study.
- Repository layout, Python package `carid`, React front end, CI, Docker.
- Code-quality gates: ruff, mypy, pytest + coverage, eslint, tsc, vitest.

## Phase 1 — Working core ✅ (this delivery)
- Data model (Make › Model › Year, images, parts, jobs, identifications, audit).
- Seed taxonomy for 20 pilot makes, 1996 – current.
- Image pipeline: quality assessment, low-light enhancement, denoising,
  face masking.
- Part detection (zero-shot detector with a layout-prior fallback).
- Embedders: CLIP-family deep model (preferred) + lightweight hand-crafted
  fallback (always available offline, used in tests).
- Vector index per part type; fused ranking; top-K with probabilities and
  year ranges.
- Collector framework: Wikimedia Commons, brochure/press sites (robots.txt),
  local folder import; opt-in social and movie-frame connectors gated by a
  legal flag. Background job runner.
- REST API and analyst UI: Identify, Results/Evidence, Database, Jobs,
  Add make, History, Benchmarks, System.
- Evaluation harness with degradation suite.

## Phase 2 — Data at scale (owner's collection host, online) — in progress
- ✅ Openverse connector with daily-quota handling; manifest export/import.
- ✅ First pilot collection run from the development sandbox (Wikimedia API is
  rate-limited there); its manifest is in `data/manifests/` so the owner's host can
  re-download the same images.
- ✅ Label quality rules from real data: titles outrank tags, out-of-scope model
  years rejected, `carid recheck-labels`.
- ✅ First real benchmark (see ALGORITHMS.md): honest grouped split; LDA adapter
  helps every mode; whole-image matching wins; accuracy limited by data volume.
- ⏳ Grow the data: Openverse app registration (higher quota), Wikimedia from the
  owner's host, brochure sites/folders, more queries per model.

1. Owner confirms the pilot make list and brochure sites (SRS §8).
2. `carid fetch-models` to cache weights; `carid collect --make <all pilot>`.
3. Curator review pass of low-confidence labels in the Database screen.
4. `carid export-bundle` → copy to the offline server → `carid import-bundle`.
5. `carid benchmark` — first real accuracy numbers; tune part weights,
   temperature and thresholds; pick the winning algorithm.

## Phase 3 — Accuracy improvements (driven by the benchmark)
- Fine-tune the embedder on the collected data with a metric-learning loss
  (ArcFace / supervised contrastive) per part type.
- Train a dedicated part detector on curator-corrected boxes.
- Use analyst feedback (FR-ID-14) as hard examples.

## Phase 4 — Expansion & hardening
- Remaining makes worldwide, regional variants, multilingual queries.
- Login and roles, PDF case reports, PostgreSQL + FAISS-GPU option.
- Load testing to NFR-1/NFR-3.

## Working agreement
- Every change: branch → automated checks → code review → merge.
- The owner is interviewed at the start of each phase (open questions in SRS §8).

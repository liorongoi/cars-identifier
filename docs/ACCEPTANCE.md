# Acceptance checklist

Traceability from SRS requirements to verification. "Auto" = covered by the
automated test suite (`make check`); "Manual" = owner walk-through.

| Requirement | Verified by |
|---|---|
| FR-DB-1/2 taxonomy & image metadata | Auto: `tests/test_taxonomy_labeler.py`, `tests/test_engine.py::test_ingest_stores_by_make_model_year_and_resolves_generation` |
| FR-DB-3 image counts per make/model/year | Auto: `tests/test_api.py::test_catalog_counts`; Manual: Database screen |
| FR-DB-4 pilot seed (20 makes) | Auto: `test_pilot_taxonomy_is_valid`, `test_seed_is_idempotent` |
| FR-DB-5 Wikidata expansion | Auto: `test_wikidata_spec_parsing`; Manual: add a make with "Wikidata" online |
| FR-DB-6 de-duplication | Auto: `test_ingest_dedup_and_rejections` |
| FR-DB-7 bundle export/import | Auto: `test_bundle_roundtrip` |
| FR-DB-8 relabel/approve/reject + audit | Auto: `test_relabel_moves_file`, `test_reject_and_restore_reference`, `test_images_browse_patch_upload` |
| FR-COL-1..3 connectors, legal gating | Auto: `tests/test_collector.py` |
| FR-COL-5 interior filter | Auto: `test_ingest_dedup_and_rejections`, `test_collector_without_classifier_skips_uncertain_web_images` |
| FR-COL-6 label inference | Auto: `test_infer_label*` |
| FR-COL-7 throttling/retry/robots | Auto: `test_polite_client_*` |
| FR-COL-8 background jobs | Auto: `tests/test_jobs_eval_bundle.py::test_job_lifecycle`, `test_running_jobs_requeued_on_start` |
| FR-COL-9 add missing make from UI | Auto: `test_add_make_manual_and_seeded`; Manual: Add make screen |
| FR-ID-1 upload formats & limits | Auto: `test_identify_validation`, `test_decode_*` |
| FR-ID-2/3 quality & enhancement | Auto: `test_quality_flags_dark_and_blurry`, `test_enhance_brightens_dark_images` |
| FR-ID-4 face masking | Auto: `test_mask_regions_fill_and_blur`; Manual: photo with a driver |
| FR-ID-5 part detection + manual boxes | Auto: `test_layout_detector_*`, `test_identify_filters_and_manual_parts` |
| FR-ID-6/7 matching & ranking | Auto: `tests/test_index_matcher.py`, `test_identify_finds_the_right_car[*]` |
| FR-ID-8/9 result table & evidence | Auto: `frontend/src/__tests__/results.test.tsx` |
| FR-ID-10 low-confidence flag | Auto: `test_identify_with_empty_database_warns` |
| FR-ID-12/13 export & history | Auto: `test_identify_flow` |
| FR-EV-1..4 benchmark | Auto: `test_split_holdout_keeps_training_images`, `test_benchmark_report` |
| NFR-2 accuracy targets | Manual: `carid benchmark` on collected data, reviewed with owner |
| NFR-5 path traversal, upload validation | Auto: `test_path_traversal_is_blocked`, `test_identify_validation` |
| NFR-6 retention | Auto: `test_query_retention_purge`, `test_no_query_storage_when_retention_zero` |

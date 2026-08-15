# Final Worktree Classification and Inventory

**Inventory Date:** 2026-08-14
**Total Untracked Entries:** 181

This document classifies all untracked files in the workspace into the required 7 categories, defining whether each path should be version-controlled, ignored, or kept as local validation data.

## Category A: SOURCE / REQUIRED (Active Application Core) (43 items)

| Path | Disposition | Description |
|---|---|---|
| `backend/app/api/advanced_search_api.py` | **Should be version-controlled** | Authoritative source and runtime code |
| `backend/app/competition_data.py` | **Should be version-controlled** | Authoritative source and runtime code |
| `backend/app/competition_evaluation.py` | **Should be version-controlled** | Authoritative source and runtime code |
| `backend/app/config/` | **Should be version-controlled** | Authoritative source and runtime code |
| `backend/app/dataset_ops.py` | **Should be version-controlled** | Authoritative source and runtime code |
| `backend/app/distributed/` | **Should be version-controlled** | Authoritative source and runtime code |
| `backend/app/indexes/advanced_faiss_index.py` | **Should be version-controlled** | Authoritative source and runtime code |
| `backend/app/indexes/index_factory.py` | **Should be version-controlled** | Authoritative source and runtime code |
| `backend/app/main.py` | **Should be version-controlled** | Authoritative source and runtime code |
| `backend/app/model_cache.py` | **Should be version-controlled** | Authoritative source and runtime code |
| `backend/app/retrieval/async_retriever.py` | **Should be version-controlled** | Authoritative source and runtime code |
| `backend/app/retrieval/batch_retriever.py` | **Should be version-controlled** | Authoritative source and runtime code |
| `backend/app/retrieval/cache.py` | **Should be version-controlled** | Authoritative source and runtime code |
| `backend/app/retrieval/candidate_resolver.py` | **Should be version-controlled** | Authoritative source and runtime code |
| `backend/app/retrieval/competition_scoring.py` | **Should be version-controlled** | Authoritative source and runtime code |
| `backend/app/retrieval/cross_encoder_reranker.py` | **Should be version-controlled** | Authoritative source and runtime code |
| `backend/app/retrieval/distributed_retriever.py` | **Should be version-controlled** | Authoritative source and runtime code |
| `backend/app/retrieval/hybrid_ranker.py` | **Should be version-controlled** | Authoritative source and runtime code |
| `backend/app/retrieval/hybrid_retriever.py` | **Should be version-controlled** | Authoritative source and runtime code |
| `backend/app/retrieval/kis_pipeline.py` | **Should be version-controlled** | Authoritative source and runtime code |
| `backend/app/retrieval/m12_pipeline.py` | **Should be version-controlled** | Authoritative source and runtime code |
| `backend/app/retrieval/m13_pipeline.py` | **Should be version-controlled** | Authoritative source and runtime code |
| `backend/app/retrieval/m14_pipeline.py` | **Should be version-controlled** | Authoritative source and runtime code |
| `backend/app/retrieval/model_scorer.py` | **Should be version-controlled** | Authoritative source and runtime code |
| `backend/app/retrieval/ninerouter_vision_scorer.py` | **Should be version-controlled** | Authoritative source and runtime code |
| `backend/app/retrieval/pipeline.py` | **Should be version-controlled** | Authoritative source and runtime code |
| `backend/app/retrieval/qa_query_decomposition.py` | **Should be version-controlled** | Authoritative source and runtime code |
| `backend/app/retrieval/query_planner.py` | **Should be version-controlled** | Authoritative source and runtime code |
| `backend/app/retrieval/ranking_metrics.py` | **Should be version-controlled** | Authoritative source and runtime code |
| `backend/app/retrieval/reranker.py` | **Should be version-controlled** | Authoritative source and runtime code |
| `backend/app/retrieval/semantic_reranker.py` | **Should be version-controlled** | Authoritative source and runtime code |
| `backend/app/retrieval/sharded_retriever.py` | **Should be version-controlled** | Authoritative source and runtime code |
| `backend/app/retrieval/siglip_reranker.py` | **Should be version-controlled** | Authoritative source and runtime code |
| `backend/app/retrieval/trake.py` | **Should be version-controlled** | Authoritative source and runtime code |
| `backend/app/retrieval/video_multimodal.py` | **Should be version-controlled** | Authoritative source and runtime code |
| `backend/app/retrieval/video_qa.py` | **Should be version-controlled** | Authoritative source and runtime code |
| `backend/app/runtime/` | **Should be version-controlled** | Authoritative source and runtime code |
| `backend/app/services/advanced_search_service.py` | **Should be version-controlled** | Authoritative source and runtime code |
| `backend/app/services/configured_search.py` | **Should be version-controlled** | Authoritative source and runtime code |
| `backend/app/utils/` | **Should be version-controlled** | Authoritative source and runtime code |
| `backend/app/video/` | **Should be version-controlled** | Authoritative source and runtime code |
| `frontend/src/index.html` | **Should be version-controlled** | Authoritative source and runtime code |
| `projectctl.py` | **Should be version-controlled** | Authoritative source and runtime code |

## Category B: TESTS (Unit, Integration, Regressions) (56 items)

| Path | Disposition | Description |
|---|---|---|
| `tests/integration/test_advanced_retrieval.py` | **Should be version-controlled** | Active automated test suite |
| `tests/integration/test_configured_search.py` | **Should be version-controlled** | Active automated test suite |
| `tests/integration/test_distributed_retrieval.py` | **Should be version-controlled** | Active automated test suite |
| `tests/integration/test_m10_routing.py` | **Should be version-controlled** | Active automated test suite |
| `tests/integration/test_m11_ranking.py` | **Should be version-controlled** | Active automated test suite |
| `tests/integration/test_m12_quality.py` | **Should be version-controlled** | Active automated test suite |
| `tests/integration/test_m13_5_evaluation.py` | **Should be version-controlled** | Active automated test suite |
| `tests/integration/test_m13_pipeline.py` | **Should be version-controlled** | Active automated test suite |
| `tests/integration/test_m13_siglip_real.py` | **Should be version-controlled** | Active automated test suite |
| `tests/integration/test_m14_9router_real.py` | **Should be version-controlled** | Active automated test suite |
| `tests/integration/test_m14_evaluation.py` | **Should be version-controlled** | Active automated test suite |
| `tests/integration/test_m14_pipeline.py` | **Should be version-controlled** | Active automated test suite |
| `tests/integration/test_m15_config_invalidation.py` | **Should be version-controlled** | Active automated test suite |
| `tests/integration/test_m15_corrupt_video_isolation.py` | **Should be version-controlled** | Active automated test suite |
| `tests/integration/test_m15_index_publication.py` | **Should be version-controlled** | Active automated test suite |
| `tests/integration/test_m15_interrupted_resume.py` | **Should be version-controlled** | Active automated test suite |
| `tests/integration/test_m15_multivideo_search.py` | **Should be version-controlled** | Active automated test suite |
| `tests/integration/test_m15_siglip2_real.py` | **Should be version-controlled** | Active automated test suite |
| `tests/integration/test_m15_video_ingestion.py` | **Should be version-controlled** | Active automated test suite |
| `tests/integration/test_m16_text_evidence.py` | **Should be version-controlled** | Active automated test suite |
| `tests/integration/test_m17_multimodal_retrieval.py` | **Should be version-controlled** | Active automated test suite |
| `tests/integration/test_m18_kis_pipeline.py` | **Should be version-controlled** | Active automated test suite |
| `tests/integration/test_m19_video_qa.py` | **Should be version-controlled** | Active automated test suite |
| `tests/integration/test_m20_trake.py` | **Should be version-controlled** | Active automated test suite |
| `tests/integration/test_m21_competition_scoring.py` | **Should be version-controlled** | Active automated test suite |
| `tests/integration/test_m23_operator_api.py` | **Should be version-controlled** | Active automated test suite |
| `tests/integration/test_m8_pipeline.py` | **Should be version-controlled** | Active automated test suite |
| `tests/integration/test_performance_layer.py` | **Should be version-controlled** | Active automated test suite |
| `tests/m15_support.py` | **Should be version-controlled** | Active automated test suite |
| `tests/unit/test_9router_vision_reranker.py` | **Should be version-controlled** | Active automated test suite |
| `tests/unit/test_candidate_resolver.py` | **Should be version-controlled** | Active automated test suite |
| `tests/unit/test_competition_data.py` | **Should be version-controlled** | Active automated test suite |
| `tests/unit/test_competition_evaluation.py` | **Should be version-controlled** | Active automated test suite |
| `tests/unit/test_device_policy.py` | **Should be version-controlled** | Active automated test suite |
| `tests/unit/test_evaluator_semantics.py` | **Should be version-controlled** | Active automated test suite |
| `tests/unit/test_frame_id_policy.py` | **Should be version-controlled** | Active automated test suite |
| `tests/unit/test_frame_record.py` | **Should be version-controlled** | Active automated test suite |
| `tests/unit/test_frame_sampler.py` | **Should be version-controlled** | Active automated test suite |
| `tests/unit/test_hybrid_ranker.py` | **Should be version-controlled** | Active automated test suite |
| `tests/unit/test_image_search.py` | **Should be version-controlled** | Active automated test suite |
| `tests/unit/test_latency.py` | **Should be version-controlled** | Active automated test suite |
| `tests/unit/test_m13_5_corpus.py` | **Should be version-controlled** | Active automated test suite |
| `tests/unit/test_m22_end_to_end_benchmark.py` | **Should be version-controlled** | Active automated test suite |
| `tests/unit/test_model_cache.py` | **Should be version-controlled** | Active automated test suite |
| `tests/unit/test_model_scorer.py` | **Should be version-controlled** | Active automated test suite |
| `tests/unit/test_multimodal_retrieval_v2.py` | **Should be version-controlled** | Active automated test suite |
| `tests/unit/test_operations.py` | **Should be version-controlled** | Active automated test suite |
| `tests/unit/test_projectctl.py` | **Should be version-controlled** | Active automated test suite |
| `tests/unit/test_projectctl_qa.py` | **Should be version-controlled** | Active automated test suite |
| `tests/unit/test_qa_generalized_validation.py` | **Should be version-controlled** | Active automated test suite |
| `tests/unit/test_ranking_metrics.py` | **Should be version-controlled** | Active automated test suite |
| `tests/unit/test_semantic_reranker.py` | **Should be version-controlled** | Active automated test suite |
| `tests/unit/test_temporal_nms.py` | **Should be version-controlled** | Active automated test suite |
| `tests/unit/test_text_backends.py` | **Should be version-controlled** | Active automated test suite |
| `tests/unit/test_video_decoder.py` | **Should be version-controlled** | Active automated test suite |
| `tests/unit/test_video_qa_m26_1.py` | **Should be version-controlled** | Active automated test suite |

## Category C: DOCS (Architecture, Checkpoints, Manuals) (16 items)

| Path | Disposition | Description |
|---|---|---|
| `docs/autonomous_m27_m31_checkpoint.md` | **Should be version-controlled** | System documentation and freeze checkpoints |
| `docs/competition_data.md` | **Should be version-controlled** | System documentation and freeze checkpoints |
| `docs/docker_deployment.md` | **Should be version-controlled** | System documentation and freeze checkpoints |
| `docs/final_engineering_stabilization_checkpoint.md` | **Should be version-controlled** | System documentation and freeze checkpoints |
| `docs/final_system_audit_and_architecture.md` | **Should be version-controlled** | System documentation and freeze checkpoints |
| `docs/final_worktree_inventory.md` | **Should be version-controlled** | Worktree inventory document |
| `docs/m15_video_ingestion.md` | **Should be version-controlled** | System documentation and freeze checkpoints |
| `docs/m16_text_evidence.md` | **Should be version-controlled** | System documentation and freeze checkpoints |
| `docs/m27_representative_evaluation.md` | **Should be version-controlled** | System documentation and freeze checkpoints |
| `docs/m27_three_video_evaluation.md` | **Should be version-controlled** | System documentation and freeze checkpoints |
| `docs/m28_retrieval_quality_v2.md` | **Should be version-controlled** | System documentation and freeze checkpoints |
| `docs/m29_temporal_refinement.md` | **Should be version-controlled** | System documentation and freeze checkpoints |
| `docs/m30_qa_quality.md` | **Should be version-controlled** | System documentation and freeze checkpoints |
| `docs/m31_final_competition_readiness.md` | **Should be version-controlled** | System documentation and freeze checkpoints |
| `docs/projectctl.md` | **Should be version-controlled** | System documentation and freeze checkpoints |
| `docs/real_sample_validation.md` | **Should be version-controlled** | System documentation and freeze checkpoints |

## Category D: CONFIG / DEPLOYMENT (Docker, Environments) (5 items)

| Path | Disposition | Description |
|---|---|---|
| `.dockerignore` | **Should be version-controlled** | Deployment configuration |
| `.env.example` | **Should be version-controlled** | Deployment configuration |
| `Dockerfile` | **Should be version-controlled** | Deployment configuration |
| `docker-compose.cuda.yml` | **Should be version-controlled** | Deployment configuration |
| `docker-compose.yml` | **Should be version-controlled** | Deployment configuration |

## Category E: GENERATED ARTIFACTS (Local Processed Roots, Indexes) (1 items)

| Path | Disposition | Description |
|---|---|---|
| `data/processed-validation/` | **Should remain local / Ignored** | Processed video and FAISS index artifacts |

## Category F: TEMP / CACHE / LOGS (Scratch Logs, Traces, Contact Sheets) (7 items)

| Path | Disposition | Description |
|---|---|---|
| `artifacts/` | **Should remain local / Ignored** | Transient logs, state artifacts, and contact sheets |
| `build_m27_gt.py` | **Local / Unclassified** | Miscellaneous untracked file |
| `contact_sheets/` | **Should remain local / Ignored** | Transient logs, state artifacts, and contact sheets |
| `make_contact_sheets.py` | **Should remain local / Ignored** | Transient logs, state artifacts, and contact sheets |
| `make_sparse_contacts.py` | **Should remain local / Ignored** | Transient logs, state artifacts, and contact sheets |
| `qa_output.txt` | **Should remain local / Ignored** | Transient logs, state artifacts, and contact sheets |
| `trace_output.txt` | **Should remain local / Ignored** | Transient logs, state artifacts, and contact sheets |

## Category G: HISTORICAL VALIDATION DATA (Baselines, Ground Truths, Benchmarks) (53 items)

| Path | Disposition | Description |
|---|---|---|
| `ablation.py` | **Should remain local / Historical archive** | Historical evaluation scripts, benchmarking baselines, and debug probes |
| `audit_gt.py` | **Should remain local / Historical archive** | Historical evaluation scripts, benchmarking baselines, and debug probes |
| `build_gt_v1.py` | **Should remain local / Historical archive** | Historical evaluation scripts, benchmarking baselines, and debug probes |
| `build_gt_v2.py` | **Should remain local / Historical archive** | Historical evaluation scripts, benchmarking baselines, and debug probes |
| `create_m27_three_video_gt.py` | **Should remain local / Historical archive** | Historical evaluation scripts, benchmarking baselines, and debug probes |
| `data/test-videos-m27/` | **Should remain local / Versioned separately** | Validation videos and test corpus data |
| `data/test-videos/` | **Should remain local / Versioned separately** | Validation videos and test corpus data |
| `data/validation/` | **Should remain local / Versioned separately** | Validation videos and test corpus data |
| `debug_qa.py` | **Should remain local / Historical archive** | Historical evaluation scripts, benchmarking baselines, and debug probes |
| `eval/baselines/` | **Should remain local / Historical archive** | Historical evaluation scripts, benchmarking baselines, and debug probes |
| `eval/data/` | **Should remain local / Historical archive** | Historical evaluation scripts, benchmarking baselines, and debug probes |
| `eval/eval_m27_three_video.py` | **Should remain local / Historical archive** | Historical evaluation scripts, benchmarking baselines, and debug probes |
| `eval/m10_routing_benchmark.py` | **Should remain local / Historical archive** | Historical evaluation scripts, benchmarking baselines, and debug probes |
| `eval/m11_ranking_benchmark.py` | **Should remain local / Historical archive** | Historical evaluation scripts, benchmarking baselines, and debug probes |
| `eval/m12_hard_negative_dataset.py` | **Should remain local / Historical archive** | Historical evaluation scripts, benchmarking baselines, and debug probes |
| `eval/m12_quality_benchmark.py` | **Should remain local / Historical archive** | Historical evaluation scripts, benchmarking baselines, and debug probes |
| `eval/m12_weight_tuning.py` | **Should remain local / Historical archive** | Historical evaluation scripts, benchmarking baselines, and debug probes |
| `eval/m13_5_corpus.py` | **Should remain local / Historical archive** | Historical evaluation scripts, benchmarking baselines, and debug probes |
| `eval/m13_5_dataset_report.py` | **Should remain local / Historical archive** | Historical evaluation scripts, benchmarking baselines, and debug probes |
| `eval/m13_5_ground_truth_benchmark.py` | **Should remain local / Historical archive** | Historical evaluation scripts, benchmarking baselines, and debug probes |
| `eval/m13_dataset.py` | **Should remain local / Historical archive** | Historical evaluation scripts, benchmarking baselines, and debug probes |
| `eval/m13_model_benchmark.py` | **Should remain local / Historical archive** | Historical evaluation scripts, benchmarking baselines, and debug probes |
| `eval/m13_real_model_quality.py` | **Should remain local / Historical archive** | Historical evaluation scripts, benchmarking baselines, and debug probes |
| `eval/m14_9router_benchmark.py` | **Should remain local / Historical archive** | Historical evaluation scripts, benchmarking baselines, and debug probes |
| `eval/m14_9router_quality.py` | **Should remain local / Historical archive** | Historical evaluation scripts, benchmarking baselines, and debug probes |
| `eval/m15_video_ingestion_benchmark.py` | **Should remain local / Historical archive** | Historical evaluation scripts, benchmarking baselines, and debug probes |
| `eval/m16_text_benchmark.py` | **Should remain local / Historical archive** | Historical evaluation scripts, benchmarking baselines, and debug probes |
| `eval/m22_end_to_end_benchmark.py` | **Should remain local / Historical archive** | Historical evaluation scripts, benchmarking baselines, and debug probes |
| `eval/m22_pipeline_benchmark.py` | **Should remain local / Historical archive** | Historical evaluation scripts, benchmarking baselines, and debug probes |
| `eval/m7_faiss_siglip_benchmark.py` | **Should remain local / Historical archive** | Historical evaluation scripts, benchmarking baselines, and debug probes |
| `eval/m8_scalability_benchmark.py` | **Should remain local / Historical archive** | Historical evaluation scripts, benchmarking baselines, and debug probes |
| `eval/m9_distributed_benchmark.py` | **Should remain local / Historical archive** | Historical evaluation scripts, benchmarking baselines, and debug probes |
| `eval/results/` | **Should remain local / Historical archive** | Historical evaluation scripts, benchmarking baselines, and debug probes |
| `eval_m27.py` | **Should remain local / Historical archive** | Historical evaluation scripts, benchmarking baselines, and debug probes |
| `eval_mini_gt.py` | **Should remain local / Historical archive** | Historical evaluation scripts, benchmarking baselines, and debug probes |
| `fusion_experiment.py` | **Should remain local / Historical archive** | Historical evaluation scripts, benchmarking baselines, and debug probes |
| `inspect_asr_ocr.py` | **Should remain local / Historical archive** | Historical evaluation scripts, benchmarking baselines, and debug probes |
| `m26_1_trace.py` | **Should remain local / Historical archive** | Historical evaluation scripts, benchmarking baselines, and debug probes |
| `print_asr.py` | **Should remain local / Historical archive** | Historical evaluation scripts, benchmarking baselines, and debug probes |
| `print_asr_v2_v3.py` | **Should remain local / Historical archive** | Historical evaluation scripts, benchmarking baselines, and debug probes |
| `prove_frames.py` | **Should remain local / Historical archive** | Historical evaluation scripts, benchmarking baselines, and debug probes |
| `run_queries.py` | **Should remain local / Historical archive** | Historical evaluation scripts, benchmarking baselines, and debug probes |
| `sample_asr.py` | **Should remain local / Historical archive** | Historical evaluation scripts, benchmarking baselines, and debug probes |
| `sample_asr_windows.py` | **Should remain local / Historical archive** | Historical evaluation scripts, benchmarking baselines, and debug probes |
| `sample_text.py` | **Should remain local / Historical archive** | Historical evaluation scripts, benchmarking baselines, and debug probes |
| `subset_matrix.py` | **Should remain local / Historical archive** | Historical evaluation scripts, benchmarking baselines, and debug probes |
| `test_api.py` | **Should remain local / Historical archive** | Historical evaluation scripts, benchmarking baselines, and debug probes |
| `test_api_v2.py` | **Should remain local / Historical archive** | Historical evaluation scripts, benchmarking baselines, and debug probes |
| `test_fusion_v2.py` | **Should remain local / Historical archive** | Historical evaluation scripts, benchmarking baselines, and debug probes |
| `trace_fusion.py` | **Should remain local / Historical archive** | Historical evaluation scripts, benchmarking baselines, and debug probes |
| `trace_qa.py` | **Should remain local / Historical archive** | Historical evaluation scripts, benchmarking baselines, and debug probes |
| `validate_artifacts.py` | **Should remain local / Historical archive** | Historical evaluation scripts, benchmarking baselines, and debug probes |
| `validate_ocr.py` | **Should remain local / Historical archive** | Historical evaluation scripts, benchmarking baselines, and debug probes |

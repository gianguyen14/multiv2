# Final Engineering Freeze Manifest

**Freeze Date:** 2026-08-14
**Final Engineering State:** FROZEN
**Active Processed Root:** `data/processed-validation/full-3-videos`
**Active Generation ID:** `gen-3d51a16ea32b4953820583c3af181b31`

---

## 1. Authoritative Source Architecture

| Component | Authoritative Implementation | Description |
|---|---|---|
| **CLI Entry Point** | `projectctl.py` | Primary operator interface (preprocess, search, kis, qa, trake, image-search, backend, doctor, models) |
| **Search Orchestrator** | `backend/app/services/configured_search.py` (`ConfiguredSearch`) | Unifies SigLIP2 visual search, lexical OCR/ASR expansion, temporal NMS, and Q&A |
| **Frame Identity & Model** | `backend/app/video/frame_record.py` (`FrameRecord`) | Canonical format `video_id:source_frame_index_zero_based:09d` |
| **Visual Ingestion** | `backend/app/video/m15_ingestion_pipeline.py` | PyAV display-order enumeration decoder, nearest-timestamp sampler, float32 L2-norm embedding |
| **Visual Model Tower** | `backend/app/embeddings/siglip2.py` (`SigLIP2Encoder`) | Local-only deterministic SigLIP2 base patch16-224 with immutable cached commit SHA identity |
| **Text Evidence Pipeline** | `backend/app/video/m16_text_pipeline.py` (`M16TextPipeline`) | Dependency-aware OCR & Faster Whisper ASR with sidecar metadata and cache validation |
| **TRAKE Sequence Aligner** | `backend/app/retrieval/trake.py` (`TRAKEAligner`) | Dynamic programming monotonic alignment within single video with strict predecessor checks |
| **API Server** | `backend/app/main.py` (`create_app`) | FastAPI backend with security headers, multipart streaming image search, numeric frame serving |
| **Web Frontend** | `frontend/src/index.html` | Static Single Page Application with vanilla JS, sanitized DOM rendering, no inline event handlers |

---

## 2. Inactive / Deprecated Historical Stacks

| Module | Status | Notes |
|---|---|---|
| `backend/app/api/search_api.py` | **DEPRECATED** | Legacy M5 unmounted prototype functions |
| `backend/app/api/advanced_search_api.py` | **DEPRECATED** | Legacy M6 unmounted prototype functions |
| `backend/app/services/search_service.py` | **DEPRECATED** | Legacy M5 prototype service |
| `backend/app/services/advanced_search_service.py` | **DEPRECATED** | Legacy M6 prototype service |
| `backend/app/retrieval/retriever.py` | **DEPRECATED** | Legacy `SigLIPFaissRetriever` |

---

## 3. Authoritative Automated Test Suite

| Test Suite | Path | Purpose |
|---|---|---|
| Configured Search | `tests/integration/test_configured_search.py` | End-to-end integration across KIS, Q&A, image search |
| Operator API & Security | `tests/integration/test_m23_operator_api.py` | API endpoints, security headers, multipart parsing, symlink/traversal defenses |
| TRAKE Alignment | `tests/integration/test_m20_trake.py` | Sequence alignment DP, monotonic ordering, candidate depth, edge cases |
| Text Evidence & Cache | `tests/integration/test_m16_text_evidence.py` | OCR/ASR cache fingerprints, invalidation, corrupt JSON handling |
| Model Cache & Identity | `tests/unit/test_model_cache.py` | Immutable snapshot commit resolution, offline checks |
| Image Search Parsing | `tests/unit/test_image_search.py` | Image MIME/format validation, normalization, and search dispatch |
| Ingestion & Frame ID | `tests/integration/test_m15_video_ingestion.py` | PyAV ordinal extraction, sampling, embedding shape |
| Index Publication & Atomicity | `tests/integration/test_m15_index_publication.py` | Atomic staging, CURRENT symlink/pointer swapping, checksum verification |
| Interrupted Resume | `tests/integration/test_m15_interrupted_resume.py` | Checkpoint resumption, manifest healing |

---

## 4. Deployment & Infrastructure Files

| File | Purpose |
|---|---|
| `Dockerfile` | Reproducible Python 3.12 slim container with FFmpeg and Tesseract OCR |
| `docker-compose.yml` | Multi-service Compose (backend, worker) with localhost-bound ports and persistent volumes |
| `docker-compose.cuda.yml` | GPU override for NVIDIA Container Toolkit |
| `.dockerignore` | Build context exclusions |
| `.env.example` | Template deployment environment variables |
| `docs/docker_deployment.md` | Complete container deployment and operator guide |

---

## 5. Active Processed Artifacts & Validation Integrity

- **Processed Root:** `data/processed-validation/full-3-videos`
- **Indexed Videos:** `L22_V001`, `L22_V002`, `L22_V003`
- **Active Generation:** `gen-3d51a16ea32b4953820583c3af181b31`
- **FAISS Vectors:** 3,578
- **Payload Count:** 3,578
- **Mapping Count:** 3,578
- **Physical Frames:** 3,578 (numeric stems `000000000.jpg` to `000034900.jpg`)
- **OCR Records:** 2,925 total across 3 videos
- **ASR Segments:** 1,122 total across 3 videos
- **Staging Directory:** `.staging` is clean and empty

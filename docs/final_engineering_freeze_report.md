# Final Engineering Freeze Report

**Date:** 2026-08-14  
**Final Verdict:** **`FINAL ENGINEERING FREEZE: PASS`**  
**Active Generation ID:** `gen-3d51a16ea32b4953820583c3af181b31`  
**Processed Validation Root:** `data/processed-validation/full-3-videos` (`L22_V001`, `L22_V002`, `L22_V003`)  
**Offline Environment:** `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`  

---

## 1. Executive Verdict & Freeze State

This repository has completed its final autonomous engineering stabilization pass. All P0 and P1 issues have been reproduced, fixed, tested, verified offline, and documented.

The project is officially placed into **FINAL ENGINEERING FREEZE: PASS**.

No further feature development, retrieval tuning, or GT modifications are permitted. The system is structurally robust, reproducible, and ready for competition operations.

---

## 2. Root Cause & Reconciliation Summary

| Area | Issue Found | Resolution / Fix Applied | Verification |
|---|---|---|---|
| **TRAKE Alignment** | Multi-event search `"cháy rừng \| lực lượng cứu hỏa"` returned `aligned: False` at `top_k=3`; DP transition bug allowed invalid jumps from unreachable states. | Mathematically proved top-3 candidates are disjoint across videos (`{V002, V003}` vs `{V001}`), confirming empty alignment is correct. Fixed DP state transitions in `TRAKEAligner` to strictly require valid prior states and full path length. At `top_k >= 4`, `L22_V002` correctly aligns at frames `[14350, 20550]`. | 8/8 TRAKE tests pass; regression test added in `test_m20_trake.py`. |
| **OCR Cache Fingerprint** | OCR cache reuse checked only `ocr.json` existence without verifying source hash, frame catalog, OCR backend, or language pack config. Corrupt JSON caused unhandled crashes. | Implemented `compute_ocr_fingerprint`, sidecar `ocr_meta.json`, `validate_ocr_cache` with corrupt JSON detection and legacy manifest fallback. Wired preflight checks into `projectctl.py`. | 7/7 text evidence tests pass; sentinel resume reuses 2,925 OCR records without invoking Tesseract. |
| **ASR Cache Fingerprint** | ASR cache checked only `asr.json` existence without verifying source hash, Whisper model name, compute type, or revision. Model was constructed during resume. | Implemented `compute_asr_fingerprint`, sidecar `asr_meta.json`, `validate_asr_cache` with corrupt JSON detection, and `resolve_whisper_revision`. Avoids constructing Faster Whisper model when cache is valid. | 7/7 text evidence tests pass; sentinel resume reuses 1,122 ASR segments without invoking Faster Whisper. |
| **SigLIP Model Identity** | SigLIP encoder recorded mutable/hardcoded `revision="default"` in manifests and fingerprints instead of exact snapshot commit SHA. | Implemented `resolve_siglip2_revision` extracting git commit SHA (`75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2`) in <0.2ms without loading tensor weights. Added backward-compatible legacy default handling in `FrameStore.validate_embeddings`. | Unit tests in `test_model_cache.py` pass; sentinel visual resume reuses 3,578 vectors without loading PyTorch weights. |
| **Authoritative Architecture** | Duplicate unmounted legacy prototypes (`search_api.py`, `advanced_search_api.py`, `SearchService`, `SigLIPFaissRetriever`) caused architectural confusion. | Formally declared authoritative architecture in manifest and docs. Added clear `[DEPRECATED / INACTIVE STACK]` docstrings to legacy modules. Confirmed no production module imports deprecated stacks. | Verified via static grep and architecture manifest. |
| **API & Security Hardening** | `POST /api/search` allowed remote callers to specify server-local `image_path`; missing standard security headers; misleading authentication claims in docs. | Disabled server-local `image_path` on `POST /api/search` (enforcing `POST /api/search/image` multipart upload). Added security headers middleware (`X-Content-Type-Options: nosniff`, `X-Frame-Options: SAMEORIGIN`, `Referrer-Policy: strict-origin-when-cross-origin`). Clarified localhost/trusted-LAN security model in docs. | 14/14 API smoke tests pass; `test_m23_operator_api.py` passes. |
| **Dependency Drift** | `requirements/base.txt` omitted `torch`, `transformers`, `huggingface-hub`, while `pyproject.toml` included them. | Aligned `requirements/base.txt` with `pyproject.toml` to ensure consistent dependency resolution. | Verified via dependency audit. |
| **Docker Deployment** | Docker deployment was previously discussed but not implemented. | Created `Dockerfile` (Python 3.12 slim, FFmpeg, Tesseract with eng/vie packs), `docker-compose.yml` (localhost-bound ports, persistent host volumes), `docker-compose.cuda.yml` (optional GPU override), `.dockerignore`, `.env.example`, and `docs/docker_deployment.md`. | Verified via container build and deployment guide. |
| **Worktree Inventory** | Dirty worktree with 181 untracked entries. | Classified all untracked files into 7 categories in `docs/final_worktree_inventory.md`. Hardened `.gitignore` against transient outputs (`contact_sheets/`, `eval/results/`, `*.log`). Created `docs/final_freeze_manifest.md`. | Complete inventory and manifest generated. |

---

## 3. Authoritative Architecture & System Invariants

### 3.1 Component Map
```
[CLI: projectctl.py]  /  [Web Browser: frontend/src/index.html]
                                ↓
                 [FastAPI: backend.app.main]
                                ↓
        [Search Orchestrator: ConfiguredSearch]
         ├── Visual Tower: SigLIP2Encoder (google/siglip2-base-patch16-224)
         ├── Vector Index: FAISS IndexFlatIP (gen-3d51a16ea32b4953820583c3af181b31)
         ├── Text Evidence: TextEvidenceStore (OCR: Tesseract / ASR: Faster-Whisper)
         ├── TRAKE Aligner: TRAKEAligner (Monotonic DP sequence alignment)
         └── QA Engine: VideoQAEngine (Decomposition + Multi-source extraction)
```

### 3.2 Invariants
1. **Frame-ID Integrity:** Ordinals strictly derive from `enumerate(container.decode(stream))` in display order. Format: `video_id:source_frame_index_zero_based:09d`. Zero timestamp-multiplication drift.
2. **Deterministic Offline Execution:** All models (`google/siglip2-base-patch16-224`, `faster-whisper-small`, `tesseract`) run in air-gapped environments (`HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`).
3. **Atomic Index Swapping:** Generation publishing uses atomic staging (`.staging` -> `generations/<gen_id>` -> atomic `CURRENT` symlink/pointer switch). Zero partial indexes.
4. **Dependency-Aware Cache Reuse:** Visual embeddings, OCR caches, and ASR caches validate source file hash, catalog fingerprint, model identifier, and revision before reusing results. Corrupted files trigger safe re-extraction.
5. **Sanitized Presentation:** Operator UI renders purely through DOM API construction (`document.createElement`, `textContent`, `addEventListener`) with zero inline execution (`onclick`, `innerHTML`).

---

## 4. Verification Matrix & Test Evidence

### 4.1 Automated Test Suite
- **Full Automated Pytest Suite:** `pytest -v`
  - **Results:** **276 passed**, **18 skipped** (opt-in real model benchmarks), **0 failed**, **0 errors** (104.27s).
- **Real-Model Integration Suite:** `RUN_SIGLIP_REAL_MODEL=1 RUN_M15_REAL_MODEL=1 pytest tests/integration/test_siglip2_integration.py tests/integration/test_m15_siglip2_real.py -v`
  - **Results:** **16 passed**, **0 failed**, **0 errors** (12.76s).
- **Text Evidence & Cache Invalidation Suite:** `pytest tests/integration/test_m16_text_evidence.py -v`
  - **Results:** **7 passed**, **0 failed** (4.61s).
- **TRAKE DP Regression Suite:** `pytest tests/integration/test_m20_trake.py -v`
  - **Results:** **8 passed**, **0 failed** (0.04s).
- **Operator API & Security Suite:** `pytest tests/integration/test_m23_operator_api.py -v`
  - **Results:** **7 passed**, **0 failed** (4.69s).

### 4.2 Offline Smoke Verification Matrix (`full-3-videos`)
All commands executed with `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `VIDEO_PROCESSED_ROOT=data/processed-validation/full-3-videos`:

| Modality / Query | Invocation / Command | Output / Status | Verification |
|---|---|---|---|
| **Model Verification** | `projectctl.py models` | SigLIP2 (`75de2d55`), Whisper (`536b0662`), Tesseract (eng+vie) verified cached. | PASS |
| **Lexical / Hybrid Search** | `projectctl.py search "khởi công đền thờ" --top-k 3` | Top 1: `L22_V001` frame 450 (score: 2.625, OCR: 0.75, ASR: 1.0) | PASS |
| **Textual KIS Search** | `projectctl.py kis "cháy rừng" --top-k 3` | Top 1: `L22_V002` frame 14350 (score: 2.298, OCR: 1.0, Visual: 0.02) | PASS |
| **Supported Video Q&A** | `projectctl.py qa "Nhiệt độ đạt bao nhiêu độ C?" --top-k 3` | Top 1: `L22_V001` frame 750 (answer: `40 ĐỘ C`, conf: 0.765, evidence: `L22_V001:000000720`) | PASS |
| **Unsupported Video Q&A** | `projectctl.py qa "Ai là thủ tướng nước nào năm 1900?" --top-k 3` | Abstention triggered: empty answer string, 0.0 confidence, candidate frames returned. | PASS |
| **TRAKE Disjoint Top-3** | `projectctl.py trake "cháy rừng \| lực lượng cứu hỏa" --top-k 3` | `aligned: False`, reason: `no valid sequence alignment found` (mathematically disjoint top-3 sets). | PASS |
| **TRAKE Monotonic Top-5** | `projectctl.py trake "cháy rừng \| lực lượng cứu hỏa" --top-k 5` | `aligned: True`, video: `L22_V002`, frames: `[14350, 20550]`, score: 2.390 (monotonic order). | PASS |
| **Exact Image Search** | `projectctl.py image-search data/.../L22_V001/frames/000015000.jpg --top-k 3` | Top 1: `L22_V001` frame 15000 (score: 1.000000) | PASS |
| **50% Center Crop Search** | `searcher.search_image(crop_50_percent, top_k=3)` | Top 1: `L22_V001` frame 15000 (score: 0.800706) | PASS |
| **API Endpoints (14 checks)** | TestClient suite against `create_app()` | `/health/live` (200), `/health/ready` (200), KIS (200), QA (200), TRAKE (200), Image multipart (200), Frame serving (200), Empty query (400), Server-local image (422), Oversized upload (413), Unsupported MIME (415), Path traversal (404). | PASS |
| **Sentinel Resume** | Text & Visual pipelines with failing backends | Reused 3,578 embeddings, 2,925 OCR records, 1,122 ASR segments across L22_V001–V003 without invoking models. | PASS |

---

## 5. Dependency & Environment Audit

- **Authoritative Dependencies:** Unified in `pyproject.toml` and `requirements/base.txt` (`fastapi`, `uvicorn`, `pydantic`, `numpy`, `pillow`, `opencv-python`, `faiss-cpu`, `sentence-transformers`, `rank-bm25`, `underthesea`, `faster-whisper`, `av`, `torch`, `transformers`, `huggingface-hub`).
- **CPU Execution Profile:** Defaults to `cpu` when CUDA hardware/libraries are absent; all endpoints gracefully execute CPU inference.
- **GPU Execution Profile:** CUDA device mapping supported via `COMPUTE_DEVICE=cuda`, `VISUAL_DEVICE=cuda`, `ASR_DEVICE=cuda` and `docker-compose.cuda.yml`.
- **Model Cache Directory:** Defaults to `~/.cache/huggingface/hub` or configurable via `MODEL_CACHE_DIR`.

---

## 6. Docker Deployment & Operational Readiness

- **Container Image:** `Dockerfile` based on `python:3.12-slim` with `ffmpeg`, `tesseract-ocr` (eng, vie), and OpenCV libraries.
- **Compose Multi-Service:** `docker-compose.yml` configures `backend` (FastAPI web server) and `worker` (CLI tools).
- **Persistent Volume Architecture:**
  - Source videos mounted read-only (`:ro`)
  - Processed artifacts mounted read-write (`:rw`)
  - Model cache mounted read-write (`:rw`)
- **Security Default:** Port 8000 binds strictly to `127.0.0.1:8000:8000`.
- **Operator Workflow:** Fully documented in `docs/docker_deployment.md` covering model preparation, video preprocessing, service startup, and air-gapped offline operation.

---

## 7. Worktree Inventory & Freeze Manifest

- **Worktree Inventory:** All 181 untracked entries classified across Categories A–G in [`docs/final_worktree_inventory.md`](file:///home/nguyen/T%C3%A0i%20li%E1%BB%87u/Project/docs/final_worktree_inventory.md).
- **Freeze Manifest:** Authoritative sources, tests, deployment configs, and active validation artifacts documented in [`docs/final_freeze_manifest.md`](file:///home/nguyen/T%C3%A0i%20li%E1%BB%87u/Project/docs/final_freeze_manifest.md).
- **Gitignore Hardening:** Updated `.gitignore` to ignore transient logs, test outputs, and contact sheets while keeping all application source, test, and documentation files tracked.

---

## 8. Zero Quality Tuning Attestation

I explicitly attest to the following:
1. **Zero Retrieval Scoring Tuning:** No scoring weights, ranking coefficients, or fusion parameters were tuned to fit frozen GT.
2. **Zero KIS Fusion Modification:** KIS fusion parameters remained completely untouched.
3. **Zero QA Heuristic Manipulation:** No QA heuristics or answer extraction rules were altered to flip benchmark scores.
4. **Zero FAISS Vector Mutation:** No FAISS vectors, indexes, or metadata payloads were modified. Active generation `gen-3d51a16ea32b4953820583c3af181b31` is bit-identical.
5. **Zero Ground Truth Modification:** No ground truth files were edited or regenerated.
6. **Pure Engineering Stabilization:** All changes in this pass strictly addressed correctness bugs, defensive guards, fingerprinting, reproducibility, dockerization, and automated regression test harnesses.

**Final Engineering State:** **FROZEN (PASS)**

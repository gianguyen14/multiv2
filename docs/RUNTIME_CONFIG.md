# Runtime Configuration Reference

This document provides a complete, authoritative reference for every environment variable, configuration flag, default value, and valid range defined in `backend/app/core/config.py` and related runtime modules.

---

## 1. Search Backend & Multimodal Embeddings

| Variable | Type | Default Value | Valid Options / Range | Description |
|---|---|---|---|---|
| `SEARCH_BACKEND` | string | `qwen3_vl` | `qwen3_vl` (production default), `siglip2` (legacy) | Canonical search backend selector. `qwen3_vl` queries the packed 1024-d FAISS index with Qwen3-VL-Embedding-2B. `siglip2` is strictly legacy and cannot query Qwen indexes. |
| `SEARCH_ENCODER` | string | `None` | (Same as `SEARCH_BACKEND`) | Accepted as a backward-compatible legacy alias for `SEARCH_BACKEND`. |
| `VIDEO_PROCESSED_ROOT` | string | `data/processed` | Valid directory path | Path to processed search directory containing `index/` (with active generation), `ocr/`, and `asr/` spools. |
| `VISION_PROCESSED_ROOT` | string | `""` | Valid directory path | Path to dense 1 FPS vision index (`aic-db-v2/vision/runtime/index`). Required for experimental image query search. |
| `QWEN3_VL_MODEL_DIR` | string | `""` | Valid directory path | Path to local Qwen3-VL-Embedding-2B weights directory. Defaults to `${MODEL_CACHE_DIR}/Qwen3-VL-Embedding-2B`. |
| `QWEN_MODEL_DIR` | string | `""` | Valid directory path | Secondary legacy fallback alias for `QWEN3_VL_MODEL_DIR`. |
| `MODEL_CACHE_DIR` | string | `models` | Directory path | Base directory for cached model weights. |
| `VECTOR_STORE` | string | `FAISS` | `FAISS`, `QDRANT`, `MILVUS` | Vector index storage provider (FAISS is the only active production provider). |
| `SIGLIP_ENABLED` | boolean | `True` | `true`, `false` | Enables SigLIP2 embedding support in legacy `ConfiguredSearch`. |
| `SIGLIP2_MODEL` | string | `google/siglip2-base-patch16-224` | Hugging Face Model ID | Checkpoint identifier for legacy SigLIP2 encoder (768-D). |
| `SIGLIP_LONG_TEXT_MODE` | string | `chunk_mean` | `chunk_mean`, `truncate` | Encoding strategy for text exceeding SigLIP token limits. |
| `SIGLIP_TEXT_MAX_LENGTH` | integer | `64` | Positive integer | Token sequence truncation length for SigLIP text inputs. |
| `SIGLIP_TEXT_CHUNK_STRIDE`| integer | `8` | Positive integer | Window sliding stride for chunked text embedding in SigLIP. |
| `SIGLIP_TEXT_MAX_CHUNKS` | integer | `8` | Positive integer | Maximum number of chunks pooled per text input. |
| `DINO_ENABLED` | boolean | `False` | `false` | Inactive placeholder flag; DINOv3 is unsupported. |
| `E5_ENABLED` | boolean | `True` | `true`, `false` | Legacy text embedding adapter toggle. |
| `BM25_ENABLED` | boolean | `True` | `true`, `false` | Inverted text index toggle for legacy hybrid scoring. |
| `CAPTION_ENABLED` | boolean | `False` | `false` | Visual captioning toggle (inactive in production). |
| `ENABLE_LLM_EXPANSION` | boolean | `False` | `true`, `false` | Global flag for LLM query reformulation. |
| `ENABLE_LLM_ANSWERING` | boolean | `True` | `true`, `false` | Global flag enabling question answering pipelines. |

---

## 2. QA Synthesis Pipeline (`QA_ANSWER_*`)

The Question Answering subsystem provides grounded answer synthesis over retrieved OCR and ASR evidence.

| Variable | Type | Default Value | Valid Options / Range | Description |
|---|---|---|---|---|
| `QA_ANSWER_BACKEND` | string | `extractive` | `extractive` (production default), `remote_llm`, `agent-lite`, `llm` | Synthesis backend. Default `extractive` slices top 100 characters. `remote_llm` is **EXPERIMENTAL** (upstream 502). |
| `QA_ANSWER_URL` | string | `""` | Valid HTTP(S) URL | Base URL of OpenAI-compatible API (e.g. `http://llm-gateway:8000/v1`). |
| `QA_ANSWER_MODEL` | string | `agent-lite` | Valid Model ID | Model name passed in chat completions payload. |
| `QA_ANSWER_API_KEY` | string | `""` | Secret String | Bearer authentication token for remote LLM calls (never commit). |
| `QA_ANSWER_REMOTE_TOP_N`| integer | `5` | `1` to `10` (Target: **`2`**) | Maximum candidate rows that may invoke remote synthesis. **Must be set to `2` in production** to keep query latency within 16 seconds. |
| `QA_ANSWER_TIMEOUT_SECONDS`| float | `8.0` | Minimum `0.5` | Per-request socket timeout for remote completions. |
| `QA_ANSWER_MAX_EVIDENCE_CHARS`| integer| `12000` | Minimum `500` | Maximum character budget of concatenated OCR/ASR evidence sent to LLM. |
| `QA_CONTEXT_BEFORE_MS` | integer | `5000` | Positive integer (ms) | Pre-event temporal context window for legacy QA. |
| `QA_CONTEXT_AFTER_MS` | integer | `5000` | Positive integer (ms) | Post-event temporal context window for legacy QA. |

---

## 3. Optical Character Recognition (OCR) & Speech Recognition (ASR)

| Variable | Type | Default Value | Valid Options / Range | Description |
|---|---|---|---|---|
| `SEARCH_ENABLE_OCR` | boolean | `true` | `true`, `false` | Toggles loading of `VIDEO_PROCESSED_ROOT/ocr/*.json` and lexical OCR score fusion. |
| `SEARCH_ENABLE_ASR` | boolean | `true` | `true`, `false` | Toggles loading of `VIDEO_PROCESSED_ROOT/asr/*.json` and lexical ASR score fusion. |
| `OCR_BACKEND` | string | `auto` | `auto`, `paddleocr`, `tesseract`, `easyocr` | Ingestion engine selector. `auto` probes CUDA and PaddleOCR, falling back to Tesseract. |
| `OCR_CPU_BACKEND` | string | `tesseract` | `tesseract`, `easyocr` | Ingestion engine when running without GPU acceleration. |
| `OCR_GPU_BACKEND` | string | `paddleocr` | `paddleocr` | Ingestion engine when running on CUDA-equipped hosts. |
| `OCR_FALLBACK_BACKEND`| string | `tesseract` | `tesseract` | Secondary fallback engine when the primary OCR engine fails. |
| `OCR_PADDLE_DEVICE` | string | `auto` | `auto`, `gpu`, `cpu` | Device allocation for PaddleOCR engine. |
| `OCR_PADDLE_MIN_CONFIDENCE`| float | `0.50` | `0.0` to `1.0` | Minimum confidence score to retain a PaddleOCR bounding box. |
| `OCR_FALLBACK_ON_EMPTY` | boolean | `true` | `true`, `false`, `1`, `0` | Triggers fallback to Tesseract if primary OCR returns no text. |
| `OCR_FALLBACK_ON_ERROR` | boolean | `true` | `true`, `false`, `1`, `0` | Triggers fallback to Tesseract if primary OCR throws an exception. |
| `OCR_FALLBACK_ON_LOW_CONFIDENCE`| boolean| `true`| `true`, `false`, `1`, `0` | Triggers fallback if average detection confidence falls below threshold. |
| `FASTER_WHISPER_MODEL` | string | `small` | `tiny`, `base`, `small`, `medium`, `large-v3` | Model checkpoint used for offline audio transcription. |

---

## 4. TRAKE & Temporal Alignment

| Variable | Type | Default Value | Valid Options / Range | Description |
|---|---|---|---|---|
| `TRAKE_COHERENCE_MODE` | string | `diagnostic` | `off`, `diagnostic`, `enforce` | Mode for temporal gap and monotonicity analysis in TRAKE results. |
| `TRAKE_CANDIDATE_VIDEOS`| integer | `10` | Positive integer | Top candidate video count evaluated during TRAKE beam search. |
| `TRAKE_BEAM_WIDTH` | integer | `30` | Positive integer | Beam width for Viterbi sequence alignment in legacy TRAKE. |
| `TRAKE_TEMPORAL_REFINE_ENABLED`| boolean| `true` | `true`, `false` | Enables local temporal window refinement in legacy TRAKE. |
| `TRAKE_TEMPORAL_REFINE_WINDOW_SECONDS`| float| `2.5` | Positive float | Refinement search radius in seconds around candidate keyframes. |
| `TRAKE_TEMPORAL_REFINE_SAMPLE_FPS`| float | `5.0` | Positive float | Frame sampling density during local video decoding. |
| `TRAKE_TEMPORAL_REFINE_MAX_REGIONS_PER_VIDEO`| integer| `3`| Positive integer | Max temporal regions inspected per candidate video. |
| `TRAKE_TEMPORAL_REFINE_MAX_TOTAL_REGIONS`| integer| `6`| Positive integer | Global ceiling on total decoded regions per TRAKE query. |
| `TRAKE_TEMPORAL_REFINE_MAX_FRAMES_PER_REGION`| integer| `50`| Positive integer | Max decoded frames evaluated per temporal region. |
| `TRAKE_TEMPORAL_REFINE_CACHE_ENABLED`| boolean| `true`| `true`, `false` | In-memory cache toggle for decoded refinement embeddings. |

> [!NOTE]
> **TRAKE 400 Bad Request Contract:** In production `QwenRuntimeSearch`, multi-event queries require a single video to satisfy all events in strictly increasing frame order ($f_1 < f_2 < \dots < f_m$). If no single video in the recalled pool covers every event monotonically, `QwenRuntimeSearch.search_trake` raises `ValueError("TRAKE: no single video covers every event in increasing frame order")`, and FastAPI correctly returns **HTTP 400 Bad Request**. This is mathematically verified and intentional behavior.

---

## 5. Raw Video Preview & Byte-Range Streaming

| Variable | Type | Default Value | Valid Options / Range | Description |
|---|---|---|---|---|
| `VIDEO_SOURCE_DIR` | string | `""` | Directory path | Local path containing raw `<video_id>.mp4` files. Enables `GET /api/video/{video_id}`. |
| `VIDEO_SOURCE_URL_TEMPLATE`| string| `""` | Valid URL template | HTTP(S) URL template containing `{video_id}` placeholder (e.g. `https://server/v/{video_id}.mp4`) for lazy on-demand streaming. |
| `VIDEO_CACHE_DIR` | string | `data/videos` | Directory path | Local destination directory where lazily downloaded raw videos are cached. |
| `VIDEO_SOURCE_DIR_HOST` | string | `./data/videos` | Host directory path | Host bind mount source for Docker Compose volume mapping. |
| `VIDEO_SOURCE_DIR_CONTAINER`| string| `/videos` | Container path | Container mount target for local raw videos (`/videos`). |

---

## 6. Query Refiner Settings

| Variable | Type | Default Value | Valid Options / Range | Description |
|---|---|---|---|---|
| `QUERY_REFINER_ENABLED` | boolean | `true` | `true`, `false` | Enables natural language query expansion into visual and text variants. |
| `QUERY_REFINER_BACKEND` | string | `auto` | `auto`, `transformers`, `deterministic` | Backend selector. `auto` uses HuggingFace Transformers if weights exist, falling back cleanly to deterministic regex. |
| `QUERY_REFINER_MODEL` | string | `Qwen/Qwen2.5-1.5B-Instruct` | Hugging Face Model ID | Local causal text LLM used for visual query rewriting (1.5B parameters; distinct from Qwen3-VL 2B embedder). |
| `QUERY_REFINER_MAX_VISUAL_VARIANTS`| integer| `4` | `1` to `8` | Maximum number of visual query variants generated per request. |
| `QUERY_REFINER_CACHE_ENABLED`| boolean| `true` | `true`, `false` | In-memory LRU cache for query reformulation strings. |
| `QUERY_REFINER_RRF_K` | integer | `60` | Positive integer | Reciprocal Rank Fusion constant $K$ used to merge variant rankings. |
| `DEBUG_QUERY_PLAN` | boolean | `false` | `true`, `false` | When `true`, returns detailed query rewriting telemetry in API responses. |
| `RERANKER_ENABLED` | boolean | `true` | `true`, `false` | Toggles candidate re-scoring based on text evidence proximity. |

---

## 7. Frame Sampling & Local Refinement (Ingestion / Legacy)

| Variable | Type | Default Value | Valid Options / Range | Description |
|---|---|---|---|---|
| `VISUAL_SAMPLING_MODE` | string | `legacy` | `legacy`, `sparse_shot` | Ingestion frame sampling strategy. `legacy` = 1.0s uniform. `sparse_shot` = 5.0s uniform + shot boundary fallback. |
| `VISUAL_GLOBAL_SAMPLE_SECONDS`| float | `5.0` | Positive float | Base cadence in seconds for sparse periodic frame extraction. |
| `VISUAL_DEDUP_ENABLED` | boolean | `false` | `true`, `false` | Toggles cosine-similarity deduplication during offline frame ingestion. |
| `VISUAL_DEDUP_THRESHOLD` | float | `0.97` | `0.0` to `1.0` | Cosine similarity threshold above which duplicate frames are dropped. |
| `LOCAL_REFINE_ENABLED` | boolean | `false` | `true`, `false` | Offline scaffolding toggle for dense window candidate decoding. |
| `LOCAL_REFINE_WINDOW_SECONDS`| float | `10.0` | Positive float | Search radius in seconds for local dense refinement. |
| `LOCAL_REFINE_INTERVAL_SECONDS`| float| `0.5` | Positive float | Sampling interval within candidate temporal windows. |
| `LOCAL_REFINE_MAX_REGIONS` | integer | `5` | Positive integer | Max candidate temporal regions refined per query. |

---

## 8. Host Networking, Security & Offline Execution

| Variable | Type | Default Value | Valid Options / Range | Description |
|---|---|---|---|---|
| `AIC_BIND_IP` | string | `127.0.0.1` | IP address (`0.0.0.0`, `127.0.0.1`) | Host IP interface to which the FastAPI service binds. |
| `AIC_PORT` | integer | `8000` | Valid TCP Port | Host TCP port exposed for the API service. |
| `ALLOWED_ORIGINS` | string | `http://localhost:3000,http://127.0.0.1:3000` | Comma-separated URLs | Allowed CORS origins. Wildcard `*` is rejected when credentials are enabled. |
| `DEBUG_API_ERRORS` | boolean | `false` | `true`, `false` | When `true`, exposes raw exception tracebacks in HTTP 503 error payloads. |
| `HF_HUB_OFFLINE` | integer | `0` | `0`, `1` | Set to `1` in isolated environments to enforce zero internet calls by HuggingFace Hub. |
| `TRANSFORMERS_OFFLINE` | integer | `0` | `0`, `1` | Set to `1` to restrict Transformers library exclusively to pre-cached local weights. |

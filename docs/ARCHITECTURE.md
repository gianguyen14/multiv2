# System Architecture: AIC 2026 Multimodal Video Retrieval System

## 1. Executive Architectural Overview

The **AIC 2026 Multimodal Video Retrieval System** is an enterprise-grade multimedia information retrieval engine engineered for the Ho Chi Minh City AI Challenge (AIC) 2026. The platform supports three competition query modes:
- **Known-Item Search (KIS)**: Cross-modal text-to-visual ad-hoc retrieval of target video keyframes.
- **Video Question Answering (QA)**: Natural language question answering over video content by fusing visual keyframe localization with lexical speech/text evidence and answer synthesis.
- **Temporal Retrieval of Actions and Key Events (TRAKE)**: Monotonic multi-event sequence alignment across temporally evolving video actions within a single video.

### 1.1 The Decoupled Read-Only Runtime Paradigm

The system adheres to a strict separation between **offline ingestion / indexing** and **online read-only retrieval runtime**:
- **Historical In-Process Paradigm (Milestones M1–M31)**: The application server previously executed video decoding, frame sampling, SigLIP2 embedding extraction, PaddleOCR, Faster-Whisper ASR, and FAISS index generation in-process. This architecture has been retired.
- **Production Paradigm (`QwenRuntimeSearch`)**: Ingestion is decoupled and executed offline. The online search service mounts pre-computed, published FAISS index generations (1024-dimensional normalized inner-product vector indices) and pre-extracted JSON text evidence spools. At query time, the system performs zero in-process video decoding, zero video writing, zero dynamic re-indexing, and zero online OCR/ASR inference. The database is opened strictly read-only.

```
+----------------------------------------------------------------------------------------------------+
|                                    OPERATOR INTERFACES                                             |
|                                                                                                    |
|   CLI Tool: python projectctl.py {search|kis|qa|trake}     Web UI: "Chi Lăng" (frontend/src/)       |
|   (Interactive & Automated Query Runner)                   (Vanilla HTML5 / CSS3 / ES6 Javascript) |
+--------------------------------------------------+-------------------------------------------------+
                                                   | HTTP / REST (Port 8000 / Dev Proxy Port 3000)
                                                   v
+----------------------------------------------------------------------------------------------------+
|                                FASTAPI APPLICATION CORE (backend/app/main.py)                      |
|                                                                                                    |
|   - Middleware: Security Headers (nosniff, SAMEORIGIN), CORS (Explicit Origins)                    |
|   - Health Endpoints: /health, /health/live, /health/ready                                         |
|   - Retrieval Routes: POST /api/search (kis, qa, trake), POST /api/search/image                    |
|   - Streaming & Media Routes: GET / HEAD /api/video/{video_id} (HTTP 206 Byte-Range streaming)     |
|   - Static Frame Serving: GET /api/frames/{video_id}/{filename}                                    |
|   - Frontend Static Host: GET /, GET /styles/{filename}, GET /scripts/{filename}                   |
+--------------------------------------------------+-------------------------------------------------+
                                                   | Dispatches search payloads
                                                   v
+----------------------------------------------------------------------------------------------------+
|                               ACTIVE SEARCH BACKEND: QwenRuntimeSearch                             |
|                           (backend/app/services/qwen_runtime_search.py)                            |
|                                                                                                    |
|   1. Query Encoding: Qwen3VlLocalEmbedder (Qwen/Qwen3-VL-Embedding-2B, MRL 1024-d, L2 norm)        |
|   2. Vector Recall: FaissSigLIPIndex (IndexFlatIP over 47,430 frames, depth: max(top_k*2, 200))     |
|   3. Visual Score Min-Max Normalization over recalled candidate set                                |
|   4. Lexical Token Matching: Pre-indexed OCR & ASR in-memory token sets (Dice + Phrase)            |
|   5. Timestamp Alignment: bisect nearest_uid() to closest indexed frame per video                  |
|   6. Deterministic Score Fusion: Score = 0.70 * Visual + 0.18 * OCR + 0.12 * ASR                   |
|   7. Additive Context: Attaches video_ocr_evidence & video_asr_evidence per video                  |
|   8. Grounded QA Answering: QAAnswerSynthesizer (default: extractive, opt-in: remote_llm)          |
|   9. Ordered TRAKE DP Alignment: Single-video monotonic frame progression enforcement              |
+--------------------------------------------------+-------------------------------------------------+
                                                   | Mounts read-only artifacts
                                                   v
+----------------------------------------------------------------------------------------------------+
|                                  READ-ONLY DATA LAYER ON DISK                                      |
|                                                                                                    |
|   Published Generations ($VIDEO_PROCESSED_ROOT/index/):                                            |
|   - CURRENT -> {"schema_version": 1, "generation_id": "gen-A-b531063ff8ce48e1b0d62739fb290c23"}    |
|   - generations/gen-A-b531063ff8ce48e1b0d62739fb290c23/                                            |
|       - frames.faiss      (47,430 vectors x 1024-d, IndexFlatIP, SHA256 validated)                 |
|       - mapping.json      (Sequential index-to-UID mapping: 47,430 entries)                        |
|       - payloads.json     (Frame metadata: video_id, timestamp_seconds, source_frame_index)        |
|       - generation.json   (Manifest & cryptographic digests: artifact_sha256)                      |
|                                                                                                    |
|   Evidence Text Spools:                                                                            |
|   - $VIDEO_PROCESSED_ROOT/ocr/*.json (PaddleOCR text extractions per video)                        |
|   - $VIDEO_PROCESSED_ROOT/asr/*.json (Faster-Whisper audio transcripts per video)                  |
|                                                                                                    |
|   Local Model Weights ($MODEL_CACHE_DIR/Qwen3-VL-Embedding-2B/):                                  |
|   - model.safetensors (~4.0 GB)                                                                    |
|   - scripts/qwen3_vl_embedding.py (Official embedder module)                                       |
+----------------------------------------------------------------------------------------------------+
```

### 1.2 Operator Frontend Architecture ("Chi Lăng")

The user console (`frontend/src/`) is built with **vanilla HTML5, CSS3, and ES6+ JavaScript**:
- **Zero Build Step**: Requires no Node.js, npm, webpack, Vite, or packaging pipelines.
- **Direct FastAPI Static Hosting**: Served directly by the Python application server via `main.py`:
  - `GET /` serves `frontend/src/index.html`.
  - `GET /styles/{filename}` serves CSS files (`frontend/src/styles/main.css`).
  - `GET /scripts/{filename}` serves JavaScript files (`api.js`, `app.js`, `shortcuts.js`).
- **Safe DOM Standards**: Exclusively uses `textContent`, `document.createElement`, and `addEventListener`. The codebase prohibits `innerHTML`, `document.write`, and `eval`.
- **Zero CORS in Standard Deployment**: Relative API URLs (`/api/search`, `/api/video/{video_id}`) eliminate cross-origin complexity when accessed via port 8000.
- **Standalone Development Runner**: `run_dev.py` provides a standard-library HTTP server that serves frontend assets on port 3000 and proxies `/api/*` calls to the FastAPI backend on port 8000.

---

## 2. Production Retrieval Pipeline (`QwenRuntimeSearch`)

When `SEARCH_BACKEND=qwen3_vl` (production default), queries are handled by `backend/app/services/qwen_runtime_search.py:QwenRuntimeSearch`.

```
                            User Search Query (Text)
                                       │
                                       ▼
    ┌──────────────────────────────────────────────────────────────────────┐
    │ 1. Text Query Encoding (backend/app/embeddings/qwen3_vl.py)          │
    │    - Model: Qwen/Qwen3-VL-Embedding-2B                               │
    │    - Instruction prepended: "Retrieve the video frame that best       │
    │      matches the described visual scene."                            │
    │    - Bounded length: max_length=256                                  │
    │    - Matryoshka Representation Learning (MRL): truncated to 1024-d   │
    │    - L2 Normalization: vector / ||vector||2 (float32)                │
    └──────────────────────────────────┬───────────────────────────────────┘
                                       │ (1024-d query vector)
                                       ▼
    ┌──────────────────────────────────────────────────────────────────────┐
    │ 2. Visual Candidate Recall (FaissSigLIPIndex.search)                 │
    │    - FAISS Index: IndexFlatIP over 47,430 indexed frames             │
    │    - Candidate depth: K_recall = min(max(top_k * 2, 200), N_total)   │
    │    - Raw cosine similarities: S_visual = Q · V_i                     │
    └──────────────────────────────────┬───────────────────────────────────┘
                                       │
                                       ▼
    ┌──────────────────────────────────────────────────────────────────────┐
    │ 3. Score Min-Max Normalization                                       │
    │    - S_norm = (S_visual - min(S)) / (max(S) - min(S))                │
    │    - Handles degenerate single-value recall gracefully               │
    └──────────────────────────────────┬───────────────────────────────────┘
                                       │
                                       ▼
    ┌──────────────────────────────────────────────────────────────────────┐
    │ 4. Lexical OCR & ASR Evidence Matching                               │
    │    - Scans pre-tokenized memory records from ocr/*.json & asr/*.json │
    │    - Punctuation removed, Unicode NFC, lowercase, stopword filtering │
    │    - Overlap Dice = 2 * |Q ∩ T| / (|Q| + |T|)                        │
    │    - Phrase Bonus = 1.0 if query token sequence in text              │
    │    - S_lexical = min(1.0, max(Dice, Phrase))                         │
    │    - Temporal projection: nearest_uid() maps text timestamp to the   │
    │      closest indexed keyframe within the same video via bisect       │
    └──────────────────────────────────┬───────────────────────────────────┘
                                       │
                                       ▼
    ┌──────────────────────────────────────────────────────────────────────┐
    │ 5. Deterministic Score Fusion                                        │
    │    Score = 0.70 * S_norm + 0.18 * S_ocr + 0.12 * S_asr               │
    └──────────────────────────────────┬───────────────────────────────────┘
                                       │
                                       ▼
    ┌──────────────────────────────────────────────────────────────────────┐
    │ 6. Additive Per-Video Evidence Attachment                            │
    │    - Locates highest-scoring OCR & ASR hits across entire video      │
    │    - Populates: video_ocr_score, video_asr_score, ocr_hit_frame_uid, │
    │      asr_hit_frame_uid, video_ocr_evidence, video_asr_evidence       │
    │    - Preserves primary ranking while granting operator visibility    │
    └──────────────────────────────────┬───────────────────────────────────┘
                                       │
                                       ▼
    ┌──────────────────────────────────────────────────────────────────────┐
    │ 7. Sorting & Ranking                                                 │
    │    - Sort order: (-Score, video_id, frame_id)                        │
    │    - Top-K truncation and rank assignment (1..K)                     │
    └──────────────────────────────────┬───────────────────────────────────┘
                                       │
                                       ▼
               Final Search Results Array ([ResultItem, ...])
```

### 2.1 Detailed Query Execution Steps

1. **Text Query Encoding (`Qwen3VlLocalEmbedder.encode_query`)**:
   - Instruction prefix: `"Retrieve the video frame that best matches the described visual scene."`.
   - Tokenization: truncated to `max_length=256`.
   - Low-memory eager CPU execution using `bfloat16` and bounded threads (`torch.set_num_threads(2)`).
   - Pooling and Matryoshka Representation Learning (MRL) truncation: embedding slice `full[:1024]`.
   - Float32 L2 normalization: `vector = (vector / np.linalg.norm(vector)).astype(np.float32)`.
   - Invariant check: vector must satisfy `shape == (1024,)` and unit norm (`np.isclose(norm, 1.0, atol=1e-5)`).

2. **Visual Candidate Recall (`FaissSigLIPIndex.search`)**:
   - Queries the published FAISS `IndexFlatIP` across 47,430 frames.
   - Dynamic recall depth: $K_{	ext{recall}} = \min(\max(	ext{top\_k} 	imes 2, 200), N_{	ext{total}})$.
   - Dot product yields exact cosine similarity since frame embeddings and query embeddings are unit-normalized.

3. **Visual Score Normalization**:
   - Visual scores undergo min-max scaling across the recalled candidate set:
     $$S_{	ext{norm}} = rac{S_{	ext{visual}} - \min(S)}{\max(S) - \min(S)}$$
   - If $\max(S) == \min(S)$, all candidates receive $1.0$ (if score $> 0$) or $0.0$.

4. **Lexical Matching & Temporal Projection**:
   - In-memory OCR (`_ocr_records`) and ASR (`_asr_records`) tokens are normalized (Unicode NFC, lowercase, whitespace consolidated, stopwords removed).
   - Lexical overlap computed via Dice coefficient and exact token phrase matching:
     $$	ext{Dice} = rac{2 \cdot |Q \cap T|}{|Q| + |T|}$$
     $$	ext{Phrase} = 1.0 \quad 	ext{if } Q \subseteq T 	ext{ and } |Q| \ge 2 \quad 	ext{else } 0.0$$
     $$S_{	ext{lexical}} = \min(1.0, \max(	ext{Dice}, 	ext{Phrase}))$$
   - `nearest_uid(timeline, video_id, timestamp)` uses binary search (`bisect_left`) to map text timestamps to the nearest indexed visual keyframe in that video.

5. **Multi-Modal Score Fusion**:
   - Candidate pool forms the union: $U = \{ 	ext{visual hits} \} \cup \{ 	ext{top OCR hits} \} \cup \{ 	ext{top ASR hits} \}$.
   - Deterministic linear fusion:
     $$	ext{Score} = 0.70 \cdot S_{	ext{norm}} + 0.18 \cdot S_{	ext{ocr}} + 0.12 \cdot S_{	ext{asr}}$$

6. **Additive Per-Video Evidence Attachment**:
   - Identifies the highest-scoring OCR and ASR occurrences anywhere in each matched video.
   - Attaches `video_ocr_score`, `video_asr_score`, `ocr_hit_frame_uid`, `asr_hit_frame_uid`, `video_ocr_evidence`, and `video_asr_evidence` to each candidate.
   - Ranking is completely preserved; this exposes video-level contextual text evidence to the operator even if the matched keyframe is not adjacent to the text timestamp.

7. **Sorting and Truncation**:
   - Sorted deterministically by `(-score, video_id, frame_id)`.
   - Truncated to `top_k`, assigning 1-based sequential `rank`.

### 2.2 Question Answering & Synthesis (`QAAnswerSynthesizer`)

For `query_type="qa"`, answer generation operates on top of the retrieved ranking:
- **Extractive Baseline (`QA_ANSWER_BACKEND=extractive`)**: Default mode. Inspects `ocr_evidence` and `asr_evidence` for the keyframe; returns up to 100 characters of raw text evidence. Produces `answer_backend="extractive"`, `answer_model="none"`, `answer_status="disabled"` (or `"no_evidence"`).
- **Remote LLM Synthesis (`QA_ANSWER_BACKEND=remote_llm`)**:
  - Implemented in `backend/app/services/qa_answer_synthesizer.py`.
  - Connects to an external OpenAI-compatible `/v1/chat/completions` endpoint (`QA_ANSWER_URL`) using bearer authentication (`QA_ANSWER_API_KEY`) and model `QA_ANSWER_MODEL` (default `agent-lite`).
  - **Latency Budgeting**: Constrained by `QA_ANSWER_REMOTE_TOP_N` (code default 5, clamped 1..10). The **production target is `QA_ANSWER_REMOTE_TOP_N=2`**. Sequential LLM calls with 8-second timeouts (`QA_ANSWER_TIMEOUT_SECONDS=8`) require bounding $N=2$ to maintain total query latency under 16 seconds. Rows ranked 3 through $K$ deterministically stay extractive without invoking remote requests.
  - **Abstention**: If the prompt cannot be answered from evidence, the remote model outputs `"Không đủ bằng chứng."`, preserved as `answer_status="abstained"`.
  - **Status**: Labelled **EXPERIMENTAL**. Upstream services frequently experience 502 Bad Gateway responses or connection timeouts. All network and JSON decode errors cleanly degrade to extractive fallback without raising 500 errors.

### 2.3 Temporal Retrieval of Actions (TRAKE)

For `query_type="trake"`, `search_trake(events)` receives an ordered list of 2 to 20 event descriptions:
1. Each event query is independently evaluated against the index via `search_single(event, top_k=max(10, top_k))`.
2. The candidates are grouped by video. For each video, greedy dynamic programming finds the highest-scoring sequence of frames strictly satisfying monotonic progression:
   $$	ext{frame\_index}_1 < 	ext{frame\_index}_2 < \dots < 	ext{frame\_index}_M$$
3. **Validation Invariant (HTTP 400)**: If no single video covers every event in strictly increasing frame order, `search_trake` raises `ValueError("TRAKE: no single video covers every event in increasing frame order")`. In `main.py`, this is caught and returned as **HTTP 400 Bad Request**. This is the documented, intended contract behavior.
4. When a sequence is found, scores are averaged across all events, and the primary keyframe is assigned to the first event (`events[0]`).

---

## 3. Database Generations & Dimensionality Contracts

The platform defines two pre-computed database generations on disk. Both maintain an immutable read-only contract.

```
                           DATABASE GENERATION TOPOLOGY

         DB v1 (Production Default)                   DB v2 (Dense Experimental)
       $VIDEO_PROCESSED_ROOT/index/                  $VISION_PROCESSED_ROOT/index/
                     │                                             │
                     ▼                                             ▼
        CURRENT -> gen-A-b531...                      CURRENT -> gen-001
                     │                                             │
     ┌───────────────┴───────────────┐             ┌───────────────┴───────────────┐
     │ - 47,430 vectors              │             │ - 470,833 vectors             │
     │ - 1024-D embedding space      │             │ - 1024-D embedding space      │
     │ - 873 videos @ 10.0s interval │             │ - 873 videos @ 1.0s (1 FPS)   │
     │ - Model: Qwen3-VL-Embed-2B    │             │ - Model: Qwen3-VL-Embed-2B    │
     │ - IndexFlatIP                 │             │ - IndexFlatIP                 │
     │ - Paired with ocr/ & asr/     │             │ - Sidecar: artifact_sha256    │
     └───────────────────────────────┘             └───────────────────────────────┘
```

### 3.1 DB v1: Production Coarse Database (`GENERATION_A`)
- **Generation Identifier**: `gen-A-b531063ff8ce48e1b0d62739fb290c23`
- **Location**: `$VIDEO_PROCESSED_ROOT/index/generations/gen-A-b531063ff8ce48e1b0d62739fb290c23/`
- **Pointer File**: `$VIDEO_PROCESSED_ROOT/index/CURRENT` containing `{"schema_version": 1, "generation_id": "gen-A-b531063ff8ce48e1b0d62739fb290c23"}`
- **Embedding Space**: **1024 dimensions**, float32, L2 unit-normalized.
- **Vector Count**: **47,430 indexed frames**.
- **Corpus Coverage**: 873 videos (`L21_V001` through `L30_V096`).
- **Sampling Frequency**: Coarse uniform 10.0-second intervals (`interval_seconds: 10.0`).
- **Encoder Identity**: `Qwen/Qwen3-VL-Embedding-2B` (backend: `qwen3_vl_embedding_2b`, contract: `visual-encoder-v2`, revision: `9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda`).
- **Index Engine**: FAISS `IndexFlatIP` (inner product cosine similarity).
- **Cryptographic Artifact Checksums (`artifact_sha256`)**:
  - `frames.faiss`: `b61ed8652f1661b8fd881cbeaa9c23516521fbe28872281fd925c77760768abb`
  - `mapping.json`: `db3cdbf6a3c184683a27c2feaa960fab8f83a6c1a2b66a2a3e9c1c4493e17d0c`
  - `payloads.json`: `1c566e07bc73bd87f09069d010ba7ec579e179085aea78463799ed6430c9c8ef`
- **Companion Spools**:
  - `$VIDEO_PROCESSED_ROOT/ocr/*.json` (PaddleOCR text extractions)
  - `$VIDEO_PROCESSED_ROOT/asr/*.json` (Faster-Whisper audio transcripts)

### 3.2 DB v2: Dense 1 FPS Vision Database (`DENSE_1FPS_V1`)
- **Generation Identifier**: `gen-001`
- **Location**: `$VISION_PROCESSED_ROOT/index/generations/gen-001/`
- **Pointer File**: `$VISION_PROCESSED_ROOT/index/CURRENT`
- **Embedding Space**: **1024 dimensions**, float32, L2 unit-normalized.
- **Vector Count**: **470,833 indexed frames**.
- **Corpus Coverage**: 873 videos (`L21_V001` through `L30_V096`).
- **Sampling Frequency**: Dense uniform 1.0-second intervals (1 FPS, `sampling_policy: "dense_1fps"`).
- **Encoder Identity**: `Qwen/Qwen3-VL-Embedding-2B`.
- **Index File Size**: ~1.93 GB (`frames.faiss`).
- **Cryptographic Artifact Checksums (`artifact_sha256.json`)**:
  - `frames.faiss`: `e37efa3ff6cb99de038a7e4378bdd7b1cf547c1397f70b72ad041957672825b4`
  - `mapping.json`: `e1112053c4563164cd27be5e139dd795c4626b0d0021c70b07f7de273d3e1477`
  - `payloads.json`: `473ccaf1c33ee95b64119be75e5b7483a29ba75c500258ca460c436a55f24ed5`
  - `generation.json`: `6f54b4a60033ae3538cfdce0df3037c8cd954f9f85ffca527b596adb24e1bd4c`
- **Operational Role**: Developed for dense frame localization and image-to-frame search; accessed via experimental branches.

---

## 4. HTTP Route Catalog & Protocols

All routes are registered in `backend/app/main.py`:

| Method | URI Path | Handler | Status Codes | Description |
|---|---|---|---|---|
| `GET` | `/` | `index()` | 200 | Serves the operator frontend console (`frontend/src/index.html`). |
| `GET` | `/styles/{filename}` | `style()` | 200, 404 | Serves frontend static stylesheets (`main.css`). Blocks path traversal. |
| `GET` | `/scripts/{filename}` | `script()` | 200, 404 | Serves frontend JavaScript files (`api.js`, `app.js`, `shortcuts.js`). |
| `GET` | `/health` | `health()` | 200 | Liveness probe returning configuration status, backend name, and device summary. |
| `GET` | `/health/live` | `health()` | 200 | Kubernetes liveness probe alias. |
| `GET` | `/health/ready` | `ready()` | 200, 503 | Readiness probe validating FAISS index presence, generation ID, and model weights. |
| `POST` | `/api/search` | `search()` | 200, 400, 503 | Primary multimodal search route supporting `kis`, `qa`, and `trake`. |
| `POST` | `/api/search/image` | `search_image_endpoint()` | 200, 400, 415, 503 | Experimental image query route. Disabled (503) under production Qwen default. |
| `GET` | `/api/frames/{video_id}/{filename}` | `frame()` | 200, 404 | Static keyframe image delivery (`Cache-Control: public, max-age=86400`). |
| `GET`, `HEAD` | `/api/video/{video_id}` | `video()` | 200, 206, 404, 416, 503 | Seekable raw video stream with full RFC 7233 HTTP Byte-Range support. |

---

## 5. Configuration & Environment Variables

System behavior is declared in `backend/app/core/config.py` and configurable via `.env`:

### 5.1 Core System & Search Backend

| Variable | Default Value | Defined In | Purpose & Notes |
|---|---|---|---|
| `SEARCH_BACKEND` | `qwen3_vl` | `main.py:111` | Primary search backend selector. `qwen3_vl` (production default) or `siglip2` (legacy). |
| `SEARCH_ENCODER` | None | `main.py:111` | Legacy alias for `SEARCH_BACKEND`. |
| `VIDEO_PROCESSED_ROOT` | `data/processed/videos` | `main.py:140`, `config.py` | Path to production DB v1 root containing `index/`, `ocr/`, and `asr/`. |
| `VISION_PROCESSED_ROOT` | None | Compose / Feature branches | Root directory for DB v2 dense 1 FPS index (`gen-001`). |
| `SEARCH_ENABLE_OCR` | `true` | `qwen_runtime_search.py:182` | Toggles in-memory OCR text evidence loading and score fusion. |
| `SEARCH_ENABLE_ASR` | `true` | `qwen_runtime_search.py:183` | Toggles in-memory ASR speech transcript loading and score fusion. |
| `ALLOWED_ORIGINS` | `localhost:3000, 127.0.0.1:3000` | `main.py:240` | Permitted origins for CORS handling. Wildcards (`*`) are disallowed when credentials are enabled. |
| `DEBUG_API_ERRORS` | `false` | `main.py:36` | Exposes internal exception types and messages in HTTP 503 response payloads. |

### 5.2 Model Configuration & Paths

| Variable | Default Value | Defined In | Purpose & Notes |
|---|---|---|---|
| `MODEL_CACHE_DIR` | `models` | `qwen3_vl.py:36` | Base directory for neural network weight checkpoints. |
| `QWEN3_VL_MODEL_DIR` | None | `qwen3_vl.py:33` | Explicit directory override for `Qwen3-VL-Embedding-2B`. |
| `QWEN_MODEL_DIR` | None | `qwen3_vl.py:33` | Alternate directory override for Qwen model weights. |
| `SIGLIP_ENABLED` | `True` | `config.py:4` | Toggles availability of SigLIP2 in legacy backend. |
| `SIGLIP2_MODEL` | `google/siglip2-base-patch16-224` | `config.py:5` | Model repository identifier for legacy SigLIP2 encoder. |
| `SIGLIP_LONG_TEXT_MODE`| `chunk_mean` | `config.py:8` | Chunk aggregation strategy for text exceeding 64 tokens (`chunk_mean` or `truncate`). |
| `SIGLIP_TEXT_MAX_LENGTH`| `64` | `config.py:9` | Token window size per text chunk in SigLIP2. |
| `SIGLIP_TEXT_CHUNK_STRIDE`| `8` | `config.py:10` | Overlap stride between consecutive text chunks. |
| `SIGLIP_TEXT_MAX_CHUNKS`| `8` | `config.py:11` | Maximum number of chunks averaged per long query. |

### 5.3 Video Preview & Streaming

| Variable | Default Value | Defined In | Purpose & Notes |
|---|---|---|---|
| `VIDEO_CACHE_DIR` | `data/videos` | `video_source.py:42` | Directory storing cached raw video MP4 files. |
| `VIDEO_SOURCE_DIR` | None | `video_source.py:46` | Local directory serving raw MP4 videos in place without caching. |
| `VIDEO_SOURCE_URL_TEMPLATE`| None | `video_source.py:50` | HTTP(S) URL template containing `{video_id}` placeholder for lazy on-demand downloads. |

### 5.4 Video Question Answering (QA)

| Variable | Default Value | Production Target | Purpose & Notes |
|---|---|---|---|
| `QA_ANSWER_BACKEND` | `extractive` | `extractive` (or `remote_llm`) | QA answer generation backend (`extractive` or `remote_llm`). |
| `QA_ANSWER_URL` | `""` | `http://.../v1` | Base URL of OpenAI-compatible chat completions service. |
| `QA_ANSWER_MODEL` | `agent-lite` | `agent-lite` | Model identifier passed to remote LLM endpoint. |
| `QA_ANSWER_API_KEY` | `""` | Secret | Bearer authorization token. |
| `QA_ANSWER_TIMEOUT_SECONDS` | `8.0` | `8.0` | Per-request network timeout for remote LLM inference. |
| `QA_ANSWER_MAX_EVIDENCE_CHARS`| `12000` | `12000` | Evidence character budget passed to system/user prompt. |
| `QA_ANSWER_REMOTE_TOP_N` | `5` | **`2`** | Maximum candidate rows triggering remote synthesis calls. Clamped to 1..10. **Target is 2** to bound latency. |

### 5.5 Inactive / Legacy Configuration Flags

To maintain strict documentation accuracy, the following configuration flags exist in code but are **INACTIVE** in the production retrieval path:
- **DINOv3 (`DINO_ENABLED=False`)**: Disabled. Not used for feature extraction or ranking.
- **E5 (`E5_ENABLED=True`)**: Bypassed. Text embeddings in production rely exclusively on Qwen3-VL.
- **TransNetV2**: Inactive. Production frame sampling relies on fixed interval temporal sampling; TransNetV2 is not active in the online search path.
- **BM25 (`BM25_ENABLED=True`)**: Replaced in production by the in-memory lexical Dice overlap and exact phrase matching engine.
- **`QUERY_REFINER_ENABLED`, `RERANKER_ENABLED`**: Active only when `SEARCH_BACKEND=siglip2` (legacy).

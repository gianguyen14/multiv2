# Search Backends: Architecture & Capability Matrix

This document provides the authoritative technical reference for the search backends in the AIC 2026 Multimodal Video Retrieval System. It contrasts the production-default **`qwen3_vl`** engine against the **`siglip2`** legacy pipeline, details experimental visual and QA synthesis capabilities, and outlines operational lifecycle states.

---

## 1. Backend Classification & Lifecycle States

The repository supports two distinct search architectures selectable at startup via `SEARCH_BACKEND` (or legacy alias `SEARCH_ENCODER`):

1. **`qwen3_vl` (PRODUCTION DEFAULT)**:
   - **Status**: **PRODUCTION DEFAULT**.
   - **Backend Class**: `backend/app/services/qwen_runtime_search.py:QwenRuntimeSearch`.
   - **Underlying Model**: `Qwen/Qwen3-VL-Embedding-2B` (1024-dimensional normalized inner-product vector space).
   - **Operational Mode**: High-performance, read-only runtime serving text queries over pre-indexed visual keyframe generations and pre-extracted OCR/ASR JSON text spools.
   - **Hardware Contract**: Highly optimized for CPU inference (`bfloat16`, 2 worker threads) as well as GPU acceleration (`float16`).

2. **`siglip2` (LEGACY)**:
   - **Status**: **LEGACY (Explicit Opt-In Only)**.
   - **Backend Class**: `backend/app/services/configured_search.py:ConfiguredSearch`.
   - **Underlying Model**: `google/siglip2-base-patch16-224` (768-dimensional normalized inner-product vector space).
   - **Operational Mode**: Historical milestone pipeline (M1–M31) with in-process query expansion (`Qwen2.5-1.5B-Instruct`), Reciprocal Rank Fusion (RRF), dynamic modality weighting, and temporal refinement.
   - **Database Compatibility**: **Zero** populated SigLIP2 FAISS indices exist on disk for the full 873-video competition dataset. The backend includes a safety guard (`_guard_index_backend`) that refuses to execute against a Qwen-built index to prevent invalid cross-embedding scoring.

3. **Image-to-Frame Search (EXPERIMENTAL)**:
   - **Status**: **EXPERIMENTAL (GPU Validation Incomplete)**.
   - **Branch / Module**: Feature branch `agent/qwen-image-search` (`73fc6fe`) and legacy `ConfiguredSearch.search_image`.
   - **Constraint**: On the production `qwen3_vl` backend, image search is explicitly disabled (`capabilities.image = False`) and returns HTTP 503 (`"image search is not supported by the active search backend"`). Validation script `scripts/validate_qwen_image_gpu.py` requires a dedicated CUDA GPU host and has not been executed on production.

4. **Grounded QA Answer Synthesis (EXPERIMENTAL)**:
   - **Status**: **IMPLEMENTED / EXPERIMENTAL**.
   - **Module**: `backend/app/services/qa_answer_synthesizer.py:QAAnswerSynthesizer`.
   - **Constraint**: Opt-in via `QA_ANSWER_BACKEND=remote_llm`. While fully implemented with OpenAI-compatible chat completion payload schemas and latency budgeting (`QA_ANSWER_REMOTE_TOP_N=2`), the upstream endpoint frequently returns **HTTP 502 Bad Gateway** or times out. The runtime gracefully falls back to deterministic extractive answering without failing search requests.

---

## 2. Comprehensive Capability Matrix

| Architectural Dimension | `qwen3_vl` (Production Default) | `siglip2` (Legacy Milestone) |
|---|---|---|
| **Lifecycle State** | **PRODUCTION DEFAULT** | **LEGACY (Deprecated)** |
| **Backend Implementation** | `QwenRuntimeSearch` | `ConfiguredSearch` |
| **Activation Flag** | `SEARCH_BACKEND=qwen3_vl` | `SEARCH_BACKEND=siglip2` |
| **Primary Vision-Language Model**| `Qwen/Qwen3-VL-Embedding-2B` | `google/siglip2-base-patch16-224` |
| **Embedding Dimension** | **1024-D** (via MRL truncation) | **768-D** |
| **Query Embedding Normalization**| Float32 L2 Unit Normalization | Float32 L2 Unit Normalization |
| **Index Data Type** | FAISS `IndexFlatIP` (Cosine Similarity) | FAISS `IndexFlatIP` (Cosine Similarity) |
| **Active Dataset on Disk** | **DB v1** (47,430 vectors, 873 videos, 10s) | None (Requires offline re-ingestion) |
| **Visual Candidate Recall Depth**| $K_{	ext{recall}} = \min(\max(	ext{top\_k} \cdot 2, 200), N)$ | $K_{	ext{recall}} = \max(	ext{top\_k} \cdot 2, 200)$ |
| **Score Normalization** | Dynamic Min-Max across recalled set | Min-Max across candidate set |
| **OCR Text Evidence Engine** | In-memory token index (Dice + Phrase) | `TextEvidenceStore` per-frame lexical |
| **ASR Speech Evidence Engine** | In-memory token index (Dice + Phrase) | `TextEvidenceStore` per-interval lexical |
| **Multi-Modal Score Fusion** | **Fixed Linear**: `0.70*Vis + 0.18*OCR + 0.12*ASR` | **RRF** ($K=60$) or dynamic environment weights |
| **Per-Video Additive Context** | **Yes** (`video_ocr_score`, `video_asr_score`) | No (Attaches only keyframe-local text) |
| **Textual KIS Queries** | **Supported (Production)** | Supported (Legacy) |
| **Video Question Answering (QA)**| **Supported** (Extractive & Remote LLM) | Supported (Extractive regex rule-based) |
| **QA Remote LLM Budget** | **`QA_ANSWER_REMOTE_TOP_N=2`** | Unsupported (Dead-code flags) |
| **Ordered TRAKE Alignment** | **Supported** (Single-video monotonic DP) | Supported (DP Aligner + TemporalRefiner) |
| **TRAKE Unmatched Error** | **HTTP 400 Bad Request** | Returns empty list `[]` |
| **Dense Temporal Refinement** | Not utilized (Bypassed) | Supported via `TemporalRefiner` (Requires raw MP4) |
| **Image-to-Frame Search** | **Disabled (HTTP 503)** (Experimental) | Implemented via SigLIP2 (No DB on disk) |
| **Video Playback Streaming** | **HTTP 206 Partial Content** (`/api/video`) | **HTTP 206 Partial Content** (`/api/video`) |
| **Static Frame Delivery** | `/api/frames/{video_id}/{filename}` | `/api/frames/{video_id}/{filename}` |
| **Online Ingestion / Decoding**| **Zero** (Pure read-only runtime) | Optional (In-process decoder support) |
| **Cross-Backend Protection** | Rejects non-Qwen index in `_initialize` | Rejects Qwen index in `_guard_index_backend` |

---

## 3. Model Identity, Revisions & Inference Contracts

### 3.1 Production Model: `Qwen3-VL-Embedding-2B`

The production retrieval pipeline relies on Alibaba Cloud's `Qwen3-VL-Embedding-2B` for cross-modal textual-visual embedding.

- **HuggingFace Repository**: `Qwen/Qwen3-VL-Embedding-2B`
- **Pinned Git Revision**: `9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda`
- **Model Fingerprint**: `5835d5fd0517b47c4ba72ebb7b2e8af8f0ab5102aada0ed34d497fef5684cd99`
- **Contract Version**: `visual-encoder-v2`
- **Disk Footprint**:
  - `model.safetensors`: ~4.0 GB
  - `config.json`: Model configuration
  - `scripts/qwen3_vl_embedding.py`: Official embedder script loaded dynamically via Python `importlib.util`
- **Text Query Embedding Contract (`Qwen3VlLocalEmbedder`)**:
  ```python
  DEFAULT_INSTRUCTION = "Retrieve the video frame that best matches the described visual scene."
  ```
  1. The user text query is combined with `DEFAULT_INSTRUCTION`.
  2. The input is tokenized and truncated to `max_length=256`.
  3. Evaluated on CPU using `torch.bfloat16` with `attn_implementation="eager"` and `threads=2`.
  4. The model produces a pooled hidden state representation.
  5. **Matryoshka Representation Learning (MRL)**: The vector is truncated to the first 1024 dimensions: `vector = full[:1024]`.
  6. **L2 Normalization**: The sliced vector is converted to float32 and divided by its Euclidean norm:
     $$\hat{v} = rac{v}{\|v\|_2}$$
  7. Shape and norm are validated before execution: `vector.shape == (1024,)` and $\| \hat{v} \|_2 pprox 1.0$.

### 3.2 Legacy Model: `SigLIP2-Base-Patch16-224`

The legacy pipeline uses Google's `google/siglip2-base-patch16-224`.

- **HuggingFace Repository**: `google/siglip2-base-patch16-224`
- **Contract Version**: `m15.1-v1`
- **Native Embedding Dimension**: **768-D**
- **Text Tokenization Contract (`SigLIP2Encoder`)**:
  - Window size: `SIGLIP_TEXT_MAX_LENGTH=64`.
  - Stride: `SIGLIP_TEXT_CHUNK_STRIDE=8`.
  - Max chunks: `SIGLIP_TEXT_MAX_CHUNKS=8`.
  - Long text handling: Queries longer than 64 tokens are divided into overlapping chunks and aggregated using `SIGLIP_LONG_TEXT_MODE=chunk_mean` (normalizing each chunk vector before taking the arithmetic mean).
- **Image Preprocessing**: Resized to 224x224 RGB, normalized via SigLIP processor, projected to 768-D.

### 3.3 Disambiguation: Qwen3-VL vs Qwen2.5

The repository references two distinct models from the Qwen family. Documentation must never conflate them:
- **`Qwen/Qwen3-VL-Embedding-2B`**: Vision-Language Embedding model (2B parameters) used for multimodal dense retrieval and text query encoding in `QwenRuntimeSearch`.
- **`Qwen/Qwen2.5-1.5B-Instruct`**: Causal language model (1.5B parameters) used exclusively in legacy `QueryRefiner` for text query decomposition and visual variant generation. It plays **zero** role in vector embedding generation.

---

## 4. Operational Invariants & Safeguards

### 4.1 Cross-Backend Index Incompatibility Protection

Because FAISS indices are serialized as generic binary files (`frames.faiss`), querying a 1024-D index with a 768-D vector (or vice-versa) causes dimension mismatch errors or completely meaningless mathematical projections. The codebase implements bidirectional protection:

1. **`QwenRuntimeSearch._initialize()` Protection**:
   ```python
   encoder_identity = bundle.metadata.get("encoder_identity") or {}
   backend_name = str(encoder_identity.get("backend", "")).lower()
   if backend_name not in {"qwen3_vl", "qwen3_vl_embedding_2b", "qwen3-vl-embedding-2b"}:
       raise RuntimeError(
           "refusing to serve a non-Qwen index with the qwen3_vl encoder: "
           f"index encoder backend is {backend_name!r}"
       )
   ```

2. **`ConfiguredSearch._guard_index_backend()` Protection**:
   ```python
   backend = str((bundle.metadata.get("encoder_identity") or {}).get("backend", "")).lower()
   if backend in {"qwen3_vl", "qwen3_vl_embedding_2b", "qwen3-vl-embedding-2b"}:
       raise RuntimeError(
           "active generation was built with Qwen3-VL embeddings "
           f"(backend={backend!r}); set SEARCH_BACKEND=qwen3_vl "
           "instead of using the siglip2 backend"
       )
   ```

### 4.2 Fail-Loud Startup Contract

In `backend/app/main.py:_build_configured_search`:
```python
backend = (os.getenv("SEARCH_BACKEND") or os.getenv("SEARCH_ENCODER") or "qwen3_vl").strip().lower()
if backend in ("qwen3_vl", "qwen3-vl", "qwen"):
    return QwenRuntimeSearch(processed_root=media_root)
if backend == "siglip2":
    return ConfiguredSearch(media_root)
raise RuntimeError(
    f"Unknown SEARCH_BACKEND={backend!r}; supported values: qwen3_vl (default), siglip2 (legacy)"
)
```
If an invalid or unrecognized backend name is supplied in the environment, the application refuses to start up. This guarantees that an unverified deployment can never silently default to an incorrect embedding space.

---

## 5. Environment Variable Reference per Backend

### 5.1 Production Backend (`SEARCH_BACKEND=qwen3_vl`)

| Environment Variable | Recommended Value | Impact & Behavior |
|---|---|---|
| `SEARCH_BACKEND` | `qwen3_vl` | Selects QwenRuntimeSearch production engine. |
| `VIDEO_PROCESSED_ROOT` | `/data/processed` | Path to root containing `index/CURRENT`, `ocr/`, `asr/`. |
| `MODEL_CACHE_DIR` | `/models` | Directory holding `Qwen3-VL-Embedding-2B/`. |
| `QWEN3_VL_MODEL_DIR` | None | Optional explicit override for Qwen model directory. |
| `SEARCH_ENABLE_OCR` | `true` | Enables lexical matching against OCR text spools. |
| `SEARCH_ENABLE_ASR` | `true` | Enables lexical matching against ASR audio spools. |
| `QA_ANSWER_BACKEND` | `extractive` | `extractive` (production default) or `remote_llm` (experimental). |
| `QA_ANSWER_REMOTE_TOP_N` | `2` | Hard candidate cap for remote LLM calls (prevents timeout). |
| `QA_ANSWER_TIMEOUT_SECONDS`| `8.0` | Timeout per remote LLM synthesis call. |

### 5.2 Legacy Backend (`SEARCH_BACKEND=siglip2`)

| Environment Variable | Default Value | Impact & Behavior |
|---|---|---|
| `SEARCH_BACKEND` | `siglip2` | Selects legacy ConfiguredSearch engine. |
| `SIGLIP_ENABLED` | `true` | Master flag for SigLIP2 model loader. |
| `SIGLIP2_MODEL` | `google/siglip2-base-patch16-224` | HuggingFace model checkpoint. |
| `VISUAL_WEIGHT` | `1.0` | Modality fusion weight for visual similarity. |
| `OCR_WEIGHT` | `1.0` | Modality fusion weight for OCR lexical overlap. |
| `ASR_WEIGHT` | `0.8` (or `1.0`) | Modality fusion weight for ASR lexical overlap. |
| `QUERY_REFINER_ENABLED` | `true` | Enables multi-path query expansion with Qwen2.5. |
| `QUERY_REFINER_RRF_K` | `60` | Constant $K$ for Reciprocal Rank Fusion. |
| `RERANKER_ENABLED` | `true` | Enables CandidateReranker cross-evidence reranking. |
| `TRAKE_TEMPORAL_REFINE_ENABLED` | `true` | Enables high-FPS local window decoding during TRAKE. |

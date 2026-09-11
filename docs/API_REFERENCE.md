# REST API Reference: AIC 2026 Multimodal Video Retrieval System

This document provides the exhaustive specification for all HTTP REST API endpoints exposed by the FastAPI server in `backend/app/main.py`.

---

## 1. Global Server Protocols & Security

### 1.1 Base URLs & Ports
- **Production FastAPI Server**: `http://127.0.0.1:8000` (configurable via `AIC_BIND_IP` and `AIC_PORT`).
- **Development Proxy Server**: `http://127.0.0.1:3000` (`run_dev.py` proxying `/api/*` to `:8000`).

### 1.2 Mandatory Security Headers
Every HTTP response issued by the FastAPI application is enriched with standard security headers:
```http
X-Content-Type-Options: nosniff
X-Frame-Options: SAMEORIGIN
Referrer-Policy: strict-origin-when-cross-origin
```

### 1.3 Cross-Origin Resource Sharing (CORS)
Governed by `ALLOWED_ORIGINS` in `.env`:
- Default: `http://localhost:3000, http://127.0.0.1:3000, http://localhost:8000, http://127.0.0.1:8000`.
- Allowed HTTP Methods: `GET`, `POST`, `OPTIONS`.
- Allowed Headers: `*`.
- Invariant: Wildcard `*` in `ALLOWED_ORIGINS` raises `RuntimeError` at startup when credentials are enabled.

---

## 2. Health & Diagnostic Endpoints

### 2.1 Liveness Probe: `GET /health` & `GET /health/live`
Returns the operational status of the service, active search backend, and compute device environment.

#### Request
```http
GET /health HTTP/1.1
Host: 127.0.0.1:8000
```

#### Response (200 OK)
```json
{
  "status": "ok",
  "search_configured": true,
  "processed_root": "/data/processed/videos",
  "search": {
    "backend": "qwen3_vl",
    "configured": true,
    "initialized": true,
    "processed_root": "/data/processed/videos",
    "model_dir": "/models/Qwen3-VL-Embedding-2B",
    "generation_id": "gen-A-b531063ff8ce48e1b0d62739fb290c23",
    "dimension": 1024,
    "ocr_records": 47430,
    "asr_records": 38120,
    "weights_present": true,
    "capabilities": {
      "kis": true,
      "qa": true,
      "trake": true,
      "image": false,
      "thumbnails": false,
      "raw_video_preview": true
    }
  },
  "compute": {
    "cuda_available": false,
    "device": "cpu",
    "torch_threads": 2
  }
}
```

---

### 2.2 Readiness Probe: `GET /health/ready`
Verifies that the published FAISS generation on disk is valid, matching the declared encoder identity, and that neural network weights exist.

#### Request
```http
GET /health/ready HTTP/1.1
Host: 127.0.0.1:8000
```

#### Response: Ready (200 OK)
```json
{
  "status": "ready",
  "search_configured": true,
  "processed_root": "/data/processed/videos",
  "search": {
    "backend": "qwen3_vl",
    "configured": true,
    "initialized": true,
    "processed_root": "/data/processed/videos",
    "model_dir": "/models/Qwen3-VL-Embedding-2B",
    "generation_id": "gen-A-b531063ff8ce48e1b0d62739fb290c23",
    "dimension": 1024,
    "ocr_records": 47430,
    "asr_records": 38120,
    "weights_present": true,
    "capabilities": {
      "kis": true,
      "qa": true,
      "trake": true,
      "image": false,
      "thumbnails": false,
      "raw_video_preview": true
    }
  },
  "compute": {
    "cuda_available": false,
    "device": "cpu",
    "torch_threads": 2
  },
  "search_readiness": {
    "ready": true,
    "generation_id": "gen-A-b531063ff8ce48e1b0d62739fb290c23"
  }
}
```

#### Response: Not Ready (503 Service Unavailable)
Returned if `VIDEO_PROCESSED_ROOT` is unset, `CURRENT` pointer is missing, or Qwen weights are absent from disk:
```json
{
  "detail": {
    "status": "not_ready",
    "reason": "Qwen3-VL model weights are not available",
    "search_configured": true,
    "processed_root": "/data/processed/videos",
    "search": { ... },
    "compute": { ... }
  }
}
```

---

## 3. Multimodal Search Endpoints

### 3.1 Text Search: `POST /api/search`
The primary retrieval interface supporting Known-Item Search (`kis`), Question Answering (`qa`), and Action Sequencing (`trake`).

#### Request Schema (`SearchRequest`)
```json
{
  "query": "string (max 2000 chars; required for kis/qa)",
  "query_type": "kis | qa | trake",
  "events": ["string", "string"],
  "top_k": 100,
  "temporal_refine": true,
  "query_refine": true,
  "rerank": true,
  "debug_query_plan": false
}
```

#### Field Specifications:
- `query` (string, optional/required): Search text for `kis` and `qa`. Maximum 2000 characters.
- `query_type` (string, required): One of `"kis"`, `"qa"`, `"trake"`. Defaults to `"kis"`.
- `events` (list of strings, required for `trake`): Ordered list of 1 to 20 event descriptions. Each string non-empty, max 2000 chars.
- `top_k` (integer, optional): Number of candidate results to return. Minimum 1, maximum 1000. Default 100.
- `temporal_refine` (boolean, optional): Toggles high-FPS local refinement in legacy pipeline. Ignored in production Qwen runtime.
- `query_refine` (boolean, optional): Toggles LLM query decomposition in legacy pipeline. Ignored in production Qwen runtime.
- `rerank` (boolean, optional): Toggles candidate reranking in legacy pipeline. Ignored in production Qwen runtime.
- `debug_query_plan` (boolean, optional): When true, attaches query plan execution timings to response.

---

### 3.2 Known-Item Search (`query_type: "kis"`)

#### Request Example
```http
POST /api/search HTTP/1.1
Host: 127.0.0.1:8000
Content-Type: application/json

{
  "query": "người phụ nữ mặc áo dài đỏ trên phố đi bộ",
  "query_type": "kis",
  "top_k": 2
}
```

#### Response (200 OK)
```json
{
  "results": [
    {
      "rank": 1,
      "video_id": "L24_V018",
      "frame_id": 14250,
      "source_frame_index_zero_based": 14250,
      "frame_uid": "L24_V018:000014250",
      "timestamp_seconds": 475.0,
      "score": 0.812450,
      "visual_score": 0.841200,
      "ocr_score": 0.0,
      "asr_score": 0.0,
      "ocr_evidence": "",
      "asr_evidence": "",
      "video_url": "/api/video/L24_V018",
      "video_ocr_score": 0.421050,
      "video_asr_score": 0.0,
      "ocr_hit_frame_uid": "L24_V018:000003000",
      "ocr_hit_timestamp_seconds": 100.0,
      "video_ocr_evidence": "HTV9 07:00:00 PHỐ ĐI BỘ NGUYỄN HUỆ",
      "video_asr_evidence": ""
    },
    {
      "rank": 2,
      "video_id": "L21_V003",
      "frame_id": 8100,
      "source_frame_index_zero_based": 8100,
      "frame_uid": "L21_V003:000008100",
      "timestamp_seconds": 270.0,
      "score": 0.741920,
      "visual_score": 0.782100,
      "ocr_score": 0.0,
      "asr_score": 0.0,
      "ocr_evidence": "",
      "asr_evidence": "",
      "video_url": "/api/video/L21_V003",
      "video_ocr_score": 0.0,
      "video_asr_score": 0.0,
      "ocr_hit_frame_uid": null,
      "ocr_hit_timestamp_seconds": null,
      "video_ocr_evidence": "",
      "video_asr_evidence": ""
    }
  ]
}
```

---

### 3.3 Video Question Answering (`query_type: "qa"`)

#### Request Example
```http
POST /api/search HTTP/1.1
Host: 127.0.0.1:8000
Content-Type: application/json

{
  "query": "Biển số xe cứu thương xuất hiện trong đoạn video là gì?",
  "query_type": "qa",
  "top_k": 2
}
```

#### Response (200 OK)
Each candidate row is augmented with question answer synthesis fields:
```json
{
  "results": [
    {
      "rank": 1,
      "video_id": "L22_V015",
      "frame_id": 3600,
      "source_frame_index_zero_based": 3600,
      "frame_uid": "L22_V015:000003600",
      "timestamp_seconds": 120.0,
      "score": 0.892010,
      "visual_score": 0.791500,
      "ocr_score": 0.950000,
      "asr_score": 0.0,
      "ocr_evidence": "CẤP CỨU 115 51B-123.45",
      "asr_evidence": "",
      "answer": "51B-123.45",
      "answer_backend": "remote_llm",
      "answer_model": "agent-lite",
      "answer_status": "ok",
      "video_url": "/api/video/L22_V015",
      "video_ocr_score": 0.950000,
      "video_asr_score": 0.0,
      "video_ocr_evidence": "CẤP CỨU 115 51B-123.45",
      "video_asr_evidence": ""
    },
    {
      "rank": 2,
      "video_id": "L22_V015",
      "frame_id": 3900,
      "source_frame_index_zero_based": 3900,
      "frame_uid": "L22_V015:000003900",
      "timestamp_seconds": 130.0,
      "score": 0.851000,
      "visual_score": 0.760000,
      "ocr_score": 0.800000,
      "asr_score": 0.0,
      "ocr_evidence": "XE CỨU THƯƠNG",
      "asr_evidence": "",
      "answer": "XE CỨU THƯƠNG",
      "answer_backend": "extractive",
      "answer_model": "none",
      "answer_status": "disabled",
      "video_url": "/api/video/L22_V015",
      "video_ocr_score": 0.950000,
      "video_asr_score": 0.0,
      "video_ocr_evidence": "CẤP CỨU 115 51B-123.45",
      "video_asr_evidence": ""
    }
  ]
}
```

#### QA Synthesis Behavior & Status Notes:
- **`answer_backend`**: `"remote_llm"` when synthesized by external model; `"extractive"` when using local text snippet.
- **`answer_status`**:
  - `"ok"`: Successfully synthesized concise Vietnamese answer ($\le 100$ characters).
  - `"abstained"`: Model returned `"Không đủ bằng chứng."`.
  - `"disabled"`: Candidate rank $> 	ext{QA\_ANSWER\_REMOTE\_TOP\_N}$ (production budget target: **2**).
  - `"no_evidence"`: No OCR or ASR evidence found for candidate.
  - `"remote_error_fallback"`: Upstream network timeout, 502 Bad Gateway, or invalid JSON. Reverts safely to extractive answer.

---

### 3.4 Action Sequencing (`query_type: "trake"`)

#### Request Example
```http
POST /api/search HTTP/1.1
Host: 127.0.0.1:8000
Content-Type: application/json

{
  "query_type": "trake",
  "events": [
    "xe ô tô màu trắng rẽ phải",
    "xe ô tô màu trắng dừng trước đèn đỏ",
    "người đi bộ qua đường"
  ],
  "top_k": 100
}
```

#### Success Response (200 OK)
Returns single best matching video satisfying monotonic temporal order across all events:
```json
{
  "results": [
    {
      "video_id": "L25_V004",
      "frame_id": 1200,
      "frame_ids": [1200, 2400, 3900],
      "source_frame_index_zero_based": 1200,
      "frame_uid": "L25_V004:000001200",
      "timestamp_seconds": 40.0,
      "score": 0.781200,
      "visual_score": 0.812000,
      "ocr_score": 0.0,
      "asr_score": 0.0,
      "events": [
        {"frame_id": 1200},
        {"frame_id": 2400},
        {"frame_id": 3900}
      ],
      "video_url": "/api/video/L25_V004",
      "video_ocr_score": 0.0,
      "video_asr_score": 0.0,
      "video_ocr_evidence": "",
      "video_asr_evidence": ""
    }
  ]
}
```

#### Failure Response: Documented Multi-Event Invariant (400 Bad Request)
When no single video in the database contains all requested events in strictly ascending chronological frame order, the backend intentionally raises `ValueError`:
```http
HTTP/1.1 400 Bad Request
Content-Type: application/json

{
  "detail": "TRAKE: no single video covers every event in increasing frame order"
}
```
**Operational Note**: This is the verified, intended contract behavior. Clients must handle HTTP 400 as an indication that the multi-event hypothesis could not be fulfilled as a coherent action sequence within a single video.

---

## 4. Image-to-Frame Search: `POST /api/search/image`

Experimental visual query endpoint allowing users to upload an image and retrieve visually matching video keyframes.

#### Request Parameters
- Query Parameter `top_k` (integer, optional): Number of results (1–1000, default 100).
- Query Parameter `raw` (boolean, optional): Bypass deduplication when true (default false).
- Supported Media Types: `image/jpeg`, `image/png`, `image/webp`, or `multipart/form-data` with form field `file`.
- Size Limit: Maximum **15 MiB** (`MAX_IMAGE_UPLOAD_BYTES = 15 * 1024 * 1024`).

#### Error Status Code Contract:
1. **HTTP 415 Unsupported Media Type**:
   Raised if `Content-Type` header is not JPEG, PNG, WebP, or multipart/form-data:
   ```json
   {
     "detail": "content type must be JPEG, PNG, WebP, or multipart/form-data"
   }
   ```

2. **HTTP 400 Bad Request**:
   - Empty upload payload: `{"detail": "empty image upload"}`.
   - Multipart missing `file` field: `{"detail": "multipart upload is missing file field"}`.
   - Malformed/unrecognized image format: `{"detail": "unsupported image format; use JPEG, PNG, or WebP"}`.
   - Corrupt or invalid image bytes: `{"detail": "invalid or corrupt image"}`.

3. **HTTP 413 Payload Too Large**:
   - Upload exceeds 15 MiB: `{"detail": "image file exceeds 15MB limit"}`.

4. **HTTP 503 Service Unavailable (Active Backend Unsupported)**:
   - When running on `SEARCH_BACKEND=qwen3_vl` (production default), image search is disabled because the packed database is indexed for text queries and GPU validation has not been performed:
   ```json
   {
     "detail": "image search is not supported by the active search backend"
   }
   ```
   - If search index is not loaded: `{"detail": "search index is not configured"}`.

---

## 5. Media Streaming & Static Asset Endpoints

### 5.1 Static Keyframe Image: `GET /api/frames/{video_id}/{filename}`
Serves pre-rendered keyframe JPEG/WebP images.

#### Request
```http
GET /api/frames/L21_V001/000000300.jpg HTTP/1.1
Host: 127.0.0.1:8000
```

#### Response (200 OK)
```http
HTTP/1.1 200 OK
Content-Type: image/jpeg
Cache-Control: public, max-age=86400

<binary image data>
```

#### Path Traversal Protection & 404
- Only extensions `.jpg`, `.webp`, `.png` are accepted.
- Requests containing `/`, `\`, or `..` immediately return `HTTP 404 Not Found`.
- Missing frame images return 404; the operator UI displays a graceful CSS placeholder card.

---

### 5.2 Raw Video Preview & HTTP 206 Streaming: `GET /api/video/{video_id}`
Provides seekable raw MP4 playback for operator verification. Supports full RFC 7233 Byte Range streaming.

#### Source Resolution Protocol (`backend/app/services/video_source.py`):
1. **Local Video Cache**: Checks `$VIDEO_CACHE_DIR/{video_id}.mp4` (default `data/videos/{video_id}.mp4`).
2. **Local Video Source**: Checks `$VIDEO_SOURCE_DIR/{video_id}.mp4` (served directly in-place, never copied).
3. **Lazy Remote Download**: If `$VIDEO_SOURCE_URL_TEMPLATE` is configured (e.g. `http://media-server/videos/{video_id}.mp4`), downloads the video into the cache directory atomically using a `.part` file with per-video threading mutexes to prevent duplicate parallel downloads.

#### Full Video Request (200 OK)
```http
GET /api/video/L21_V001 HTTP/1.1
Host: 127.0.0.1:8000
```
Response:
```http
HTTP/1.1 200 OK
Content-Type: video/mp4
Content-Length: 52428800
Accept-Ranges: bytes
Cache-Control: public, max-age=86400

<binary video stream>
```

#### Range Request: Standard Chunk (206 Partial Content)
The client requests byte range `0-1048575` (1 MiB initial buffer):
```http
GET /api/video/L21_V001 HTTP/1.1
Host: 127.0.0.1:8000
Range: bytes=0-1048575
```
Response:
```http
HTTP/1.1 206 Partial Content
Content-Type: video/mp4
Content-Range: bytes 0-1048575/52428800
Content-Length: 1048576
Accept-Ranges: bytes
Cache-Control: public, max-age=86400

<binary video chunk>
```

#### Range Request: Suffix Specification (206 Partial Content)
The client requests the final 512 KiB of the video file:
```http
GET /api/video/L21_V001 HTTP/1.1
Host: 127.0.0.1:8000
Range: bytes=-524288
```
Response:
```http
HTTP/1.1 206 Partial Content
Content-Type: video/mp4
Content-Range: bytes 51904512-52428799/52428800
Content-Length: 524288
Accept-Ranges: bytes
Cache-Control: public, max-age=86400

<binary video chunk>
```

#### Streaming Implementation:
Streams via generator `_iter_file_range(path, start, end, chunk_size=1048576)`. Only requested byte slices are read into memory, preventing memory bloat on large video collections.

#### Range Errors (416 Range Not Satisfiable):
Returned if range header syntax is invalid, if multiple comma-separated ranges are requested, or if `start >= file_size`:
```http
HTTP/1.1 416 Range Not Satisfiable
Content-Range: bytes */52428800

{"detail": "invalid byte range"}
```

---

## 6. Frontend Static Delivery Endpoints

- `GET /`: Serves `frontend/src/index.html`.
- `GET /styles/{filename}`: Serves CSS assets (`main.css`) with directory-traversal protection.
- `GET /scripts/{filename}`: Serves JavaScript modules (`api.js`, `app.js`, `shortcuts.js`) with directory-traversal protection.

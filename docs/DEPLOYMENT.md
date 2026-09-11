# Production Deployment Guide — AIC 2026 Multimodal Video Retrieval

This guide provides authoritative instructions for deploying, configuring, and operating the AIC 2026 Multimodal Retrieval service using standalone Uvicorn or Docker Compose.

---

## 1. Core Architecture & Runtime Requirements

The production system runs a lightweight, read-only search runtime over pre-ingested FAISS index generations and OCR/ASR spools:

* **Production Default Encoder:** `Qwen/Qwen3-VL-Embedding-2B` (1024-dimensional normalized vectors via Matryoshka Representation Learning).
* **Legacy Encoder:** `google/siglip2-base-patch16-224` (768-dimensional; retained solely as legacy opt-in, cannot query a Qwen index).
* **Frontend:** Vanilla HTML5 / ES6 JavaScript / CSS served directly by FastAPI or `run_dev.py`. Requires **no Node.js, npm, or build step**.
* **Process Model:** Exactly **1 Uvicorn worker** per instance (`--workers 1`) to prevent duplicating the ~4.0 GB Qwen model weights across processes.

---

## 2. Minimum Viable Run Command (Bare Metal / Local Python)

To run the search service directly with Python and Uvicorn without Docker:

### 2.1 Set Environment Variables

```bash
# ------------------------------------------------------------------------------
# REQUIRED CONFIGURATION
# ------------------------------------------------------------------------------
# 1. Select the production search backend
export SEARCH_BACKEND=qwen3_vl

# 2. Path to the processed data directory containing:
#    - index/ (with CURRENT file pointing to active generation, e.g. gen-A-...)
#    - ocr/   (extracted OCR JSON files)
#    - asr/   (extracted Whisper ASR JSON files)
export VIDEO_PROCESSED_ROOT=/opt/aic/data/aic-db-v1/runtime

# 3. Path to local Qwen3-VL-Embedding-2B weights directory containing:
#    model.safetensors, config.json, and scripts/qwen3_vl_embedding.py
export QWEN3_VL_MODEL_DIR=/opt/aic/models/Qwen3-VL-Embedding-2B

# ------------------------------------------------------------------------------
# OPTIONAL CONFIGURATION
# ------------------------------------------------------------------------------
# Local directory containing raw <video_id>.mp4 files for seekable streaming preview
export VIDEO_SOURCE_DIR=/opt/aic/data/videos

# Optional remote download template if videos are fetched on demand
# export VIDEO_SOURCE_URL_TEMPLATE="https://media-server/videos/{video_id}.mp4"
# export VIDEO_CACHE_DIR=/opt/aic/cache/videos

# Grounded QA synthesis settings (Production default is extractive)
export QA_ANSWER_BACKEND=extractive
# If testing experimental remote LLM:
# export QA_ANSWER_BACKEND=remote_llm
# export QA_ANSWER_URL=http://llm-gateway:8000/v1
# export QA_ANSWER_MODEL=agent-lite
# export QA_ANSWER_API_KEY=secret-token
# export QA_ANSWER_REMOTE_TOP_N=2
```

### 2.2 Execute Uvicorn

```bash
# Launch FastAPI application with single worker
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --workers 1
```

Once started, the Chi Lăng web console is immediately accessible at `http://localhost:8000`.

---

## 3. Docker Compose Deployment

The repository provides production-grade container orchestration configurations.

### 3.1 Actual Compose Service Names

> [!IMPORTANT]
> The primary [`docker-compose.yml`](file:///tmp/docs-agy2/docker-compose.yml) defines the following service names:
> * **`aic`**: The main FastAPI application service (exposes port 8000).
> * **`aic-cli`**: Interactive CLI helper (under the `tools` profile, executes `projectctl.py`).
>
> *(Note: The release file `docker-compose.release.yml` uses service names `backend` and `worker` for pre-built registry images).*

### 3.2 Standard Compose Startup (CPU Mode)

```bash
# 1. Setup environment file
cp .env.example .env
# Edit .env with valid paths for VIDEO_PROCESSED_ROOT and QWEN3_VL_MODEL_DIR

# 2. Build and start the service in background
docker compose build aic
docker compose up -d aic

# 3. Inspect container logs
docker compose logs -f aic
```

To run CLI maintenance commands inside the container environment:

```bash
docker compose --profile tools run --rm aic-cli doctor
```

---

## 4. GPU Acceleration & Hardware Profiles (GPU Required)

Running deep learning inference on GPU accelerates query embedding and offline processing. Note the service name targets for each compose file:

### 4.1 Production GPU Override (`docker-compose.gpu.yml`)

The file [`docker-compose.gpu.yml`](file:///tmp/docs-agy2/docker-compose.gpu.yml) provides NVIDIA container device reservations for the **`aic`** service:

```yaml
services:
  aic:
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    environment:
      - CUDA_VISIBLE_DEVICES=all
      - TORCH_CUDA_ARCH_LIST=Auto
```

**Startup Command:**
```bash
# Requires host with NVIDIA driver and nvidia-container-toolkit installed
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d aic
```

### 4.2 Legacy CUDA Override (`docker-compose.cuda.yml`)

The file [`docker-compose.cuda.yml`](file:///tmp/docs-agy2/docker-compose.cuda.yml) targets the **`backend`** and **`worker`** services used in `docker-compose.release.yml`:

```bash
# Used only with docker-compose.release.yml
docker compose -f docker-compose.release.yml -f docker-compose.cuda.yml up -d backend
```

---

## 5. Health Check Endpoints & Monitoring

The service exposes comprehensive health and readiness probes in `backend/app/main.py`:

### 5.1 Liveness Probe (`GET /health` or `GET /health/live`)

* **Endpoint:** `http://127.0.0.1:8000/health/live`
* **Response Status:** HTTP 200 OK
* **Payload:**
  ```json
  {
    "status": "ok",
    "search_configured": true,
    "processed_root": "/opt/aic/data/aic-db-v1/runtime",
    "search": {
      "backend": "qwen3_vl",
      "configured": true,
      "initialized": true,
      "generation_id": "gen-A-b531063ff8ce48e1b0d62739fb290c23",
      "dimension": 1024,
      "ocr_records": 94499,
      "asr_records": 14867,
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
      "device": "cpu",
      "cuda_available": false
    }
  }
  ```

### 5.2 Readiness Probe (`GET /health/ready`)

* **Endpoint:** `http://127.0.0.1:8000/health/ready`
* **Verification Gates:**
  1. Verifies that the search handler is configured.
  2. Confirms active index generation (`CURRENT`) exists and is valid.
  3. Verifies that the Qwen3-VL model weights (`model.safetensors`, `config.json`, `scripts/qwen3_vl_embedding.py`) exist on disk.
  4. Confirms the FAISS index dimension and encoder match `qwen3_vl`.
* **Response Codes:**
  * `200 OK`: `{"status": "ready", "generation_id": "gen-A-..."}`
  * `503 Service Unavailable`: Returned if any artifact is missing or corrupted.

---

## 6. Video Preview & HTTP 206 Streaming

When `VIDEO_SOURCE_DIR` is set to a directory of raw `.mp4` videos:
* The backend enables `GET /api/video/{video_id}`.
* Requests support standard HTTP `Range: bytes=start-end` headers.
* Streamed using a 1 MiB chunk generator (`_iter_file_range`), ensuring raw video files are never loaded into application memory.
* Clicking any search result in the frontend opens a modal player automatically seeking to `timestamp_seconds`.

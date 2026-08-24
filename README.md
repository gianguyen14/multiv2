<div align="center">

# 🎬 Unified Video Retrieval

### Local-first multimodal video retrieval

**Text → Frames · Video Q&A · TRAKE · Image Search · OCR · ASR · Temporal Refinement**

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Recommended-2496ED?logo=docker&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-API-009688?logo=fastapi&logoColor=white)
![FAISS](https://img.shields.io/badge/FAISS-Vector_Search-0467DF)
![SigLIP2](https://img.shields.io/badge/SigLIP2-768D-FF6F00)
![Status](https://img.shields.io/badge/Release-1.1.0--rc2_prevalidation-orange)

**English · [Tiếng Việt](README.vi.md)**

*A local/offline-friendly retrieval stack built to find the right video, the right frame, and the right temporal sequence.*

</div>

---

## ✨ What this system does

| Mode | Input | Output | Main signals |
|---|---|---|---|
| 🔎 **Textual KIS** | Natural-language description | Ranked `video_id`, `frame_id` | SigLIP2 + OCR + ASR + fusion |
| 💬 **Video Q&A** | Question about video content | Evidence frames for answer handling | Visual + OCR + ASR evidence |
| 🧭 **TRAKE** | Ordered semantic events | One video + ordered keyframes | Coarse retrieval + temporal refinement + DP alignment |
| 🖼️ **Image Search** | Query image | Visually similar frames | SigLIP2 image embeddings |
| 🔤 **OCR / ASR** | Frames + audio | Searchable text evidence | Tesseract + Faster Whisper |

The retrieval layer supports Vietnamese and English query variants, evidence-aware reranking, temporal deduplication, and an optional local QueryRefiner with deterministic fallback.

---

## 🧠 Architecture at a glance

```text
                               ┌─────────────────────┐
                               │      USER QUERY     │
                               │ text / image / QA   │
                               │ ordered TRAKE events│
                               └──────────┬──────────┘
                                          │
                                          ▼
                              ┌───────────────────────┐
                              │   Query Intelligence  │
                              │ VI/EN · lexical · LLM │
                              └───────────┬───────────┘
                                          │
                    ┌─────────────────────┼─────────────────────┐
                    │                     │                     │
                    ▼                     ▼                     ▼
             ┌─────────────┐       ┌─────────────┐       ┌─────────────┐
             │   SigLIP2   │       │     OCR     │       │     ASR     │
             │ visual/text │       │  Tesseract  │       │   Whisper   │
             └──────┬──────┘       └──────┬──────┘       └──────┬──────┘
                    │                     │                     │
                    └─────────────────────┼─────────────────────┘
                                          ▼
                                ┌───────────────────┐
                                │ Candidate Fusion  │
                                │ RRF + reranking   │
                                └─────────┬─────────┘
                                          │
                        ┌─────────────────┼─────────────────┐
                        │                 │                 │
                        ▼                 ▼                 ▼
                      KIS               Q&A              TRAKE
                                                          │
                                                          ▼
                                                Dense TemporalRefiner
                                                          │
                                                          ▼
                                               Ordered DP alignment
```

### Frame identity is authoritative

A result `frame_id` is the **zero-based ordinal emitted by sequential PyAV decoding in display order**.

```python
for frame_id, frame in enumerate(container.decode(stream)):
    ...
```

The system does **not** reconstruct authoritative frame IDs using `timestamp × FPS`. This keeps ingestion, retrieval, evaluation, and temporal refinement aligned with the source video.

---

# 🚀 Quick Start with Docker

Docker is the recommended runtime on **Linux** and **Windows 10/11**.

The image contains the core Linux dependencies used by the application, including Python 3.12, FFmpeg, Tesseract, Git, GCC and G++.

## Requirements

### Linux

- Docker Engine
- Docker Compose v2 (`docker compose`)

### Windows 10/11

- Docker Desktop
- WSL2 backend enabled
- Git for Windows or Git inside WSL2

For Windows, keeping videos, model caches, and processed artifacts inside the cloned repository is the simplest setup.

## 1. Clone

```bash
git clone git@github.com:gianguyen14/multiv2.git
cd multiv2
```

> The repository is private, so configure GitHub SSH access on the machine first.

## 2. Create local storage

Linux / WSL2:

```bash
mkdir -p data/videos data/processed models cache logs
```

PowerShell:

```powershell
New-Item -ItemType Directory -Force data/videos, data/processed, models, cache, logs
```

Put source videos in:

```text
data/videos/
```

Default Docker mounts:

```text
Host                 Container
----------------------------------------------
./data         -->   /data              read/write
./models       -->   /models            read/write
./cache        -->   /cache             read/write
./logs         -->   /logs              read/write
```

## 3. Build the image

```bash
docker compose build
```

## 4. Check the runtime

```bash
docker compose --profile tools run --rm aic-cli env --check
docker compose --profile tools run --rm aic-cli doctor
```

`doctor` is the main readiness check before preprocessing data or serving queries.

## 5. Prepare the models

Prepare the default visual and ASR models:

```bash
docker compose --profile tools run --rm aic-cli models --prepare
```

Inspect model availability:

```bash
docker compose --profile tools run --rm aic-cli models
```

Optional local QueryRefiner model:

```bash
docker compose --profile tools run --rm aic-cli models --prepare --query-refiner
```

If the QueryRefiner model is unavailable, search can use the deterministic fallback path.

## 6. Preprocess and index videos

```bash
docker compose --profile tools run --rm aic-cli preprocess /data/videos
```

This runs the configured ingestion pipeline and publishes searchable artifacts under the mounted processed directory. Compatible completed work can be resumed instead of repeated from scratch.

Check state afterward:

```bash
docker compose --profile tools run --rm aic-cli status
```

## 7. Start the application

```bash
docker compose up -d aic
```

Check containers:

```bash
docker compose ps
```

Follow logs:

```bash
docker compose logs -f aic
```

Open the web interface:

```text
http://127.0.0.1:8000
```

Health endpoints:

```text
http://127.0.0.1:8000/health/live
http://127.0.0.1:8000/health/ready
http://127.0.0.1:8000/health
```

Linux / WSL2:

```bash
curl http://127.0.0.1:8000/health/ready
```

PowerShell:

```powershell
Invoke-WebRequest http://127.0.0.1:8000/health/ready
```

---

# 🎮 How to use the system

There are three main interfaces:

1. **Web UI** — the easiest way to search and inspect results interactively.
2. **`projectctl.py` CLI** — useful for development, experiments, batch work and validation.
3. **FastAPI** — useful for integrations and custom frontends.

## Option A — Web UI

Start the application service:

```bash
docker compose up -d aic
```

Then open:

```text
http://127.0.0.1:8000
```

The frontend is served directly by FastAPI, so the normal Docker workflow does not require a separate frontend server.

---

## Option B — CLI with `projectctl.py`

Inside Docker, use the `aic-cli` tools-profile service:

```bash
docker compose --profile tools run --rm aic-cli --help
```

### 🔎 Textual KIS

Find frames matching a description:

```bash
docker compose --profile tools run --rm aic-cli \
  kis "một người phụ nữ mặc áo dài" --top-k 20
```

Example queries:

```text
"a red car crossing an intersection"
"người đàn ông đang đứng trước màn hình lớn"
"biển số xe 79H-6072"
```

### 💬 Video Q&A

Retrieve evidence for a question:

```bash
docker compose --profile tools run --rm aic-cli \
  qa "Nhiệt độ hiển thị trên màn hình là bao nhiêu?" --top-k 20
```

Q&A retrieval combines available visual, OCR and ASR evidence before downstream answer handling.

### 🧭 TRAKE

TRAKE searches for an **ordered event sequence inside the same video**.

Pipe-separated syntax:

```bash
docker compose --profile tools run --rm aic-cli \
  trake "người đứng yên | bắt đầu chạy | nhảy lên | tiếp đất" --top-k 30
```

JSON syntax:

```bash
docker compose --profile tools run --rm aic-cli \
  trake '["đứng", "chạy đà", "nhảy", "tiếp đất"]' --top-k 30
```

TRAKE can decode and search more densely around promising temporal regions, then enforce monotonic event ordering.

Disable dense temporal refinement for diagnostics:

```bash
docker compose --profile tools run --rm aic-cli \
  trake "event one | event two" --no-temporal-refine
```

### 🖼️ Image-to-frame search

If the query image is available inside the container:

```bash
docker compose --profile tools run --rm aic-cli \
  image-search /data/videos/query.jpg --top-k 20
```

You can also upload an image through the HTTP API from the host.

### 🧠 Inspect the query plan

```bash
docker compose --profile tools run --rm aic-cli \
  query-plan "biển số xe 79H-6072" --task kis --json
```

This is useful for understanding query expansion, lexical terms, and the local QueryRefiner path.

### 🩺 Diagnostics

```bash
docker compose --profile tools run --rm aic-cli doctor
docker compose --profile tools run --rm aic-cli status
docker compose --profile tools run --rm aic-cli info
docker compose --profile tools run --rm aic-cli smoke
```

### 📦 Dataset validation

```bash
docker compose --profile tools run --rm aic-cli \
  dataset verify /data/videos
```

Run the repository's representative validation workflow:

```bash
docker compose --profile tools run --rm aic-cli validate-dataset
```

---

## Option C — HTTP API

The active backend exposes the retrieval API from `backend.app.main`.

### KIS request

```bash
curl -X POST http://127.0.0.1:8000/api/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "một người phụ nữ mặc áo dài",
    "query_type": "kis",
    "top_k": 20
  }'
```

### Q&A request

```bash
curl -X POST http://127.0.0.1:8000/api/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Nhiệt độ hiển thị là bao nhiêu?",
    "query_type": "qa",
    "top_k": 20
  }'
```

### TRAKE request

```bash
curl -X POST http://127.0.0.1:8000/api/search \
  -H "Content-Type: application/json" \
  -d '{
    "query_type": "trake",
    "events": ["đứng", "chạy", "nhảy", "tiếp đất"],
    "top_k": 30,
    "temporal_refine": true,
    "query_refine": true,
    "rerank": true
  }'
```

### Image search request

```bash
curl -X POST "http://127.0.0.1:8000/api/search/image?top_k=20" \
  -F "file=@query.jpg"
```

Supported image formats are JPEG, PNG and WebP. The API limits image uploads to 15 MB.

### Debug the QueryPlan through the API

```bash
curl -X POST http://127.0.0.1:8000/api/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "biển số xe 79H-6072",
    "query_type": "kis",
    "top_k": 20,
    "debug_query_plan": true
  }'
```

---

# 🧩 Typical end-to-end workflow

```text
1. Put videos in data/videos/
           │
           ▼
2. docker compose build
           │
           ▼
3. models --prepare
           │
           ▼
4. doctor
           │
           ▼
5. preprocess /data/videos
           │
           ▼
6. status
           │
           ▼
7. docker compose up -d aic
           │
           ▼
8. Open Web UI / run KIS / Q&A / TRAKE / image search
           │
           ▼
9. benchmark / tune
```

---

# 🐳 Docker configuration

## Override data/model locations

Compose supports:

```text
AIC_DATA_DIR
AIC_MODELS_DIR
AIC_CACHE_DIR
AIC_LOGS_DIR
AIC_BIND_IP
AIC_PORT
```

Example on Linux / WSL2:

```bash
AIC_DATA_DIR=/mnt/aic-data \
AIC_MODELS_DIR=/mnt/retrieval-models \
AIC_CACHE_DIR=/mnt/retrieval-cache \
AIC_LOGS_DIR=/mnt/retrieval-logs \
docker compose up -d aic
```

For Windows, repository-relative paths are recommended unless Docker Desktop has access to the external drive or path.

## Offline mode

After required weights are already cached in the mounted model directory:

Linux / WSL2:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 docker compose up -d aic
```

PowerShell:

```powershell
$env:HF_HUB_OFFLINE="1"
$env:TRANSFORMERS_OFFLINE="1"
docker compose up -d aic
```

Verify offline model readiness:

```bash
docker compose --profile tools run --rm aic-cli models --verify-offline
```

## Stop / rebuild

```bash
docker compose down
```

After source changes:

```bash
docker compose up --build -d aic
```

---

# ⚡ NVIDIA GPU / CUDA

The repository includes a CUDA Compose override:

```bash
docker compose \
  -f docker-compose.yml \
  -f docker-compose.gpu.yml \
  up --build -d aic
```

Host prerequisites:

- **Linux:** compatible NVIDIA driver and Docker GPU/container runtime support.
- **Windows:** NVIDIA driver with WSL2 GPU support and Docker Desktop using the WSL2 backend.

Verify Docker GPU access before running the CUDA override.

> ⚠️ **RC2 OCR note:** the CUDA override still contains experimental GPU OCR routing. PaddleOCR GPU is **not** part of the accepted RC2 runtime path yet. For RC2 reproducibility, use the Tesseract OCR path until CUDA/Paddle integration is validated separately.

---

# 🔬 Retrieval design

### Sparse globally, dense locally

The permanent FAISS index stays sparse. For TRAKE, dense decoding and embedding happen only inside bounded temporal regions around promising coarse hits.

### Evidence before cosmetics

OCR and ASR evidence can promote candidates that visual similarity alone would miss, including text, numbers, signage and spoken details.

### Deterministic ordering

Fusion and reranking are designed to preserve stable ordering and deterministic tie-breaking wherever possible.

### Fail-open optional intelligence

Optional query-refinement components can fall back to deterministic parsing instead of making the whole retrieval path unavailable.

---

# 🛠️ Development without Docker

```bash
python -m pip install -e .
python projectctl.py env --check
python projectctl.py doctor
pytest
```

Run the local server:

```bash
python projectctl.py dev
```

Open:

```text
http://127.0.0.1:8000
```

Use the checked-out revision's help output as the authoritative CLI reference:

```bash
python projectctl.py --help
```

---

# 📁 Repository layout

```text
backend/       active application and retrieval pipeline
frontend/      FastAPI-served operator UI
eval/          evaluation and benchmark utilities
scripts/       diagnostics, validation, dataset and experiment helpers
tests/         unit and integration tests
docs/          architecture, deployment and engineering notes
projectctl.py  operator CLI / project entry point
```

## Documentation

- [Project Control CLI](docs/projectctl.md)
- [Architecture](ARCHITECTURE.md)
- [Deployment](docs/DEPLOYMENT.md)
- [Engineering / Agent Rules](AGENTS.md)

---

# 🎯 Current status

**`1.1.0-rc2` prevalidation source**

The release candidate is being validated against the target NVIDIA GPU environment before promotion. Performance claims should be based on representative datasets and recorded validation results.

<div align="center">

### Built for retrieval quality, temporal correctness and reproducible experimentation.

**KIS · Q&A · TRAKE · OCR · ASR · SigLIP2 · FAISS · FastAPI**

</div>

<div align="center">

# 🎬 Unified AIC Retrieval

### Multimodal Video Retrieval for Ho Chi Minh City AI Challenge 2026

**Text → Frames · Video Q&A · TRAKE · Image Search · OCR · ASR · Temporal Refinement**

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Recommended-2496ED?logo=docker&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-API-009688?logo=fastapi&logoColor=white)
![FAISS](https://img.shields.io/badge/FAISS-Vector_Search-0467DF)
![SigLIP2](https://img.shields.io/badge/SigLIP2-768D-FF6F00)
![Status](https://img.shields.io/badge/Release-1.1.0--rc2_prevalidation-orange)

*A local/offline-friendly retrieval stack built to find the right video, the right frame, and the right temporal sequence — fast.*

</div>

---

## ✨ What this system does

| Mode | Input | Output | Main signals |
|---|---|---|---|
| 🔎 **Textual KIS** | Natural-language description | Ranked `video_id`, `frame_id` | SigLIP2 + OCR + ASR + fusion |
| 💬 **Video Q&A** | Question about video content | Evidence frames + answer path | Visual + OCR + ASR evidence |
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
mkdir -p data/test-videos data/processed models
```

PowerShell:

```powershell
New-Item -ItemType Directory -Force data/test-videos, data/processed, models
```

Put your source videos in:

```text
data/test-videos/
```

Default Docker mounts:

```text
Host                       Container
------------------------------------------------
./data/test-videos   -->   /data/videos       read-only
./data/processed     -->   /data/processed    read/write
./models             -->   /models            read/write
```

## 3. Build the image

```bash
docker compose build
```

## 4. Check the runtime

```bash
docker compose --profile tools run --rm worker env --check
docker compose --profile tools run --rm worker doctor
```

## 5. Prepare the models

Prepare the default visual + ASR models:

```bash
docker compose --profile tools run --rm worker models --prepare
```

Inspect model availability:

```bash
docker compose --profile tools run --rm worker models
```

Optional local query-refiner model:

```bash
docker compose --profile tools run --rm worker models --prepare --query-refiner
```

If the QueryRefiner model is unavailable, search can use the deterministic fallback path.

## 6. Preprocess / index videos

```bash
docker compose --profile tools run --rm worker preprocess /data/videos
```

This performs the configured ingestion pipeline and publishes searchable artifacts under the mounted processed directory. Preprocessing resumes compatible work when possible.

Check state afterward:

```bash
docker compose --profile tools run --rm worker status
```

## 7. Start the application

```bash
docker compose up -d backend
```

Check containers:

```bash
docker compose ps
```

Follow logs:

```bash
docker compose logs -f backend
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

You can use the project in three ways:

1. **Web UI** — easiest interactive workflow.
2. **`projectctl.py` CLI** — best for development, experiments, batch work and validation.
3. **FastAPI** — best for external applications and custom frontends.

## Option A — Web UI

Start the backend:

```bash
docker compose up -d backend
```

Then open:

```text
http://127.0.0.1:8000
```

The frontend is served directly by FastAPI, so no second frontend server is required for the normal Docker workflow.

---

## Option B — CLI with `projectctl.py`

Inside Docker, use the `worker` service:

```bash
docker compose --profile tools run --rm worker --help
```

### 🔎 Textual KIS

Find frames matching a description:

```bash
docker compose --profile tools run --rm worker \
  kis "một người phụ nữ mặc áo dài" --top-k 20
```

Useful for queries such as:

```text
"a red car crossing an intersection"
"người đàn ông đang đứng trước màn hình lớn"
"biển số xe 79H-6072"
```

### 💬 Video Q&A

Retrieve evidence for a question:

```bash
docker compose --profile tools run --rm worker \
  qa "Nhiệt độ hiển thị trên màn hình là bao nhiêu?" --top-k 20
```

Q&A retrieval combines available visual, OCR and ASR evidence before downstream answer handling.

### 🧭 TRAKE

TRAKE searches for an **ordered event sequence inside the same video**.

Pipe-separated syntax:

```bash
docker compose --profile tools run --rm worker \
  trake "người đứng yên | bắt đầu chạy | nhảy lên | tiếp đất" --top-k 30
```

JSON syntax:

```bash
docker compose --profile tools run --rm worker \
  trake '["đứng", "chạy đà", "nhảy", "tiếp đất"]' --top-k 30
```

TRAKE can use dense temporal refinement around coarse candidate regions and then enforce monotonic event ordering.

Disable dense temporal refinement for diagnostics:

```bash
docker compose --profile tools run --rm worker \
  trake "event one | event two" --no-temporal-refine
```

### 🖼️ Image-to-frame search

If the query image is available inside the container:

```bash
docker compose --profile tools run --rm worker \
  image-search /data/videos/query.jpg --top-k 20
```

Or use the HTTP image endpoint from the host; see the API examples below.

### 🧠 Inspect the query plan

```bash
docker compose --profile tools run --rm worker \
  query-plan "biển số xe 79H-6072" --task kis --json
```

This is useful for understanding query expansion, lexical terms, and the local QueryRefiner path.

### 🩺 Diagnostics

```bash
docker compose --profile tools run --rm worker doctor
docker compose --profile tools run --rm worker status
docker compose --profile tools run --rm worker info
docker compose --profile tools run --rm worker smoke
```

### 📦 Dataset validation

```bash
docker compose --profile tools run --rm worker \
  dataset verify /data/videos
```

For the repository's representative validation workflow:

```bash
docker compose --profile tools run --rm worker validate-dataset
```

### 📊 Evaluation

Competition-style internal evaluation:

```bash
docker compose --profile tools run --rm worker \
  evaluate --competition --ground-truth /path/to/ground_truth
```

The checked-in competition scorer is an **internal provisional metric**, not an official competition scoring claim.

---

## Option C — HTTP API

The active service exposes the retrieval API from `backend.app.main`.

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
1. Put videos in data/test-videos/
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
7. docker compose up -d backend
           │
           ▼
8. Open Web UI / run KIS / Q&A / TRAKE / image search
           │
           ▼
9. evaluate / benchmark / tune
```

---

# 🐳 Docker configuration

## Override data/model locations

Compose supports:

```text
VIDEOS_DIR
PROCESSED_DIR
MODELS_DIR
HF_HUB_OFFLINE
TRANSFORMERS_OFFLINE
```

Example on Linux / WSL2:

```bash
VIDEOS_DIR=/mnt/videos \
PROCESSED_DIR=/mnt/aic-processed \
MODELS_DIR=/mnt/aic-models \
docker compose up -d backend
```

For Windows, repository-relative paths are recommended unless Docker Desktop has access to the external drive/path.

## Offline mode

After required weights are already cached in the mounted model directory:

Linux / WSL2:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 docker compose up -d backend
```

PowerShell:

```powershell
$env:HF_HUB_OFFLINE="1"
$env:TRANSFORMERS_OFFLINE="1"
docker compose up -d backend
```

Verify offline model readiness:

```bash
docker compose --profile tools run --rm worker models --verify-offline
```

## Stop / rebuild

```bash
docker compose down
```

After source changes:

```bash
docker compose up --build -d backend
```

---

# ⚡ NVIDIA GPU / CUDA

The repository includes a CUDA Compose override:

```bash
docker compose \
  -f docker-compose.yml \
  -f docker-compose.cuda.yml \
  up --build -d backend
```

Host prerequisites:

- **Linux:** compatible NVIDIA driver and Docker GPU/container runtime support.
- **Windows:** NVIDIA driver with WSL2 GPU support and Docker Desktop using the WSL2 backend.

Verify Docker GPU access before running the CUDA override.

> ⚠️ **RC2 OCR note:** the CUDA override still contains experimental GPU OCR routing. PaddleOCR GPU is **not** part of the accepted RC2 runtime path yet. For RC2 reproducibility, use the Tesseract OCR path until CUDA/Paddle integration is validated separately.

---

# 🔬 Retrieval design

### Sparse globally, dense locally

The permanent FAISS index stays sparse. For TRAKE, dense decoding/embedding happens only inside bounded temporal regions around promising coarse hits.

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

The release candidate is being validated against the target NVIDIA GPU environment before promotion. Performance claims should be based on representative competition data and recorded validation results.

<div align="center">

### Built for retrieval quality, temporal correctness and reproducible experimentation.

**KIS · Q&A · TRAKE · OCR · ASR · SigLIP2 · FAISS · FastAPI**

</div>

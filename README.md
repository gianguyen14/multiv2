# Unified AIC Retrieval

Multimodal video retrieval system for **AIC 2026**, focused on fast and reliable retrieval from long-form video collections.

The project combines visual embeddings, speech, OCR, temporal reasoning, and evidence-aware ranking in a local/offline-friendly pipeline.

## What it supports

- **Textual KIS** — retrieve the most relevant video frames from a natural-language description.
- **Video Q&A** — retrieve supporting evidence and return the associated answer.
- **TRAKE** — retrieve an ordered sequence of semantic keyframes for multi-event temporal queries.
- **Image-to-frame search** — use an image as the query against indexed video frames.
- **OCR / ASR evidence** — combine text visible in frames with spoken content from video audio.

## Core stack

- **SigLIP2** — visual/text embeddings.
- **FAISS** — vector search over normalized frame embeddings.
- **Faster Whisper** — speech recognition.
- **Tesseract OCR** — OCR baseline for the current release path.
- **PyAV / FFmpeg** — authoritative video decoding and media processing.
- **FastAPI** — retrieval API.

## Retrieval pipeline

```text
Video
  │
  ├─ Decode / sample frames
  ├─ Visual embedding (SigLIP2)
  ├─ ASR (Faster Whisper)
  ├─ OCR (Tesseract)
  │
  ▼
FAISS + metadata index
  │
  ▼
Query intelligence
  │
  ├─ visual / semantic path
  ├─ OCR lexical path
  ├─ ASR lexical path
  └─ Vietnamese / English query variants
  │
  ▼
Fusion + evidence-aware reranking
  │
  ├─ KIS
  ├─ Q&A
  └─ TRAKE + temporal refinement
```

## Important frame-ID invariant

Authoritative frame IDs are **zero-based frame ordinals produced by sequential PyAV decoding in display order**.

They are never reconstructed from `timestamp × FPS`. This keeps indexed frames, evaluation output, and temporal refinement aligned with the source video.

## TRAKE coarse-to-fine retrieval

TRAKE first retrieves sparse global candidates, then performs dense temporal refinement only around promising regions. Ordered events are aligned monotonically inside the same video.

This keeps the permanent index compact while allowing higher temporal resolution where it matters.

## Current status

**1.1.0-rc2 prevalidation source**

The current release candidate is being validated on the target NVIDIA GPU environment before promotion. PaddleOCR GPU is not part of the accepted RC2 runtime path; the current OCR baseline is Tesseract.

## Docker quick start

Docker is the recommended way to run the project because the image already includes the Linux runtime dependencies used by the application, including Python 3.12, FFmpeg, Tesseract, Git, GCC and G++.

### Requirements

**Linux**

- Docker Engine
- Docker Compose v2 (`docker compose`)

**Windows 10/11**

- Docker Desktop
- WSL2 backend enabled in Docker Desktop
- Run the commands below from PowerShell, Windows Terminal, or a WSL2 shell

For the simplest Windows setup, keep videos, processed data, and model caches inside the cloned repository so Docker can use relative paths without Windows drive-path conversion issues.

### 1. Clone the repository

```bash
git clone git@github.com:gianguyen14/multiv2.git
cd multiv2
```

Because the repository is private, configure GitHub SSH access on the machine before cloning.

### 2. Create local data directories

```bash
mkdir -p data/test-videos data/processed models
```

On PowerShell, the equivalent is:

```powershell
New-Item -ItemType Directory -Force data/test-videos, data/processed, models
```

Place input videos in:

```text
data/test-videos/
```

The default Compose configuration mounts:

```text
./data/test-videos  -> /data/videos      (read-only)
./data/processed    -> /data/processed   (read/write)
./models            -> /models           (read/write)
```

### 3. Build and start the backend

```bash
docker compose up --build -d backend
```

The first build can take a while because system and Python dependencies must be installed.

Check container status:

```bash
docker compose ps
```

Follow logs:

```bash
docker compose logs -f backend
```

The API is bound to localhost by default:

```text
http://127.0.0.1:8000
```

Health endpoint:

```text
http://127.0.0.1:8000/health/live
```

Linux / WSL2 health check:

```bash
curl http://127.0.0.1:8000/health/live
```

PowerShell health check:

```powershell
Invoke-WebRequest http://127.0.0.1:8000/health/live
```

### 4. Run projectctl inside Docker

The `worker` service is a tools profile whose entry point is `projectctl.py`.

Show available commands:

```bash
docker compose --profile tools run --rm worker --help
```

Run diagnostics:

```bash
docker compose --profile tools run --rm worker doctor
```

Inspect models:

```bash
docker compose --profile tools run --rm worker models
```

Prepare configured local models when required:

```bash
docker compose --profile tools run --rm worker models --prepare
```

Run preprocessing against the mounted video directory:

```bash
docker compose --profile tools run --rm worker preprocess /data/videos
```

Use `python projectctl.py --help` or the worker help output from the checked-out revision as the authoritative command list.

### 5. Override data/model locations

The Compose file supports these environment variables:

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

For Windows, relative repository paths are recommended unless Docker Desktop has file sharing configured for the external drive/path.

### 6. Offline mode

After the required model files are already present under the mounted model directory, run with Hugging Face / Transformers network access disabled:

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

### 7. Stop the stack

```bash
docker compose down
```

To rebuild after source changes:

```bash
docker compose up --build -d backend
```

### NVIDIA GPU / CUDA

The repository includes `docker-compose.cuda.yml` as a CUDA override:

```bash
docker compose -f docker-compose.yml -f docker-compose.cuda.yml up --build -d backend
```

GPU prerequisites depend on the host:

- Linux: NVIDIA driver plus Docker/NVIDIA container GPU support.
- Windows: NVIDIA driver with WSL2 GPU support and Docker Desktop using the WSL2 backend.

Verify Docker can see the GPU before using the CUDA override.

> **RC2 note:** the CUDA Compose override currently contains experimental OCR GPU routing. PaddleOCR GPU is **not** part of the accepted RC2 runtime path yet. For RC2 acceptance/reproducibility, use the Tesseract OCR path until CUDA/Paddle integration has been validated separately.

## Development without Docker

Install the project in editable mode:

```bash
python -m pip install -e .
```

Run the test suite:

```bash
pytest
```

Verify imports:

```bash
python -c "import backend; import backend.app; print('imports: OK')"
```

## Project control CLI

The repository includes `projectctl.py` as the operator entry point for local project workflows.

```bash
python projectctl.py --help
```

Use the command help from the checked-out revision as the authoritative list of available operations.

## Repository layout

```text
backend/       application and retrieval pipeline
eval/          evaluation and benchmark utilities
scripts/       diagnostics, validation, dataset, and experiment helpers
tests/         unit and integration tests
docs/          additional project documentation
projectctl.py  operator CLI
```

## Documentation

- [Architecture](ARCHITECTURE.md)
- [Engineering / agent rules](AGENTS.md)

## Design goals

The project prioritizes:

- deterministic frame identity;
- reproducible offline execution;
- sparse global indexing with bounded dense refinement;
- evidence-aware ranking instead of visual similarity alone;
- graceful fallback when optional components are unavailable;
- competition-oriented ranking quality and temporal diversity.

---

**Unified AIC Retrieval** is under active development for AIC 2026. Performance claims should be based on representative competition data and the corresponding validation results.

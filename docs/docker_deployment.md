# Docker Deployment Guide

## 1. Overview & Architecture

The containerized deployment packages the Python runtime, system utilities (FFmpeg, Tesseract OCR), and application dependencies into a single reproducible OCI image.

### Storage & Volume Model
Containers are stateless. All mutable and persistent data reside on host mounts:
- **Source Videos (`/data/videos`)**: Mounted **read-only** (`:ro`).
- **Processed Artifacts (`/data/processed`)**: Mounted **read-write** (`:rw`). Contains frame extractions, embeddings, OCR/ASR caches, and atomic FAISS generation indexes.
- **Model Cache (`/models`)**: Mounted **read-write** (`:rw`). Holds cached SigLIP2 and Faster Whisper model weights.

### Security Model & Localhost Default
The API currently contains **no built-in authentication or authorization**.
By default, `docker-compose.yml` publishes the service port explicitly bound to the localhost interface:
```yaml
ports:
  - "127.0.0.1:8000:8000"
```
**Do not expose port 8000 to `0.0.0.0` or public internet interfaces without placing a reverse proxy (e.g. Nginx, Caddy, Cloudflare Tunnel) in front with TLS termination, rate limiting, and authentication.**

---

## 2. Standard Operator Workflow

### Step 1: Initial Setup
Copy the example environment configuration:
```bash
cp .env.example .env
mkdir -p data/test-videos data/processed models
```

### Step 2: Build the Container Image
```bash
docker compose build
# Or with podman:
# podman build -t aic-retrieval:latest .
```

### Step 3: Online First-Time Model Preparation
Download and cache the SigLIP2 and Faster Whisper models to the persistent `/models` directory:
```bash
docker compose run --rm worker models --prepare
```

### Step 4: Video Preprocessing
Run offline preprocessing on the mounted video directory:
```bash
docker compose run --rm worker preprocess /data/videos
```
*Note: Preprocessing does not run automatically on backend startup to ensure fast and deterministic service boot.*

### Step 5: Start the Backend Service
Start the FastAPI search backend:
```bash
docker compose up -d backend
```

Check health:
```bash
curl http://127.0.0.1:8000/health/ready
```

Access the Web UI in your browser:
```
http://127.0.0.1:8000/
```

### Step 6: Shutdown
```bash
docker compose down
```

---

## 3. Strict Offline Mode

After model weights have been downloaded to `/models`, the system can run in strict offline air-gapped mode:
Set the following environment variables in `.env`:
```env
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
```

Verify cached models offline:
```bash
docker compose run --rm worker models
```

---

## 4. GPU Acceleration (CUDA)

To enable NVIDIA GPU acceleration for SigLIP2 image encoding and Faster Whisper ASR:

### Prerequisites
- Host NVIDIA Driver installed
- NVIDIA Container Toolkit (`nvidia-container-toolkit`) installed and configured

### Running with GPU Override
```bash
docker compose -f docker-compose.yml -f docker-compose.cuda.yml up -d backend
```

*Note on GPU Status: CUDA container execution requires a verified host GPU and NVIDIA Container Toolkit. On systems without NVIDIA GPU hardware, deployment defaults to CPU inference.*

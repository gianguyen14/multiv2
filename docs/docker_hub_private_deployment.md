# Docker Hub Private Deployment Guide

## 1. Overview & Image Specifications

The official application image is published to a private Docker Hub repository:

- **Target Repository:** `gianguyen14/aic-retrieval`
- **Visibility:** **PRIVATE** (Requires Docker Hub authentication and repository access rights)
- **Pinned Production Tag:** `gianguyen14/aic-retrieval:1.0.0`
- **Convenience Alias:** `gianguyen14/aic-retrieval:latest`
- **Digest:** `sha256:24fb0c9500ff46d797717ee2e97ec2c8df0c4cb7848bebf4ab9b2961bf9d1cc9`
- **Target Platform:** `linux/amd64`


---

## 2. Data & Volume Architecture

The Docker Hub image is fully stateless and self-contained with all application source code, system libraries (FFmpeg, Tesseract OCR), and Python dependencies.

**The image does NOT contain:**
- Source video datasets
- Processed frames / frame catalog manifests
- Vector indexes (FAISS index files)
- Text evidence caches (OCR / Faster Whisper ASR JSON)
- Frozen ground truth (GT) data
- Model weight caches (SigLIP2 / Whisper weights)
- Environment secrets / Docker credentials

### Volume Layout
| Container Mount Point | Host Volume Default | Access Mode | Purpose |
|---|---|---|---|
| `/data/videos` | `./data/test-videos` | Read-Only (`:ro`) | Source video files (`.mp4`) |
| `/data/processed` | `./data/processed` | Read-Write (`:rw`) | Extracted frames, embeddings, text caches, and FAISS index generations |
| `/models` | `./models` | Read-Write (`:rw`) | Hugging Face & Faster Whisper cached model snapshots |

---

## 3. End-User Deployment Procedure

### Step 1: Prerequisites
- Docker Engine (v24.0+) & Docker Compose (v2.20+)
- Access permissions to the private Docker Hub repository `gianguyen14/aic-retrieval`

### Step 2: Authentication on New Machine
Because the repository is **PRIVATE**, any new machine must authenticate with Docker Hub:
```bash
docker login
```
*(Enter authorized Docker Hub credentials when prompted. Do not commit or hardcode credentials in configuration files).*

### Step 3: Pull the Release Image
```bash
docker pull gianguyen14/aic-retrieval:1.0.0
```

### Step 4: Environment & Directory Setup
Clone or download the deployment configuration:
```bash
# Create local volume directories
mkdir -p data/test-videos data/processed models

# Prepare environment file if custom paths/ports are needed
cp .env.example .env
```

### Step 5: Pull Image via Release Compose
```bash
docker compose -f docker-compose.release.yml pull
```

### Step 6: Initial Model Preparation (First-time Online Setup)
Run model downloading to the persistent `/models` volume:
```bash
docker compose -f docker-compose.release.yml run --rm worker models --prepare
```

### Step 7: Video Ingestion & Preprocessing
Preprocess source videos from `/data/videos`:
```bash
docker compose -f docker-compose.release.yml run --rm worker preprocess /data/videos
```

### Step 8: Start the Backend Service
Start the search API and web console:
```bash
docker compose -f docker-compose.release.yml up -d backend
```

Verify service status:
```bash
docker compose -f docker-compose.release.yml ps
```

Check live logs:
```bash
docker compose -f docker-compose.release.yml logs -f backend
```

---

## 4. Health Verification & Browser Access

- **Liveness Probe:**
  ```bash
  curl -f http://127.0.0.1:8000/health/live
  ```
- **Readiness Probe:**
  ```bash
  curl -f http://127.0.0.1:8000/health/ready
  ```
- **Web Console UI:**
  Navigate to `http://127.0.0.1:8000/` in a web browser.

---

## 5. Teardown
```bash
docker compose -f docker-compose.release.yml down
```

---

## 6. Offline / Air-Gapped Deployment

To deploy in an isolated, air-gapped environment without internet access:

1. **On an internet-connected host:**
   ```bash
   docker login
   docker pull gianguyen14/aic-retrieval:1.0.0
   docker save -o aic-retrieval-1.0.0.tar gianguyen14/aic-retrieval:1.0.0
   ```
2. **Download Model Snapshots:**
   Run model preparation on the online machine into `./models`.
3. **Transfer Artifacts:**
   Transfer `aic-retrieval-1.0.0.tar`, `./models` directory, `./data/processed` directory, and `docker-compose.release.yml` to the target machine via secure transfer.
4. **On the Air-Gapped Target Machine:**
   ```bash
   docker load -i aic-retrieval-1.0.0.tar
   ```
   Set offline flags in `.env`:
   ```env
   HF_HUB_OFFLINE=1
   TRANSFORMERS_OFFLINE=1
   ```
   Start the service:
   ```bash
   docker compose -f docker-compose.release.yml up -d backend
   ```

---

## 7. Optional GPU Acceleration (CUDA)

For NVIDIA GPU acceleration on hosts equipped with NVIDIA Container Toolkit:
```bash
docker compose -f docker-compose.release.yml -f docker-compose.cuda.yml up -d backend
```
*Note: GPU container runtime execution requires host NVIDIA drivers and `nvidia-container-toolkit`.*

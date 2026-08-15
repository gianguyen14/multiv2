# Multimodal Video Retrieval System — Deployment & Operations Guide

## 1. System Overview & Architecture

The HCM City AI Challenge 2026 Multimodal Video Retrieval System is packaged as a high-performance, self-contained containerized service.

```
                           USER BROWSER / API CLIENT
                                      │
                                      ▼
                           FastAPI Backend (:8000)
                                      │
                         ┌────────────┼────────────┐
                         │                         │
                   /api/search               /health/ready
                         │
                         ▼
                   ConfiguredSearch
                         │
                   Query Intelligence (VI / EN / Lexical Plan)
                         │
          ┌──────────────┼──────────────┐
          │              │              │
       SigLIP2       Faster-Whisper   Tesseract / PaddleOCR
      (Visual)           (ASR)                (OCR)
          │              │              │
          └──────────────┼──────────────┘
                         │
                   RRF Fusion (K=60)
                         │
                 CandidateReranker
                         │
                   Temporal NMS
                         │
                 Top-N Ranked Frames
```

---

## 2. Host Requirements

### CPU Mode (Baseline)
- **OS**: Linux (Ubuntu 22.04+, Debian 12+, RHEL 9+, Fedora) or Windows 10/11 (Docker Desktop with WSL2).
- **RAM**: Minimum 8 GB recommended (16 GB for concurrent operations).
- **Disk**: 10+ GB free storage for models and index storage.
- **Software**: Docker Engine 24.0+ and Docker Compose v2.20+.

### CUDA GPU Mode (Accelerated)
- **GPU**: NVIDIA GPU with Turing/Ampere/Ada/Hopper architecture (Pascal+ supported with appropriate compute capabilities).
- **Driver**: NVIDIA Linux Driver 525+ / Windows NVIDIA Driver with WSL2 GPU passthrough.
- **Container Toolkit**: NVIDIA Container Toolkit (`nvidia-container-toolkit` on Linux, enabled in Docker Desktop for Windows).

---

## 3. One-Command Deployment

### Linux
```bash
# 1. Clone repository and navigate to root
cd /path/to/Project

# 2. Deploy (CPU mode)
./deploy.sh

# 2b. Deploy with CUDA GPU acceleration (if NVIDIA hardware is present)
./deploy.sh --cuda

# 3. Stop containers
./stop.sh
```

### Windows (PowerShell)
```powershell
# 1. Open PowerShell in project directory
cd C:\Path\To\Project

# 2. Deploy (CPU mode)
.\deploy.ps1

# 2b. Deploy with CUDA GPU acceleration
.\deploy.ps1 -Cuda

# 3. Stop containers
.\stop.ps1
```

---

## 4. Persistent Volume Layout

The deployment mounts three directories between the host and container:

| Host Path | Container Mount | Mode | Purpose |
|---|---|---|---|
| `./data/videos` | `/data/videos` | `ro` (read-only) | Source video files |
| `./data/processed` | `/data/processed` | `rw` (read-write) | Frame indices, OCR/ASR metadata, query cache |
| `./models` | `/models` | `rw` (read-write) | Local model weights (SigLIP2, Faster-Whisper, Qwen, PaddleOCR) |

---

## 5. Offline Operation & Model Preparation

All models can be prepared while online, then operated in strict offline mode without network connectivity.

```bash
# 1. Prepare visual model (SigLIP2)
python projectctl.py models --prepare --visual

# 2. Prepare ASR model (Faster-Whisper small)
python projectctl.py models --prepare --asr

# 3. Prepare Query Refiner model
python projectctl.py models --prepare --query-refiner

# 4. Verify offline readiness
python projectctl.py models --verify-offline --all
```

Set `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` in `.env` to enforce zero outgoing network requests at inference.

---

## 6. Preflight Health Checks & System Doctor

Run the system doctor to verify environment readiness before competition operations:

```bash
python projectctl.py doctor
```

Example output:
```
=== AIC 2026 System Doctor Preflight ===
[PASS           ] Docker Runtime      : docker installed
[PASS           ] Processed Root      : data/processed-validation/three-video-final
[PASS           ] CURRENT Index       : VALID (gen-3d51a16ea32b4953820583c3af181b31)
[PASS           ] Disk Free Space     : 340.8 GB free
[PASS           ] SigLIP2 Weights     : cached locally
[PASS           ] Faster-Whisper      : cached locally
[PASS (FALLBACK)] Query Refiner       : deterministic fallback ready
[PASS           ] Tesseract OCR       : tesseract present (vie=True, eng=True)
[INFO           ] GPU / CUDA          : NOT AVAILABLE (CPU MODE)
[PASS           ] FFmpeg Decoder      : ffmpeg installed
----------------------------------------
Overall Preflight Verdict: PASS
```

---

## 7. Security & Network Configuration

- **Localhost Default Bind**: The container binds to `127.0.0.1:8000` by default.
- **Unauthenticated API**: The API is unauthenticated by design for low-latency competition use.
- **Production Exposure**: If exposed beyond localhost or a trusted private LAN, put the service behind a reverse proxy (e.g. Nginx, Caddy, Cloudflare Tunnel) with TLS and authentication.
- **Registry Tokens**: Docker registry tokens are read from environment variables (`DOCKER_HUB_TOKEN`) or interactive standard input—never commit credentials to version control.

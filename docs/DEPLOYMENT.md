# AIC 2026 Multimodal Video Retrieval — Production Deployment Guide

This document outlines the standard production deployment, configuration, update, and rollback workflows for the AIC 2026 Multimodal Retrieval service using Docker.

---

## 1. Directory Structure

Recommended production server layout:

```text
/opt/aic/
├── app/        # Cloned Git repository
├── data/       # Persistent indexed data and video metadata
│   ├── videos/
│   └── processed/
├── models/     # Persistent ML model weights (offline cache)
├── cache/      # Persistent HuggingFace / PyTorch caches
│   ├── huggingface/
│   └── torch/
└── logs/       # Application logs (if file logging is enabled)
```

---

## 2. Quick Start Installation

```bash
# 1. Clone repository to /opt/aic/app
git clone git@github.com:gianguyen14/multiv2.git /opt/aic/app
cd /opt/aic/app

# 2. Setup environment configuration
cp .env.example .env

# 3. Create persistent directories
mkdir -p /opt/aic/data /opt/aic/models /opt/aic/cache /opt/aic/logs

# 4. Build and start service via deploy script
./deploy/install.sh

# Or using docker compose directly:
docker compose build
docker compose up -d
```

---

## 3. Recommended Production Environment (`.env`)

Configure the following visual sampling and local refinement parameters in `.env`:

```ini
# Production Visual Sampling Configuration
VISUAL_SAMPLING_MODE=sparse_shot
VISUAL_GLOBAL_SAMPLE_SECONDS=5.0
VISUAL_DEDUP_ENABLED=true
VISUAL_DEDUP_THRESHOLD=0.97

# Query-Time Local Dense Refinement
LOCAL_REFINE_ENABLED=true
LOCAL_REFINE_WINDOW_SECONDS=10.0
LOCAL_REFINE_INTERVAL_SECONDS=0.5
LOCAL_REFINE_MAX_REGIONS=5
```

Current runtime limitations:

- `sparse_shot` always provides deterministic sparse periodic sampling. The
  bundled `TransNetV2Adapter` does not yet implement inference and no TransNet
  weights or TensorFlow runtime are shipped, so the standard CLI currently
  adds no shot representatives. A tested injected `ShotDetector` remains
  supported by the ingestion pipeline.
- The local-refinement region generation, 0.5-second sampling, and scoring
  module is unit-tested, but `ConfiguredSearch` does not yet have a raw-video
  frame provider and therefore does not invoke it. `LOCAL_REFINE_ENABLED=true`
  is forward-compatible configuration, not an active API-search stage in this
  revision. No local vectors are persisted.

These limitations should be resolved before claiming shot-coverage or local
refinement quality improvements from the production service.

### Rollback Configuration
To restore the legacy 1.0-second uniform sampling baseline without deduplication or local refinement, update `.env`:

```ini
VISUAL_SAMPLING_MODE=legacy
VISUAL_DEDUP_ENABLED=false
LOCAL_REFINE_ENABLED=false
```

---

## 4. Operational Commands

### Check Status & Health
```bash
./deploy/status.sh
# Or:
docker compose ps
curl -s http://127.0.0.1:8000/health/live
```

### View Application Logs
```bash
docker compose logs -f --tail=100 aic
```

### Update to Latest Version
```bash
./deploy/update.sh
# Or manually:
git pull --ff-only
docker compose up -d --build
```

### Rollback to Previous Version / Commit
```bash
./deploy/rollback.sh <git-commit-or-tag>
# Example: ./deploy/rollback.sh 291044b
```

### Stop / Restart Service
```bash
# Restart container
docker compose restart aic

# Stop container (Preserves data volumes)
docker compose down

# IMPORTANT: Never run 'docker compose down -v' in production to avoid volume loss.
```

---

## 5. GPU Acceleration (Optional)

To enable NVIDIA GPU acceleration:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d
```

For a fail-fast GPU acceptance run with real, pre-cached models and three real
videos, configure the `AIC_*_DIR` variables documented in the script and run:

```bash
./scripts/validate_rc2_gpu.sh
```

The script builds only the local `aic-retrieval:rc2-validation` tag by default
and never pushes or overwrites a Docker Hub tag.

---

## 6. Reverse Proxy & Security

* The AIC API binds to `127.0.0.1:8000` by default.
* For external access, proxy through Nginx, Caddy, or a Tailscale funnel with TLS termination.
* Container executes as non-root user `appuser` (UID 1000).
* Container does not require privileged mode or access to `/var/run/docker.sock`.

# Docker Release Record 1.0.0

**Release Date:** 2026-08-14  
**Release Status:** **SUCCESSFULLY PUBLISHED**  

---

## 1. Registry & Target Information

- **Repository:** `gianguyen14/aic-retrieval`
- **Visibility:** **PRIVATE**
- **Release Version:** `1.0.0`
- **Published Tags:**
  - `gianguyen14/aic-retrieval:1.0.0` (Pinned immutable release)
  - `gianguyen14/aic-retrieval:latest` (Convenience alias)
- **Image ID:** `sha256:24fb0c9500ff46d797717ee2e97ec2c8df0c4cb7848bebf4ab9b2961bf9d1cc9`
- **Target OS / Architecture:** `linux/amd64`
- **Image Size:** `3,372,797,565` bytes (~3.37 GB)
- **Build Timestamp:** `2026-08-14T20:23:47+07:00`
- **Dockerfile Used:** `Dockerfile`

---

## 2. Remote Digest & Manifest Resolution

- **Version 1.0.0 Digest:** `sha256:24fb0c9500ff46d797717ee2e97ec2c8df0c4cb7848bebf4ab9b2961bf9d1cc9`
- **Latest Tag Digest:** `sha256:24fb0c9500ff46d797717ee2e97ec2c8df0c4cb7848bebf4ab9b2961bf9d1cc9`
- **Remote Manifest Inspection:**
  - Verified via `docker buildx imagetools inspect gianguyen14/aic-retrieval:1.0.0`
  - Manifest Type: `application/vnd.oci.image.index.v1+json`
  - Sub-manifest Linux AMD64: `sha256:87024b95a53ce1de812f3e39c79d39208db812c7aedc39a5baabe5b8895cbd80`
- **Pull Verification:**
  - Verified via authenticated pull: `docker pull gianguyen14/aic-retrieval:1.0.0`
  - Status: Image up to date, digest verified matching.


---

## 3. Build & Smoke Test Results

| Step | Verification Command | Result | Notes |
|---|---|---|---|
| **Build Context Safety** | `.dockerignore` evaluation | PASS | Build context size: 810 kB; no video, model, or cache artifacts transferred. |
| **Docker Build** | `docker build -t gianguyen14/aic-retrieval:1.0.0 .` | PASS | Exit code 0, layers assembled cleanly. |
| **Image Content Audit** | `docker run --rm ... inspect /app, /data, /models` | PASS | Verified zero videos, FAISS indexes, GT, model weights, or `.env` files inside image. |
| **Local CLI Smoke** | `docker run --rm ... python projectctl.py --help` | PASS | Exit code 0, CLI dispatcher loads and outputs help. |
| **Compose Local Smoke** | `docker compose up -d backend` | PASS | Container started, `/health/live` returned 200 OK. |
| **Release Compose Validation** | `docker compose -f docker-compose.release.yml config` | PASS | Verified `backend` and `worker` resolve to `gianguyen14/aic-retrieval:1.0.0`. |
| **Push 1.0.0** | `docker push gianguyen14/aic-retrieval:1.0.0` | PASS | Exit code 0, pushed digest captured. |
| **Push latest** | `docker push gianguyen14/aic-retrieval:latest` | PASS | Exit code 0, existing layers reused. |
| **Pull Verification** | `docker pull gianguyen14/aic-retrieval:1.0.0` | PASS | Exit code 0, served directly from Docker Hub registry. |

---

## 4. CUDA Status

- **Configuration Present:** `docker-compose.cuda.yml` is provided for NVIDIA Container Toolkit GPU device passthrough.
- **Runtime Execution Status:** **CPU RUNTIME VERIFIED** (Host environment ran in CPU mode; GPU configuration is statically structured for compatible NVIDIA hosts).

---

## 5. Security Attestation

- **Docker Credentials Exposed:** **NO** (Zero credential inspection or printing).
- **`.env` File Embedded:** **NO** (`.dockerignore` strictly excludes `.env` and `.env.*`).
- **Private Data / Videos Embedded:** **NO** (All video datasets, frames, and FAISS vector indexes reside purely in external persistent host volumes).
- **Model Cache Embedded:** **NO** (`/models` is an external host volume).
- **Ground Truth Embedded:** **NO**.

---

## 6. Deployment Files Summary

- Release Compose File: [`docker-compose.release.yml`](file:///home/nguyen/T%C3%A0i%20li%E1%BB%87u/Project/docker-compose.release.yml)
- Deployment Guide: [`docs/docker_hub_private_deployment.md`](file:///home/nguyen/T%C3%A0i%20li%E1%BB%87u/Project/docs/docker_hub_private_deployment.md)

# AIC Full System LAN Usage Guide

## Recommended — Docker Hub CPU (normal finals deployment)

Normal operators do not need to clone GitHub or build an image locally. The application comes from Docker Hub; the production DB, model, videos, and cache stay on the host and are mounted into one container.

1. Install Docker Engine.
2. Supply these external resources:
   - `/opt/aic/data/aic-db-v1/runtime`
   - `/opt/aic/models/Qwen3-VL-Embedding-2B`
   - `/opt/aic/videos`
   - `/opt/aic/cache`
3. Pull the verified immutable image:

```bash
docker pull gianguyen14/aic-retrieval:cpu-18132237cca7
```

4. Start one container with host ports 3000 and 8000 mapped to internal port 8000:

```bash
docker run -d --name aic-retrieval --restart unless-stopped \
  -p 0.0.0.0:3000:8000 -p 0.0.0.0:8000:8000 \
  -v /opt/aic/data/aic-db-v1/runtime:/data/runtime:ro \
  -v /opt/aic/models/Qwen3-VL-Embedding-2B:/models/Qwen3-VL-Embedding-2B:ro \
  -v /opt/aic/videos:/videos:ro -v /opt/aic/cache:/cache:rw \
  -e VIDEO_PROCESSED_ROOT=/data/runtime \
  -e MODEL_CACHE_DIR=/models \
  -e QWEN3_VL_MODEL_DIR=/models/Qwen3-VL-Embedding-2B \
  -e SEARCH_BACKEND=qwen3_vl -e QA_ANSWER_BACKEND=extractive \
  -e QUERY_REFINER_ENABLED=false -e SEARCH_ENABLE_OCR=true -e SEARCH_ENABLE_ASR=true \
  -e RERANKER_ENABLED=true -e VIDEO_SOURCE_DIR=/videos -e VIDEO_CACHE_DIR=/cache/videos \
  -e ALLOWED_ORIGINS=http://SERVER_IP:3000 \
  gianguyen14/aic-retrieval:cpu-18132237cca7
```

5. Verify:

```bash
curl -fsS http://127.0.0.1:8000/health/ready
curl -I http://127.0.0.1:3000/
ss -lntp | grep -E ':3000|:8000'
```

6. Open from another LAN machine: `http://SERVER_IP:3000`.
7. Run the finals smoke script when the repository ops script is available.

`0.0.0.0` is a bind address, not a browser URL. Do not open `http://0.0.0.0:3000`.

The CPU release tags resolve to:

```text
cpu-18132237cca7
cpu-finals
sha256:31200173184ca6fe4b6887b6bef3a5b8cfec69a8784303bbd9ce4f0733a293e2
```

Do not use historical `finals-20260917` for this release.

## Recommended — Docker Hub GPU

No GPU image is currently published. Current host has no verified NVIDIA runtime and the CUDA build could not complete within available disk space. Therefore `GPU_RUNTIME_VALIDATION=BLOCKED_NO_GPU_HOST` and no `gpu-finals` tag is documented as available.

On a future qualified GPU host:

```bash
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
docker pull gianguyen14/aic-retrieval:gpu-GPU_SHA
```

Run the same mounts/env as CPU, add `--gpus all`, set `COMPUTE_DEVICE=cuda` and `VISUAL_DEVICE=cuda`, then verify `torch.cuda.is_available()` inside the container and run full KIS/QA/TRAKE/video/range smoke. Do not invent or pull a GPU tag until it actually exists.

See `ops/DOCKER_GPU_GUIDE.md` for the complete gate.

## Download deployment files without cloning

When only deployment files are needed:

```bash
mkdir -p /opt/aic/config
curl -fsSL https://raw.githubusercontent.com/gianguyen14/multiv2/main/docker-compose.cpu.yml -o /opt/aic/config/compose.cpu.yml
curl -fsSL https://raw.githubusercontent.com/gianguyen14/multiv2/main/ops/docker/.env.cpu.example -o /opt/aic/config/.env.cpu
```

Edit the paths and `SERVER_IP`, then use `docker compose ... pull` and `up -d`. The application remains the Docker Hub image; no source checkout is required.

## Native host (secondary)

Native host operation is retained for this repository's development/runtime maintenance. Check the actual CLI before starting:

```bash
python3 projectctl.py --help
python3 projectctl.py backend --help
python3 projectctl.py frontend --help
python3 projectctl.py dev --help
```

The actual commands are separate Uvicorn launches. `dev` does not start both services. For the current host-native layout, set the environment from the repository guide and run:

```bash
python3 projectctl.py backend --host 0.0.0.0 --port 8000
python3 projectctl.py frontend --host 0.0.0.0 --port 3000
```

Do not start a second Qwen instance on a low-RAM host. Native process control uses the PID/process manager that started the service; do not use `kill -9` except as a last resort.

## Developer / build from source (secondary)

Only developers need this path:

```bash
git clone https://github.com/gianguyen14/multiv2.git /home/hermes/aic
cd /home/hermes/aic
git fetch origin
git checkout main
git pull --ff-only origin main
SOURCE_SHA=$(git rev-parse HEAD)
```

CPU build:

```bash
bash ops/docker/build-cpu.sh
```

GPU build requires a CUDA build host and is not runtime-valid on this host:

```bash
TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124 bash ops/docker/build-gpu.sh
```

Never build a release from uncommitted production source. Never include model/DB/video/cache/secrets in the image.

## Production environment contract

```text
SEARCH_BACKEND=qwen3_vl
QA_ANSWER_BACKEND=extractive
QUERY_REFINER_ENABLED=false
VIDEO_PROCESSED_ROOT=/data/runtime
QWEN3_VL_MODEL_DIR=/models/Qwen3-VL-Embedding-2B
```

Current retrieval flow:

```text
KIS/QA query -> Qwen text embedding -> FAISS visual retrieval -> OCR/ASR evidence -> weighted fusion -> ranked results
QA -> same retrieval -> extractive answer handling
TRAKE -> event embeddings -> per-event retrieval -> same-video monotonic temporal sequence
```

The current fusion weights are visual `0.70`, OCR `0.18`, ASR `0.12`. The frontend does not call an LLM directly. Frame JPEG preview may be unavailable in the packed DB; video preview plus returned timestamp is the fallback.

## Hardware qualification and smoke

From a checkout containing the ops scripts:

```bash
bash ops/qualify_finals_hardware.sh
bash ops/finals_smoke.sh
```

Profile priority is validated CUDA, then qualified FP32 CPU with >=16 GiB and >=2 GiB headroom, then BF16 safe fallback. RAM size alone never enables FP32.

## Health, ports, firewall

```bash
curl -fsS http://SERVER_IP:8000/health
curl -fsS http://127.0.0.1:8000/health/ready
curl -I http://SERVER_IP:3000/
ss -lntp | grep -E ':3000|:8000'
hostname -I
ip addr
```

If UFW is active, restrict to the LAN subnet:

```bash
sudo ufw status
sudo ufw allow from 192.168.1.0/24 to any port 3000 proto tcp
sudo ufw allow from 192.168.1.0/24 to any port 8000 proto tcp
```

Never expose unauthenticated ports directly to the public Internet.

## Docker lifecycle

```bash
docker ps
docker logs -f aic-retrieval
docker stop aic-retrieval
docker rm aic-retrieval
docker pull gianguyen14/aic-retrieval:cpu-NEW_SHA
```

Upgrade or rollback by rerunning the same command with a different immutable image tag. Do not delete external DB/model/video/cache mounts.

## Troubleshooting

| Problem | Check | Fix |
|---|---|---|
| LAN unreachable | `ss`, `hostname -I`, firewall | Publish `0.0.0.0:3000/8000`, use server IPv4, allow LAN subnet |
| Health fails | `docker logs`, `/health/ready` | Check DB `CURRENT`, model mount, permissions, and exact v1 generation |
| Port occupied | `ss -lntp` | Stop the old container/process; do not duplicate Qwen |
| CORS error | Browser origin vs `ALLOWED_ORIGINS` | Set exact `http://SERVER_IP:3000`, recreate container |
| Model missing | `test -f .../model.safetensors` | Correct model mount; never bake weights into image |
| DB not ready | `test -f .../index/CURRENT` | Correct DB mount and preserve generation symlink targets |
| Frame JPEG 404 | UI/API frame endpoint | Use video seek fallback; do not repack automatically |
| Video 404 | `VIDEO_SOURCE_DIR`, video mount | Correct `/videos` mount and matching video IDs |
| TRAKE 400/no match | API detail | Strict monotonic same-video no-match can be valid; use known-good events |
| RAM low | `free -h`, `docker stats` | BF16 safe profile, one container/process, no swap thrashing |
| Query slow | latency and `docker stats` | Qwen CPU embedding is the bottleneck; qualify hardware before dtype change |
| GPU unavailable | `nvidia-smi`, `--gpus all` | Use CPU release; GPU is not validated on this host |

Do not put secrets in compose/env examples or image labels.

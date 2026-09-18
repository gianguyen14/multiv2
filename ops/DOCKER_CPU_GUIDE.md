# AIC CPU Docker deployment (Docker Hub first)

This is the normal finals/operator path. It does not require cloning GitHub or building locally.

## 1. Requirements

- Docker Engine on a Linux server.
- At least 9 GiB RAM for the BF16 safe fallback. Do not select FP32 without a finals-host qualification showing at least 16 GiB RAM and at least 2 GiB post-startup headroom.
- External production resources supplied by the operator:
  - `/opt/aic/data/aic-db-v1/runtime`
  - `/opt/aic/models/Qwen3-VL-Embedding-2B`
  - `/opt/aic/videos`
  - `/opt/aic/cache`

The image contains application code and dependencies only. It does not contain the model, DB, videos, cache, credentials, or paper/experiment artifacts.

## 2. Pull the release image

Use the immutable tag for this release:

```bash
docker pull gianguyen14/aic-retrieval:cpu-18132237cca7
```

The human-readable alias is also available:

```bash
docker pull gianguyen14/aic-retrieval:cpu-finals
```

Both tags resolve to the same verified Docker Hub manifest digest:

```text
sha256:31200173184ca6fe4b6887b6bef3a5b8cfec69a8784303bbd9ce4f0733a293e2
```

Do not use the historical `finals-20260917` tag for this release.

## 3. Prepare external mounts

```bash
sudo mkdir -p /opt/aic/{data,models,videos,cache,config}
sudo test -f /opt/aic/data/aic-db-v1/runtime/index/CURRENT
sudo test -f /opt/aic/models/Qwen3-VL-Embedding-2B/model.safetensors
sudo test -d /opt/aic/videos
```

The DB must be the verified production v1 runtime. Keep the DB and model mounts read-only. The cache is writable.

## 4. Start with `docker run`

Replace `SERVER_IP` in `ALLOWED_ORIGINS` with the server's LAN IPv4. `0.0.0.0` is the bind/publish address; users open `http://SERVER_IP:3000`, not `http://0.0.0.0:3000`.

```bash
docker run -d \
  --name aic-retrieval \
  --restart unless-stopped \
  -p 0.0.0.0:3000:8000 \
  -p 0.0.0.0:8000:8000 \
  -v /opt/aic/data/aic-db-v1/runtime:/data/runtime:ro \
  -v /opt/aic/models/Qwen3-VL-Embedding-2B:/models/Qwen3-VL-Embedding-2B:ro \
  -v /opt/aic/videos:/videos:ro \
  -v /opt/aic/cache:/cache:rw \
  -e VIDEO_PROCESSED_ROOT=/data/runtime \
  -e MODEL_CACHE_DIR=/models \
  -e QWEN3_VL_MODEL_DIR=/models/Qwen3-VL-Embedding-2B \
  -e SEARCH_BACKEND=qwen3_vl \
  -e QA_ANSWER_BACKEND=extractive \
  -e QUERY_REFINER_ENABLED=false \
  -e SEARCH_ENABLE_OCR=true \
  -e SEARCH_ENABLE_ASR=true \
  -e RERANKER_ENABLED=true \
  -e VIDEO_SOURCE_DIR=/videos \
  -e VIDEO_CACHE_DIR=/cache/videos \
  -e ALLOWED_ORIGINS=http://SERVER_IP:3000 \
  -e HF_HUB_OFFLINE=1 \
  -e TRANSFORMERS_OFFLINE=1 \
  gianguyen14/aic-retrieval:cpu-18132237cca7
```

One container runs one Uvicorn/FastAPI process on internal `0.0.0.0:8000`; host ports 3000 and 8000 both map to it. This does not start a second Qwen instance.

## 5. Verify

```bash
docker ps
docker logs -f aic-retrieval
curl -fsS http://127.0.0.1:8000/health/ready
curl -I http://127.0.0.1:3000/
curl -I http://127.0.0.1:8000/
ss -lntp | grep -E ':3000|:8000'
docker top aic-retrieval
```

The health response must show `qwen3_vl`, `/data/runtime`, the Qwen model path, and initialized production v1 generation. `docker top` should show one application Uvicorn process.

Run the smoke script only when the repository's `ops/finals_smoke.sh` is available locally. Otherwise perform the equivalent endpoint checks from `ops/DOCKER_CPU_GUIDE.md`; the image itself does not contain the repository's ops scripts.

## 6. LAN access

```bash
hostname -I
ip addr
```

Open from another LAN machine:

```text
http://SERVER_IP:3000
```

Backend:

```text
http://SERVER_IP:8000/health
http://SERVER_IP:8000/health/ready
```

If UFW is active, restrict access to the LAN subnet where possible:

```bash
sudo ufw status
sudo ufw allow from 192.168.1.0/24 to any port 3000 proto tcp
sudo ufw allow from 192.168.1.0/24 to any port 8000 proto tcp
```

Do not expose unauthenticated ports directly to the public Internet.

## 7. Stop, upgrade, rollback

```bash
docker stop aic-retrieval
docker rm aic-retrieval
```

Upgrade by pulling an immutable tag, then rerun the same command with the new tag:

```bash
docker pull gianguyen14/aic-retrieval:cpu-NEW_SHA
```

Rollback by rerunning the same mounts and environment with the previous immutable tag. Do not delete the external DB or model.

## 8. Download-only Compose option

Normal operators may download only the release Compose/env files; the application still comes from Docker Hub:

```bash
mkdir -p /opt/aic/config
curl -fsSL https://raw.githubusercontent.com/gianguyen14/multiv2/main/docker-compose.cpu.yml -o /opt/aic/config/compose.cpu.yml
curl -fsSL https://raw.githubusercontent.com/gianguyen14/multiv2/main/ops/docker/.env.cpu.example -o /opt/aic/config/.env.cpu
```

Review all paths and replace `SERVER_IP`, then run:

```bash
docker compose --env-file /opt/aic/config/.env.cpu -f /opt/aic/config/compose.cpu.yml pull
docker compose --env-file /opt/aic/config/.env.cpu -f /opt/aic/config/compose.cpu.yml up -d
docker compose --env-file /opt/aic/config/.env.cpu -f /opt/aic/config/compose.cpu.yml ps
```

The compose file's `build:` section is for developers; normal deployment uses `pull` and the image tag.

## 9. Developer / build from source

This is not required for normal finals deployment. Developers may clone the repository, verify the source SHA, and use:

```bash
bash ops/docker/build-cpu.sh
```

Never build from an uncommitted source state for a release.

# AIC CPU Docker Deployment Guide

## Scope

This is the CPU deployment path for the current Qwen production backend. It keeps one FastAPI process, one Qwen model instance, and one read-only production DB mount.

## Requirements

- Linux server with Docker Engine and Docker Compose plugin.
- At least 9 GiB RAM for BF16 safe fallback; 16 GiB or more is required before considering FP32, with at least 2 GiB measured headroom after startup and no swap pressure.
- Production DB mounted at `AIC_DATA_ROOT`.
- Qwen model mounted at `AIC_MODEL_ROOT`.
- Raw videos mounted at `AIC_VIDEO_ROOT`.

## Install Docker

Use your distribution's official Docker Engine instructions. Verify:

```bash
docker --version
docker compose version
```

If `docker compose` is unavailable, install the Docker Compose plugin before continuing. Do not use an unverified third-party installer.

## Prepare host resources

The example paths match this server:

```bash
sudo mkdir -p /home/hermes/aic/cache/videos
sudo test -f /home/hermes/aic/data/aic-db-v1/runtime/index/CURRENT
sudo test -f /home/hermes/aic/models/Qwen3-VL-Embedding-2B/model.safetensors
sudo test -d /video/video
```

The image does not contain these resources.

## Clone and pin source

```bash
git clone https://github.com/gianguyen14/multiv2.git /home/hermes/aic
cd /home/hermes/aic
git fetch origin
git checkout main
git pull --ff-only origin main
git rev-parse HEAD
```

For reproducible deployment, record the full SHA and use an image tag containing that SHA.

## Configure CPU environment

```bash
cd /home/hermes/aic
cp ops/docker/.env.cpu.example .env.cpu
sed -i 's/SERVER_IP/192.168.1.50/g' .env.cpu
```

Replace `192.168.1.50` with the actual server IPv4. Review every path before starting:

```bash
sed -n '1,120p' .env.cpu
```

Do not put passwords, API tokens, GitHub tokens, or Docker tokens in `.env.cpu`.

## Pull or build image

Immutable pull:

```bash
docker pull gianguyen14/aic-retrieval:finals-20260917
docker image inspect gianguyen14/aic-retrieval:finals-20260917
```

Build from the pinned repository source:

```bash
SOURCE_SHA=$(git rev-parse HEAD)
docker build --pull -t gianguyen14/aic-retrieval:sha-${SOURCE_SHA:0:12} .
```

The `.dockerignore` excludes data, models, caches, results, logs, tests/docs, paper files, credentials, and environment files.

## Start CPU Docker

The container has one FastAPI process on internal `0.0.0.0:8000`. Both host ports map to that same process; this does not create a second Qwen instance.

```bash
docker compose --env-file .env.cpu -f docker-compose.cpu.yml up -d
```

Check status/logs:

```bash
docker compose --env-file .env.cpu -f docker-compose.cpu.yml ps
docker compose --env-file .env.cpu -f docker-compose.cpu.yml logs -f aic
```

The host publishes:

```text
0.0.0.0:3000 -> container:8000
0.0.0.0:8000 -> container:8000
```

Users open `http://SERVER_IP:3000` or `http://SERVER_IP:8000`; the two URLs reach the same application process. `0.0.0.0` is only a bind address and is not a browser URL.

## Verify CPU runtime

```bash
docker compose --env-file .env.cpu -f docker-compose.cpu.yml ps
docker inspect --format '{{.State.Health.Status}}' aic-retrieval
curl -fsS http://127.0.0.1:8000/health/ready
curl -fsS http://127.0.0.1:3000/health/ready
curl -I http://127.0.0.1:3000/
curl -I http://127.0.0.1:8000/
ss -lntp | grep -E ':3000|:8000'
```

The health payload must identify `qwen3_vl`, the production DB root inside the container (`/data/runtime`), Qwen weights present, dimension 1024, and initialized search.

Run finals smoke:

```bash
BACKEND_URL=http://127.0.0.1:8000 FRONTEND_URL=http://127.0.0.1:3000 bash ops/finals_smoke.sh
```

Expected checks: backend health, frontend HTTP, KIS, QA, known-good TRAKE, video preview, and video range all PASS.

## LAN access

Find the server IPv4:

```bash
hostname -I
ip addr
```

From a laptop on the same LAN:

```text
http://SERVER_IP:3000
```

Backend/API:

```text
http://SERVER_IP:8000
http://SERVER_IP:8000/health
```

If UFW is active, check first:

```bash
sudo ufw status
```

Then, if the LAN is `192.168.1.0/24`, an administrator may allow only that subnet:

```bash
sudo ufw allow from 192.168.1.0/24 to any port 3000 proto tcp
sudo ufw allow from 192.168.1.0/24 to any port 8000 proto tcp
```

Never expose unauthenticated ports 3000/8000 directly to the public Internet.

## Use the UI

KIS: choose Textual KIS, enter a natural-language visual description, search, inspect score/video/frame/timestamp, then open the video and seek to the returned timestamp.

QA: choose Q&A and enter a question. Default QA is extractive evidence handling, not generative LLM synthesis. OCR/ASR evidence may appear with the result.

TRAKE: choose TRAKE, add ordered events, and search. The result must be one video with event frame IDs in increasing order. A no-match response is a valid semantic outcome; do not weaken the constraint.

Current weighted evidence is visual 0.70, OCR 0.18, and ASR 0.12. The DB contains pre-extracted evidence spools; query-time OCR/ASR is lookup/scoring, not full live re-ingestion.

The packed DB may not contain JPEG frame previews. `Frame image unavailable` is expected in that case. Use video preview and seek using `timestamp_seconds`; never derive authoritative frame identity with `timestamp * FPS`.

## CPU dtype profiles

- `SAFE_CPU_LOW_RAM`: current BF16 production path. Use below 16 GiB or whenever FP32 qualification fails.
- `FAST_CPU_FP32`: do not enable solely from RAM size. It requires >=16 GiB, >=2 GiB measured post-startup headroom, no swap pressure, embedding/retrieval compatibility, and KIS/QA/TRAKE/frontend/video smoke PASS.

The current compose file intentionally does not silently switch dtype. Any FP32 selection must be qualified on the actual finals host before changing the image/runtime configuration.

## Stop, restart, update, rollback

Stop without deleting volumes:

```bash
docker compose --env-file .env.cpu -f docker-compose.cpu.yml down
```

Restart the same immutable image:

```bash
docker compose --env-file .env.cpu -f docker-compose.cpu.yml restart aic
```

Upgrade to another immutable tag:

```bash
sed -i 's/^IMAGE_TAG=.*/IMAGE_TAG=sha-NEW_FULL_TAG/' .env.cpu
docker compose --env-file .env.cpu -f docker-compose.cpu.yml pull
docker compose --env-file .env.cpu -f docker-compose.cpu.yml up -d
docker compose --env-file .env.cpu -f docker-compose.cpu.yml ps
```

Rollback by setting the previous immutable tag and repeating pull/up. Never use `latest` for finals.

Remove the container and network while preserving host mounts:

```bash
docker compose --env-file .env.cpu -f docker-compose.cpu.yml down
```

Remove an image only after confirming no running container uses it:

```bash
docker image rm gianguyen14/aic-retrieval:TAG
```

Do not remove `/home/hermes/aic/data/aic-db-v1/runtime`, model, raw video, or cache directories as part of ordinary uninstall.

## Troubleshooting

| Problem | Diagnosis | Fix |
|---|---|---|
| Docker daemon unavailable | `docker info` | Start Docker Engine; do not change application config |
| Docker socket permission denied | `id`, `docker info` | Add the operator to the Docker group according to local policy, then start a new login session |
| Compose plugin missing | `docker compose version` | Install the official Compose plugin |
| Port 3000/8000 occupied | `ss -lntp \| grep -E ':3000|:8000'` | Stop the correct old container/process; never launch duplicate Qwen instances |
| LAN unreachable | `ss -lntp`, firewall, `hostname -I` | Confirm published ports bind `0.0.0.0`, use server IPv4, allow LAN subnet |
| CORS error | `Origin` browser URL vs `ALLOWED_ORIGINS` | Set exact `http://SERVER_IP:3000` origin in `.env.cpu` and recreate container |
| Health not ready | `docker logs aic-retrieval`, `/health/ready` | Check DB `index/CURRENT`, model mount, permissions, and container health |
| DB CURRENT missing | `test -f HOST/data/.../index/CURRENT` | Correct `AIC_DATA_ROOT`; keep it read-only |
| FAISS/mapping mismatch | health/logs | Use matching production DB generation; do not rebuild during deployment |
| Model missing | `test -f HOST/models/.../model.safetensors` | Correct `AIC_MODEL_ROOT`; do not bake model into image |
| Model permission issue | `docker exec aic-retrieval id`, host permissions | Grant read permission to container UID 1000 without making model writable |
| RAM OOM/swap | `free -h`, `docker stats` | Use BF16 safe profile, one container, one worker; stop before swap thrash |
| CUDA unavailable | `nvidia-smi`, `docker run --rm --gpus all ...` | Use CPU compose or fix driver/toolkit before GPU selection |
| GPU OOM | `docker logs`, `nvidia-smi` | Stop GPU container, use compatible GPU/profile; do not silently fall back and claim GPU PASS |
| KIS/QA slow | `docker stats`, response latency | Qwen CPU embedding is the bottleneck; qualify hardware before dtype change |
| TRAKE no match | API `detail` | Valid strict same-video monotonic no-match; use a known-good sequence for smoke |
| Frame JPEG unavailable | UI message or frame 404 | Use video seek fallback; do not repack DB automatically |
| Video preview 404 | `VIDEO_SOURCE_DIR`, host video file | Correct `AIC_VIDEO_ROOT` and matching video IDs |
| Container restart loop | `docker compose ps`, logs | Read first startup error; check mounts/permissions/model/DB; do not keep restarting blindly |
| Pull digest mismatch | `docker image inspect`, registry digest | Pull the exact immutable SHA tag and verify digest before startup |

## Diagnostic command

```bash
bash ops/docker/diagnose.sh
```

It prints system/runtime diagnostics but no credential values.

# AIC Docker Operations

The current release uses one FastAPI process inside the container on port 8000. Host ports 3000 and 8000 both map to that same internal port, so the two URLs do not create two Qwen instances.

## CPU

```bash
cp ops/docker/.env.cpu.example .env.cpu
# Replace SERVER_IP and review paths in .env.cpu
docker compose --env-file .env.cpu -f docker-compose.cpu.yml up -d
docker compose --env-file .env.cpu -f docker-compose.cpu.yml ps
BACKEND_URL=http://127.0.0.1:8000 FRONTEND_URL=http://127.0.0.1:3000 bash ops/finals_smoke.sh
```

See `ops/DOCKER_CPU_GUIDE.md` for A-Z setup, CPU profiles, LAN, updates, rollback, and troubleshooting.

## GPU

```bash
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
cp ops/docker/.env.gpu.example .env.gpu
# Replace SERVER_IP and review paths in .env.gpu
docker compose --env-file .env.gpu -f docker-compose.gpu.yml up -d
docker compose --env-file .env.gpu -f docker-compose.gpu.yml ps
BACKEND_URL=http://127.0.0.1:8000 FRONTEND_URL=http://127.0.0.1:3000 bash ops/finals_smoke.sh
```

See `ops/DOCKER_GPU_GUIDE.md` for NVIDIA Container Toolkit, CUDA qualification, VRAM monitoring, and fallback procedures.

## Full system usage

See `ops/FULL_SYSTEM_USAGE_GUIDE.md` for host-native and Docker usage, KIS/QA/TRAKE, LAN access, firewall/CORS, health, and recovery.

Do not put model, DB, videos, caches, or credentials into the image. Mount them externally.

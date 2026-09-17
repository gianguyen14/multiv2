# AIC GPU Docker Deployment Guide

## Scope

This guide qualifies the same AIC application on an NVIDIA CUDA host. It does not assume a GPU model or CUDA version. The GPU host must have a supported NVIDIA driver and NVIDIA Container Toolkit; the container supplies the PyTorch CUDA runtime selected by the GPU compose build.

## Host prerequisites

```bash
nvidia-smi
docker --version
docker compose version
docker info
```

The following must work before AIC startup:

```bash
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

If the test image cannot be pulled, use an approved local CUDA test image. Do not claim GPU validation from host `nvidia-smi` alone.

Host responsibilities:

- NVIDIA kernel driver
- NVIDIA Container Toolkit
- Docker Engine and Compose plugin

Container responsibilities:

- CUDA-enabled PyTorch wheel selected by `TORCH_INDEX_URL`
- AIC Python/runtime dependencies
- application code only; model/DB/videos remain external mounts

## Prepare and pin source

```bash
git clone https://github.com/gianguyen14/multiv2.git /home/hermes/aic
cd /home/hermes/aic
git fetch origin
git checkout main
git pull --ff-only origin main
git rev-parse HEAD
```

## Prepare external resources

The GPU deployment uses the same host resources as CPU:

```text
AIC_DATA_ROOT=/home/hermes/aic/data/aic-db-v1/runtime
AIC_MODEL_ROOT=/home/hermes/aic/models/Qwen3-VL-Embedding-2B
AIC_VIDEO_ROOT=/video/video
AIC_CACHE_ROOT=/home/hermes/aic/cache/videos
```

Verify:

```bash
test -f /home/hermes/aic/data/aic-db-v1/runtime/index/CURRENT
test -f /home/hermes/aic/models/Qwen3-VL-Embedding-2B/model.safetensors
test -d /video/video
```

## Configure GPU environment

```bash
cd /home/hermes/aic
cp ops/docker/.env.gpu.example .env.gpu
sed -i 's/SERVER_IP/192.168.1.50/g' .env.gpu
sed -n '1,140p' .env.gpu
```

The example uses a CUDA 12.4 PyTorch wheel index. Verify that the selected wheel, NVIDIA driver, and actual GPU are compatible before build. Do not infer compatibility from the CUDA index URL alone.

## Build or pull image

The GPU compose file uses the same application Dockerfile and asks the build to install a CUDA PyTorch wheel before installing the requirements. Because the current requirements file specifies unpinned `torch`, validate the resolved wheel in the build output and inside the container.

```bash
SOURCE_SHA=$(git rev-parse HEAD)
docker build --pull \
  --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124 \
  -t gianguyen14/aic-retrieval:gpu-${SOURCE_SHA:0:12} .
```

Alternatively pull a previously qualified immutable GPU tag:

```bash
docker pull gianguyen14/aic-retrieval:GPU_TAG
```

Do not use a GPU tag that has only static build validation when finals runtime validation is required.

## Validate CUDA inside the image

```bash
docker run --rm --gpus all \
  -e CUDA_VISIBLE_DEVICES=all \
  gianguyen14/aic-retrieval:gpu-SHORT_SHA \
  python -c 'import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else None); print(torch.version.cuda)'
```

A GPU profile is not accepted if `torch.cuda.is_available()` is false or the application silently reports CPU fallback.

## Start GPU Docker

```bash
docker compose --env-file .env.gpu -f docker-compose.gpu.yml up -d
```

Check:

```bash
docker compose --env-file .env.gpu -f docker-compose.gpu.yml ps
docker compose --env-file .env.gpu -f docker-compose.gpu.yml logs -f aic
nvidia-smi
docker exec aic-retrieval python -c 'import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else None); print(torch.version.cuda)'
```

The single application process listens on internal `0.0.0.0:8000`; host ports map `3000:8000` and `8000:8000`.

## GPU qualification gate

Run only after container startup:

```bash
curl -fsS http://127.0.0.1:8000/health/ready
curl -fsS http://127.0.0.1:3000/health/ready
BACKEND_URL=http://127.0.0.1:8000 FRONTEND_URL=http://127.0.0.1:3000 bash ops/finals_smoke.sh
```

Also record:

```bash
docker stats --no-stream aic-retrieval
nvidia-smi --query-gpu=name,memory.total,memory.used,driver_version --format=csv
```

The GPU profile is eligible only when:

- CUDA is visible inside the container;
- the Qwen model actually runs on CUDA according to health/runtime diagnostics;
- embeddings are finite, normalized, and dimension 1024;
- no VRAM OOM occurs;
- KIS, QA, known-good TRAKE, frontend, video preview, and byte-range checks pass;
- a known query remains compatible with the expected production result.

## GPU dtype and semantics

Keep the existing model, processor, instruction, pooling, MRL truncation to 1024 dimensions, and L2 normalization. Do not choose BF16/FP16 merely because the GPU exists; validate the actual GPU capability and retrieval compatibility first. The current source adapter defaults to BF16 and the official model script selects CUDA when PyTorch CUDA is available; verify the runtime payload before calling the deployment GPU-qualified.

## LAN access

```bash
hostname -I
ss -lntp | grep -E ':3000|:8000'
```

Open from another LAN machine:

```text
http://SERVER_IP:3000
```

Backend:

```text
http://SERVER_IP:8000
http://SERVER_IP:8000/health
```

Use explicit `ALLOWED_ORIGINS=http://SERVER_IP:3000,http://SERVER_IP:8000` and restrict firewall rules to the LAN subnet where possible.

## GPU monitoring and troubleshooting

| Problem | Check | Fix |
|---|---|---|
| `nvidia-smi` fails on host | `nvidia-smi` | Install/configure the approved NVIDIA driver |
| Docker cannot see GPU | `docker run --rm --gpus all ... nvidia-smi` | Install/configure NVIDIA Container Toolkit and restart Docker service only during planned maintenance |
| PyTorch CUDA false | `docker exec ... python -c 'import torch; ...'` | Verify CUDA wheel, driver compatibility, and GPU passthrough |
| App falls back to CPU | `/health` compute payload and logs | Do not claim GPU PASS; stop and correct device configuration |
| GPU OOM | `nvidia-smi`, `docker stats`, logs | Stop container, use a GPU with sufficient VRAM or approved safe dtype; do not change model semantics casually |
| Driver/CUDA mismatch | `nvidia-smi`, `torch.version.cuda` | Select a compatible container wheel/image; never randomly change CUDA versions |
| KIS/QA/TRAKE fail | `bash ops/finals_smoke.sh` | Inspect exact endpoint/log; keep CPU fallback separate and explicitly qualified |

## Stop, update, rollback

```bash
docker compose --env-file .env.gpu -f docker-compose.gpu.yml down
docker compose --env-file .env.gpu -f docker-compose.gpu.yml restart aic
```

For an immutable upgrade or rollback, edit only `IMAGE_TAG` in `.env.gpu`, then:

```bash
docker compose --env-file .env.gpu -f docker-compose.gpu.yml pull
docker compose --env-file .env.gpu -f docker-compose.gpu.yml up -d
docker compose --env-file .env.gpu -f docker-compose.gpu.yml ps
```

Never force-push source or overwrite an existing immutable image tag.

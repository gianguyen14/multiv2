# AIC GPU Docker deployment (Docker Hub first)

GPU is an optional finals profile. The current Docker Hub GPU tags have not been published because this host has no NVIDIA GPU and the attempted CUDA build exhausted local disk space. Do not use a nonexistent GPU tag.

## 1. Qualify the GPU host

The finals host must have an NVIDIA driver, NVIDIA Container Toolkit, Docker Engine, and sufficient VRAM:

```bash
nvidia-smi
docker --version
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

If these fail, use the CPU image. Do not claim GPU runtime validation from host `nvidia-smi` alone.

## 2. GPU image status

Current verified status:

```text
GPU_DOCKER_HUB_TAG: NOT AVAILABLE
GPU_RUNTIME_VALIDATION: BLOCKED_NO_GPU_HOST
```

A GPU image must not be pulled until `gpu-finals` and an immutable `gpu-SHA` tag are actually published after a successful build. No GPU image is documented as ready by this release.

## 3. External resources

The operator supplies the same external resources as CPU:

```text
/opt/aic/data/aic-db-v1/runtime
/opt/aic/models/Qwen3-VL-Embedding-2B
/opt/aic/videos
/opt/aic/cache
```

Verify:

```bash
sudo test -f /opt/aic/data/aic-db-v1/runtime/index/CURRENT
sudo test -f /opt/aic/models/Qwen3-VL-Embedding-2B/model.safetensors
sudo test -d /opt/aic/videos
```

## 4. Start after a GPU tag is published

Replace `GPU_SHA` only with a real published immutable tag:

```bash
docker pull gianguyen14/aic-retrieval:gpu-GPU_SHA
```

Then run one container/process:

```bash
docker run -d \
  --name aic-retrieval \
  --restart unless-stopped \
  --gpus all \
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
  -e COMPUTE_DEVICE=cuda \
  -e VISUAL_DEVICE=cuda \
  -e ASR_DEVICE=cuda \
  -e ALLOWED_ORIGINS=http://SERVER_IP:3000 \
  -e HF_HUB_OFFLINE=1 \
  -e TRANSFORMERS_OFFLINE=1 \
  gianguyen14/aic-retrieval:gpu-GPU_SHA
```

`0.0.0.0` is only the bind address. Users open `http://SERVER_IP:3000` and `http://SERVER_IP:8000`.

## 5. GPU runtime gate

```bash
docker exec aic-retrieval python -c 'import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else None); print(torch.version.cuda)'
curl -fsS http://127.0.0.1:8000/health/ready
curl -I http://127.0.0.1:3000/
docker top aic-retrieval
nvidia-smi
```

Accept GPU only when CUDA is visible inside the container, the Qwen model actually uses CUDA, no VRAM OOM occurs, health is ready, one application process is running, and KIS/QA/known-good TRAKE/video/range smoke passes. Keep the existing model, instruction, pooling, 1024-D truncation, normalization, and retrieval semantics.

## 6. Download-only Compose option

Once a real GPU tag exists, an operator can download only the deployment files:

```bash
mkdir -p /opt/aic/config
curl -fsSL https://raw.githubusercontent.com/gianguyen14/multiv2/main/docker-compose.gpu.yml -o /opt/aic/config/compose.gpu.yml
curl -fsSL https://raw.githubusercontent.com/gianguyen14/multiv2/main/ops/docker/.env.gpu.example -o /opt/aic/config/.env.gpu
```

Review paths and `SERVER_IP`, then:

```bash
docker compose --env-file /opt/aic/config/.env.gpu -f /opt/aic/config/compose.gpu.yml pull
docker compose --env-file /opt/aic/config/.env.gpu -f /opt/aic/config/compose.gpu.yml up -d
docker compose --env-file /opt/aic/config/.env.gpu -f /opt/aic/config/compose.gpu.yml ps
```

## 7. Developer / build from source

This is secondary and not required for normal deployment. On an NVIDIA build host, a developer may clone the repository and run:

```bash
TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124 bash ops/docker/build-gpu.sh
```

Then the image must be live-qualified on the actual GPU host before any `gpu-finals` tag is pushed. Never publish a runtime-unverified image as GPU-ready.

# Qwen GPU Docker images

Source freeze: `568e1256b5e16ecfaa66f44ac0f5478286c2e7`.

The Dockerfile is a single application image definition with an explicit PyTorch CUDA wheel index. Build variants are distinguished by dependency index:

- V100/Volta: CUDA 11.8 PyTorch wheels, FP16 policy, tag `gpu-v100-568e125`.
- Modern GPU: CUDA 12.4 PyTorch wheels, auto BF16/FP16 policy, tag `gpu-modern-568e125`.

The build host used for this audit has no NVIDIA GPU, so both GPU runtime validations are `PENDING_HARDWARE_VALIDATION`. Docker build/push must not be described as GPU hardware validation.

## Build

From the frozen branch and required SHA:

```bash
bash ops/docker/build-gpu-qwen.sh
```

The script fails before build if `git rev-parse HEAD` is not the required source SHA.

## CPU/import smoke

```bash
for image in \
  gianguyen14/aic-retrieval:gpu-v100-568e125 \
  gianguyen14/aic-retrieval:gpu-modern-568e125; do
  docker run --rm --entrypoint python "$image" --version
  docker run --rm --entrypoint python "$image" -c \
    'import torch,av,numpy,faiss,transformers; print(torch.__version__, torch.version.cuda); print("imports ok")'
  docker run --rm --entrypoint python "$image" projectctl.py --help
  docker run --rm --entrypoint python "$image" -c \
    'from backend.app.main import app; print(app.title)'
done
```

## Real GPU validation

Only run on a host with NVIDIA Container Toolkit and an actual target GPU:

```bash
docker run --rm --gpus all --entrypoint python \
  gianguyen14/aic-retrieval:gpu-v100-568e125 -c \
  'import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0)); print(torch.cuda.get_device_capability(0))'
```

Then mount the model and a probe image:

```bash
docker run --rm --gpus all \
  -v /workspace/models:/models:ro \
  -v /workspace/probe.jpg:/tmp/probe.jpg:ro \
  -e QWEN3_VL_MODEL_DIR=/models/Qwen3-VL-Embedding-2B \
  gianguyen14/aic-retrieval:gpu-v100-568e125 \
  python scripts/validate_gpu_ingest.py \
    --model /models/Qwen3-VL-Embedding-2B \
    --image /tmp/probe.jpg --device cuda:0 --dtype float16
```

Do not claim V100, Blackwell, MIG, throughput, VRAM, or CUDA PASS without real execution.

## Runtime mounts

Do not bake model/data/video/index/cache into the image. Mount them externally:

```text
/models
/data
/videos
/output
```

Set:

```text
INGEST_BACKEND=qwen3_vl
GPU_STRICT=true
QWEN3_VL_MODEL_DIR=/models/Qwen3-VL-Embedding-2B
VIDEO_PROCESSED_ROOT=/data/processed
```

Production DB/CURRENT remains read-only and is never modified by image build or import smoke.

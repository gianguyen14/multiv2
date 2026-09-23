# AIC GPU Image Builds

Two CUDA profiles are produced by `ops/docker/build-gpu-qwen.sh`:

- V100/Volta: CUDA 11.8, PyTorch 2.7.1+cu118, torchvision 0.22.1+cu118, FP16 policy.
- Modern GPU (Ampere/Ada/Hopper/Blackwell): CUDA 12.8, PyTorch 2.7.1+cu128, torchvision 0.22.1+cu128, auto BF16/FP16 policy. This includes RTX 5090, RTX PRO 6000, and MIG-partitioned modern GPUs where the runtime exposes the required CUDA capability.

Both profiles build with `INSTALL_YOLO=true` (Ultralytics included, weights external).

## Tags

Tags are derived from the actual packaging HEAD SHA at build time:

    gianguyen14/aic-retrieval:gpu-v100-<7-char-SHA>
    gianguyen14/aic-retrieval:gpu-modern-<7-char-SHA>

No frozen feature SHA is required; the script works from any commit.

## Build

```bash
cd /path/to/multiv2
bash ops/docker/build-gpu-qwen.sh
```

Override image repository:

```bash
IMAGE_REPOSITORY=myorg/aic-retrieval bash ops/docker/build-gpu-qwen.sh
```

## Counting-ready images

Both GPU images include Ultralytics (INSTALL_YOLO=true at build time).
COUNTING_ENABLED defaults to false; enable at runtime:

    COUNTING_ENABLED=true
    OBJECT_DETECTOR_MODEL=/models/yolo/yolo11n.pt   # external mount required

## Container smoke checks

```bash
# Dependency integrity
docker run --rm --entrypoint pip <IMAGE> check

# PyTorch / CUDA version (V100 example)
docker run --rm <IMAGE> python -c "
import torch, torchvision
assert torch.__version__ == '2.7.1+cu118'
assert torchvision.__version__ == '0.22.1+cu118'
assert torch.version.cuda == '11.8'
print(torch.__version__, torchvision.__version__, torch.version.cuda)
"

# Ultralytics + adapter
docker run --rm <IMAGE> python -c "
import ultralytics; print(ultralytics.__version__)
from backend.app.vision.yolo_detector import YoloDetector; print('ok')
"

# GPU validation script
docker run --rm <IMAGE> python scripts/validate_gpu_ingest.py --help
```

## External mounts required at runtime

| Host path | Container path | Purpose |
|---|---|---|
| DB v1 runtime | /data/runtime | FAISS index, OCR/ASR spools |
| Qwen3-VL model | /models/Qwen3-VL-Embedding-2B | embedding weights |
| YOLO weights | /models/yolo/ | optional detector weights |
| Video files | /videos | raw MP4 source |
| Video cache | /cache/videos | transcoded preview cache |
| Detection cache | /cache/detections | YOLO result JSON cache |

## GPU runtime validation

Without a real NVIDIA host, GPU runtime is blocked:

    GPU_RUNTIME_VALIDATION=BLOCKED_NO_GPU_HOST

Run `scripts/validate_gpu_ingest.py` on an authorized GPU host to confirm
embedding parity before declaring GPU ready.

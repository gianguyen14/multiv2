#!/usr/bin/env bash
# Build CUDA GPU images for the AIC retrieval service.
# Supports two profiles:
#   gpu-v100   — CUDA 11.8 / PyTorch 2.7.1+cu118 (V100 / Volta)
#   gpu-modern — CUDA 12.8 / PyTorch 2.7.1+cu128 (Ampere / Ada / Hopper)
#
# Counting-ready images include Ultralytics (INSTALL_YOLO=true).
# Weights remain on external /models mounts; COUNTING_ENABLED defaults to false.
#
# Tags use the actual packaged-source HEAD SHA.
set -euo pipefail

SOURCE_SHA="$(git rev-parse HEAD)"

REPO="${IMAGE_REPOSITORY:-gianguyen14/aic-retrieval}"
PLATFORM="linux/amd64"

# ── V100 (CUDA 11.8) ──────────────────────────────────────────────────────────
echo "Building V100 image from $SOURCE_SHA"
docker build --pull --platform "$PLATFORM" \
  --build-arg TORCH_INDEX_URL="https://download.pytorch.org/whl/cu118" \
  --build-arg TORCH_VERSION="2.7.1+cu118" \
  --build-arg TORCHVISION_VERSION="0.22.1+cu118" \
  --build-arg INSTALL_YOLO="true" \
  --build-arg SOURCE_REVISION="$SOURCE_SHA" \
  -t "$REPO:gpu-v100-${SOURCE_SHA:0:7}" .

# Dependency integrity guard
docker run --rm --entrypoint pip "$REPO:gpu-v100-${SOURCE_SHA:0:7}" check

# Hard version assertion (torch, torchvision, CUDA)
docker run --rm "$REPO:gpu-v100-${SOURCE_SHA:0:7}" python -c "
import torch, torchvision
assert torch.__version__ == '2.7.1+cu118', torch.__version__
assert torchvision.__version__ == '0.22.1+cu118', torchvision.__version__
assert torch.version.cuda == '11.8', torch.version.cuda
print('V100 torch OK:', torch.__version__, torchvision.__version__, torch.version.cuda)
"

# Ultralytics + adapter smoke
docker run --rm "$REPO:gpu-v100-${SOURCE_SHA:0:7}" python -c "
import ultralytics; print('ultralytics', ultralytics.__version__)
from backend.app.vision.yolo_detector import YoloDetector; print('yolo adapter import ok')
"

# Validation script available
docker run --rm "$REPO:gpu-v100-${SOURCE_SHA:0:7}" python scripts/validate_gpu_ingest.py --help > /dev/null

echo "V100: $REPO:gpu-v100-${SOURCE_SHA:0:7}  OK"

# ── Modern GPU (CUDA 12.8) ────────────────────────────────────────────────────
echo "Building modern GPU image from $SOURCE_SHA"
docker build --pull --platform "$PLATFORM" \
  --build-arg TORCH_INDEX_URL="https://download.pytorch.org/whl/cu128" \
  --build-arg TORCH_VERSION="2.7.1+cu128" \
  --build-arg TORCHVISION_VERSION="0.22.1+cu128" \
  --build-arg INSTALL_YOLO="true" \
  --build-arg SOURCE_REVISION="$SOURCE_SHA" \
  -t "$REPO:gpu-modern-${SOURCE_SHA:0:7}" .

docker run --rm --entrypoint pip "$REPO:gpu-modern-${SOURCE_SHA:0:7}" check

docker run --rm "$REPO:gpu-modern-${SOURCE_SHA:0:7}" python -c "
import torch, torchvision
assert torch.__version__ == '2.7.1+cu128', torch.__version__
assert torchvision.__version__ == '0.22.1+cu128', torchvision.__version__
assert torch.version.cuda == '12.8', torch.version.cuda
print('Modern torch OK:', torch.__version__, torchvision.__version__, torch.version.cuda)
"

docker run --rm "$REPO:gpu-modern-${SOURCE_SHA:0:7}" python -c "
import ultralytics; print('ultralytics', ultralytics.__version__)
from backend.app.vision.yolo_detector import YoloDetector; print('yolo adapter import ok')
"

docker run --rm "$REPO:gpu-modern-${SOURCE_SHA:0:7}" python scripts/validate_gpu_ingest.py --help > /dev/null

echo "Modern: $REPO:gpu-modern-${SOURCE_SHA:0:7}  OK"

echo ""
echo "Built:"
echo "  $REPO:gpu-v100-${SOURCE_SHA:0:7}"
echo "  $REPO:gpu-modern-${SOURCE_SHA:0:7}"

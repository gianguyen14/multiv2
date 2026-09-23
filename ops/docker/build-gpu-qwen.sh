#!/usr/bin/env bash
# Build CUDA GPU images for the AIC retrieval service.
# Supports two profiles:
#   gpu-v100  — CUDA 11.8 / PyTorch 2.7.1+cu118 (V100 / Volta)
#   gpu-modern — CUDA 12.8 / PyTorch 2.7.1+cu128 (Ampere / Ada / Hopper)
#
# Tags use the actual packaged-source HEAD SHA, not a frozen feature SHA,
# so the script works from any commit on this branch.
set -euo pipefail

SOURCE_SHA="$(git rev-parse HEAD)"

REPO="${IMAGE_REPOSITORY:-gianguyen14/aic-retrieval}"
PLATFORM="linux/amd64"

echo "Building V100 image from $SOURCE_SHA"
docker build --pull --platform "$PLATFORM" \
  --build-arg TORCH_INDEX_URL="https://download.pytorch.org/whl/cu118" \
  --build-arg TORCH_VERSION="2.7.1+cu118" \
  --build-arg TORCHVISION_VERSION="0.22.1+cu118" \
  --build-arg SOURCE_REVISION="$SOURCE_SHA" \
  -t "$REPO:gpu-v100-${SOURCE_SHA:0:7}" .
# Verify dependency consistency after install
docker run --rm --entrypoint pip "$REPO:gpu-v100-${SOURCE_SHA:0:7}" check

echo "Building modern GPU image from $SOURCE_SHA"
docker build --pull --platform "$PLATFORM" \
  --build-arg TORCH_INDEX_URL="https://download.pytorch.org/whl/cu128" \
  --build-arg TORCH_VERSION="2.7.1+cu128" \
  --build-arg TORCHVISION_VERSION="0.22.1+cu128" \
  --build-arg SOURCE_REVISION="$SOURCE_SHA" \
  -t "$REPO:gpu-modern-${SOURCE_SHA:0:7}" .
docker run --rm --entrypoint pip "$REPO:gpu-modern-${SOURCE_SHA:0:7}" check

echo "Built:"
echo "  $REPO:gpu-v100-${SOURCE_SHA:0:7}"
echo "  $REPO:gpu-modern-${SOURCE_SHA:0:7}"

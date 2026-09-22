#!/usr/bin/env bash
set -euo pipefail

SOURCE_SHA="$(git rev-parse HEAD)"
REQUIRED_SHA="568e1256b5e16ecfaa66f44ac0f5478286e2c8cb"
[[ "$SOURCE_SHA" == "$REQUIRED_SHA" ]] || { echo "SOURCE_SHA_MISMATCH: $SOURCE_SHA" >&2; exit 2; }

REPO="gianguyen14/aic-retrieval"
PLATFORM="linux/amd64"

echo "Building V100 image from $SOURCE_SHA"
docker build --pull --platform "$PLATFORM" \
  --build-arg TORCH_INDEX_URL="https://download.pytorch.org/whl/cu118" \
  --build-arg TORCH_VERSION="2.7.1+cu118" \
  --build-arg TORCHVISION_VERSION="0.22.1+cu118" \
  --build-arg SOURCE_REVISION="$SOURCE_SHA" \
  -t "$REPO:gpu-v100-${SOURCE_SHA:0:7}" .

echo "Building modern GPU image from $SOURCE_SHA"
docker build --pull --platform "$PLATFORM" \
  --build-arg TORCH_INDEX_URL="https://download.pytorch.org/whl/cu124" \
  --build-arg TORCH_VERSION="2.6.0+cu124" \
  --build-arg TORCHVISION_VERSION="0.21.0+cu124" \
  --build-arg SOURCE_REVISION="$SOURCE_SHA" \
  -t "$REPO:gpu-modern-${SOURCE_SHA:0:7}" .

# GPU Docker build: uses a CUDA-specific PyTorch wheel index.
: "${TORCH_INDEX_URL:=https://download.pytorch.org/whl/cu124}"
SOURCE_SHA=$(git rev-parse HEAD)
docker build --pull --progress=plain \
  --build-arg TORCH_INDEX_URL="$TORCH_INDEX_URL" \
  -t gianguyen14/aic-retrieval:gpu-${SOURCE_SHA:0:12} \
  .

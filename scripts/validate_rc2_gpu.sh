#!/usr/bin/env bash
# RC2 GPU acceptance runner. It never pushes an image and requires real local
# model/video resources for the model and ingestion gates.
set -Eeuo pipefail

log() { echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*"; }
fail() { log "ERROR: $*"; exit 1; }

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

IMAGE_TAG="${AIC_VALIDATION_IMAGE_TAG:-aic-retrieval:rc2-validation}"
MODELS_DIR="${AIC_MODELS_DIR:-/opt/aic/models}"
CACHE_DIR="${AIC_CACHE_DIR:-/opt/aic/cache}"
VIDEOS_DIR="${AIC_VALIDATION_VIDEOS_DIR:-/opt/aic/videos-smoke}"
PROCESSED_DIR="${AIC_VALIDATION_PROCESSED_DIR:-/opt/aic/processed-rc2-acceptance}"
LOG_DIR="${AIC_VALIDATION_LOG_DIR:-logs}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

for command in docker nvidia-smi "${PYTHON_BIN}"; do
    command -v "${command}" >/dev/null 2>&1 || fail "required command is missing: ${command}"
done
[[ -d "${MODELS_DIR}" ]] || fail "model directory is missing: ${MODELS_DIR}"
[[ -d "${VIDEOS_DIR}" ]] || fail "validation video directory is missing: ${VIDEOS_DIR}"
mkdir -p "${CACHE_DIR}" "${PROCESSED_DIR}" "${LOG_DIR}"

log "GATE A - Docker and NVIDIA runtime"
docker version
docker info >/dev/null
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi

log "GATE B - Build isolated validation image ${IMAGE_TAG}"
BUILD_STARTED="$(date +%s)"
docker build --pull -t "${IMAGE_TAG}" .
BUILD_SECONDS="$(( $(date +%s) - BUILD_STARTED ))"
IMAGE_ID="$(docker image inspect --format '{{.Id}}' "${IMAGE_TAG}")"
[[ -n "${IMAGE_ID}" ]] || fail "built image has no ID"

COMMON_RUN=(
    --rm --gpus all
    -e HF_HOME=/cache/huggingface
    -e HF_HUB_OFFLINE=1
    -e TRANSFORMERS_OFFLINE=1
    -v "${MODELS_DIR}:/models:ro"
    -v "${CACHE_DIR}:/cache"
)

log "GATE C - Runtime imports and CUDA visibility"
docker run "${COMMON_RUN[@]}" "${IMAGE_TAG}" python -c '
import av, faiss, torch
print("torch", torch.__version__)
print("faiss", getattr(faiss, "__version__", "unknown"))
print("av", av.__version__)
assert torch.cuda.is_available(), "CUDA is not visible inside the release image"
print("gpu", torch.cuda.get_device_name(0))
'
docker run --rm "${IMAGE_TAG}" python -m pip check

log "GATE D - Entrypoint and provenance without a Git working tree"
docker run --rm "${IMAGE_TAG}" python -c '
from pathlib import Path
from tempfile import TemporaryDirectory
from backend.app.runtime.operations import write_run_manifest
with TemporaryDirectory() as directory:
    root = Path(directory)
    source = root / "source.mp4"
    source.write_bytes(b"fixture")
    result = write_run_manifest(root / "run.json", "validation", source, root, {}, {})
    assert result["git_commit"] is None
print("provenance fallback: OK")
'

log "GATE E - Reject multiple installed NCCL package families"
docker run --rm "${IMAGE_TAG}" python -c '
from importlib.metadata import distributions
packages = sorted({
    (dist.metadata.get("Name") or "").lower()
    for dist in distributions()
    if (dist.metadata.get("Name") or "").lower().startswith("nvidia-nccl-")
})
print("NCCL distributions", packages)
assert len(packages) <= 1, f"multiple NCCL package families installed: {packages}"
'

log "GATE F - Offline CPU regression suite on the checked-out source"
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    "${PYTHON_BIN}" -m pytest -m 'not slow and not real_model and not network and not gpu' \
    | tee "${LOG_DIR}/rc2_pytest.log"

log "GATE G - Query-refiner model is available offline"
docker run "${COMMON_RUN[@]}" "${IMAGE_TAG}" \
    python projectctl.py models --verify-offline --query-refiner --json

log "GATE H - Three-video GPU ingestion with intended frame policy"
docker run "${COMMON_RUN[@]}" \
    -e VISUAL_SAMPLING_MODE=sparse_shot \
    -e VISUAL_GLOBAL_SAMPLE_SECONDS=5.0 \
    -e VISUAL_DEDUP_ENABLED=true \
    -e VISUAL_DEDUP_THRESHOLD=0.97 \
    -v "${VIDEOS_DIR}:/videos:ro" \
    -v "${PROCESSED_DIR}:/data/processed" \
    "${IMAGE_TAG}" python projectctl.py ingest /videos \
    --processed-root /data/processed --device cuda --limit 3 --json

log "GATE I - KIS query against the newly published index"
docker run "${COMMON_RUN[@]}" \
    -e VIDEO_PROCESSED_ROOT=/data/processed \
    -v "${PROCESSED_DIR}:/data/processed:ro" \
    "${IMAGE_TAG}" python projectctl.py kis "sample KIS query" \
    --top-k 10 --no-query-refine --json

log "GATE J - Write acceptance summary"
SUMMARY_PATH="${LOG_DIR}/rc2_acceptance_summary.json"
"${PYTHON_BIN}" - "${SUMMARY_PATH}" "${IMAGE_TAG}" "${IMAGE_ID}" "${BUILD_SECONDS}" <<'PY'
import json
import sys

path, image_tag, image_id, build_seconds = sys.argv[1:]
summary = {
    "status": "PASS",
    "image_tag": image_tag,
    "image_id": image_id,
    "build_seconds": int(build_seconds),
    "gates": {letter: "PASS" for letter in "ABCDEFGHI"},
}
with open(path, "w", encoding="utf-8") as handle:
    json.dump(summary, handle, indent=2)
    handle.write("\n")
PY

log "All mandatory RC2 GPU acceptance gates passed"

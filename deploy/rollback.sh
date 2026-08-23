#!/usr/bin/env bash
# ==============================================================================
# AIC 2026 — Production Rollback Script
# Usage: ./deploy/rollback.sh <git-commit-hash-or-tag>
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

if [ $# -lt 1 ]; then
    echo "Usage: $0 <git-commit-hash-or-tag>"
    echo "Example: $0 main@{1} or $0 v1.0.0"
    exit 1
fi

TARGET_REVISION="$1"

echo "========================================="
echo " Rolling back AIC Service to: ${TARGET_REVISION}"
echo "========================================="

# 1. Checkout target revision
echo "Checking out ${TARGET_REVISION}..."
git checkout "${TARGET_REVISION}"

# 2. Rebuild Docker image for rollback revision
echo "Rebuilding Docker image..."
docker compose build

# 3. Restart container stack (persistent volumes untouched)
echo "Restarting service..."
docker compose up -d

echo "========================================="
echo " Rollback to ${TARGET_REVISION} complete!"
docker compose ps
echo "========================================="

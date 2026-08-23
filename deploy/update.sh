#!/usr/bin/env bash
# ==============================================================================
# AIC 2026 — Production Update Script
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

echo "========================================="
echo " Updating AIC 2026 Retrieval Service"
echo "========================================="

# 1. Pull latest code fast-forward only
echo "Pulling latest code from git..."
git pull --ff-only

# 2. Rebuild Docker image
echo "Building updated Docker image..."
docker compose build

# 3. Restart container with updated image (preserving persistent volumes)
echo "Recreating container with updated image..."
docker compose up -d

echo "========================================="
echo " Update complete! Current status:"
docker compose ps
echo "========================================="

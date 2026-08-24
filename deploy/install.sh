#!/usr/bin/env bash
# ==============================================================================
# AIC 2026 — Production Installation Script
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

echo "========================================="
echo " AIC 2026 Retrieval System Installation"
echo "========================================="

# 1. Verify Docker and Docker Compose
if ! command -v docker >/dev/null 2>&1; then
    echo "ERROR: Docker is not installed or not in PATH."
    exit 1
fi

if ! docker compose version >/dev/null 2>&1; then
    echo "ERROR: Docker Compose v2 plugin is not installed."
    exit 1
fi

# 2. Check for .env file
if [ ! -f .env ]; then
    echo "Creating .env from .env.example..."
    cp .env.example .env
    echo "NOTICE: .env created. Review and adjust environment variables if needed."
fi

# 3. Create persistent directories on host
AIC_DATA_DIR="${AIC_DATA_DIR:-./data}"
AIC_MODELS_DIR="${AIC_MODELS_DIR:-./models}"
AIC_CACHE_DIR="${AIC_CACHE_DIR:-./cache}"
AIC_LOGS_DIR="${AIC_LOGS_DIR:-./logs}"

echo "Creating host persistent directories..."
mkdir -p "${AIC_DATA_DIR}" "${AIC_MODELS_DIR}" "${AIC_CACHE_DIR}" "${AIC_LOGS_DIR}"

# 4. Build Docker image
echo "Building production Docker image (aic:latest)..."
docker compose build

# 5. Start container stack
echo "Starting AIC service..."
docker compose up -d

echo "========================================="
echo " AIC Service successfully installed and started!"
echo " Status:   ./deploy/status.sh"
echo " Logs:     docker compose logs -f aic"
echo "========================================="

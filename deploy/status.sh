#!/usr/bin/env bash
# ==============================================================================
# AIC 2026 — Service Status & Health Inspection Script
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

echo "========================================="
echo " AIC Service Container Status"
echo "========================================="
docker compose ps

echo ""
echo "========================================="
echo " HTTP Health Check"
echo "========================================="
if command -v curl >/dev/null 2>&1; then
    curl -sS -i http://127.0.0.1:8000/health/live || echo "Healthcheck failed (service might still be initializing)."
else
    python3 -c "import urllib.request; resp = urllib.request.urlopen('http://127.0.0.1:8000/health/live'); print(resp.read().decode())" || echo "Healthcheck request failed."
fi

echo ""
echo "========================================="
echo " Recent Logs (tail 50 lines)"
echo "========================================="
docker compose logs --tail=50 aic

#!/usr/bin/env bash
# Read-only deployment diagnostics. Never prints credential values.
set -u
printf 'OS: '; . /etc/os-release 2>/dev/null && printf '%s\n' "${PRETTY_NAME:-unknown}" || printf 'unknown\n'
printf 'KERNEL: '; uname -sr
printf 'CPU: '; lscpu 2>/dev/null | awk -F: '/Model name/ {gsub(/^ +| +$/, "", $2);print $2;exit}'
printf 'PHYSICAL_CORES: '; lscpu 2>/dev/null | awk -F: '/Core\(s\) per socket/ {gsub(/^ +| +$/, "", $2);print $2;exit}'
printf 'RAM: '; free -h 2>/dev/null | awk '/^Mem:/ {print $2 " total, " $7 " available"}'
printf 'SWAP: '; free -h 2>/dev/null | awk '/^Swap:/ {print $2 " total, " $3 " used"}'
printf 'DOCKER: '; docker --version 2>/dev/null || printf 'unavailable\n'
printf 'COMPOSE: '; docker compose version 2>/dev/null || printf 'unavailable\n'
printf 'DISK: '; df -h . 2>/dev/null | awk 'NR==2 {print $4 " free on " $1}'
printf 'PORTS:\n'; ss -lntp 2>/dev/null | grep -E ':3000|:8000' || printf 'none on 3000/8000\n'
printf 'CONTAINERS:\n'; docker ps --format 'table {{.Names}}\t{{.Image}}\t{{.Status}}\t{{.Ports}}' 2>/dev/null || printf 'docker unavailable\n'
printf 'GPU: '; if command -v nvidia-smi >/dev/null 2>&1; then nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || printf 'nvidia-smi failed\n'; else printf 'none detected\n'; fi
printf 'PYTORCH_CUDA: '; python3 -c 'import torch; print(torch.cuda.is_available(), torch.version.cuda or "NONE")' 2>/dev/null || printf 'unavailable\n'
printf 'HEALTH: '; curl -fsS --max-time 10 http://127.0.0.1:8000/health/ready 2>/dev/null | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d.get("status"), d.get("processed_root"))' || printf 'unavailable\n'

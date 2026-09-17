#!/usr/bin/env bash
# Qualify the actual finals host without starting or changing the service.
set -u
printf 'CPU_MODEL: '; lscpu 2>/dev/null | awk -F: '/Model name/ {gsub(/^ +| +$/, "", $2); print $2; exit}'
printf 'PHYSICAL_CORES: '; lscpu 2>/dev/null | awk -F: '/Core\(s\) per socket/ {gsub(/^ +| +$/, "", $2); print $2; exit}'
printf 'RAM_TOTAL: '; free -h 2>/dev/null | awk '/^Mem:/ {print $2}'
printf 'RAM_AVAILABLE: '; free -h 2>/dev/null | awk '/^Mem:/ {print $7}'
if command -v nvidia-smi >/dev/null 2>&1; then
  gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | paste -sd ';' -)
  vram=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | paste -sd ';' -)
  cuda=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | paste -sd ';' -)
  printf 'GPU: %s\n' "${gpu:-UNKNOWN}"
  printf 'GPU_VRAM: %s\n' "${vram:-UNKNOWN}"
  printf 'CUDA_DRIVER: %s\n' "${cuda:-UNKNOWN}"
else
  printf 'GPU: NONE_DETECTED\nGPU_VRAM: N/A\nCUDA_DRIVER: N/A\n'
fi
printf 'CUDA_AVAILABLE: '
PYTHONPATH="${PYTHONPATH:-}" python3 -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || printf 'UNKNOWN\n'
printf 'PYTORCH_CUDA: '
PYTHONPATH="${PYTHONPATH:-}" python3 -c 'import torch; print(torch.version.cuda or "NONE")' 2>/dev/null || printf 'UNKNOWN\n'
printf 'DISK_FREE: '; df -h . 2>/dev/null | awk 'NR==2 {print $4}'

# AIC 2026 Docker CPU/GPU Release Operations

This directory contains deployment profiles that keep one FastAPI process, one Qwen model instance, and externally mounted runtime data.

Files:

- `.env.cpu.example`: safe BF16/CPU profile template.
- `.env.gpu.example`: NVIDIA qualification profile template.
- `../DOCKER_CPU_GUIDE.md`: CPU installation and operation guide.
- `../DOCKER_GPU_GUIDE.md`: GPU installation and qualification guide.
- `../diagnose.sh`: read-only host/container diagnostics.

Copy an example to a local untracked `.env.cpu` or `.env.gpu`, replace `SERVER_IP`, review all mount paths, and never commit the resulting file.

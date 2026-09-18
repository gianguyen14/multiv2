# AIC Docker operations

Normal finals deployment is Docker Hub image-first. No GitHub clone or local build is required.

## CPU quick start

```bash
mkdir -p /opt/aic/config
curl -fsSL https://raw.githubusercontent.com/gianguyen14/multiv2/main/ops/docker/release/compose.cpu.yml -o /opt/aic/config/compose.cpu.yml
curl -fsSL https://raw.githubusercontent.com/gianguyen14/multiv2/main/ops/docker/release/.env.cpu.example -o /opt/aic/config/.env.cpu
# Replace SERVER_IP and verify external mount paths.
docker compose --env-file /opt/aic/config/.env.cpu -f /opt/aic/config/compose.cpu.yml pull
docker compose --env-file /opt/aic/config/.env.cpu -f /opt/aic/config/compose.cpu.yml up -d
docker compose --env-file /opt/aic/config/.env.cpu -f /opt/aic/config/compose.cpu.yml ps
```

The verified CPU image is:

```text
gianguyen14/aic-retrieval:cpu-18132237cca7
sha256:31200173184ca6fe4b6887b6bef3a5b8cfec69a8784303bbd9ce4f0733a293e2
```

## GPU status

No GPU image is currently published. Use `ops/DOCKER_GPU_GUIDE.md` for the qualification gate. Do not use `gpu-finals` until that tag exists and the actual GPU host passes runtime smoke.

## Guides

- `ops/DOCKER_CPU_GUIDE.md`: Docker Hub CPU deployment, mounts, LAN, health, lifecycle, rollback.
- `ops/DOCKER_GPU_GUIDE.md`: GPU prerequisites and runtime gate.
- `ops/FULL_SYSTEM_USAGE_GUIDE.md`: complete LAN/system/operator guide.

The image contains application/runtime dependencies only. Model, DB, videos, cache, and credentials remain external.

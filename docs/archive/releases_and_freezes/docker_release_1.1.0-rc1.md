# Docker Release Record: 1.1.0-rc1

## Image Identification
- **Repository**: `gianguyen14/aic-retrieval`
- **Tag**: `1.1.0-rc1`
- **Architecture**: `linux/amd64`
- **Local Image ID**: `sha256:975c9f19ead207283583723a03ba84f9d302550d2e61017e1b31494314d75915`
- **Local Image Size**: `3.37 GB` (unpacked)
- **Remote OCI Index Digest**: `sha256:975c9f19ead207283583723a03ba84f9d302550d2e61017e1b31494314d75915`
- **Remote Platform Manifest Digest (`linux/amd64`)**: `sha256:e8ec6dd7cb113394b7bca02ab5680cd1e04488bec187b89b78a02dc0b1078e7d`

## Registry Verification
- **Registry Host**: `docker.io`
- **Push Timestamp**: `2026-08-15 04:38:21 UTC`
- **Manifest Verification**: Verified via `docker manifest inspect gianguyen14/aic-retrieval:1.1.0-rc1`
- **Pull Test Verification**: Verified via `docker pull gianguyen14/aic-retrieval:1.1.0-rc1` (Status: Image is up to date)

## Deployment Verification
- **CLI Help**: `docker run --rm gianguyen14/aic-retrieval:1.1.0-rc1 python projectctl.py --help` -> Exit Code 0
- **Compose Config**: `docker compose -f docker-compose.release.yml config` -> Exit Code 0
- **Health Endpoints**:
  - `/health/live` -> HTTP 200 OK
  - `/health/ready` -> HTTP 200 OK (with valid volume mounts)

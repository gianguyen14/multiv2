# Candidate-frame object counting

Counting is a query-time, optional capability. Qwen first retrieves candidate frames; the detector runs only on up to `COUNTING_TOP_N` candidates (default 10, hard maximum 30). Sampled images under the processed video directory are used directly. Ingest, FAISS generation identity, and Qwen embeddings are unchanged.

## Configuration

```dotenv
COUNTING_ENABLED=true
COUNTING_TOP_N=10
COUNTING_MAX_CANDIDATES=30
COUNTING_CACHE_ROOT=/cache/detections
OBJECT_DETECTOR_BACKEND=yolo
OBJECT_DETECTOR_MODEL=/models/yolo/yolo11n.pt
OBJECT_DETECTOR_DEVICE=auto
OBJECT_DETECTOR_CONFIDENCE=0.25
```

Install the optional adapter with `pip install -r requirements/yolo.txt` or `pip install '.[yolo]'`. The model file is provisioned separately at the configured path. A missing file or disabled feature leaves normal search available and returns an explicit detector-unavailable status for count requests. The runtime never downloads weights automatically.

For a clean CPU environment, install the repository CPU requirements and then the optional adapter while keeping the selected PyTorch build fixed:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple -r requirements/cpu.txt
python -m pip freeze | grep -E '^(torch|torchvision)==' > /tmp/multiv2-torch-constraints.txt
python -m pip install --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple -c /tmp/multiv2-torch-constraints.txt -r requirements/yolo.txt
python -m pip check
```

For Docker, build with `--build-arg INSTALL_YOLO=true`, mount the model directory at `/models/yolo`, and set the environment variables above. For example, `INSTALL_YOLO=true COUNTING_ENABLED=true docker compose -f docker-compose.yml build aic` builds the optional runtime; `INSTALL_YOLO=true COUNTING_ENABLED=true docker compose -f docker-compose.yml up -d aic` starts it when `.env` supplies the model and data paths. The default image excludes Ultralytics. The CPU profile can use the same flag with `docker compose -f docker-compose.cpu.yml build aic`.

## Scope and results

Static count requests such as “Có bao nhiêu xe máy ở ngã tư?” produce a detector-backed count for each selected candidate frame. The answer is the count on the highest-ranked successfully processed frame; counts from different timestamps are never summed. A query about objects passing over time returns `tracking_required` with `supported=false`. Unique temporal counting needs a future tracking/ROI implementation.

Unknown labels are reported as `unsupported_target`; the service does not substitute a different class. Detection cache entries use `COUNTING_CACHE_ROOT` (Docker default `/cache/detections`; local fallback `VIDEO_PROCESSED_ROOT/detections/`), keyed by backend, exact model SHA-256, confidence, and cache schema. They do not participate in FAISS generation identity. Query-plan caching is disabled in the Qwen runtime so the read-only index mount is not mutated.

## License

The adapter pins `ultralytics==8.4.160` from the official [v8.4.160 release](https://github.com/ultralytics/ultralytics/releases/tag/v8.4.160). The package declares AGPL-3.0 in the [official source](https://github.com/ultralytics/ultralytics/blob/v8.4.160/ultralytics/__init__.py). Deployments that distribute the software or provide it as a network service must review AGPL-3.0 obligations with their compliance/legal owner. Ultralytics offers a separate Enterprise license; no commercial license is included here. The model weights are not included and must be reviewed for their own provenance and terms.

The integration smoke used the official [Ultralytics bus sample](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/assets/bus.jpg) and the official `yolo11n.pt` asset from the [Ultralytics assets v8.4.0 release](https://github.com/ultralytics/assets/releases/tag/v8.4.0). Neither image nor model weights are stored in this repository.

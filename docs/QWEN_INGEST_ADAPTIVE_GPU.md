# Qwen production ingest and adaptive GPU policy

Status at audit time: `QWEN_INGEST_CODE_PATH=IMPLEMENTED`, `GPU_HARDWARE_VALIDATION=PENDING_HARDWARE_VALIDATION`, `PRODUCTION_VECTOR_EQUIVALENCE=UNVERIFIED`.

## Contract

`INGEST_BACKEND=qwen3_vl` selects `backend.app.embeddings.ingest_encoder.QwenImageIngestEncoder`. The encoder wraps the official local Qwen3-VL script and produces:

- model family: `Qwen/Qwen3-VL-Embedding-2B`
- MRL output dimension: exactly 1024
- output dtype: float32
- normalization: L2
- finite-value and unit-norm validation
- explicit encoder identity in manifest and generation metadata

`INGEST_BACKEND=siglip2` remains an explicit legacy path. It must not write into a Qwen production generation. Qwen ingest rejects an encoder identity or dimension that is not `qwen3_vl`/1024 before index publication.

The active production DB was not modified. No production `CURRENT` pointer was used by validation.

## Device policy

`backend/app/runtime/ingest_policy.py` probes CUDA availability, device index, name, compute capability, visible/free/total VRAM and BF16 support. It chooses:

- Volta-like compute capability 7.x: float16 under `QWEN_DTYPE=auto`.
- Ampere/Blackwell compute capability 8.x+: bfloat16 only when PyTorch reports BF16 support; otherwise float16.
- CPU/no GPU: float32 under auto policy.

`GPU_STRICT=true` fails before corpus processing when the requested CUDA device or kernel smoke is unavailable. The policy is capability-based; it does not match GPU names.

`initial_batch_size()` uses visible free VRAM where available and is bounded by `GPU_BATCH_MIN`/`GPU_BATCH_MAX`. The Qwen encoder halves the batch and retries on CUDA OOM until batch 1; batch-1 OOM is fatal. This retry path is code-tested only; no GPU OOM run was available on the audit host.

## Commands

CPU/preflight without writing the production DB:

```bash
INGEST_BACKEND=qwen3_vl \
GPU_STRICT=false \
python3 projectctl.py ingest tests/fixtures/test_5s.mp4 \
  --processed-root /tmp/aic-qwen-ingest-validation \
  --preflight-only --json
```

Real GPU validation on an authorized NVIDIA host:

```bash
PYTHONPATH=. python3 scripts/validate_gpu_ingest.py \
  --model /models/Qwen3-VL-Embedding-2B \
  --image /path/to/one-sampled-frame.jpg \
  --device cuda:0 --dtype auto
```

The script performs a real CUDA kernel smoke and one Qwen image embedding. It exits with `PENDING_HARDWARE_VALIDATION` when CUDA is unavailable and never processes a corpus in that case.

Direct ingest CLI:

```bash
INGEST_BACKEND=qwen3_vl GPU_STRICT=true \
python3 backend/app/video/ingest.py \
  --input /path/to/videos \
  --output /path/to/new-qwen-workspace \
  --ingest-backend qwen3_vl --gpu-strict --device cuda:0
```

Do not set the output to the active production DB until an independent compatibility review has recovered and verified the historical production image-vector contract.

## Hardware matrix

| Hardware | Code policy | Real validation |
|---|---|---|
| CPU | float32 fallback | PASS for policy/unit behavior; ingest model run not performed | 
| RTX 3060 | capability-based, conservative batch | PENDING_HARDWARE_VALIDATION |
| A40 | capability-based, larger safe batch | PENDING_HARDWARE_VALIDATION |
| RTX 5090 | Blackwell-compatible runtime required | PENDING_HARDWARE_VALIDATION |
| RTX PRO 6000 Blackwell | capability-based | PENDING_HARDWARE_VALIDATION |
| RTX PRO 6000 MIG | visible device treated as `cuda:0`; free VRAM controls batch | PENDING_HARDWARE_VALIDATION |
| Tesla V100 PCIe/SXM2 | compute capability 7.0 → float16/eager policy | PENDING_HARDWARE_VALIDATION |

No Docker image split was claimed or published. The existing GPU Compose profile was only updated with ingest policy variables; Docker build/runtime validation remains pending where the Compose plugin or NVIDIA host is unavailable.

## Limitations

- The current main branch did not contain a verified production Qwen frame-ingest path before this change. The experimental `agent/qwen-image-search` branch supplied evidence for an image adapter, but exact equivalence to stored production frame vectors remains unverified.
- OCR/ASR adaptive worker and persisted timeline reuse were not changed in this milestone. Existing OCR/ASR behavior remains in place.
- Decode-to-RAM and grouped fsync performance changes were not made because they require a broader correctness/performance benchmark before changing resume semantics.
- No real GPU, CUDA kernel, Qwen image model load, throughput, VRAM, or CPU/GPU ranking consistency result is claimed on the current host.

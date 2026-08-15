# Experimental C++ acceleration layer

This branch keeps Python as the orchestration/runtime API and moves selected CPU-bound retrieval kernels to a C++17 extension.

## Goals

- Preserve the public Python API and deterministic ranking semantics.
- Keep a pure-Python fallback so the repository remains usable without a C++ toolchain.
- Make the C++ path mandatory in CI and in the experimental Docker image so ABI/compiler failures are visible.
- Benchmark each kernel against its Python reference before considering promotion.
- Do not rewrite model orchestration simply for language consistency. SigLIP2, PyTorch, FAISS and other heavy libraries already execute most numerical work in native/CUDA code.

## Runtime selection

`UVR_NATIVE_CORE` controls dispatch:

| Value | Behavior |
|---|---|
| `auto` | Use C++ when available; otherwise use the Python reference implementation. |
| `cpp` | Require C++; missing extension or kernel errors are fatal. |
| `python` | Force the Python reference implementation. |

The default is `auto`.

Inspect the active backend:

```bash
python - <<'PY'
from backend.app.native import native_status
print(native_status())
PY
```

## Build

Normal editable install attempts to build the extension but can fall back when the compiler is unavailable:

```bash
python -m pip install -e .
```

Require a successful native build:

```bash
UVR_NATIVE_STRICT_BUILD=1 python -m pip install -e .
```

Or build in place:

```bash
UVR_NATIVE_STRICT_BUILD=1 python setup.py build_ext --inplace
```

The implementation uses the CPython C API directly and does not require pybind11 or NumPy C headers. Numerical buffers are consumed through Python's buffer protocol.

## Kernels

### TRAKE ordered dynamic programming

`backend.app.retrieval.trake.TRAKEAligner` dispatches per-video ordered event alignment through `backend.app.native.align_trake_events`.

The C++ implementation preserves:

- strictly increasing frame order;
- optional maximum transition gap;
- transition penalty semantics;
- deterministic tie-breaking toward the lexicographically smaller frame path;
- the Python layer's final cross-video tie-breaking.

This kernel is active on this branch when `UVR_NATIVE_CORE=auto` and the extension exists.

### Temporal score smoothing

`backend.app.native.smooth_scores` implements the local pooling rule used by TemporalRefiner and returns a float32 NumPy view backed by native-produced bytes.

The kernel is implemented and parity-tested. Wiring it into every refinement call should be promoted only after representative profiling confirms it is beneficial relative to NumPy on the target workload.

### Temporal candidate-region merging

`backend.app.native.merge_temporal_regions` implements bounded candidate-window construction, overlap merging, score-priority limiting and deterministic start-frame ordering.

The kernel is implemented and parity-tested. It is available for incremental TemporalRefiner integration without changing frame-ID semantics.

### Temporal NMS

`backend.app.native.temporal_nms_indices` preserves stable candidate order while suppressing same-video frames inside a configurable frame gap.

The kernel is implemented and parity-tested. It is suitable for the repeated KIS/image-search deduplication path.

## Benchmark

The benchmark is synthetic and does not download models or require processed video data:

```bash
UVR_NATIVE_STRICT_BUILD=1 python setup.py build_ext --inplace
python scripts/benchmark_native_core.py --iterations 7
```

It verifies Python/C++ parity before printing median latency and speedup for:

- temporal smoothing;
- temporal NMS;
- temporal region merging;
- TRAKE dynamic programming.

No speedup threshold is hard-coded. Promotion should depend on representative retrieval workloads, not synthetic numbers alone.

## Docker

The experimental Dockerfile compiles the extension with `UVR_NATIVE_STRICT_BUILD=1` and verifies `UVR_NATIVE_CORE=cpp` during the image build. Runtime defaults back to `auto` after the build check.

## CI

The experimental workflow installs `build-essential`, builds the extension in strict mode, verifies native availability, and executes the current CPU suite with `UVR_NATIVE_CORE=cpp`. The Docker gate also checks the compiled extension inside the image.

## Non-goals

This branch is not a full rewrite of FastAPI, QueryRefiner, model loading, OCR/ASR orchestration or the operator CLI. Those components benefit more from Python's ecosystem and already call optimized native libraries for their expensive numerical operations.

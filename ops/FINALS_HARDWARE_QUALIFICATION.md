# Finals hardware qualification runbook

Run this on the actual finals host from the repository root:

```bash
bash ops/qualify_finals_hardware.sh
```

The command is observational only. It does not start, stop, restart, or reconfigure the service. It reports CPU, RAM, GPU/CUDA, PyTorch CUDA, and disk availability.

Selection procedure:

1. If `CUDA_AVAILABLE=True`, record GPU model, VRAM, driver/CUDA versions, and qualify the existing Qwen model contract on GPU. Select `CUDA` only after no-OOM embedding validation and live KIS, QA, known-good TRAKE, result-compatibility, and video-preview smoke tests.
2. Otherwise start the normal full runtime with the exact current production configuration and measure RSS/headroom. Select `FAST_CPU_FP32` only when RAM is at least 16 GiB, measured post-startup headroom is at least 2 GiB, swap pressure is absent, and the required 3-KIS/1-QA/1-known-good-TRAKE/frontend/video smoke passes.
3. Otherwise select `SAFE_CPU_LOW_RAM`, retaining the current BF16 production path, DB, fusion, and search settings unchanged.

Priority: validated CUDA, then validated FP32 CPU, then BF16 safe fallback. RAM size alone never activates FP32.

Current-host qualification facts:

- `SAFE_CPU_LOW_RAM` is the only approved profile for the current 9-GiB host.
- FP32 was retrieval-compatible in the completed validation, but its measured full-stack RSS was approximately 8.2 GB, leaving approximately 1.0 GiB headroom; it is not approved here.
- The currently running service must remain BF16 until a separate finals-host qualification passes.

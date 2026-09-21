"""Capability-based runtime policy for offline visual ingest."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import os
from typing import Any


@dataclass(frozen=True)
class GPUCapability:
    available: bool
    name: str | None = None
    index: int | None = None
    compute_capability: tuple[int, int] | None = None
    total_vram_bytes: int | None = None
    free_vram_bytes: int | None = None
    bf16_supported: bool = False
    fp16_supported: bool = False
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        if self.compute_capability is not None:
            value["compute_capability"] = list(self.compute_capability)
        return value


def probe_gpu(device: str = "auto") -> GPUCapability:
    """Probe the selected CUDA device without loading an ML model."""
    try:
        import torch
    except Exception as exc:
        return GPUCapability(False, error=f"torch unavailable: {exc}")

    requested = str(device or "auto").strip().lower()
    if requested == "cpu":
        return GPUCapability(False, error="CPU requested")
    if not torch.cuda.is_available():
        return GPUCapability(False, error="CUDA is unavailable")
    try:
        index = int(requested.split(":", 1)[1]) if ":" in requested else 0
        if index >= torch.cuda.device_count():
            return GPUCapability(False, error=f"CUDA device {index} is unavailable")
        props = torch.cuda.get_device_properties(index)
        free_bytes, total_bytes = torch.cuda.mem_get_info(index)
        cc = torch.cuda.get_device_capability(index)
        bf16 = bool(getattr(torch.cuda, "is_bf16_supported", lambda **_: False)(including_emulation=False))
        return GPUCapability(
            True,
            name=str(props.name),
            index=index,
            compute_capability=(int(cc[0]), int(cc[1])),
            total_vram_bytes=int(total_bytes),
            free_vram_bytes=int(free_bytes),
            bf16_supported=bf16,
            fp16_supported=True,
        )
    except Exception as exc:
        return GPUCapability(False, error=f"CUDA probe failed: {type(exc).__name__}: {exc}")


def kernel_smoke(device: str = "cuda:0") -> dict[str, Any]:
    """Run a real CUDA allocation/matmul/synchronize smoke test."""
    try:
        import torch
        if not torch.cuda.is_available():
            return {"status": "BLOCKED", "reason": "CUDA is unavailable"}
        with torch.inference_mode():
            left = torch.ones((64, 64), device=device, dtype=torch.float16)
            right = torch.ones((64, 64), device=device, dtype=torch.float16)
            result = left @ right
            torch.cuda.synchronize(device)
            finite = bool(torch.isfinite(result).all().item())
            del left, right, result
            torch.cuda.empty_cache()
        return {"status": "PASS" if finite else "FAIL", "device": device}
    except Exception as exc:
        return {"status": "FAIL", "reason": f"{type(exc).__name__}: {exc}"}


def select_dtype(policy: str = "auto", capability: GPUCapability | None = None) -> str:
    value = str(policy or "auto").strip().lower()
    if value not in {"auto", "bfloat16", "float16", "float32"}:
        raise ValueError("QWEN_DTYPE must be auto, bfloat16, float16, or float32")
    if value != "auto":
        return value
    if capability and capability.available:
        if capability.compute_capability and capability.compute_capability[0] >= 8 and capability.bf16_supported:
            return "bfloat16"
        return "float16"
    return "float32"


def initial_batch_size(capability: GPUCapability | None, minimum: int = 1, maximum: int = 32) -> int:
    minimum, maximum = int(minimum), int(maximum)
    if minimum < 1 or maximum < minimum:
        raise ValueError("invalid ingest batch bounds")
    if not capability or not capability.available or not capability.total_vram_bytes:
        return minimum
    gib = capability.free_vram_bytes / (1024 ** 3) if capability.free_vram_bytes else capability.total_vram_bytes / (1024 ** 3)
    if gib <= 8:
        candidate = 1
    elif gib <= 12:
        candidate = 2
    elif gib <= 20:
        candidate = 4
    elif gib <= 32:
        candidate = 8
    elif gib <= 48:
        candidate = 16
    else:
        candidate = 24
    return max(minimum, min(maximum, candidate))

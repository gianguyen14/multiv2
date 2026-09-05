"""Qwen3-VL-Embedding-2B query encoder adapter.

The adapter wraps the *official* Qwen embedder script that ships inside the
model directory (``<model_dir>/scripts/qwen3_vl_embedding.py``) with the same
low-memory query-only contract used by the verified AIC query runtime:

- CPU, bfloat16, eager attention, bounded thread pool;
- ``max_length=256`` text truncation;
- Matryoshka (MRL) truncation of the pooled embedding to the index dimension
  (1024 for the packed AIC DB);
- manual float32 L2 normalization after truncation.

The index is read-only and never written by this module.  The SigLIP2 encoder
path is untouched; selecting this encoder is an explicit deployment choice.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np

DEFAULT_MODEL_DIR_NAME = "Qwen3-VL-Embedding-2B"
DEFAULT_INSTRUCTION = "Retrieve the video frame that best matches the described visual scene."


def resolve_model_dir(explicit: Optional[str] = None) -> Path:
    """Resolve the local Qwen model directory from configuration or defaults."""
    raw = explicit or os.getenv("QWEN3_VL_MODEL_DIR") or os.getenv("QWEN_MODEL_DIR")
    if raw:
        return Path(raw)
    cache_dir = os.getenv("MODEL_CACHE_DIR", "models")
    return Path(cache_dir) / DEFAULT_MODEL_DIR_NAME


def model_script_path(model_dir) -> Path:
    return Path(model_dir) / "scripts" / "qwen3_vl_embedding.py"


def weights_available(model_dir) -> bool:
    """Cheap presence check; does not load weights."""
    model_dir = Path(model_dir)
    return (
        (model_dir / "model.safetensors").is_file()
        and (model_dir / "config.json").is_file()
        and model_script_path(model_dir).is_file()
    )


class Qwen3VlLocalEmbedder:
    """Lazily loads the official Qwen embedder from a local model directory."""

    def __init__(
        self,
        model_dir=None,
        *,
        max_length: int = 256,
        instruction: str = DEFAULT_INSTRUCTION,
        torch_dtype: str = "bfloat16",
        attn_implementation: str = "eager",
        low_cpu_mem_usage: bool = True,
        threads: int = 2,
    ):
        self.model_dir = Path(resolve_model_dir(model_dir))
        self.max_length = int(max_length)
        self.instruction = instruction
        self.torch_dtype = torch_dtype
        self.attn_implementation = attn_implementation
        self.low_cpu_mem_usage = low_cpu_mem_usage
        self.threads = int(threads)
        self._embedder = None

    # -- availability -----------------------------------------------------

    @property
    def weights_present(self) -> bool:
        return weights_available(self.model_dir)

    # -- loading ----------------------------------------------------------

    def _ensure_loaded(self):
        if self._embedder is not None:
            return
        import torch

        torch.set_num_threads(self.threads)
        script = model_script_path(self.model_dir)
        if not script.is_file():
            raise RuntimeError(
                f"Qwen embedder script not found at {script}; "
                "the model directory must contain scripts/qwen3_vl_embedding.py"
            )
        spec = importlib.util.spec_from_file_location("qwen3_vl_embedding_backend", script)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Cannot load Qwen embedder script from {script}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)

        dtype = getattr(torch, self.torch_dtype, None)
        if not isinstance(dtype, torch.dtype):
            raise RuntimeError(f"Unsupported torch dtype: {self.torch_dtype}")
        self._embedder = module.Qwen3VLEmbedder(
            model_name_or_path=str(self.model_dir),
            max_length=self.max_length,
            torch_dtype=dtype,
            low_cpu_mem_usage=self.low_cpu_mem_usage,
            attn_implementation=self.attn_implementation,
        )

    # -- encoding ---------------------------------------------------------

    def encode_query(self, query: str, dimension: int) -> np.ndarray:
        """Return a float32 (``dimension``,) L2-normalized query embedding.

        Mirrors the verified runtime: pool the last token, MRL-truncate to the
        index dimension, then L2-normalize in float32.
        """
        import torch

        self._ensure_loaded()
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be non-empty text")
        dimension = int(dimension)
        if dimension <= 0:
            raise ValueError("dimension must be positive")

        embedder = self._embedder
        assert embedder is not None, "embedder must be loaded"
        full = embedder.process(
            [{"text": query, "instruction": self.instruction}], normalize=False
        )[0]
        vector = full[:dimension].detach().cpu().to(torch.float32).numpy()
        raw_norm = float(np.linalg.norm(vector))
        if raw_norm <= 0.0 or not np.isfinite(raw_norm):
            raise RuntimeError(f"Invalid encoder vector norm: {raw_norm}")
        vector = (vector / raw_norm).astype(np.float32, copy=False)
        norm = float(np.linalg.norm(vector))
        if vector.shape != (dimension,) or not np.isclose(norm, 1.0, atol=1e-5):
            raise RuntimeError(
                f"Encoder contract mismatch: shape={vector.shape}, norm={norm}"
            )
        return vector

    def encode_text(self, texts, dimension: int):
        """Batch convenience used by tests/alternate callers."""
        vectors = [self.encode_query(text, dimension) for text in texts]
        return np.stack(vectors).astype(np.float32, copy=False)

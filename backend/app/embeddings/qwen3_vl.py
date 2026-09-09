import importlib.util
import os
import sys
import threading
from pathlib import Path
from typing import Optional

import numpy as np

DEFAULT_MODEL_DIR_NAME = "Qwen3-VL-Embedding-2B"
DEFAULT_INSTRUCTION = "Retrieve the video frame that best matches the described visual scene."


def resolve_model_dir(explicit: Optional[str] = None) -> Path:
    raw = explicit or os.getenv("QWEN3_VL_MODEL_DIR") or os.getenv("QWEN_MODEL_DIR")
    if raw:
        return Path(raw)
    cache_dir = os.getenv("MODEL_CACHE_DIR", "models")
    return Path(cache_dir) / DEFAULT_MODEL_DIR_NAME


def model_script_path(model_dir) -> Path:
    return Path(model_dir) / "scripts" / "qwen3_vl_embedding.py"


def weights_available(model_dir) -> bool:
    model_dir = Path(model_dir)
    return (
        (model_dir / "model.safetensors").is_file()
        and (model_dir / "config.json").is_file()
        and model_script_path(model_dir).is_file()
    )


class Qwen3VlLocalEmbedder:
    """Official Qwen3-VL local embedder adapter for text and image queries."""

    def __init__(self, model_dir=None, *, max_length=256,
                 instruction: str = DEFAULT_INSTRUCTION,
                 torch_dtype: str = "bfloat16", attn_implementation: str = "eager",
                 low_cpu_mem_usage: bool = True, threads: int = 2):
        self.model_dir = Path(resolve_model_dir(model_dir))
        # Matches dense DB ingestion: 4096..1310720 pixels and 8192 tokens.
        self.max_length = max(int(max_length), 8192)
        self.min_pixels = 4096
        self.max_pixels = 1310720
        self.image_max_length = self.max_length
        self.instruction = instruction
        self.torch_dtype = torch_dtype
        self.attn_implementation = attn_implementation
        self.low_cpu_mem_usage = low_cpu_mem_usage
        self.threads = int(threads)
        self._embedder = None
        self._load_lock = threading.Lock()

    @property
    def weights_present(self) -> bool:
        return weights_available(self.model_dir)

    def _ensure_loaded(self):
        if self._embedder is not None:
            return
        with self._load_lock:
            if self._embedder is not None:
                return
            self._load_impl()

    def _load_impl(self):
        if self._embedder is not None:
            return
        import torch
        torch.set_num_threads(self.threads)
        script = model_script_path(self.model_dir)
        if not script.is_file():
            raise RuntimeError(f"Qwen embedder script not found at {script}")
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
            model_name_or_path=str(self.model_dir), max_length=self.max_length,
            torch_dtype=dtype, low_cpu_mem_usage=self.low_cpu_mem_usage,
            attn_implementation=self.attn_implementation,
        )

    @staticmethod
    def _normalize(full, dimension: int):
        import torch
        dimension = int(dimension)
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        vector = full[:dimension].detach().cpu().to(torch.float32).numpy()
        raw_norm = float(np.linalg.norm(vector))
        if vector.shape != (dimension,) or raw_norm <= 0.0 or not np.isfinite(raw_norm):
            raise RuntimeError(f"Invalid encoder vector: shape={vector.shape}, norm={raw_norm}")
        vector = (vector / raw_norm).astype(np.float32, copy=False)
        if not np.isclose(float(np.linalg.norm(vector)), 1.0, atol=1e-5):
            raise RuntimeError("Encoder output is not L2 normalized")
        return vector

    def _process_one(self, *, text=None, image=None, dimension: int):
        self._ensure_loaded()
        if text is not None and (not isinstance(text, str) or not text.strip()):
            raise ValueError("query must be non-empty text")
        if image is not None:
            from PIL import Image
            if not isinstance(image, Image.Image):
                raise ValueError("image must be a PIL image")
        embedder = self._embedder
        if embedder is None:
            raise RuntimeError("Qwen embedder failed to load")
        if image is not None:
            embedder.max_length = self.image_max_length
            embedder.min_pixels = self.min_pixels
            embedder.max_pixels = self.max_pixels
        payload: dict[str, object] = {"instruction": self.instruction}
        if text is not None:
            payload["text"] = text
        if image is not None:
            payload["image"] = image
        full = embedder.process([payload], normalize=False)[0]
        return self._normalize(full, dimension)

    def encode_query(self, query: str, dimension: int) -> np.ndarray:
        return self._process_one(text=query, dimension=dimension)

    def encode_image(self, image, dimension: int) -> np.ndarray:
        return self._process_one(image=image, dimension=dimension)

    def encode_text(self, texts, dimension: int):
        return np.stack([self.encode_query(text, dimension) for text in texts]).astype(np.float32, copy=False)

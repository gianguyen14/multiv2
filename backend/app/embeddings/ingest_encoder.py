"""Ingest encoder factory and strict vector contract."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from backend.app.runtime.ingest_policy import initial_batch_size, kernel_smoke, probe_gpu, select_dtype

QWEN_INGEST_DIM = 1024


def _validate_vectors(values, expected_count: int, dimension: int) -> np.ndarray:
    result = np.asarray(values, dtype=np.float32)
    if result.shape != (expected_count, dimension):
        raise RuntimeError(f"embedding contract mismatch: expected {(expected_count, dimension)}, got {result.shape}")
    if not np.isfinite(result).all():
        raise RuntimeError("embedding contract mismatch: non-finite vector")
    norms = np.linalg.norm(result, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-5):
        raise RuntimeError("embedding contract mismatch: vectors are not L2 normalized")
    return result


class QwenImageIngestEncoder:
    """Adapter exposing the batch interface required by VideoIngestionPipeline."""

    embedding_dim = QWEN_INGEST_DIM

    def __init__(self, model_dir=None, *, device="auto", dtype="auto", batch_size=None, strict_gpu=False, batch_min=1, batch_max=32):
        from backend.app.embeddings.qwen3_vl import Qwen3VlLocalEmbedder

        self.model_dir = Path(model_dir) if model_dir else None
        self.capability = probe_gpu(device)
        if strict_gpu and not self.capability.available:
            raise RuntimeError(f"GPU_STRICT requested but CUDA is unavailable: {self.capability.error}")
        if device.startswith("cuda") and not self.capability.available:
            raise RuntimeError(f"Qwen ingest requested {device} but CUDA is unavailable: {self.capability.error}")
        self.device = f"cuda:{self.capability.index}" if self.capability.available else "cpu"
        self.dtype = select_dtype(dtype, self.capability)
        self.batch_size = int(batch_size) if batch_size else None
        self.batch_min = int(batch_min)
        self.batch_max = int(batch_max)
        if self.batch_min < 1 or self.batch_max < self.batch_min:
            raise ValueError("invalid Qwen batch bounds")
        if self.batch_size is None:
            self.batch_size = initial_batch_size(self.capability, self.batch_min, self.batch_max)
        elif self.batch_size < self.batch_min or self.batch_size > self.batch_max:
            raise ValueError("explicit Qwen batch size is outside configured bounds")
        self.effective_batch_size = self.batch_size
        self._embedder = Qwen3VlLocalEmbedder(
            model_dir=model_dir,
            max_length=8192,
            torch_dtype=self.dtype,
            attn_implementation="eager",
            device=self.device,
        )

    def identity(self):
        return {
            "backend": "qwen3_vl",
            "provider": "qwen3_vl",
            "model_name": "Qwen/Qwen3-VL-Embedding-2B",
            "model_dir": str(self._embedder.model_dir),
            "embedding_dim": QWEN_INGEST_DIM,
            "normalization": "l2",
            "dtype": self.dtype,
            "output_dtype": "float32",
            "contract_version": "qwen3-vl-image-ingest-v1",
            "selected_batch_size": self.batch_size,
        }

    def get_model_info(self):
        return {**self.identity(), "device": self.device, "gpu": self.capability.to_dict(), "effective_batch_size": self.effective_batch_size}

    def load_model(self):
        """Load model weights before corpus decode; fail early on runtime incompatibility."""
        self._embedder._ensure_loaded()

    def encode_image(self, images, batch_size=None, normalize=True):
        if not normalize:
            raise ValueError("Qwen production ingest requires normalization")
        if not isinstance(images, list):
            images = [images]
        if not images:
            return np.zeros((0, QWEN_INGEST_DIM), dtype=np.float32)
        values = []
        size = int(batch_size or self.batch_size or 1)
        self.effective_batch_size = size
        offset = 0
        while offset < len(images):
            batch = images[offset:offset + size]
            try:
                encoded = self._embedder._ensure_loaded() or None
                del encoded
                # Official adapter accepts image payloads; output is normalized here.
                import torch
                raw = self._embedder._embedder.process(
                    [{"image": image, "instruction": self._embedder.instruction} for image in batch],
                    normalize=False,
                )
                vectors = []
                for value in raw:
                    vector = value[:QWEN_INGEST_DIM].detach().cpu().to(torch.float32).numpy()
                    norm = float(np.linalg.norm(vector))
                    if norm <= 0 or not np.isfinite(norm):
                        raise RuntimeError("invalid Qwen image embedding norm")
                    vectors.append((vector / norm).astype(np.float32, copy=False))
                values.extend(vectors)
                offset += len(batch)
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower() and size > 1 and self.device.startswith("cuda"):
                    if self.batch_size and size == 1:
                        raise
                    size = max(1, size // 2)
                    self.effective_batch_size = size
                    import torch
                    torch.cuda.empty_cache()
                    continue
                raise
        return _validate_vectors(values, len(images), QWEN_INGEST_DIM)

    def clear_cache(self):
        self._embedder._embedder = None
        try:
            import torch
            if self.device.startswith("cuda"):
                torch.cuda.empty_cache()
        except Exception:
            pass


def create_ingest_encoder(config, model_dir=None):
    backend = config.ingest_backend.strip().lower()
    if backend == "siglip2":
        from backend.app.embeddings.siglip2 import SigLIP2Encoder
        return SigLIP2Encoder(device=config.device, force_download=False, local_files_only=True)
    if backend == "qwen3_vl":
        return QwenImageIngestEncoder(
            model_dir=model_dir,
            device=config.device,
            dtype=config.qwen_dtype,
            batch_size=config.embed_batch_size,
            strict_gpu=config.gpu_strict,
            batch_min=config.qwen_batch_min,
            batch_max=config.qwen_batch_max,
        )
    raise ValueError(f"unsupported ingest backend: {backend!r}")

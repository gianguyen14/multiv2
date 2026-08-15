"""SigLIP2 encoder for joint image-text embedding space.

This module provides deterministic text and image encoding using the SigLIP2 model.
Supports CPU and CUDA inference with model caching.
"""

import hashlib
import os
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoProcessor, SiglipModel

from backend.app.core.config import SIGLIP2_MODEL, SIGLIP_ENABLED
from backend.app.runtime.device_policy import resolve_device


def resolve_siglip2_revision(model_name: str, cache_dir: Optional[Path] = None) -> str:
    try:
        from huggingface_hub import snapshot_download
        from backend.app.model_cache import model_cache_dir
        actual_cache = model_cache_dir("huggingface", cache_dir)
        path = snapshot_download(repo_id=model_name, cache_dir=actual_cache, local_files_only=True)
        p = Path(path)
        if p.parent.name == "snapshots":
            return p.name
        return str(path)
    except Exception:
        return "default"


class SigLIP2Encoder:
    """SigLIP2 encoder for joint image-text embeddings.

    Provides:
    - encode_text(): text → normalized vector
    - encode_image(): image → normalized vector
    - Both in the same joint embedding space
    - L2 normalization for cosine similarity
    - Batch inference support
    - CPU/CUDA automatic detection
    - Model caching for fast reload
    - Deterministic inference
    """

    DEFAULT_MODEL_NAME = SIGLIP2_MODEL

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        cache_dir: Optional[Path] = None,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        force_download: bool = False,
        local_files_only: bool = False,
        revision: Optional[str] = None,
    ):
        """Initialize the SigLIP2 encoder.

        Args:
            model_name: HuggingFace model identifier.
            cache_dir: Directory for model caching.
            device: "cpu", "cuda", or None for auto-detection.
            dtype: torch.dtype (float16 for CUDA, float32 for CPU).
            force_download: If True, re-download model even if cached.
        """
        if not SIGLIP_ENABLED:
            raise RuntimeError("SigLIP2 is disabled. Set SIGLIP_ENABLED=true in config.")

        from backend.app.model_cache import model_cache_dir
        self.model_name = model_name
        self.cache_dir = model_cache_dir("huggingface", cache_dir)
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.device_selection = resolve_device("visual", "torch", device,
            component_env="VISUAL_DEVICE")
        self.device = self.device_selection.device
        self.dtype = dtype or (torch.float16 if self.device.startswith("cuda") else torch.float32)
        self.auto_batch_size = 16 if self.device.startswith("cuda") else 8
        self.last_batch_size = None
        self.revision = revision or resolve_siglip2_revision(self.model_name, self.cache_dir)


        self.force_download = force_download
        self.local_files_only = local_files_only
        self._model: Optional[SiglipModel] = None
        self._processor: Optional[AutoProcessor] = None
        self._initialized = False

        # Deterministic settings
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def _load_model(self) -> None:
        """Load model and processor (lazy initialization)."""
        if self._initialized:
            return

        cache_dir = str(self.cache_dir) if self.cache_dir else None
        try:
            processor = AutoProcessor.from_pretrained(
                self.model_name,
                cache_dir=cache_dir,
                force_download=self.force_download,
                local_files_only=self.local_files_only,
            )
            model = SiglipModel.from_pretrained(
                self.model_name,
                cache_dir=cache_dir,
                force_download=self.force_download,
                local_files_only=self.local_files_only,
                torch_dtype=self.dtype,
                low_cpu_mem_usage=True,
            ).to(self.device)
            model.eval()
        except Exception as exc:
            if self.local_files_only:
                raise RuntimeError(f"SigLIP2 model {self.model_name} is not available locally. Run: python projectctl.py models --prepare --visual") from exc
            raise
        self._processor = processor
        self._model = model
        self._initialized = True

    @property
    def model(self) -> SiglipModel:
        """Get the loaded model (triggers lazy load)."""
        if not self._initialized:
            self._load_model()
        return self._model

    @property
    def processor(self) -> AutoProcessor:
        """Get the loaded processor (triggers lazy load)."""
        if not self._initialized:
            self._load_model()
        return self._processor

    @property
    def embedding_dim(self) -> int:
        """Get the embedding dimension."""
        return self.model.config.text_config.hidden_size

    def encode_text(
        self,
        texts: Union[str, List[str]],
        batch_size: Optional[int] = None,
        normalize: bool = True,
    ) -> np.ndarray:
        """Encode text(s) into normalized embedding vectors.

        Args:
            texts: Single text string or list of text strings.
            batch_size: Batch size for inference.
            normalize: If True, L2-normalize the embeddings.

        Returns:
            numpy array of shape (n_texts, embedding_dim) with float32 values.
        """
        if isinstance(texts, str):
            texts = [texts]

        self._load_model()
        batch_size = batch_size or self.auto_batch_size
        self.last_batch_size = batch_size

        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]

            # Process text
            inputs = self.processor(
                text=batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=64,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                text_outputs = self.model.get_text_features(**inputs)
                # text_outputs is a BaseModelOutputWithPooling, get pooled_output
                text_embeddings = text_outputs.pooler_output
                # text_embeddings shape: (batch_size, hidden_size)

            embeddings = text_embeddings.detach().cpu().to(torch.float32).numpy()
            all_embeddings.append(embeddings)

        result = np.vstack(all_embeddings) if all_embeddings else np.zeros((0, self.embedding_dim), dtype=np.float32)

        if normalize:
            # L2 normalization for cosine similarity
            norms = np.linalg.norm(result, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            result = result / norms

        return result.astype(np.float32)

    def encode_image(
        self,
        images: Union[Image.Image, List[Image.Image], str, List[str], Path, List[Path]],
        batch_size: Optional[int] = None,
        normalize: bool = True,
    ) -> np.ndarray:
        """Encode image(s) into normalized embedding vectors.

        Args:
            images: Single image (PIL Image, path string, or Path) or list thereof.
            batch_size: Batch size for inference.
            normalize: If True, L2-normalize the embeddings.

        Returns:
            numpy array of shape (n_images, embedding_dim) with float32 values.
        """
        if not isinstance(images, list):
            images = [images]

        self._load_model()
        automatic_batch = batch_size is None
        batch_size = batch_size or self.auto_batch_size
        self.last_batch_size = batch_size

        all_embeddings = []
        i = 0
        while i < len(images):
            batch = []
            for image in images[i:i + batch_size]:
                if isinstance(image, (str, Path)):
                    with Image.open(image) as opened:
                        batch.append(opened.convert("RGB"))
                elif isinstance(image, Image.Image):
                    batch.append(image.convert("RGB"))
                else:
                    raise TypeError(f"Unsupported image type: {type(image)}")
            try:
                inputs = self.processor(images=batch, return_tensors="pt", padding=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                with torch.inference_mode():
                    image_embeddings = self.model.get_image_features(**inputs).pooler_output
                embeddings = image_embeddings.detach().cpu().to(torch.float32).numpy()
            except torch.cuda.OutOfMemoryError:
                if not automatic_batch or not self.device.startswith("cuda") or batch_size == 1:
                    raise
                batch_size = max(1, batch_size // 2)
                self.last_batch_size = batch_size
                torch.cuda.empty_cache()
                continue
            all_embeddings.append(embeddings)
            i += len(batch)

        result = np.vstack(all_embeddings) if all_embeddings else np.zeros((0, self.embedding_dim), dtype=np.float32)

        if normalize:
            norms = np.linalg.norm(result, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            result = result / norms

        return result.astype(np.float32)

    def encode_text_image_pairs(
        self,
        texts: List[str],
        images: List[Union[Image.Image, str, Path]],
        batch_size: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Encode paired texts and images.

        Args:
            texts: List of text strings.
            images: List of images (PIL, path, or Path).
            batch_size: Batch size.

        Returns:
            Tuple of (text_embeddings, image_embeddings) both normalized.
        """
        assert len(texts) == len(images), "Texts and images must have same length"
        text_emb = self.encode_text(texts, batch_size=batch_size)
        image_emb = self.encode_image(images, batch_size=batch_size)
        return text_emb, image_emb

    def similarity(
        self,
        text_embeddings: np.ndarray,
        image_embeddings: np.ndarray,
    ) -> np.ndarray:
        """Compute cosine similarity between text and image embeddings.

        Args:
            text_embeddings: (n_texts, dim)
            image_embeddings: (n_images, dim)

        Returns:
            Similarity matrix (n_texts, n_images)
        """
        # Both already L2 normalized
        return text_embeddings @ image_embeddings.T

    def clear_cache(self) -> None:
        """Clear model from memory."""
        self._model = None
        self._processor = None
        self._initialized = False
        if self.device.startswith("cuda"):
            torch.cuda.empty_cache()

    def identity(self) -> dict:
        return {
            "provider": "huggingface-transformers",
            "model_name": self.model_name,
            "revision": self.revision,
            "embedding_dim": self.get_model_info()["embedding_dim"],
            "normalization": "l2",
            "contract_version": "m15.1-v1",
        }


    def get_model_info(self) -> dict:
        """Get model information."""
        return {
            "model_name": self.model_name,
            "embedding_dim": 768,
            "requested_device": self.device_selection.requested,
            "device": self.device,
            "device_source": self.device_selection.source,
            "device_fallback": self.device_selection.fallback,
            "dtype": str(self.dtype),
            "automatic_batch_size": self.auto_batch_size,
            "last_batch_size": self.last_batch_size,
            "cache_dir": str(self.cache_dir) if self.cache_dir else None,
            "initialized": self._initialized,
        }


def get_siglip2_encoder(
    model_name: str = SigLIP2Encoder.DEFAULT_MODEL_NAME,
    cache_dir: Optional[Path] = None,
    device: Optional[str] = None,
    dtype: Optional[torch.dtype] = None,
) -> SigLIP2Encoder:
    """Factory function to create a SigLIP2 encoder.

    Args:
        model_name: HuggingFace model identifier.
        cache_dir: Model cache directory.
        device: "cpu", "cuda", or None for auto.
        dtype: torch.dtype.

    Returns:
        Configured SigLIP2Encoder instance.
    """
    return SigLIP2Encoder(
        model_name=model_name,
        cache_dir=cache_dir,
        device=device,
        dtype=dtype,
    )
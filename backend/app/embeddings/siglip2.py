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

from backend.app.core.config import SIGLIP_ENABLED


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

    DEFAULT_MODEL_NAME = "google/siglip2-base-patch16-224"
    DEFAULT_CACHE_DIR = Path.home() / ".cache" / "siglip2"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        cache_dir: Optional[Path] = None,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        force_download: bool = False,
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

        self.model_name = model_name
        self.cache_dir = Path(cache_dir) if cache_dir else self.DEFAULT_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Device detection
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Dtype selection
        if dtype is None:
            self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        else:
            self.dtype = dtype

        self.force_download = force_download
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

        # Load processor
        self._processor = AutoProcessor.from_pretrained(
            self.model_name,
            cache_dir=str(self.cache_dir),
            force_download=self.force_download,
        )

        # Load model
        self._model = SiglipModel.from_pretrained(
            self.model_name,
            cache_dir=str(self.cache_dir),
            force_download=self.force_download,
            torch_dtype=self.dtype,
            low_cpu_mem_usage=True,
        ).to(self.device)

        self._model.eval()
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
        batch_size: int = 32,
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
        batch_size: int = 32,
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
        # Convert all inputs to PIL Images
        if not isinstance(images, list):
            images = [images]

        pil_images = []
        for img in images:
            if isinstance(img, (str, Path)):
                pil_images.append(Image.open(img).convert("RGB"))
            elif isinstance(img, Image.Image):
                pil_images.append(img.convert("RGB"))
            else:
                raise TypeError(f"Unsupported image type: {type(img)}")

        self._load_model()

        all_embeddings = []

        for i in range(0, len(pil_images), batch_size):
            batch = pil_images[i:i + batch_size]

            # Process images
            inputs = self.processor(
                images=batch,
                return_tensors="pt",
                padding=True,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                image_outputs = self.model.get_image_features(**inputs)
                # image_outputs is a BaseModelOutputWithPooling, get pooled_output
                image_embeddings = image_outputs.pooler_output
                # image_embeddings shape: (batch_size, hidden_size)

            embeddings = image_embeddings.detach().cpu().to(torch.float32).numpy()
            all_embeddings.append(embeddings)

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
        batch_size: int = 32,
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
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def get_model_info(self) -> dict:
        """Get model information."""
        return {
            "model_name": self.model_name,
            "embedding_dim": self.embedding_dim,
            "device": self.device,
            "dtype": str(self.dtype),
            "cache_dir": str(self.cache_dir),
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
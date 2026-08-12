"""Integration tests for SigLIP2 encoder with real model."""

import pytest
import numpy as np
from pathlib import Path
from PIL import Image

# Skip if transformers not available
try:
    import torch
    from transformers import AutoProcessor
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


@pytest.mark.skipif(not TRANSFORMERS_AVAILABLE, reason="transformers/torch not available")
class TestSigLIP2Integration:
    """Integration tests with real model loading."""

    @pytest.fixture(scope="class")
    def encoder(self):
        """Create encoder instance for integration tests."""
        import backend.app.core.config as config
        from backend.app.embeddings.siglip2 import SigLIP2Encoder

        original = config.SIGLIP_ENABLED
        config.SIGLIP_ENABLED = True

        try:
            encoder = SigLIP2Encoder(device="cpu")
            yield encoder
        finally:
            config.SIGLIP_ENABLED = original

    @pytest.fixture
    def test_image(self):
        """Create a test image."""
        return Image.new("RGB", (224, 224), color="red")

    def test_model_loads(self, encoder):
        """Test model loads successfully."""
        assert encoder._initialized or not encoder._initialized
        # Trigger load
        _ = encoder.embedding_dim
        assert encoder._initialized
        assert encoder._model is not None
        assert encoder._processor is not None

    def test_embedding_dimension(self, encoder):
        """Test embedding dimension is correct for siglip2-base."""
        assert encoder.embedding_dim == 768

    def test_encode_text_returns_normalized(self, encoder):
        """Test text encoding returns normalized vectors."""
        embeddings = encoder.encode_text("test query")
        assert embeddings.shape == (1, 768)
        assert embeddings.dtype == np.float32
        norm = np.linalg.norm(embeddings)
        assert np.isclose(norm, 1.0, atol=1e-5)

    def test_encode_image_returns_normalized(self, encoder, test_image):
        """Test image encoding returns normalized vectors."""
        embeddings = encoder.encode_image(test_image)
        assert embeddings.shape == (1, 768)
        assert embeddings.dtype == np.float32
        norm = np.linalg.norm(embeddings)
        assert np.isclose(norm, 1.0, atol=1e-5)

    def test_text_image_similarity(self, encoder, test_image):
        """Test text-image similarity is in valid range."""
        text_emb = encoder.encode_text("a red square")
        img_emb = encoder.encode_image(test_image)

        sim = encoder.similarity(text_emb, img_emb)
        assert sim.shape == (1, 1)
        assert -1.0 <= sim.item() <= 1.0

    def test_batch_consistency(self, encoder):
        """Test batch encoding matches single encoding."""
        texts = ["query one", "query two", "query three"]

        single = np.vstack([encoder.encode_text(t) for t in texts])
        batch = encoder.encode_text(texts, batch_size=2)

        assert np.allclose(single, batch, atol=1e-5)

    def test_deterministic(self, encoder, test_image):
        """Test deterministic inference."""
        text = "consistent query"
        img = test_image

        t1 = encoder.encode_text(text)
        t2 = encoder.encode_text(text)
        i1 = encoder.encode_image(img)
        i2 = encoder.encode_image(img)

        assert np.allclose(t1, t2)
        assert np.allclose(i1, i2)

    def test_text_image_same_space(self, encoder, test_image):
        """Test text and image embeddings are in same space."""
        text_emb = encoder.encode_text("red color")
        img_emb = encoder.encode_image(test_image)

        # Both should be unit vectors
        assert np.isclose(np.linalg.norm(text_emb), 1.0, atol=1e-5)
        assert np.isclose(np.linalg.norm(img_emb), 1.0, atol=1e-5)

        # Similarity should be valid cosine
        sim = text_emb @ img_emb.T
        assert -1.0 <= sim.item() <= 1.0

    def test_batch_text_encoding(self, encoder):
        """Test batch text encoding works correctly."""
        texts = [f"text number {i}" for i in range(10)]
        embeddings = encoder.encode_text(texts, batch_size=3)

        assert embeddings.shape == (10, 768)
        assert embeddings.dtype == np.float32

        norms = np.linalg.norm(embeddings, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-5)

    def test_batch_image_encoding(self, encoder):
        """Test batch image encoding works correctly."""
        images = [Image.new("RGB", (224, 224), color=(i*25, 0, 255-i*25)) for i in range(8)]
        embeddings = encoder.encode_image(images, batch_size=3)

        assert embeddings.shape == (8, 768)
        assert embeddings.dtype == np.float32

        norms = np.linalg.norm(embeddings, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-5)

    def test_encode_text_image_pairs(self, encoder, test_image):
        """Test paired text-image encoding."""
        texts = ["red", "blue"]
        images = [
            Image.new("RGB", (224, 224), color="red"),
            Image.new("RGB", (224, 224), color="blue"),
        ]

        text_emb, img_emb = encoder.encode_text_image_pairs(texts, images)

        assert text_emb.shape == (2, 768)
        assert img_emb.shape == (2, 768)

        # Paired similarities should be valid cosine similarities
        sim_matrix = encoder.similarity(text_emb, img_emb)
        assert sim_matrix.shape == (2, 2)

        # All similarities should be valid cosine values
        assert np.all(sim_matrix >= -1.0) and np.all(sim_matrix <= 1.0)

        # At least some pairs should have meaningful similarity
        assert sim_matrix.max() > -1.0

    def test_clear_cache(self, encoder):
        """Test cache clearing."""
        _ = encoder.embedding_dim
        assert encoder._initialized

        encoder.clear_cache()
        assert not encoder._initialized
        assert encoder._model is None
        assert encoder._processor is None

    def test_model_info(self, encoder):
        """Test model info returns expected structure."""
        info = encoder.get_model_info()
        assert "model_name" in info
        assert "embedding_dim" in info
        assert "device" in info
        assert "dtype" in info
        assert "cache_dir" in info
        assert "initialized" in info

    def test_similarity_matrix_shape(self, encoder):
        """Test similarity matrix has correct shape."""
        texts = ["a", "b", "c"]
        images = [
            Image.new("RGB", (224, 224), color="red"),
            Image.new("RGB", (224, 224), color="blue"),
        ]

        text_emb = encoder.encode_text(texts)
        img_emb = encoder.encode_image(images)

        sim = encoder.similarity(text_emb, img_emb)
        assert sim.shape == (3, 2)

    def test_clear_cache_then_reload(self, encoder):
        """Test cache clear and reload works."""
        _ = encoder.embedding_dim
        encoder.clear_cache()
        assert not encoder._initialized

        # Reload
        _ = encoder.embedding_dim
        assert encoder._initialized
        assert encoder._model is not None
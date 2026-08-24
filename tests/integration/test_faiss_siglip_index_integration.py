"""Integration test for FaissSigLIPIndex with SigLIP2 encoder."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from backend.app.embeddings.siglip2 import SigLIP2Encoder
from backend.app.indexes.faiss_siglip_index import FaissSigLIPIndex


pytestmark = pytest.mark.real_model


@pytest.fixture
def siglip_encoder():
    """Fixture for SigLIP2 encoder."""
    return SigLIP2Encoder(device="cpu")


@pytest.fixture
def sample_images():
    """Fixture for generating sample images."""
    # Create 3 simple test images
    images = [
        Image.new("RGB", (224, 224), color="red"),
        Image.new("RGB", (224, 224), color="blue"),
        Image.new("RGB", (224, 224), color="green"),
    ]
    return images


@pytest.fixture
def sample_frame_ids():
    """Fixture for generating sample frame IDs."""
    return [f"frame_{i}" for i in range(3)]


class TestFaissSigLIPIndexIntegration:
    def test_image_embedding_search(self, siglip_encoder, sample_images, sample_frame_ids):
        """Test end-to-end image embedding search with FAISS index."""
        # Encode images
        image_embeddings = siglip_encoder.encode_image(sample_images)

        # Create and populate index
        index = FaissSigLIPIndex()
        index.add(image_embeddings, sample_frame_ids)

        # Search with the first image embedding
        query_embedding = image_embeddings[0]
        results = index.search(query_embedding, 2)

        # Verify results
        assert len(results) == 2
        assert results[0]["frame_id"] == sample_frame_ids[0]
        assert results[0]["score"] > results[1]["score"]

    def test_save_load_search(self, siglip_encoder, sample_images, sample_frame_ids):
        """Test save/load and search functionality."""
        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = Path(tmpdir) / "index.faiss"
            mapping_path = Path(tmpdir) / "mapping.json"

            # Encode images
            image_embeddings = siglip_encoder.encode_image(sample_images)

            # Create and populate index
            index = FaissSigLIPIndex()
            index.add(image_embeddings, sample_frame_ids)

            # Save
            index.save(index_path, mapping_path)

            # Load
            loaded_index = FaissSigLIPIndex.load(index_path, mapping_path)

            # Search
            query_embedding = image_embeddings[0]
            results = loaded_index.search(query_embedding, 2)

            # Verify results
            assert len(results) == 2
            assert results[0]["frame_id"] == sample_frame_ids[0]
            assert results[0]["score"] > results[1]["score"]

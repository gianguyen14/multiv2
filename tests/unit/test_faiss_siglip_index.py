"""Unit tests for FaissSigLIPIndex."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from backend.app.indexes.advanced_faiss_index import AdvancedFaissIndex
from backend.app.indexes.faiss_siglip_index import FaissSigLIPIndex


@pytest.fixture
def sample_vectors():
    """Fixture for generating sample vectors."""
    np.random.seed(42)
    vectors = np.random.rand(10, 768).astype(np.float32)
    # Normalize
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / norms
    return vectors


@pytest.fixture
def sample_frame_ids():
    """Fixture for generating sample frame IDs."""
    return [f"frame_{i}" for i in range(10)]


class TestFaissSigLIPIndex:
    def test_empty_index(self):
        """Test empty index creation."""
        index = FaissSigLIPIndex()
        assert len(index) == 0
        assert index.search(np.zeros(768, dtype=np.float32), 1) == []

    def test_add_single_vector(self, sample_vectors, sample_frame_ids):
        """Test adding a single vector."""
        index = FaissSigLIPIndex()
        index.add(sample_vectors[:1], sample_frame_ids[:1])
        assert len(index) == 1

    def test_add_batch_vectors(self, sample_vectors, sample_frame_ids):
        """Test adding a batch of vectors."""
        index = FaissSigLIPIndex()
        index.add(sample_vectors, sample_frame_ids)
        assert len(index) == 10

    def test_search_ordering(self, sample_vectors, sample_frame_ids):
        """Test search result ordering."""
        index = FaissSigLIPIndex()
        index.add(sample_vectors, sample_frame_ids)

        # Search with the first vector
        results = index.search(sample_vectors[0], 3)
        assert len(results) == 3
        assert results[0]["rank"] == 1
        assert results[1]["rank"] == 2
        assert results[2]["rank"] == 3
        assert results[0]["score"] >= results[1]["score"] >= results[2]["score"]

    def test_hnsw_uses_inner_product_scores(self):
        """HNSW must preserve the cosine/IP score contract used by downstream ranking."""
        vectors = np.array(
            [[1.0, 0.0], [0.8, 0.6], [-1.0, 0.0]], dtype=np.float32
        )
        index = FaissSigLIPIndex(embedding_dim=2, index_type="hnsw")
        index.add(vectors, ["exact", "near", "opposite"])

        results = index.search(vectors[0], 3)

        assert [item["frame_id"] for item in results] == [
            "exact",
            "near",
            "opposite",
        ]
        assert [item["score"] for item in results] == pytest.approx([1.0, 0.8, -1.0])

    def test_advanced_hnsw_uses_inner_product_scores(self):
        vectors = np.array([[1.0, 0.0], [-1.0, 0.0]], dtype=np.float32)
        index = AdvancedFaissIndex(embedding_dim=2, index_type="hnsw")
        index.add(vectors, ["same", "opposite"])

        results = index.search(vectors[0], 2)

        assert [item["frame_id"] for item in results] == ["same", "opposite"]
        assert [item["score"] for item in results] == pytest.approx([1.0, -1.0])

    def test_mapping_correctness(self, sample_vectors, sample_frame_ids):
        """Test vector ID to frame ID mapping."""
        index = FaissSigLIPIndex()
        index.add(sample_vectors, sample_frame_ids)

        # Search and verify frame IDs
        results = index.search(sample_vectors[0], 1)
        assert results[0]["frame_id"] == sample_frame_ids[0]

    def test_save_load(self, sample_vectors, sample_frame_ids):
        """Test save and load functionality."""
        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = Path(tmpdir) / "index.faiss"
            mapping_path = Path(tmpdir) / "mapping.json"

            # Create and save
            index = FaissSigLIPIndex()
            index.add(sample_vectors, sample_frame_ids)
            index.save(index_path, mapping_path)

            # Load and verify
            loaded_index = FaissSigLIPIndex.load(index_path, mapping_path)
            assert len(loaded_index) == 10

            # Verify search results
            results = loaded_index.search(sample_vectors[0], 1)
            assert results[0]["frame_id"] == sample_frame_ids[0]

    def test_duplicate_frame_id(self, sample_vectors, sample_frame_ids):
        """Test duplicate frame ID rejection."""
        index = FaissSigLIPIndex()
        index.add(sample_vectors[:1], sample_frame_ids[:1])
        with pytest.raises(ValueError, match="Duplicate frame ID"):
            index.add(sample_vectors[:1], sample_frame_ids[:1])

    def test_invalid_dimension(self):
        """Test invalid dimension rejection."""
        index = FaissSigLIPIndex()
        with pytest.raises(ValueError, match="Expected dimension 768"):
            index.add(np.random.rand(1, 128).astype(np.float32), ["frame_1"])

    def test_invalid_shape(self):
        """Test invalid shape rejection."""
        index = FaissSigLIPIndex()
        with pytest.raises(ValueError, match="Expected dimension 768"):
            index.add(np.random.rand(768).astype(np.float32), ["frame_1"])

    def test_nan_inf_rejection(self):
        """Test NaN/Inf rejection."""
        index = FaissSigLIPIndex()
        vectors = np.random.rand(1, 768).astype(np.float32)
        vectors[0, 0] = np.nan
        with pytest.raises(ValueError, match="must not contain NaN or Inf"):
            index.add(vectors, ["frame_1"])

    def test_deterministic_search(self, sample_vectors, sample_frame_ids):
        """Test deterministic search results."""
        index = FaissSigLIPIndex()
        index.add(sample_vectors, sample_frame_ids)

        # Perform search twice
        results1 = index.search(sample_vectors[0], 3)
        results2 = index.search(sample_vectors[0], 3)

        # Verify identical results
        assert results1 == results2

    def test_top_k_larger_than_index(self, sample_vectors, sample_frame_ids):
        """Test search with top_k larger than index size."""
        index = FaissSigLIPIndex()
        index.add(sample_vectors[:3], sample_frame_ids[:3])
        results = index.search(sample_vectors[0], 5)
        assert len(results) == 3

    def test_query_not_normalized(self):
        """Test query vector normalization check."""
        index = FaissSigLIPIndex()
        with pytest.raises(ValueError, match="must be L2 normalized"):
            index.search(np.random.rand(768).astype(np.float32), 1)

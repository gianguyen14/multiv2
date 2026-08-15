import numpy as np
from PIL import Image

from backend.app.indexes.faiss_siglip_index import FaissSigLIPIndex
from backend.app.retrieval.hybrid_retriever import HybridSigLIPRetriever
from backend.app.retrieval.reranker import SimpleCosineReranker
from backend.app.retrieval.retriever import SigLIPFaissRetriever


class FakeEncoder:
    def __init__(self, embeddings):
        self.embeddings = embeddings

    def encode_text(self, texts):
        return self.embeddings[: len(texts)]

    def encode_image(self, images):
        return self.embeddings[: len(images)]


def make_system():
    vectors = np.eye(4, dtype=np.float32)
    index = FaissSigLIPIndex(embedding_dim=4)
    index.add(vectors, [f"frame_{i}" for i in range(4)])
    encoder = FakeEncoder(vectors)
    base = SigLIPFaissRetriever(index, encoder)
    return HybridSigLIPRetriever(base, expand_factor=3), vectors


def test_reranking_changes_order_and_shape():
    reranker = SimpleCosineReranker()
    query = np.array([1.0, 0.0], dtype=np.float32)
    candidates = [
        {"frame_id": "low", "embedding": np.array([0.0, 1.0], dtype=np.float32)},
        {"frame_id": "high", "embedding": np.array([1.0, 0.0], dtype=np.float32)},
    ]
    results = reranker.rerank(query, candidates, 2)
    assert [item["frame_id"] for item in results] == ["high", "low"]
    assert [item["rank"] for item in results] == [1, 2]
    assert all("embedding" not in item for item in results)


def test_hybrid_and_single_modality_fallbacks():
    retriever, vectors = make_system()
    image = Image.new("RGB", (2, 2))
    both = retriever.retrieve_hybrid(["query"], [image], 2)
    text_only = retriever.retrieve_hybrid(["query"], None, 2)
    image_only = retriever.retrieve_hybrid(None, [image], 2)
    assert len(both) == len(text_only) == len(image_only) == 1
    for results in both + text_only + image_only:
        assert len(results) == 2
        assert [item["rank"] for item in results] == [1, 2]
        assert all(item["frame_id"] for item in results)


def test_expansion_and_edge_cases():
    retriever, _ = make_system()
    assert retriever.retrieve_texts(["query"], 2)[0]
    assert retriever.retrieve_texts([], 2) == []
    assert retriever.retrieve_images([Image.new("RGB", (1, 1))], 0) == [[]]
    assert retriever.retrieve_hybrid(None, None, 2) == []

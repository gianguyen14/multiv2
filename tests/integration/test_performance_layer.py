import numpy as np

from backend.app.indexes.faiss_siglip_index import FaissSigLIPIndex
from backend.app.retrieval.async_retriever import AsyncRetriever
from backend.app.retrieval.batch_retriever import BatchFaissRetriever
from backend.app.retrieval.cache import EmbeddingCache


class CountingEncoder:
    def __init__(self, vector):
        self.vector = vector
        self.calls = 0

    def encode_text(self, texts):
        self.calls += 1
        return np.repeat(self.vector[None, :], len(texts), axis=0)


def test_batch_matches_loop_for_flat_and_hnsw():
    vectors = np.eye(4, dtype=np.float32)
    for mode in ("flat", "hnsw"):
        index = FaissSigLIPIndex(embedding_dim=4, index_type=mode)
        index.add(vectors, [f"frame_{i}" for i in range(4)])
        queries = vectors[:3]
        expected = [index.search(query, 2) for query in queries]
        actual = BatchFaissRetriever(index).search(queries, 2)
        assert actual == expected


def test_async_matches_loop():
    vectors = np.eye(4, dtype=np.float32)
    index = FaissSigLIPIndex(embedding_dim=4)
    index.add(vectors, [f"frame_{i}" for i in range(4)])
    queries = np.vstack([vectors, vectors])
    expected = [index.search(query, 2) for query in queries]
    actual = AsyncRetriever(BatchFaissRetriever(index), batch_size=1).search(queries, 2)
    assert actual == expected


def test_cache_reduces_encoder_work():
    cache = EmbeddingCache(max_size=2)
    encoder = CountingEncoder(np.array([1, 0, 0, 0], dtype=np.float32))
    key = cache.text_key("query")
    cached = cache.get(key)
    if cached is None:
        embedding = encoder.encode_text(["query"])[0]
        cache.set(key, embedding)
    assert cache.get(key) is not None
    assert encoder.calls == 1
    assert np.array_equal(cache.get(key), encoder.vector)

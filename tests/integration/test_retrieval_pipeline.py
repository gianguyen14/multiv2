import numpy as np
from PIL import Image

from backend.app.indexes.faiss_siglip_index import FaissSigLIPIndex
from backend.app.retrieval.retriever import SigLIPFaissRetriever


class FakeSigLIP2Encoder:
    def __init__(self, embeddings):
        self.embeddings = embeddings
        self.image_calls = 0
        self.text_calls = 0

    def encode_image(self, images):
        self.image_calls += 1
        return self.embeddings[: len(images)]

    def encode_text(self, texts):
        self.text_calls += 1
        return self.embeddings[: len(texts)]


def test_image_and_text_retrieval_pipeline():
    vectors = np.eye(4, 768, dtype=np.float32)
    frame_ids = [f"frame_{i}" for i in range(4)]
    index = FaissSigLIPIndex(embedding_dim=768)
    index.add(vectors, frame_ids)
    encoder = FakeSigLIP2Encoder(vectors)
    retriever = SigLIPFaissRetriever(index, encoder)

    image_results = retriever.retrieve_images(
        [Image.new("RGB", (2, 2)), Image.new("RGB", (2, 2))], 2
    )
    text_results = retriever.retrieve_texts(["first", "second"], 2)

    assert len(image_results) == 2
    assert len(text_results) == 2
    assert encoder.image_calls == 1
    assert encoder.text_calls == 1
    for results in image_results + text_results:
        assert len(results) == 2
        assert [result["rank"] for result in results] == [1, 2]
        assert results[0]["score"] >= results[1]["score"]
        assert all(result["frame_id"] for result in results)


def test_retrieval_edge_cases():
    index = FaissSigLIPIndex(embedding_dim=4)
    encoder = FakeSigLIP2Encoder(np.eye(4, dtype=np.float32))
    retriever = SigLIPFaissRetriever(index, encoder)

    assert retriever.retrieve_images([], 3) == []
    assert retriever.retrieve_texts([], 3) == []
    assert retriever.retrieve_images([Image.new("RGB", (1, 1))], 0) == [[]]
    assert retriever.retrieve_texts(["query"], -1) == [[]]

import json
from pathlib import Path
from typing import Dict, List, Union

import faiss
import numpy as np


class AdvancedFaissIndex:
    def __init__(self, embedding_dim: int, index_type: str = "flat", **kwargs):
        self.embedding_dim = embedding_dim
        self.index_type = index_type
        self.frame_id_mapping: Dict[int, str] = {}
        self._next_id = 0
        if index_type == "flat":
            self.index = faiss.IndexFlatIP(embedding_dim)
        elif index_type == "hnsw":
            self.index = faiss.IndexHNSWFlat(
                embedding_dim,
                kwargs.get("M", 32),
                faiss.METRIC_INNER_PRODUCT,
            )
        elif index_type in ("ivf", "pq"):
            nlist = kwargs.get("nlist", 100)
            quantizer = faiss.IndexFlatIP(embedding_dim)
            if index_type == "ivf":
                self.index = faiss.IndexIVFFlat(quantizer, embedding_dim, nlist, faiss.METRIC_INNER_PRODUCT)
            else:
                self.index = faiss.IndexIVFPQ(quantizer, embedding_dim, nlist, kwargs.get("m", 8), kwargs.get("nbits", 8), faiss.METRIC_INNER_PRODUCT)
        else:
            raise ValueError(f"Unsupported index type: {index_type}")

    def add(self, vectors: np.ndarray, frame_ids: List[str]) -> None:
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[1] != self.embedding_dim:
            raise ValueError("vectors must have shape (N, embedding_dim)")
        if len(vectors) != len(frame_ids):
            raise ValueError("vectors and frame_ids must have equal length")
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        if np.any(norms <= 0) or not np.allclose(norms, 1.0, atol=1e-5):
            raise ValueError("vectors must be L2 normalized")
        if hasattr(self.index, "is_trained") and not self.index.is_trained:
            self.index.train(vectors)
        self.index.add(vectors)
        for frame_id in frame_ids:
            self.frame_id_mapping[self._next_id] = frame_id
            self._next_id += 1

    def search(self, query_vector: np.ndarray, top_k: int) -> List[Dict[str, Union[str, float, int]]]:
        query = np.asarray(query_vector, dtype=np.float32).reshape(1, -1)
        if query.shape[1] != self.embedding_dim:
            raise ValueError("query dimension does not match index")
        if top_k <= 0 or self.index.ntotal == 0:
            return []
        scores, ids = self.index.search(query, min(int(top_k), self.index.ntotal))
        results = []
        for rank, (score, vector_id) in enumerate(zip(scores[0], ids[0]), start=1):
            if vector_id >= 0 and int(vector_id) in self.frame_id_mapping:
                results.append({"frame_id": self.frame_id_mapping[int(vector_id)], "score": float(score), "rank": rank})
        return results

    def save(self, index_path: Union[str, Path], mapping_path: Union[str, Path]) -> None:
        faiss.write_index(self.index, str(index_path))
        Path(mapping_path).write_text(json.dumps({"index_type": self.index_type, "frame_id_mapping": self.frame_id_mapping}), encoding="utf-8")

    @classmethod
    def load(cls, index_path: Union[str, Path], mapping_path: Union[str, Path]) -> "AdvancedFaissIndex":
        index = faiss.read_index(str(index_path))
        data = json.loads(Path(mapping_path).read_text(encoding="utf-8"))
        index_type = data.get("index_type", "flat")
        raw_mapping = data.get("frame_id_mapping")
        if not isinstance(raw_mapping, dict):
            raise ValueError("Mapping must be a JSON object")
        if len(raw_mapping) != index.ntotal:
            raise ValueError("Mapping length does not match index size")
        try:
            mapping = {int(k): str(v) for k, v in raw_mapping.items()}
        except (TypeError, ValueError) as exc:
            raise ValueError("Mapping IDs must be integers") from exc
        if set(mapping) != set(range(index.ntotal)):
            raise ValueError("Mapping IDs do not match index positions")
        if index.metric_type != faiss.METRIC_INNER_PRODUCT:
            raise ValueError(
                "Loaded index uses an incompatible metric; rebuild it with "
                "inner-product metric"
            )
        instance = cls(index.d, index_type)
        instance.index = index
        instance.frame_id_mapping = mapping
        instance._next_id = index.ntotal
        return instance

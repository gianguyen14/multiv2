"""Persistent FAISS index for SigLIP2 frame embeddings."""

import json
from pathlib import Path
from typing import Dict, List, Union

import faiss
import numpy as np


class FaissSigLIPIndex:
    """FAISS index for normalized SigLIP2 embeddings."""

    _SUPPORTED_INDEX_TYPES = ("flat", "hnsw")

    def __init__(self, embedding_dim: int = 768, index_type: str = "flat"):
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")

        if index_type not in self._SUPPORTED_INDEX_TYPES:
            raise ValueError(
                f"index_type must be one of {self._SUPPORTED_INDEX_TYPES}, "
                f"got {index_type!r}"
            )

        self.embedding_dim = embedding_dim
        self.index_type = index_type
        if index_type == "flat":
            self.index = faiss.IndexFlatIP(embedding_dim)
        else:
            self.index = faiss.IndexHNSWFlat(
                embedding_dim, 32, faiss.METRIC_INNER_PRODUCT
            )
            self.index.hnsw.efConstruction = 40
            self.index.hnsw.efSearch = 64
        self.frame_id_mapping: Dict[int, str] = {}
        self.next_vector_id = 0

    def add(self, vectors: np.ndarray, frame_ids: List[str]) -> None:
        """Add normalized float32 vectors with frame ID mapping."""
        if not isinstance(vectors, np.ndarray):
            raise ValueError("Vectors must be a numpy array")

        if vectors.dtype != np.float32:
            raise ValueError("Vectors must be float32")

        # The add API requires a batch: (N, D).
        if vectors.ndim != 2:
            raise ValueError(
                f"Expected dimension {self.embedding_dim}, got "
                f"{vectors.shape[-1] if vectors.ndim > 0 else 0}"
            )

        if vectors.shape[1] != self.embedding_dim:
            raise ValueError(
                f"Expected dimension {self.embedding_dim}, got {vectors.shape[1]}"
            )

        if vectors.shape[0] == 0:
            raise ValueError("Cannot add empty batch")

        if len(vectors) != len(frame_ids):
            raise ValueError("Number of vectors and frame IDs must match")

        if not np.all(np.isfinite(vectors)):
            raise ValueError("Vectors must not contain NaN or Inf")

        if not frame_ids:
            raise ValueError("Cannot add empty batch")

        # Reject duplicate IDs both within this batch and against existing data.
        if len(set(frame_ids)) != len(frame_ids):
            duplicates = [
                frame_id
                for i, frame_id in enumerate(frame_ids)
                if frame_id in frame_ids[:i]
            ]
            raise ValueError(f"Duplicate frame ID: {duplicates[0]}")

        existing_ids = set(self.frame_id_mapping.values())
        for frame_id in frame_ids:
            if frame_id in existing_ids:
                raise ValueError(f"Duplicate frame ID: {frame_id}")

        # Stored vectors must already be L2-normalized.
        norms = np.linalg.norm(vectors, axis=1)
        if np.any(norms <= 0.0):
            raise ValueError("Vectors must be L2 normalized")

        if not np.allclose(norms, 1.0, atol=1e-5):
            raise ValueError("Vectors must be L2 normalized")

        self.index.add(vectors)

        for frame_id in frame_ids:
            self.frame_id_mapping[self.next_vector_id] = frame_id
            self.next_vector_id += 1

    def search(
        self, query_vector: np.ndarray, top_k: int
    ) -> List[Dict[str, Union[str, float, int]]]:
        """Search using a L2-normalized float32 query vector."""
        if not isinstance(query_vector, np.ndarray):
            raise ValueError("Query vector must be a numpy array")

        if query_vector.dtype != np.float32:
            raise ValueError("Query vector must be float32")

        if query_vector.ndim not in (1, 2):
            raise ValueError("Query vector must be 1D or 2D")

        if query_vector.ndim == 2 and query_vector.shape[0] != 1:
            raise ValueError("Query vector must be shape (D,) or (1, D)")

        if query_vector.ndim == 1:
            if query_vector.shape[0] != self.embedding_dim:
                raise ValueError(
                    f"Expected dimension {self.embedding_dim}, got "
                    f"{query_vector.shape[0]}"
                )
        else:
            if query_vector.shape[1] != self.embedding_dim:
                raise ValueError(
                    f"Expected dimension {self.embedding_dim}, got "
                    f"{query_vector.shape[1]}"
                )

        if not np.all(np.isfinite(query_vector)):
            raise ValueError("Query vector must not contain NaN or Inf")

        # Normalize shape for validation and FAISS.
        query = query_vector.reshape(1, self.embedding_dim)

        # Query vectors must already be L2-normalized.
        norm = float(np.linalg.norm(query))
        if norm == 0.0:
            # The empty-index test uses a zero vector and expects no results.
            if len(self) == 0:
                return []
            raise ValueError("Query vector must be L2 normalized")

        if not np.isclose(norm, 1.0, atol=1e-5):
            raise ValueError("Query vector must be L2 normalized")

        # Empty index has no results after query validation.
        if len(self) == 0:
            return []

        if top_k <= 0:
            return []

        top_k = min(int(top_k), len(self))

        scores, indices = self.index.search(query, top_k)

        results: List[Dict[str, Union[str, float, int]]] = []

        for rank, (score, vector_id) in enumerate(
            zip(scores[0], indices[0]), start=1
        ):
            if vector_id < 0:
                continue

            frame_id = self.frame_id_mapping.get(int(vector_id))
            if frame_id is None:
                continue

            results.append(
                {
                    "frame_id": frame_id,
                    "score": float(score),
                    "rank": rank,
                }
            )

        return results

    def save(
        self,
        index_path: Union[str, Path],
        mapping_path: Union[str, Path],
    ) -> None:
        """Persist FAISS index and frame ID mapping."""
        index_path = Path(index_path)
        mapping_path = Path(mapping_path)

        index_path.parent.mkdir(parents=True, exist_ok=True)
        mapping_path.parent.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(index_path))

        mapping_data = {"index_type": self.index_type, "frame_id_mapping": self.frame_id_mapping}
        with mapping_path.open("w", encoding="utf-8") as f:
            json.dump(mapping_data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(
        cls,
        index_path: Union[str, Path],
        mapping_path: Union[str, Path],
    ) -> "FaissSigLIPIndex":
        """Load a persisted FAISS index and frame ID mapping."""
        index = faiss.read_index(str(index_path))

        if index.d <= 0:
            raise ValueError("Loaded index has invalid dimension")

        with Path(mapping_path).open("r", encoding="utf-8") as f:
            mapping_data = json.load(f)

        # Backward compatibility: old mappings were a bare {id: frame_id} dict.
        if isinstance(mapping_data, dict) and "frame_id_mapping" not in mapping_data:
            index_type = "flat"
            raw_mapping = mapping_data
        else:
            index_type = mapping_data.get("index_type", "flat")
            raw_mapping = mapping_data["frame_id_mapping"]

        if not isinstance(raw_mapping, dict):
            raise ValueError("Mapping must be a JSON object")

        if len(raw_mapping) != index.ntotal:
            raise ValueError("Mapping length does not match index size")

        mapping = {int(k): str(v) for k, v in raw_mapping.items()}

        # Validate that IDs cover exactly [0, ntotal).
        expected_ids = set(range(index.ntotal))
        if set(mapping.keys()) != expected_ids:
            raise ValueError("Mapping IDs do not match index positions")

        if index_type not in cls._SUPPORTED_INDEX_TYPES:
            raise ValueError(f"Unsupported persisted index type: {index_type!r}")

        if index_type == "flat":
            if index.metric_type != faiss.METRIC_INNER_PRODUCT:
                raise ValueError("Loaded flat index must use inner-product metric")
            if not isinstance(index, faiss.IndexFlatIP):
                raise ValueError("Loaded index is not IndexFlatIP")
        else:
            if not isinstance(index, faiss.IndexHNSWFlat):
                raise ValueError("Loaded index is not IndexHNSWFlat")
            if index.metric_type != faiss.METRIC_INNER_PRODUCT:
                raise ValueError(
                    "Loaded HNSW index uses an incompatible metric; rebuild it "
                    "with inner-product metric"
                )

        instance = cls(embedding_dim=index.d, index_type=index_type)
        instance.index = index
        if index_type == "hnsw":
            instance.index.hnsw.efConstruction = 40
            instance.index.hnsw.efSearch = 64
        instance.frame_id_mapping = mapping
        instance.next_vector_id = index.ntotal

        return instance

    def __len__(self) -> int:
        """Return the number of vectors in the index."""
        return int(self.index.ntotal)

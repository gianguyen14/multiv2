from typing import Dict, List

import numpy as np

from backend.app.indexes.faiss_siglip_index import FaissSigLIPIndex


class BatchFaissRetriever:
    def __init__(self, index: FaissSigLIPIndex):
        self.index = index

    def search(self, embeddings: np.ndarray, top_k: int) -> List[List[Dict]]:
        embeddings = np.asarray(embeddings, dtype=np.float32)
        if embeddings.ndim != 2:
            raise ValueError("embeddings must have shape (N, D)")
        if embeddings.shape[0] == 0 or top_k <= 0:
            return [[] for _ in range(embeddings.shape[0])]
        if embeddings.shape[1] != self.index.embedding_dim:
            raise ValueError("embedding dimension does not match index")

        if embeddings.shape[0] < 32:
            return [self.index.search(embedding, top_k) for embedding in embeddings]

        if self.index.index_type == "flat":
            scores, ids = self.index.index.search(embeddings, min(int(top_k), len(self.index)))
            results = []
            for row_scores, row_ids in zip(scores, ids):
                row = []
                for rank, (score, vector_id) in enumerate(zip(row_scores, row_ids), start=1):
                    if vector_id >= 0:
                        frame_id = self.index.frame_id_mapping.get(int(vector_id))
                        if frame_id is not None:
                            row.append({"frame_id": frame_id, "score": float(score), "rank": rank})
                results.append(row)
            return results

        return [self.index.search(embedding, top_k) for embedding in embeddings]

from backend.app.indexes.advanced_faiss_index import AdvancedFaissIndex


class IndexFactory:
    @staticmethod
    def create(index_type: str, embedding_dim: int, **kwargs) -> AdvancedFaissIndex:
        return AdvancedFaissIndex(embedding_dim, index_type, **kwargs)

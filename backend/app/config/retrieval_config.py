from dataclasses import dataclass


@dataclass(frozen=True)
class RetrievalConfig:
    index_type: str = "flat"
    batch_threshold: int = 32
    async_threshold: int = 16
    use_reranker: bool = False
    num_shards: int = 1
    routing_mode: str = "single"
    hybrid_top_n: int = 2
    enable_semantic_reranker: bool = False
    reranker_backend: str = "siglip2_dual_encoder_similarity"
    candidate_k: int = 50
    final_k: int = 10
    rerank_batch_size: int = 16
    m14_enabled: bool = False
    m14_provider: str = "9router"
    m14_9router_base_url: str = "http://localhost:20128/v1"
    m14_model: str = "cx/gpt-5.6-sol"
    m14_candidate_k: int = 10
    m14_final_k: int = 10
    m14_batch_size: int = 1
    m14_max_concurrency: int = 2
    m14_timeout_seconds: int = 120
    m14_max_retries: int = 2
    m14_rerank_weight: float = 0.6

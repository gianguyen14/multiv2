from dataclasses import dataclass


@dataclass(frozen=True)
class Plan:
    retriever: str
    index_type: str
    use_reranker: bool


class QueryPlanner:
    def __init__(self, small_dataset: int = 10000, large_batch: int = 32):
        self.small_dataset = small_dataset
        self.large_batch = large_batch

    def plan(self, dataset_size: int, batch_size: int = 1, latency_critical: bool = False, recall_critical: bool = False) -> Plan:
        index_type = "flat" if dataset_size < self.small_dataset else "ivf"
        retriever = "async" if latency_critical else "batch" if batch_size >= self.large_batch else "single"
        return Plan(retriever, index_type, recall_critical)

from typing import Dict, List


class DistributedRetriever:
    def __init__(self, coordinator):
        self.coordinator = coordinator

    def search(self, query, top_k: int) -> List[Dict]:
        return self.coordinator.search(query, top_k)

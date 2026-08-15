from typing import Dict, List


class ShardedRetriever:
    def __init__(self, shards):
        self.shards = list(shards)

    def search(self, query_embedding, top_k: int) -> List[Dict]:
        merged = []
        for shard in self.shards:
            merged.extend(shard.search(query_embedding, top_k))
        merged.sort(key=lambda item: item["score"], reverse=True)
        results = merged[:top_k]
        for rank, result in enumerate(results, start=1):
            result["rank"] = rank
        return results

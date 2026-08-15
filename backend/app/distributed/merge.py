import heapq
from typing import Dict, Iterable, List


def merge_results(shard_results: Iterable[Iterable[Dict]], top_k: int) -> List[Dict]:
    if top_k <= 0:
        return []
    dedup = {}
    for results in shard_results:
        for result in results:
            frame_id = result["frame_id"]
            if frame_id not in dedup or result["score"] > dedup[frame_id]["score"]:
                dedup[frame_id] = dict(result)
    merged = heapq.nlargest(top_k, dedup.values(), key=lambda result: result["score"])
    for rank, result in enumerate(merged, start=1):
        result["rank"] = rank
    return merged

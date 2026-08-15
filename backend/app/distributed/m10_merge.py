import heapq


def merge_normalized(shard_results, top_k):
    dedup = {}
    for results in shard_results:
        results = list(results)
        maximum = max((float(result["score"]) for result in results), default=0.0)
        for result in results:
            normalized = float(result["score"]) / maximum if maximum > 0 else 0.0
            candidate = dict(result)
            candidate["score"] = normalized
            frame_id = candidate["frame_id"]
            if frame_id not in dedup or normalized > dedup[frame_id]["score"]:
                dedup[frame_id] = candidate
    merged = heapq.nlargest(max(0, top_k), dedup.values(), key=lambda result: result["score"])
    for rank, result in enumerate(merged, start=1):
        result["rank"] = rank
    return merged

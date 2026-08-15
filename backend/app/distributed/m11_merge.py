import heapq


def merge_zscore(shard_results, top_k):
    dedup = {}
    for results in shard_results:
        results = list(results)
        scores = [float(result["score"]) for result in results]
        if scores:
            mean = sum(scores) / len(scores)
            variance = sum((score - mean) ** 2 for score in scores) / len(scores)
            std = variance ** 0.5
        else:
            mean, std = 0.0, 0.0
        for result in results:
            normalized = (float(result["score"]) - mean) / std if std else 0.0
            candidate = dict(result, score=normalized)
            frame_id = candidate["frame_id"]
            if frame_id not in dedup or normalized > dedup[frame_id]["score"]:
                dedup[frame_id] = candidate
    output = heapq.nlargest(max(0, top_k), dedup.values(), key=lambda item: item["score"])
    for rank, result in enumerate(output, start=1):
        result["rank"] = rank
    return output


def merge_hybrid(image_results, text_results, top_k, image_weight=0.5, text_weight=0.5):
    combined = {}
    for result in image_results:
        combined.setdefault(result["frame_id"], {})["image"] = result["score"]
    for result in text_results:
        combined.setdefault(result["frame_id"], {})["text"] = result["score"]
    results = []
    for frame_id, scores in combined.items():
        score = image_weight * scores.get("image", 0.0) + text_weight * scores.get("text", 0.0)
        results.append({"frame_id": frame_id, "score": score})
    results = heapq.nlargest(max(0, top_k), results, key=lambda item: item["score"])
    for rank, result in enumerate(results, start=1):
        result["rank"] = rank
    return results

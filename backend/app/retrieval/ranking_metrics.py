import math
from typing import Mapping, Sequence, Union

Relevance = Union[Sequence[str], Mapping[str, float]]


def _relevance(relevance: Relevance) -> dict[str, float]:
    if isinstance(relevance, Mapping):
        return {str(key): float(value) for key, value in relevance.items() if value > 0}
    return {str(item): 1.0 for item in relevance}


def _unique_at_k(predictions: Sequence[str], k: int) -> list[str]:
    if k <= 0:
        return []
    seen = set()
    output = []
    for item in predictions:
        item = str(item)
        if item not in seen:
            seen.add(item)
            output.append(item)
        if len(output) == k:
            break
    return output


def recall_at_k(predictions: Sequence[str], relevance: Relevance, k: int) -> float:
    relevant = _relevance(relevance)
    if not relevant or k <= 0:
        return 0.0
    return sum(item in relevant for item in _unique_at_k(predictions, k)) / len(relevant)


def precision_at_k(predictions: Sequence[str], relevance: Relevance, k: int) -> float:
    if k <= 0:
        return 0.0
    relevant = _relevance(relevance)
    ranked = _unique_at_k(predictions, k)
    return sum(item in relevant for item in ranked) / k


def mrr_at_k(predictions: Sequence[str], relevance: Relevance, k: int) -> float:
    relevant = _relevance(relevance)
    for rank, item in enumerate(_unique_at_k(predictions, k), start=1):
        if item in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(predictions: Sequence[str], relevance: Relevance, k: int) -> float:
    relevant = _relevance(relevance)
    if not relevant or k <= 0:
        return 0.0
    ranked = _unique_at_k(predictions, k)
    dcg = sum((2 ** relevant.get(item, 0.0) - 1) / math.log2(rank + 1) for rank, item in enumerate(ranked, start=1))
    ideal = sorted(relevant.values(), reverse=True)[:k]
    idcg = sum((2 ** gain - 1) / math.log2(rank + 1) for rank, gain in enumerate(ideal, start=1))
    return dcg / idcg if idcg else 0.0


def ranking_metrics(predictions: Sequence[str], relevance: Relevance, k: int) -> dict[str, float]:
    return {
        "recall": recall_at_k(predictions, relevance, k),
        "precision": precision_at_k(predictions, relevance, k),
        "mrr": mrr_at_k(predictions, relevance, k),
        "ndcg": ndcg_at_k(predictions, relevance, k),
    }

import math
from abc import ABC, abstractmethod
from typing import Any, Iterable


def validate_scores(scores, expected_count: int) -> list[float]:
    values = list(scores)
    if len(values) != expected_count:
        raise ValueError("model scorer returned the wrong number of scores")
    validated = []
    for score in values:
        if isinstance(score, bool):
            raise ValueError("model scorer returned a non-numeric score")
        try:
            value = float(score)
        except (TypeError, ValueError) as exc:
            raise ValueError("model scorer returned a non-numeric score") from exc
        if not math.isfinite(value):
            raise ValueError("model scorer returned a non-finite score")
        validated.append(value)
    return validated


class ModelScorer(ABC):
    backend_name = "model_scorer"

    @abstractmethod
    def score_batch(self, query: Any, candidates: Iterable[dict]) -> list[float]:
        raise NotImplementedError

    def score(self, query: Any, candidate: dict) -> float:
        return self.score_batch(query, [candidate])[0]

    @property
    def diagnostics(self) -> dict:
        return {"model_backend": self.backend_name}


class DeterministicTestScorer(ModelScorer):
    backend_name = "deterministic_test_scorer"

    def __init__(self, score_key: str = "relevance"):
        self.score_key = score_key
        self.seen_candidate_ids: list[str] = []

    def score_batch(self, query: Any, candidates: Iterable[dict]) -> list[float]:
        candidates = list(candidates)
        self.seen_candidate_ids = [str(candidate.get("candidate_id", candidate.get("frame_id", ""))) for candidate in candidates]
        return [float(candidate.get(self.score_key, 0.0)) for candidate in candidates]

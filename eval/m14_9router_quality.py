"""M14 joint-relevance metric plumbing over a caller-supplied corpus.

The checked-in corpus is a synthetic contract fixture, not quality evidence.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from backend.app.indexes.faiss_siglip_index import FaissSigLIPIndex
from backend.app.retrieval.model_scorer import validate_scores
from backend.app.retrieval.ranking_metrics import ranking_metrics
from eval.m13_5_corpus import load_corpus
from eval.m13_5_ground_truth_benchmark import bootstrap_delta


def hard_negative_error(
    ranking: Sequence[dict],
    relevance: dict[str, int],
    hard_negative_ids: set[str],
) -> int:
    """Return one when a judged hard negative outranks every grade-three hit."""

    ids = [str(item["candidate_id"]) for item in ranking]
    grade_three_positions = [
        index for index, item in enumerate(ids) if relevance.get(item) == 3
    ]
    if not grade_three_positions:
        return 0
    best_positive = min(grade_three_positions)
    return int(any(item in hard_negative_ids for item in ids[:best_positive]))


def _normalize(rows: Any) -> np.ndarray:
    rows = np.asarray(rows, dtype=np.float32)
    if rows.ndim != 2 or not np.all(np.isfinite(rows)):
        raise ValueError("encoder output must be a finite two-dimensional array")
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("encoder output contains a zero-norm vector")
    return rows / norms


def _order(ids: Sequence[str], scores: Sequence[float]) -> list[str]:
    return sorted(ids, key=lambda item: (-float(scores[ids.index(item)]), item))


def run_quality(
    *,
    corpus_root: str | Path = "eval/data/m13_5",
    candidate_ks: Sequence[int] = (10, 24),
    encoder: Any,
    scorer: Any,
    bootstrap_repetitions: int = 2_000,
) -> dict[str, Any]:
    """Compare FAISS, exact, joint-model, and hybrid ranks on one pool."""

    if not candidate_ks or min(candidate_ks) <= 0:
        raise ValueError("candidate cutoffs must be positive")
    corpus = load_corpus(corpus_root)
    images = []
    for candidate in corpus.candidates:
        with Image.open(candidate.image_path) as image:
            images.append(image.convert("RGB").copy())
    image_vectors = _normalize(encoder.encode_image(images, batch_size=16))
    query_vectors = _normalize(
        encoder.encode_text([query.text for query in corpus.queries], batch_size=1)
    )
    if len(image_vectors) != len(corpus.candidates) or len(query_vectors) != len(
        corpus.queries
    ):
        raise ValueError("encoder returned the wrong number of vectors")
    if image_vectors.shape[1] != query_vectors.shape[1]:
        raise ValueError("image and text embedding dimensions do not match")

    candidate_ids = [candidate.candidate_id for candidate in corpus.candidates]
    similarities = query_vectors @ image_vectors.T
    index = FaissSigLIPIndex(image_vectors.shape[1])
    index.add(image_vectors.astype(np.float32, copy=False), candidate_ids)
    candidate_position = {
        candidate_id: position for position, candidate_id in enumerate(candidate_ids)
    }
    results = {}
    for requested_k in candidate_ks:
        candidate_k = min(int(requested_k), len(candidate_ids))
        rankings = {name: [] for name in ("faiss", "siglip_exact", "m14", "hybrid")}
        metric_rows = {name: {"mrr": [], "ndcg": []} for name in rankings}
        error_rows = {name: [] for name in rankings}
        outcomes = Counter({"improved": 0, "unchanged": 0, "regressed": 0})
        per_query = []
        signal_differences = []

        for query_index, query in enumerate(corpus.queries):
            hits = index.search(
                query_vectors[query_index].astype(np.float32, copy=False),
                candidate_k,
            )
            pool_ids = [str(hit["frame_id"]) for hit in hits]
            pool_indices = [
                candidate_position[candidate_id] for candidate_id in pool_ids
            ]
            exact_scores = [
                float(similarities[query_index, index]) for index in pool_indices
            ]
            candidates = [
                {
                    "candidate_id": candidate_id,
                    "image_path": str(corpus.candidates[index].image_path),
                    "metadata": corpus.candidates[index].metadata,
                    "siglip_score": exact_score,
                }
                for candidate_id, index, exact_score in zip(
                    pool_ids, pool_indices, exact_scores, strict=True
                )
            ]
            model_scores = validate_scores(
                scorer.score_batch(query.text, candidates), len(candidates)
            )
            model_min, model_max = min(model_scores), max(model_scores)
            model_span = model_max - model_min
            normalized_model = [
                (score - model_min) / model_span if model_span else 0.0
                for score in model_scores
            ]
            exact_min, exact_max = min(exact_scores), max(exact_scores)
            exact_span = exact_max - exact_min
            normalized_exact = [
                (score - exact_min) / exact_span if exact_span else 0.0
                for score in exact_scores
            ]
            hybrid_scores = [
                (exact_score + model_score) / 2.0
                for exact_score, model_score in zip(
                    normalized_exact, normalized_model, strict=True
                )
            ]

            query_rankings = {
                "faiss": list(pool_ids),
                "siglip_exact": _order(pool_ids, exact_scores),
                "m14": _order(pool_ids, model_scores),
                "hybrid": _order(pool_ids, hybrid_scores),
            }
            hard_negatives = {
                candidate.candidate_id
                for candidate in corpus.candidates
                if candidate.candidate_id in pool_ids
                and candidate.metadata.get("hard_negative") is True
                and query.relevance.get(candidate.candidate_id, 0) == 0
            }
            for name, ranked_ids in query_rankings.items():
                rankings[name].append(ranked_ids)
                metrics = ranking_metrics(ranked_ids, query.relevance, 10)
                metric_rows[name]["mrr"].append(metrics["mrr"])
                metric_rows[name]["ndcg"].append(metrics["ndcg"])
                error_rows[name].append(
                    hard_negative_error(
                        [{"candidate_id": candidate_id} for candidate_id in ranked_ids],
                        query.relevance,
                        hard_negatives,
                    )
                )

            faiss_ndcg = metric_rows["faiss"]["ndcg"][-1]
            m14_ndcg = metric_rows["m14"]["ndcg"][-1]
            outcome = (
                "improved"
                if m14_ndcg > faiss_ndcg
                else "regressed" if m14_ndcg < faiss_ndcg else "unchanged"
            )
            outcomes[outcome] += 1
            signal_differences.extend(
                abs(exact_score - model_score)
                for exact_score, model_score in zip(
                    normalized_exact, normalized_model, strict=True
                )
            )
            per_query.append(
                {
                    "query_id": query.query_id,
                    "candidate_ids": pool_ids,
                    "rankings": query_rankings,
                    "outcome": outcome,
                }
            )

        metrics = {
            name: {
                metric: float(np.mean(values))
                for metric, values in method_metrics.items()
            }
            for name, method_metrics in metric_rows.items()
        }
        deltas = {
            name: {
                metric: metrics[name][metric] - metrics["faiss"][metric]
                for metric in ("mrr", "ndcg")
            }
            for name in ("siglip_exact", "m14", "hybrid")
        }
        results[str(requested_k)] = {
            "metrics": metrics,
            "deltas_from_faiss@10": deltas,
            "signal_comparisons": {
                "faiss_vs_m14": {
                    "mean_absolute_difference": (
                        float(np.mean(signal_differences))
                        if signal_differences
                        else 0.0
                    )
                }
            },
            "query_outcomes": dict(outcomes),
            "bootstrap": {
                metric: bootstrap_delta(
                    metric_rows["faiss"][metric],
                    metric_rows["m14"][metric],
                    repetitions=bootstrap_repetitions,
                    seed=14_000 + candidate_k,
                )
                for metric in ("mrr", "ndcg")
            },
            "hard_negative_error_rate": {
                name: float(np.mean(values)) for name, values in error_rows.items()
            },
            "per_query": per_query,
            "scorer_diagnostics": dict(getattr(scorer, "diagnostics", {})),
        }

    return {
        "run_metadata": {
            "dataset_fingerprint": corpus.fingerprint,
            "dataset_kind": corpus.dataset_kind,
            "quality_claims_allowed": corpus.quality_claims_allowed,
            "encoder": getattr(encoder, "model_name", type(encoder).__name__),
            "scorer": getattr(scorer, "backend_name", type(scorer).__name__),
        },
        "results_by_candidate_k": results,
    }

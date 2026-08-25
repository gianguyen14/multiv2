"""Same-candidate metric plumbing for an M13.5-format corpus.

The checked-in corpus is explicitly synthetic and may only be used for contract
testing.  Callers must inspect ``quality_claims_allowed`` before reporting its
metrics as real retrieval quality.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from backend.app.indexes.faiss_siglip_index import FaissSigLIPIndex
from backend.app.retrieval.ranking_metrics import ranking_metrics, recall_at_k
from eval.m13_5_corpus import EvaluationCorpus, corpus_statistics, load_corpus


def _normalized(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or not np.all(np.isfinite(values)):
        raise ValueError("encoder output must be a finite two-dimensional array")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("encoder output contains a zero-norm vector")
    return values / norms


def _stable_order(scores: np.ndarray, candidate_ids: Sequence[str]) -> list[int]:
    return sorted(
        range(len(candidate_ids)),
        key=lambda index: (-float(scores[index]), candidate_ids[index]),
    )


def _mean_metrics(
    rankings: Sequence[Sequence[str]],
    relevances: Sequence[dict[str, int]],
    final_ks: Sequence[int],
) -> dict[str, dict[str, float]]:
    output = {}
    for final_k in final_ks:
        rows = [
            ranking_metrics(ranking, relevance, final_k)
            for ranking, relevance in zip(rankings, relevances, strict=True)
        ]
        output[str(final_k)] = {
            name: float(np.mean([row[name] for row in rows]))
            for name in ("recall", "precision", "mrr", "ndcg")
        }
    return output


def bootstrap_delta(
    baseline: Sequence[float],
    treatment: Sequence[float],
    *,
    repetitions: int = 2_000,
    seed: int = 0,
) -> dict[str, float | int]:
    """Return a deterministic paired bootstrap interval for treatment-baseline."""

    baseline_values = np.asarray(baseline, dtype=np.float64)
    treatment_values = np.asarray(treatment, dtype=np.float64)
    if baseline_values.shape != treatment_values.shape or baseline_values.ndim != 1:
        raise ValueError(
            "baseline and treatment must be equally sized one-dimensional samples"
        )
    if baseline_values.size == 0 or repetitions <= 0:
        raise ValueError(
            "bootstrap requires observations and a positive repetition count"
        )
    delta = treatment_values - baseline_values
    rng = np.random.default_rng(seed)
    samples = delta[rng.integers(0, len(delta), size=(repetitions, len(delta)))].mean(
        axis=1
    )
    return {
        "mean_delta": float(delta.mean()),
        "ci95_lower": float(np.quantile(samples, 0.025)),
        "ci95_upper": float(np.quantile(samples, 0.975)),
        "probability_improved": float(np.mean(samples > 0)),
        "repetitions": repetitions,
    }


def _encode_corpus(
    corpus: EvaluationCorpus, encoder: Any, batch_size: int
) -> tuple[np.ndarray, np.ndarray]:
    images = []
    for candidate in corpus.candidates:
        with Image.open(candidate.image_path) as image:
            images.append(image.convert("RGB").copy())
    image_embeddings = _normalized(encoder.encode_image(images, batch_size=batch_size))
    query_embeddings = _normalized(
        encoder.encode_text(
            [query.text for query in corpus.queries], batch_size=batch_size
        )
    )
    if len(image_embeddings) != len(corpus.candidates) or len(query_embeddings) != len(
        corpus.queries
    ):
        raise ValueError("encoder returned the wrong number of vectors")
    if image_embeddings.shape[1] != query_embeddings.shape[1]:
        raise ValueError("image and text embedding dimensions do not match")
    return image_embeddings, query_embeddings


def run_benchmark(
    corpus_root: str | Path,
    *,
    candidate_ks: Sequence[int] = (10, 24),
    final_ks: Sequence[int] = (5, 10),
    batch_size: int = 16,
    repetitions: int = 2_000,
    encoder: Any,
) -> dict[str, Any]:
    """Evaluate all signals over each FAISS candidate pool without pool drift."""

    if not candidate_ks or not final_ks or min(candidate_ks) <= 0 or min(final_ks) <= 0:
        raise ValueError("candidate and final cutoffs must be positive")
    corpus = load_corpus(corpus_root)
    image_embeddings, query_embeddings = _encode_corpus(corpus, encoder, batch_size)
    candidate_ids = [candidate.candidate_id for candidate in corpus.candidates]
    relevances = [query.relevance for query in corpus.queries]
    similarity = query_embeddings @ image_embeddings.T
    index = FaissSigLIPIndex(image_embeddings.shape[1])
    index.add(image_embeddings.astype(np.float32, copy=False), candidate_ids)
    candidate_position = {
        candidate_id: position for position, candidate_id in enumerate(candidate_ids)
    }
    results_by_k = {}

    for requested_k in candidate_ks:
        candidate_k = min(int(requested_k), len(candidate_ids))
        rankings = {name: [] for name in ("faiss", "dot_product", "siglip2", "hybrid")}
        per_query = []
        score_differences = []
        faiss_ndcg = []
        siglip_ndcg = []
        for query_index, query in enumerate(corpus.queries):
            hits = index.search(
                query_embeddings[query_index].astype(np.float32, copy=False),
                candidate_k,
            )
            faiss_ids = [str(hit["frame_id"]) for hit in hits]
            faiss_scores = {str(hit["frame_id"]): float(hit["score"]) for hit in hits}
            pool_ids = list(faiss_ids)
            pool = [candidate_position[candidate_id] for candidate_id in pool_ids]
            pool_scores = similarity[query_index, pool]

            # Exact scores are recomputed over the identical FAISS pool so
            # rerank comparisons cannot benefit from candidate-pool drift.
            exact_order = _stable_order(pool_scores, pool_ids)
            exact_ids = [pool_ids[index] for index in exact_order]
            siglip_ids = list(exact_ids)
            retrieval_position = {
                candidate_id: position
                for position, candidate_id in enumerate(faiss_ids)
            }
            exact_position = {
                candidate_id: position
                for position, candidate_id in enumerate(exact_ids)
            }
            hybrid_ids = sorted(
                pool_ids,
                key=lambda item: (
                    retrieval_position[item] + exact_position[item],
                    item,
                ),
            )
            for method, ranked in (
                ("faiss", faiss_ids),
                ("dot_product", exact_ids),
                ("siglip2", siglip_ids),
                ("hybrid", hybrid_ids),
            ):
                rankings[method].append(ranked)

            final_k = max(final_ks)
            base_score = ranking_metrics(faiss_ids, query.relevance, final_k)["ndcg"]
            new_score = ranking_metrics(siglip_ids, query.relevance, final_k)["ndcg"]
            faiss_ndcg.append(base_score)
            siglip_ndcg.append(new_score)
            outcome = (
                "improved"
                if new_score > base_score
                else "regressed" if new_score < base_score else "unchanged"
            )
            score_differences.extend(
                abs(faiss_scores[candidate_id] - float(pool_scores[index]))
                for index, candidate_id in enumerate(pool_ids)
            )
            per_query.append(
                {
                    "query_id": query.query_id,
                    "candidate_ids": pool_ids,
                    "rankings": {name: ranked[-1] for name, ranked in rankings.items()},
                    "outcome": outcome,
                }
            )

        metrics = {
            method: _mean_metrics(method_rankings, relevances, final_ks)
            for method, method_rankings in rankings.items()
        }
        candidate_recall = float(
            np.mean(
                [
                    recall_at_k(ranking, relevance, candidate_k)
                    for ranking, relevance in zip(
                        rankings["faiss"], relevances, strict=True
                    )
                ]
            )
        )
        results_by_k[str(requested_k)] = {
            "candidate_count": candidate_k,
            "metrics": metrics,
            "candidate_recall": candidate_recall,
            "comparisons": {
                "faiss_vs_siglip2": bootstrap_delta(
                    faiss_ndcg,
                    siglip_ndcg,
                    repetitions=repetitions,
                    seed=13_500 + candidate_k,
                )
            },
            "score_differences": {
                "faiss_vs_siglip": {
                    "mean_absolute_difference": (
                        float(np.mean(score_differences)) if score_differences else 0.0
                    ),
                    "maximum_absolute_difference": (
                        float(np.max(score_differences)) if score_differences else 0.0
                    ),
                }
            },
            "per_query": per_query,
        }

    return {
        "run_metadata": {
            "dataset_fingerprint": corpus.fingerprint,
            "dataset_kind": corpus.dataset_kind,
            "quality_claims_allowed": corpus.quality_claims_allowed,
            "encoder": getattr(encoder, "model_name", type(encoder).__name__),
            "candidate_ks": list(candidate_ks),
            "final_ks": list(final_ks),
            "corpus": corpus_statistics(corpus),
        },
        "results_by_candidate_k": results_by_k,
    }

"""Deterministic hard-negative fixture for the M12 retrieval quality gate."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class HardNegativeDataset:
    candidates: tuple[dict, ...]
    query_embedding: np.ndarray
    relevance: dict[str, int]


def build_hard_negative_dataset() -> HardNegativeDataset:
    """Return a small distributed fixture where retrieval score alone fails.

    The duplicate frame models the same logical candidate being returned by two
    shards.  Relevant frames deliberately have lower first-stage scores than
    hard negatives, so the fixture exercises both de-duplication and semantic
    reranking instead of merely checking a happy path.
    """

    rows = (
        ("negative_billboard", 0.99, 0, "shard-a"),
        ("negative_red_car", 0.97, 0, "shard-b"),
        ("negative_shop_sign", 0.95, 0, "shard-a"),
        ("positive_bicycle_close", 0.83, 3, "shard-b"),
        ("positive_bicycle_wide", 0.80, 2, "shard-c"),
        ("positive_rider", 0.78, 1, "shard-a"),
        ("neutral_street", 0.75, 0, "shard-c"),
        ("negative_red_car", 0.72, 0, "shard-c"),
    )
    candidates = tuple(
        {
            "frame_id": frame_id,
            "retrieval_score": retrieval_score,
            "score": retrieval_score,
            "relevance": relevance,
            "shard_id": shard_id,
        }
        for frame_id, retrieval_score, relevance, shard_id in rows
    )
    relevance = {
        candidate["frame_id"]: candidate["relevance"]
        for candidate in candidates
        if candidate["relevance"] > 0
    }
    return HardNegativeDataset(
        candidates=candidates,
        query_embedding=np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        relevance=relevance,
    )

"""Deterministic Near-Duplicate Frame Filtering Module.

Filters near-duplicate sampled video frames based on pairwise cosine similarity
of normalized SigLIP embeddings while strictly protecting shot representative keyframes.
"""

from __future__ import annotations

import math
from typing import Any, List, Optional, Set, Tuple
import numpy as np


def filter_near_duplicate_frames(
    records: List[Any],
    embeddings: np.ndarray,
    protected_source_frame_indices: Optional[Set[int]] = None,
    threshold: float = 0.97,
    enabled: bool = True,
) -> Tuple[List[Any], np.ndarray, List[int]]:
    """Filter consecutive near-duplicate frames chronologically.

    Args:
        records: List of FrameRecord objects (must have source_frame_index_zero_based).
        embeddings: Float32 numpy array of shape (N, D).
        protected_source_frame_indices: Set of source frame indices that must never be dropped.
        threshold: Cosine similarity threshold in [0.0, 1.0]. Consecutive frames with
                   similarity >= threshold are dropped unless protected.
        enabled: If False, returns input records and embeddings unchanged.

    Returns:
        Tuple of (retained_records, retained_embeddings, retained_indices).
    """
    if not enabled:
        return list(records), embeddings, list(range(len(records)))

    if (
        threshold is None
        or not isinstance(threshold, (int, float))
        or math.isnan(threshold)
        or math.isinf(threshold)
        or threshold < 0.0
        or threshold > 1.0
    ):
        raise ValueError("visual dedup threshold must be a finite number between 0.0 and 1.0")

    if not isinstance(embeddings, np.ndarray):
        embeddings = np.asarray(embeddings, dtype=np.float32)
    if embeddings.ndim != 2 or embeddings.dtype != np.float32:
        raise ValueError("embeddings must be a 2D float32 array")
    if len(records) != len(embeddings):
        raise ValueError(
            f"Mismatched records count ({len(records)}) and embeddings count ({len(embeddings)})"
        )
    if not np.isfinite(embeddings).all():
        raise ValueError("embeddings must contain only finite values")
    if len(records) == 0:
        return [], embeddings, []

    # Normalize vectors to ensure exact cosine similarity via dot product
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    if np.any(norms <= 0.0):
        raise ValueError("embeddings must not contain zero-norm vectors")
    norm_embs = embeddings / norms

    protected_set: Set[int] = set(protected_source_frame_indices or [])

    # Rule 11: First frame in a video is always retained
    retained_indices: List[int] = [0]
    last_retained_emb = norm_embs[0]

    for i in range(1, len(records)):
        cand_rec = records[i]
        cand_emb = norm_embs[i]
        cand_source_idx = getattr(cand_rec, "source_frame_index_zero_based", None)

        # Rule 7: Protected shot representatives are always kept
        if cand_source_idx is not None and cand_source_idx in protected_set:
            retained_indices.append(i)
            last_retained_emb = cand_emb  # Rule 13: Protected frame becomes the new reference
            continue

        # Rule 6 & 12: Compute cosine similarity against previous RETAINED frame
        cos_sim = float(np.dot(cand_emb, last_retained_emb))

        # Rule 6 & 16D: Drop if similarity >= threshold, keep if < threshold
        if cos_sim >= threshold:
            # Drop candidate
            continue
        else:
            retained_indices.append(i)
            last_retained_emb = cand_emb

    retained_records = [records[idx] for idx in retained_indices]
    retained_embeddings = embeddings[retained_indices]

    return retained_records, retained_embeddings, retained_indices

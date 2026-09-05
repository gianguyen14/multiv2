"""Deterministic retrieval benchmark metrics and query loading helpers."""

from __future__ import annotations

import json
import math
import statistics
import time
from collections import Counter, OrderedDict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class BenchmarkQueryInput:
    query_id: str
    task_type: str
    query_text: str
    events: tuple[str, ...] = ()
    ground_truth_video_ids: list[str] = field(default_factory=list)
    accepted_frame_intervals: list[list[int]] = field(default_factory=list)


@dataclass(frozen=True)
class ConcentrationMetrics:
    depth: int
    total_candidates: int
    unique_videos: int
    top_video_id: str | None
    top_video_count: int
    top_video_share: float
    top_3_video_share: float
    top_5_video_share: float
    hhi: float


def compute_hhi(shares: Iterable[float]) -> float:
    values = [float(share) for share in shares]
    if any(not math.isfinite(share) or share < 0.0 for share in values):
        raise ValueError("shares must be finite and non-negative")
    return float(sum(share * share for share in values))


def compute_concentration(
    candidates: Sequence[Mapping[str, Any]], depth: int = 100
) -> ConcentrationMetrics:
    if not isinstance(depth, int) or isinstance(depth, bool) or depth < 1:
        raise ValueError("depth must be an integer >= 1")
    considered = list(candidates[:depth])
    counts = Counter(
        str(candidate["video_id"])
        for candidate in considered
        if candidate.get("video_id") not in (None, "")
    )
    total = len(considered)
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    shares = [count / total for _, count in ordered] if total else []
    top_video_id, top_video_count = ordered[0] if ordered else (None, 0)

    return ConcentrationMetrics(
        depth=depth,
        total_candidates=total,
        unique_videos=len(counts),
        top_video_id=top_video_id,
        top_video_count=top_video_count,
        top_video_share=shares[0] if shares else 0.0,
        top_3_video_share=float(sum(shares[:3])),
        top_5_video_share=float(sum(shares[:5])),
        hhi=compute_hhi(shares),
    )


def compute_rank_metrics(ranks: Sequence[int | None]) -> dict[str, Any]:
    normalized = [
        (
            rank
            if isinstance(rank, int) and not isinstance(rank, bool) and rank > 0
            else None
        )
        for rank in ranks
    ]
    total = len(normalized)
    valid = [rank for rank in normalized if rank is not None]
    hits = {
        f"Hit@{cutoff}": (
            sum(rank is not None and rank <= cutoff for rank in normalized) / total
            if total
            else 0.0
        )
        for cutoff in (1, 5, 10, 20, 50, 100)
    }
    return {
        "total_queries": total,
        "evaluated_queries": len(valid),
        "hits": hits,
        "mrr": (sum(1.0 / rank for rank in valid) / total if total else 0.0),
        "median_rank": statistics.median(valid) if valid else None,
    }


def compute_latency_stats(latencies_ms: Sequence[float]) -> dict[str, float]:
    values = np.asarray(latencies_ms, dtype=np.float64)
    if values.size == 0:
        return {
            "count": 0,
            "min_ms": 0.0,
            "max_ms": 0.0,
            "mean_ms": 0.0,
            "p50_ms": 0.0,
            "p95_ms": 0.0,
            "p99_ms": 0.0,
        }
    if values.ndim != 1 or not np.isfinite(values).all() or np.any(values < 0.0):
        raise ValueError("latencies must be a finite non-negative sequence")
    return {
        "count": int(values.size),
        "min_ms": float(values.min()),
        "max_ms": float(values.max()),
        "mean_ms": float(values.mean()),
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
        "p99_ms": float(np.percentile(values, 99)),
    }


def _matches_ground_truth(
    candidate: Mapping[str, Any],
    video_ids: set[str],
    intervals: Sequence[Sequence[int]],
) -> bool:
    if str(candidate.get("video_id", "")) not in video_ids:
        return False
    if not intervals:
        return True
    frame_id = candidate.get("source_frame_index_zero_based", candidate.get("frame_id"))
    return isinstance(frame_id, int) and any(
        len(interval) == 2 and interval[0] <= frame_id <= interval[1]
        for interval in intervals
    )


def _stage_has_ground_truth(
    rows: Any, video_ids: set[str], intervals: Sequence[Sequence[int]]
) -> bool:
    if isinstance(rows, Mapping):
        iterable = []
        for key, value in rows.items():
            if isinstance(key, tuple) and len(key) >= 2:
                iterable.append(
                    {"video_id": key[0], "source_frame_index_zero_based": key[1]}
                )
            elif isinstance(value, Mapping):
                iterable.append(value)
    else:
        iterable = rows or []
    return any(
        _matches_ground_truth(row, video_ids, intervals)
        for row in iterable
        if isinstance(row, Mapping)
    )


def localize_failure(
    *,
    gt_video_ids: Sequence[str],
    gt_intervals: Sequence[Sequence[int]],
    plan: Any,
    channel_hits: Mapping[str, Any],
    fused_results: Sequence[Mapping[str, Any]],
    reranked_results: Sequence[Mapping[str, Any]],
    final_results: Sequence[Mapping[str, Any]],
    target_rank: int | None,
    visual_concentration: ConcentrationMetrics | None,
) -> tuple[str, dict[str, Any], str | None]:
    evidence = {"target_rank": target_rank, "query_plan_available": plan is not None}
    if target_rank is not None:
        return "SUCCESS", evidence, "final"
    if not gt_video_ids:
        return "UNKNOWN", evidence, None

    video_ids = {str(video_id) for video_id in gt_video_ids}
    if visual_concentration and (
        visual_concentration.top_video_share >= 0.8 or visual_concentration.hhi >= 0.7
    ):
        evidence["visual_concentration"] = visual_concentration
        return "CANDIDATE_CONCENTRATION", evidence, "visual_retrieval"

    visual_rows = {
        channel: rows
        for channel, rows in channel_hits.items()
        if "visual" in channel.lower()
    }
    if not any(
        _stage_has_ground_truth(rows, video_ids, gt_intervals)
        for rows in visual_rows.values()
    ):
        return "VISUAL_RECALL_FAILURE", evidence, "visual_retrieval"
    if not _stage_has_ground_truth(fused_results, video_ids, gt_intervals):
        return "FUSION_FAILURE", evidence, "fusion"
    if not _stage_has_ground_truth(reranked_results, video_ids, gt_intervals):
        return "RERANK_FAILURE", evidence, "rerank"
    if not _stage_has_ground_truth(final_results, video_ids, gt_intervals):
        return "FINAL_SELECTION_FAILURE", evidence, "final"
    return "UNKNOWN", evidence, None


def simulate_diversity_soft_cap(
    candidates: Sequence[Mapping[str, Any]], max_per_video: int, top_k: int
) -> list[dict[str, Any]]:
    if max_per_video < 1 or top_k < 1:
        return []
    counts: Counter[str] = Counter()
    selected = []
    for candidate in candidates:
        video_id = str(candidate.get("video_id", ""))
        if counts[video_id] >= max_per_video:
            continue
        selected.append(dict(candidate))
        counts[video_id] += 1
        if len(selected) == top_k:
            break
    return selected


def simulate_round_robin_diversification(
    candidates: Sequence[Mapping[str, Any]], top_k: int
) -> list[dict[str, Any]]:
    if top_k < 1:
        return []
    queues: OrderedDict[str, list[Mapping[str, Any]]] = OrderedDict()
    for candidate in candidates:
        queues.setdefault(str(candidate.get("video_id", "")), []).append(candidate)
    selected = []
    offset = 0
    while len(selected) < top_k:
        added = False
        for queue in queues.values():
            if offset < len(queue):
                selected.append(dict(queue[offset]))
                added = True
                if len(selected) == top_k:
                    break
        if not added:
            break
        offset += 1
    return selected


class ProductionBenchmarkRunner:
    def __init__(
        self,
        search_handler: (
            Callable[[BenchmarkQueryInput, int], Sequence[Mapping[str, Any]]] | None
        ) = None,
    ):
        self.search_handler = search_handler

    @staticmethod
    def load_queries(path: str | Path) -> list[BenchmarkQueryInput]:
        path = Path(path)
        if path.suffix.lower() == ".jsonl":
            rows = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        else:
            payload = json.loads(path.read_text(encoding="utf-8"))
            rows = payload.get("queries", []) if isinstance(payload, dict) else payload
        if not isinstance(rows, list):
            raise ValueError("benchmark query file must contain a list")

        queries = []
        seen = set()
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError("benchmark query rows must be objects")
            query_id = str(row.get("id", row.get("query_id", ""))).strip()
            if not query_id or query_id in seen:
                raise ValueError("query IDs must be non-empty and unique")
            seen.add(query_id)
            task_type = str(row.get("type", row.get("task", "kis"))).lower()
            events = tuple(str(event) for event in row.get("events", []) if str(event))
            query_text = str(
                row.get("query", row.get("text", " -> ".join(events)))
            ).strip()
            ground_truth = row.get("ground_truth") or {}
            video_ids = ground_truth.get("video_ids", row.get("video_ids", []))
            single_video = ground_truth.get("video_id", row.get("video_id"))
            if single_video:
                video_ids = [single_video, *video_ids]
            video_ids = list(dict.fromkeys(str(video_id) for video_id in video_ids))
            intervals = ground_truth.get(
                "frame_ranges", row.get("accepted_frame_intervals", [])
            )
            single_interval = row.get("accepted_frame_interval")
            if single_interval:
                intervals = [single_interval, *intervals]
            queries.append(
                BenchmarkQueryInput(
                    query_id=query_id,
                    task_type=task_type,
                    query_text=query_text,
                    events=events,
                    ground_truth_video_ids=video_ids,
                    accepted_frame_intervals=[list(interval) for interval in intervals],
                )
            )
        return queries

    def run(
        self, queries: Sequence[BenchmarkQueryInput], top_k: int = 100
    ) -> dict[str, Any]:
        if self.search_handler is None:
            raise RuntimeError("search handler is not configured")
        ranks: list[int | None] = []
        latencies = []
        concentrations = []
        for query in queries:
            started = time.perf_counter()
            rows = list(self.search_handler(query, top_k))
            latencies.append((time.perf_counter() - started) * 1000.0)
            rank = next(
                (
                    index
                    for index, row in enumerate(rows, start=1)
                    if _matches_ground_truth(
                        row,
                        set(query.ground_truth_video_ids),
                        query.accepted_frame_intervals,
                    )
                ),
                None,
            )
            ranks.append(rank)
            concentrations.append(compute_concentration(rows, depth=top_k))
        return {
            "rank_metrics": compute_rank_metrics(ranks),
            "latency": compute_latency_stats(latencies),
            "concentration": [metric.__dict__ for metric in concentrations],
        }

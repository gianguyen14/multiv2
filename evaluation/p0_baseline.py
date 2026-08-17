#!/usr/bin/env python3
"""P0 baseline evaluator for KIS, QA, and TRAKE.

This module is intentionally dependency-free. It evaluates precomputed prediction
JSONL files against annotated authoritative-frame intervals. It does not run the
retrieval system or download models.

QA exact-match scoring in this evaluator is an internal diagnostic metric only;
official external QA scoring semantics are intentionally not assumed.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

SCHEMA_VERSION = "p0-baseline-v1"
TASKS = {"kis", "qa", "trake"}


class EvaluationInputError(ValueError):
    pass


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, 1):
            text = raw.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError as exc:
                raise EvaluationInputError(f"{path}:{line_no}: invalid JSON") from exc
            if not isinstance(row, dict):
                raise EvaluationInputError(f"{path}:{line_no}: row must be an object")
            query_id = row.get("query_id")
            if not isinstance(query_id, str) or not query_id.strip():
                raise EvaluationInputError(f"{path}:{line_no}: query_id must be a non-empty string")
            if query_id in seen:
                raise EvaluationInputError(f"{path}:{line_no}: duplicate query_id {query_id!r}")
            seen.add(query_id)
            rows.append(row)
    if not rows:
        raise EvaluationInputError(f"{path}: no JSONL rows")
    return rows


def _validate_interval(interval: Any, context: str) -> tuple[int, int]:
    if not isinstance(interval, list) or len(interval) != 2:
        raise EvaluationInputError(f"{context}: interval must be [start, end]")
    start, end = interval
    if not isinstance(start, int) or not isinstance(end, int):
        raise EvaluationInputError(f"{context}: interval bounds must be integers")
    if start < 0 or end < 0 or start > end:
        raise EvaluationInputError(f"{context}: invalid authoritative-frame interval {interval!r}")
    return start, end


def _validate_ground_truth(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        query_id = row["query_id"]
        task = row.get("task_type")
        if task not in TASKS:
            raise EvaluationInputError(f"{query_id}: unsupported task_type {task!r}")
        gt = row.get("ground_truth")
        if not isinstance(gt, dict):
            raise EvaluationInputError(f"{query_id}: ground_truth must be an object")

        if task in {"kis", "qa"}:
            answerable = bool(gt.get("answerable", True)) if task == "qa" else True
            if answerable:
                video_id = gt.get("video_id")
                if not isinstance(video_id, str) or not video_id:
                    raise EvaluationInputError(f"{query_id}: ground_truth.video_id is required")
                intervals = gt.get("valid_frame_intervals")
                if not isinstance(intervals, list) or not intervals:
                    raise EvaluationInputError(f"{query_id}: valid_frame_intervals must be non-empty")
                for idx, interval in enumerate(intervals):
                    _validate_interval(interval, f"{query_id}:valid_frame_intervals[{idx}]")
                exact = gt.get("exact_frame_id")
                if exact is not None and (not isinstance(exact, int) or exact < 0):
                    raise EvaluationInputError(f"{query_id}: exact_frame_id must be a non-negative integer")
            if task == "qa":
                answers = gt.get("accepted_answers", [])
                if not isinstance(answers, list) or any(not isinstance(v, str) for v in answers):
                    raise EvaluationInputError(f"{query_id}: accepted_answers must be a list of strings")

        if task == "trake":
            video_id = gt.get("video_id")
            if not isinstance(video_id, str) or not video_id:
                raise EvaluationInputError(f"{query_id}: ground_truth.video_id is required")
            intervals = gt.get("event_intervals")
            if not isinstance(intervals, list) or not intervals:
                raise EvaluationInputError(f"{query_id}: event_intervals must be non-empty")
            for idx, interval in enumerate(intervals):
                _validate_interval(interval, f"{query_id}:event_intervals[{idx}]")
            exact_frames = gt.get("exact_event_frames")
            if exact_frames is not None:
                if (
                    not isinstance(exact_frames, list)
                    or len(exact_frames) != len(intervals)
                    or any(not isinstance(v, int) or v < 0 for v in exact_frames)
                ):
                    raise EvaluationInputError(
                        f"{query_id}: exact_event_frames must match event_intervals length"
                    )


def _normalize_answer(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).casefold().split())


def _in_any_interval(frame_id: Any, intervals: Iterable[list[int]]) -> bool:
    if not isinstance(frame_id, int):
        return False
    return any(start <= frame_id <= end for start, end in intervals)


def _first_rank(predicate, rows: list[dict[str, Any]]) -> int | None:
    for rank, row in enumerate(rows, 1):
        if predicate(row):
            return rank
    return None


def _safe_div(num: float, den: float) -> float | None:
    return None if den == 0 else num / den


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * fraction
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    weight = pos - lo
    return ordered[lo] * (1.0 - weight) + ordered[hi] * weight


def _evaluate_kis(gt_rows: list[dict[str, Any]], predictions: dict[str, dict[str, Any]]) -> dict[str, Any]:
    n = len(gt_rows)
    video_hits = {1: 0, 5: 0, 20: 0}
    frame_hits = {1: 0, 5: 0, 20: 0}
    reciprocal_ranks: list[float] = []
    frame_distances: list[float] = []

    for gt_row in gt_rows:
        query_id = gt_row["query_id"]
        gt = gt_row["ground_truth"]
        pred = predictions.get(query_id, {})
        results = pred.get("results", [])
        if not isinstance(results, list):
            results = []
        results = [r for r in results if isinstance(r, dict)]

        video_id = gt["video_id"]
        intervals = gt["valid_frame_intervals"]
        video_rank = _first_rank(lambda r: r.get("video_id") == video_id, results)
        frame_rank = _first_rank(
            lambda r: r.get("video_id") == video_id
            and _in_any_interval(r.get("frame_id"), intervals),
            results,
        )

        for k in (1, 5, 20):
            if video_rank is not None and video_rank <= k:
                video_hits[k] += 1
            if frame_rank is not None and frame_rank <= k:
                frame_hits[k] += 1

        reciprocal_ranks.append(0.0 if frame_rank is None else 1.0 / frame_rank)
        exact = gt.get("exact_frame_id")
        if exact is not None and frame_rank is not None:
            frame_id = results[frame_rank - 1].get("frame_id")
            if isinstance(frame_id, int):
                frame_distances.append(abs(frame_id - exact))

    return {
        "queries": n,
        "VR@1": video_hits[1] / n,
        "VR@5": video_hits[5] / n,
        "VR@20": video_hits[20] / n,
        "FIR@1": frame_hits[1] / n,
        "FIR@5": frame_hits[5] / n,
        "FIR@20": frame_hits[20] / n,
        "MRR": statistics.fmean(reciprocal_ranks) if reciprocal_ranks else None,
        "MFD_first_interval_hit": statistics.fmean(frame_distances) if frame_distances else None,
        "MFD_hit_queries": len(frame_distances),
    }


def _evaluate_qa(gt_rows: list[dict[str, Any]], predictions: dict[str, dict[str, Any]]) -> dict[str, Any]:
    total = len(gt_rows)
    answerable = 0
    unanswerable = 0
    video_hits = 0
    frame_hits = 0
    exact_answer_hits = 0
    grounded_exact_hits = 0
    negative_abstentions = 0

    for gt_row in gt_rows:
        query_id = gt_row["query_id"]
        gt = gt_row["ground_truth"]
        pred = predictions.get(query_id, {})
        result = pred.get("result", {})
        if not isinstance(result, dict):
            result = {}
        is_answerable = bool(gt.get("answerable", True))
        answer = _normalize_answer(result.get("answer"))

        if not is_answerable:
            unanswerable += 1
            if answer == "":
                negative_abstentions += 1
            continue

        answerable += 1
        video_match = result.get("video_id") == gt["video_id"]
        frame_match = video_match and _in_any_interval(
            result.get("frame_id"), gt["valid_frame_intervals"]
        )
        video_hits += int(video_match)
        frame_hits += int(frame_match)

        accepted = {_normalize_answer(v) for v in gt.get("accepted_answers", [])}
        answer_match = bool(answer) and answer in accepted
        exact_answer_hits += int(answer_match)
        grounded_exact_hits += int(answer_match and frame_match)

    return {
        "queries": total,
        "answerable_queries": answerable,
        "unanswerable_queries": unanswerable,
        "evidence_video_recall@1": _safe_div(video_hits, answerable),
        "evidence_frame_recall@1": _safe_div(frame_hits, answerable),
        "internal_exact_match_accuracy_answerable": _safe_div(exact_answer_hits, answerable),
        "evidence_grounded_internal_exact_rate": _safe_div(grounded_exact_hits, answerable),
        "negative_abstention_rate": _safe_div(negative_abstentions, unanswerable),
        "official_qa_scoring_semantics": "UNRESOLVED",
    }


def _evaluate_trake(gt_rows: list[dict[str, Any]], predictions: dict[str, dict[str, Any]]) -> dict[str, Any]:
    queries = len(gt_rows)
    monotonic_count = 0
    complete_count = 0
    event_hits = 0
    total_events = 0
    frame_errors: list[float] = []

    for gt_row in gt_rows:
        query_id = gt_row["query_id"]
        gt = gt_row["ground_truth"]
        pred = predictions.get(query_id, {})
        result = pred.get("result", {})
        if not isinstance(result, dict):
            result = {}

        frames = result.get("frames", [])
        if not isinstance(frames, list):
            frames = []
        intervals = gt["event_intervals"]
        total_events += len(intervals)
        valid_count = len(frames) == len(intervals) and all(isinstance(v, int) for v in frames)
        monotonic = valid_count and all(frames[i] < frames[i + 1] for i in range(len(frames) - 1))
        monotonic_count += int(monotonic)
        video_match = result.get("video_id") == gt["video_id"]

        local_hits = 0
        if valid_count and video_match:
            for frame_id, interval in zip(frames, intervals):
                if interval[0] <= frame_id <= interval[1]:
                    local_hits += 1
            exact_frames = gt.get("exact_event_frames")
            if exact_frames is not None:
                frame_errors.extend(abs(frame_id - exact) for frame_id, exact in zip(frames, exact_frames))
        event_hits += local_hits
        complete_count += int(video_match and monotonic and local_hits == len(intervals))

    return {
        "queries": queries,
        "valid_monotonic_sequence_rate": monotonic_count / queries,
        "event_hit_recall": _safe_div(event_hits, total_events),
        "complete_sequence_accuracy": complete_count / queries,
        "mean_event_frame_error": statistics.fmean(frame_errors) if frame_errors else None,
        "frame_error_events": len(frame_errors),
    }


def evaluate(ground_truth_rows: list[dict[str, Any]], prediction_rows: list[dict[str, Any]], scope: str) -> dict[str, Any]:
    _validate_ground_truth(ground_truth_rows)
    predictions = {row["query_id"]: row for row in prediction_rows}
    category_counts = Counter(row.get("category_id", "UNSPECIFIED") for row in ground_truth_rows)
    by_task = {task: [row for row in ground_truth_rows if row["task_type"] == task] for task in TASKS}

    latency_values = [
        float(row["latency_ms"])
        for row in prediction_rows
        if isinstance(row.get("latency_ms"), (int, float)) and row["latency_ms"] >= 0
    ]

    return {
        "schema_version": SCHEMA_VERSION,
        "measurement_scope": scope,
        "ground_truth_queries": len(ground_truth_rows),
        "prediction_rows": len(prediction_rows),
        "category_counts": dict(sorted(category_counts.items())),
        "metrics": {
            "kis": _evaluate_kis(by_task["kis"], predictions) if by_task["kis"] else None,
            "qa": _evaluate_qa(by_task["qa"], predictions) if by_task["qa"] else None,
            "trake": _evaluate_trake(by_task["trake"], predictions) if by_task["trake"] else None,
            "runtime": {
                "samples": len(latency_values),
                "p50_latency_ms": _percentile(latency_values, 0.50),
                "p95_latency_ms": _percentile(latency_values, 0.95),
            },
        },
        "notes": {
            "frame_metric": "annotated authoritative-frame intervals; no universal +/-N tolerance",
            "qa_exact_match": "internal diagnostic only; official scoring semantics unresolved",
            "performance_thresholds": "TO_BE_ESTABLISHED_FROM_BASELINE",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scope", default="UNSPECIFIED", help="e.g. SYNTHETIC_CI_SMOKE or REAL_BASELINE")
    args = parser.parse_args()

    ground_truth = _load_jsonl(args.ground_truth)
    predictions = _load_jsonl(args.predictions)
    report = evaluate(ground_truth, predictions, args.scope)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

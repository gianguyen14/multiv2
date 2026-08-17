from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

CATEGORY_IDS = tuple(f"C{i:02d}" for i in range(1, 19))
TASK_TYPES = {"kis", "qa", "trake"}
DEFAULT_CUTOFFS = (1, 5, 20)
SCHEMA_VERSION = "p0-baseline-v1"
FAILURE_TAXONOMY = {
    "QUERY_PARSE_MISS",
    "TRANSLATION_MISS",
    "VISUAL_MODEL_MISS",
    "SAMPLING_MISS",
    "OCR_MISS",
    "ASR_MISS",
    "EXACT_TEXT_MISS",
    "FUSION_MISS",
    "RERANK_MISS",
    "DEDUP_MISS",
    "TEMPORAL_REFINEMENT_MISS",
    "TRAKE_ALIGNMENT_MISS",
    "FRAME_MAPPING_MISS",
    "RESULT_SERIALIZATION_MISS",
    "OPERATOR_UI_MISS",
    "DATASET_ANNOTATION_ISSUE",
    "AMBIGUOUS_QUERY",
}


class EvaluationDataError(ValueError):
    pass


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            text = raw.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError as exc:
                raise EvaluationDataError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(row, dict):
                raise EvaluationDataError(f"{path}:{line_number}: row must be an object")
            query_id = row.get("query_id")
            if not isinstance(query_id, str) or not query_id.strip():
                raise EvaluationDataError(f"{path}:{line_number}: query_id must be a non-empty string")
            if query_id in seen:
                raise EvaluationDataError(f"{path}:{line_number}: duplicate query_id {query_id!r}")
            seen.add(query_id)
            rows.append(row)
    if not rows:
        raise EvaluationDataError(f"{path}: no JSONL records")
    return rows


def load_ground_truth(path: str | Path) -> list[dict[str, Any]]:
    rows = _read_jsonl(path)
    validate_ground_truth(rows)
    return rows


def load_predictions(path: str | Path) -> dict[str, dict[str, Any]]:
    rows = _read_jsonl(path)
    predictions: dict[str, dict[str, Any]] = {}
    for row in rows:
        results = row.get("results", [])
        if not isinstance(results, list):
            raise EvaluationDataError(f"prediction {row['query_id']}: results must be a list")
        predictions[row["query_id"]] = row
    return predictions


def _valid_interval(value: Any) -> bool:
    return (
        isinstance(value, (list, tuple))
        and len(value) == 2
        and all(isinstance(v, int) and not isinstance(v, bool) for v in value)
        and 0 <= value[0] <= value[1]
    )


def _validate_intervals(value: Any, context: str) -> list[list[int]]:
    if not isinstance(value, list) or not value:
        raise EvaluationDataError(f"{context}: valid frame intervals must be a non-empty list")
    normalized: list[list[int]] = []
    for index, interval in enumerate(value):
        if not _valid_interval(interval):
            raise EvaluationDataError(f"{context}[{index}]: interval must be [start, end] with 0 <= start <= end")
        normalized.append([int(interval[0]), int(interval[1])])
    return normalized


def _validate_localization(record: Mapping[str, Any], query_id: str, *, allow_negative: bool) -> None:
    ground_truth = record.get("ground_truth")
    if not isinstance(ground_truth, Mapping):
        raise EvaluationDataError(f"{query_id}: ground_truth must be an object")
    expect_no_result = ground_truth.get("expect_no_result", False)
    if not isinstance(expect_no_result, bool):
        raise EvaluationDataError(f"{query_id}: ground_truth.expect_no_result must be boolean")
    if expect_no_result:
        if not allow_negative:
            raise EvaluationDataError(f"{query_id}: expect_no_result is not valid for this task")
        return

    video_id = ground_truth.get("video_id")
    if not isinstance(video_id, str) or not video_id.strip():
        raise EvaluationDataError(f"{query_id}: ground_truth.video_id is required")
    intervals = _validate_intervals(ground_truth.get("valid_frame_intervals"), f"{query_id}:valid_frame_intervals")
    exact_frame = ground_truth.get("exact_frame_id")
    if exact_frame is not None:
        if not isinstance(exact_frame, int) or isinstance(exact_frame, bool) or exact_frame < 0:
            raise EvaluationDataError(f"{query_id}: exact_frame_id must be a non-negative integer")
        if not any(start <= exact_frame <= end for start, end in intervals):
            raise EvaluationDataError(f"{query_id}: exact_frame_id must lie inside a valid frame interval")


def validate_ground_truth(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    seen: set[str] = set()
    by_task: Counter[str] = Counter()
    by_category: Counter[str] = Counter()

    for position, record in enumerate(records, 1):
        query_id = record.get("query_id")
        if not isinstance(query_id, str) or not query_id.strip():
            raise EvaluationDataError(f"record {position}: query_id is required")
        if query_id in seen:
            raise EvaluationDataError(f"record {position}: duplicate query_id {query_id}")
        seen.add(query_id)

        category_id = record.get("category_id")
        if category_id not in CATEGORY_IDS:
            raise EvaluationDataError(f"{query_id}: category_id must be C01..C18")
        by_category[str(category_id)] += 1

        task_type = record.get("task_type")
        if task_type not in TASK_TYPES:
            raise EvaluationDataError(f"{query_id}: task_type must be one of {sorted(TASK_TYPES)}")
        by_task[str(task_type)] += 1

        failure_category = record.get("failure_category")
        if failure_category is not None and failure_category not in FAILURE_TAXONOMY:
            raise EvaluationDataError(f"{query_id}: unknown failure_category {failure_category}")

        if task_type == "kis":
            if not isinstance(record.get("query"), str) or not record["query"].strip():
                raise EvaluationDataError(f"{query_id}: query is required for KIS")
            _validate_localization(record, query_id, allow_negative=True)

        elif task_type == "qa":
            if not isinstance(record.get("query"), str) or not record["query"].strip():
                raise EvaluationDataError(f"{query_id}: query is required for QA")
            qa_ground_truth = record.get("qa_ground_truth")
            if not isinstance(qa_ground_truth, Mapping):
                raise EvaluationDataError(f"{query_id}: qa_ground_truth must be an object")
            answerable = qa_ground_truth.get("answerable", True)
            if not isinstance(answerable, bool):
                raise EvaluationDataError(f"{query_id}: qa_ground_truth.answerable must be boolean")
            answers = qa_ground_truth.get("accepted_answers", [])
            if not isinstance(answers, list) or any(not isinstance(v, str) for v in answers):
                raise EvaluationDataError(f"{query_id}: accepted_answers must be a list of strings")
            if answerable and not any(v.strip() for v in answers):
                raise EvaluationDataError(f"{query_id}: answerable QA requires at least one accepted answer")
            ground_truth = record.get("ground_truth")
            if answerable:
                _validate_localization(record, query_id, allow_negative=False)
            elif ground_truth is not None:
                _validate_localization(record, query_id, allow_negative=True)

        else:
            events = record.get("events")
            if not isinstance(events, list) or not events or not all(isinstance(v, str) and v.strip() for v in events):
                raise EvaluationDataError(f"{query_id}: non-empty events are required for TRAKE")
            truth = record.get("trake_ground_truth")
            if not isinstance(truth, Mapping):
                raise EvaluationDataError(f"{query_id}: trake_ground_truth must be an object")
            video_id = truth.get("video_id")
            if not isinstance(video_id, str) or not video_id.strip():
                raise EvaluationDataError(f"{query_id}: trake_ground_truth.video_id is required")
            intervals = truth.get("event_intervals")
            if not isinstance(intervals, list) or len(intervals) != len(events) or not intervals:
                raise EvaluationDataError(f"{query_id}: event_intervals must align one-to-one with events")
            normalized = _validate_intervals(intervals, f"{query_id}:event_intervals")
            if any(a[0] >= b[0] for a, b in zip(normalized, normalized[1:])):
                raise EvaluationDataError(f"{query_id}: TRAKE event intervals must be chronologically ordered")
            event_frames = truth.get("event_frames")
            if event_frames is not None:
                if (
                    not isinstance(event_frames, list)
                    or len(event_frames) != len(normalized)
                    or any(not isinstance(v, int) or isinstance(v, bool) or v < 0 for v in event_frames)
                ):
                    raise EvaluationDataError(f"{query_id}: event_frames must align one-to-one with events")
                for frame, (start, end) in zip(event_frames, normalized):
                    if not start <= frame <= end:
                        raise EvaluationDataError(f"{query_id}: every event_frame must lie inside its event interval")

    return {
        "records": len(records),
        "by_task": dict(sorted(by_task.items())),
        "by_category": dict(sorted(by_category.items())),
        "valid": True,
    }


def _normalize_answer(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).casefold().split())


def _in_intervals(frame_id: Any, intervals: Iterable[Sequence[int]]) -> bool:
    if not isinstance(frame_id, int) or isinstance(frame_id, bool):
        return False
    return any(start <= frame_id <= end for start, end in intervals)


def _first_rank(results: Sequence[Mapping[str, Any]], predicate) -> int | None:
    for rank, result in enumerate(results, 1):
        if predicate(result):
            return rank
    return None


def _mean(values: Iterable[float]) -> float | None:
    values = list(values)
    return statistics.fmean(values) if values else None


def _safe_rate(num: int, den: int) -> float | None:
    return None if den == 0 else num / den


def evaluate_kis(records: Sequence[Mapping[str, Any]], predictions: Mapping[str, Mapping[str, Any]], cutoffs=DEFAULT_CUTOFFS) -> dict[str, Any]:
    positive = [r for r in records if r["task_type"] == "kis" and not r["ground_truth"].get("expect_no_result", False)]
    negative = [r for r in records if r["task_type"] == "kis" and r["ground_truth"].get("expect_no_result", False)]
    per_query: list[dict[str, Any]] = []
    reciprocal_ranks: list[float] = []
    frame_distances: list[float] = []

    for record in positive:
        truth = record["ground_truth"]
        results = predictions.get(record["query_id"], {}).get("results", [])
        results = results if isinstance(results, list) else []
        video_id = truth["video_id"]
        intervals = truth["valid_frame_intervals"]
        video_rank = _first_rank(results, lambda r: isinstance(r, Mapping) and r.get("video_id") == video_id)
        frame_rank = _first_rank(
            results,
            lambda r: isinstance(r, Mapping)
            and r.get("video_id") == video_id
            and _in_intervals(r.get("frame_id"), intervals),
        )
        reciprocal_ranks.append(0.0 if frame_rank is None else 1.0 / frame_rank)
        exact_frame = truth.get("exact_frame_id")
        if exact_frame is not None and frame_rank is not None:
            frame_id = results[frame_rank - 1].get("frame_id")
            if isinstance(frame_id, int) and not isinstance(frame_id, bool):
                frame_distances.append(abs(frame_id - exact_frame))
        item: dict[str, Any] = {
            "query_id": record["query_id"],
            "category_id": record["category_id"],
            "expected": "TARGET",
            "first_video_hit_rank": video_rank,
            "first_interval_hit_rank": frame_rank,
        }
        for cutoff in cutoffs:
            item[f"vr@{cutoff}"] = float(video_rank is not None and video_rank <= cutoff)
            item[f"fir@{cutoff}"] = float(frame_rank is not None and frame_rank <= cutoff)
        per_query.append(item)

    negative_abstentions = 0
    for record in negative:
        results = predictions.get(record["query_id"], {}).get("results", [])
        results = results if isinstance(results, list) else []
        abstained = len(results) == 0
        negative_abstentions += int(abstained)
        per_query.append({
            "query_id": record["query_id"],
            "category_id": record["category_id"],
            "expected": "NO_RESULT",
            "abstained": abstained,
        })

    aggregate: dict[str, Any] = {
        "queries": len(positive) + len(negative),
        "positive_queries": len(positive),
        "negative_queries": len(negative),
        "mrr_positive": _mean(reciprocal_ranks),
        "mfd_first_interval_hit": _mean(frame_distances),
        "mfd_hit_queries": len(frame_distances),
        "negative_abstention_rate": _safe_rate(negative_abstentions, len(negative)),
        "negative_false_positive_rate": _safe_rate(len(negative) - negative_abstentions, len(negative)),
    }
    for cutoff in cutoffs:
        aggregate[f"vr@{cutoff}"] = _mean(v[f"vr@{cutoff}"] for v in per_query if v["expected"] == "TARGET")
        aggregate[f"fir@{cutoff}"] = _mean(v[f"fir@{cutoff}"] for v in per_query if v["expected"] == "TARGET")
    return {"aggregate": aggregate, "per_query": per_query}


def evaluate_qa(records: Sequence[Mapping[str, Any]], predictions: Mapping[str, Mapping[str, Any]], cutoffs=DEFAULT_CUTOFFS) -> dict[str, Any]:
    qa_records = [r for r in records if r["task_type"] == "qa"]
    per_query: list[dict[str, Any]] = []
    answerable_count = 0
    unanswerable_count = 0
    exact_hits = 0
    grounded_exact_hits = 0
    negative_abstentions = 0
    localizable_count = 0
    evidence_video_hits = {k: 0 for k in cutoffs}
    evidence_frame_hits = {k: 0 for k in cutoffs}

    for record in qa_records:
        qa_truth = record["qa_ground_truth"]
        answerable = bool(qa_truth.get("answerable", True))
        results = predictions.get(record["query_id"], {}).get("results", [])
        results = results if isinstance(results, list) else []
        top = results[0] if results and isinstance(results[0], Mapping) else {}
        predicted_answer = _normalize_answer(top.get("answer"))
        item: dict[str, Any] = {
            "query_id": record["query_id"],
            "category_id": record["category_id"],
            "answerable": answerable,
        }

        truth = record.get("ground_truth")
        has_localization = isinstance(truth, Mapping) and not truth.get("expect_no_result", False)
        if has_localization:
            localizable_count += 1
            video_id = truth["video_id"]
            intervals = truth["valid_frame_intervals"]
            for cutoff in cutoffs:
                subset = [r for r in results[:cutoff] if isinstance(r, Mapping)]
                video_hit = any(r.get("video_id") == video_id for r in subset)
                frame_hit = any(r.get("video_id") == video_id and _in_intervals(r.get("frame_id"), intervals) for r in subset)
                evidence_video_hits[cutoff] += int(video_hit)
                evidence_frame_hits[cutoff] += int(frame_hit)
                item[f"evr@{cutoff}"] = float(video_hit)
                item[f"efr@{cutoff}"] = float(frame_hit)
        else:
            for cutoff in cutoffs:
                item[f"evr@{cutoff}"] = None
                item[f"efr@{cutoff}"] = None

        if answerable:
            answerable_count += 1
            accepted = {_normalize_answer(v) for v in qa_truth.get("accepted_answers", []) if _normalize_answer(v)}
            answer_match = bool(predicted_answer) and predicted_answer in accepted
            exact_hits += int(answer_match)
            top_frame_hit = False
            if has_localization and top:
                top_frame_hit = top.get("video_id") == truth["video_id"] and _in_intervals(top.get("frame_id"), truth["valid_frame_intervals"])
            grounded_exact_hits += int(answer_match and top_frame_hit)
            item["internal_exact_match"] = float(answer_match)
            item["abstained"] = not bool(predicted_answer)
            item["evidence_grounded_internal_exact"] = float(answer_match and top_frame_hit)
        else:
            unanswerable_count += 1
            abstained = not bool(predicted_answer)
            negative_abstentions += int(abstained)
            item["internal_exact_match"] = None
            item["abstained"] = abstained
            item["evidence_grounded_internal_exact"] = None
        per_query.append(item)

    aggregate: dict[str, Any] = {
        "queries": len(qa_records),
        "answerable_queries": answerable_count,
        "unanswerable_queries": unanswerable_count,
        "localizable_queries": localizable_count,
        "internal_exact_match_accuracy_answerable": _safe_rate(exact_hits, answerable_count),
        "evidence_grounded_internal_exact_rate": _safe_rate(grounded_exact_hits, answerable_count),
        "negative_abstention_rate": _safe_rate(negative_abstentions, unanswerable_count),
        "official_qa_scoring_semantics": "UNRESOLVED",
    }
    for cutoff in cutoffs:
        aggregate[f"evr@{cutoff}"] = _safe_rate(evidence_video_hits[cutoff], localizable_count)
        aggregate[f"efr@{cutoff}"] = _safe_rate(evidence_frame_hits[cutoff], localizable_count)
    return {"aggregate": aggregate, "per_query": per_query}


def _trake_metrics(result: Mapping[str, Any], truth: Mapping[str, Any]) -> dict[str, Any]:
    frames = result.get("frame_ids", [])
    intervals = truth["event_intervals"]
    video_match = result.get("video_id") == truth["video_id"]
    valid_shape = isinstance(frames, list) and len(frames) == len(intervals) and all(
        isinstance(v, int) and not isinstance(v, bool) for v in frames
    )
    if not valid_shape:
        return {
            "video_match": float(video_match),
            "event_hit_recall": 0.0,
            "complete_sequence_accuracy": 0.0,
            "monotonic": 0.0,
            "mean_event_frame_error": None,
        }
    monotonic = all(a < b for a, b in zip(frames, frames[1:]))
    hits = [start <= frame <= end for frame, (start, end) in zip(frames, intervals)] if video_match else [False] * len(intervals)
    exact = truth.get("event_frames")
    frame_error = None
    if isinstance(exact, list) and len(exact) == len(frames) and video_match:
        frame_error = statistics.fmean(abs(frame - target) for frame, target in zip(frames, exact))
    return {
        "video_match": float(video_match),
        "event_hit_recall": sum(hits) / len(intervals),
        "complete_sequence_accuracy": float(video_match and monotonic and all(hits)),
        "monotonic": float(monotonic),
        "mean_event_frame_error": frame_error,
    }


def evaluate_trake(records: Sequence[Mapping[str, Any]], predictions: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    trake_records = [r for r in records if r["task_type"] == "trake"]
    per_query: list[dict[str, Any]] = []
    for record in trake_records:
        results = predictions.get(record["query_id"], {}).get("results", [])
        top = results[0] if isinstance(results, list) and results and isinstance(results[0], Mapping) else {}
        metrics = _trake_metrics(top, record["trake_ground_truth"])
        per_query.append({"query_id": record["query_id"], "category_id": record["category_id"], **metrics})
    frame_errors = [v["mean_event_frame_error"] for v in per_query if v["mean_event_frame_error"] is not None]
    aggregate = {
        "queries": len(trake_records),
        "video_match_rate": _mean(v["video_match"] for v in per_query),
        "event_hit_recall": _mean(v["event_hit_recall"] for v in per_query),
        "complete_sequence_accuracy": _mean(v["complete_sequence_accuracy"] for v in per_query),
        "valid_monotonic_sequence_rate": _mean(v["monotonic"] for v in per_query),
        "mean_event_frame_error": _mean(frame_errors),
    }
    return {"aggregate": aggregate, "per_query": per_query}


def category_coverage(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    counts = {category: 0 for category in CATEGORY_IDS}
    for record in records:
        category_id = record.get("category_id")
        if category_id in counts:
            counts[category_id] += 1
    covered = sum(value > 0 for value in counts.values())
    return {"counts": counts, "covered": covered, "target": len(CATEGORY_IDS), "complete": covered == len(CATEGORY_IDS)}


def _percentile(values: Sequence[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def latency_summary(predictions: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    values = [
        float(row["latency_ms"])
        for row in predictions.values()
        if isinstance(row.get("latency_ms"), (int, float)) and not isinstance(row.get("latency_ms"), bool) and row["latency_ms"] >= 0
    ]
    return {
        "samples": len(values),
        "p50_ms": _percentile(values, 0.50),
        "p95_ms": _percentile(values, 0.95),
    }


def evaluate(records: Sequence[Mapping[str, Any]], predictions: Mapping[str, Mapping[str, Any]], *, scope: str = "UNSPECIFIED", cutoffs=DEFAULT_CUTOFFS) -> dict[str, Any]:
    validation = validate_ground_truth(records)
    return {
        "schema_version": SCHEMA_VERSION,
        "measurement_scope": scope,
        "measurement_status": "MEASURED_FROM_SUPPLIED_PREDICTIONS",
        "ground_truth_validation": validation,
        "category_coverage": category_coverage(records),
        "missing_prediction_query_ids": sorted(record["query_id"] for record in records if record["query_id"] not in predictions),
        "kis": evaluate_kis(records, predictions, cutoffs),
        "qa": evaluate_qa(records, predictions, cutoffs),
        "trake": evaluate_trake(records, predictions),
        "latency": latency_summary(predictions),
        "notes": {
            "frame_metric": "annotated authoritative-frame intervals; no universal +/-N-frame tolerance",
            "qa_exact_match": "internal diagnostic only; official QA scoring semantics unresolved",
            "performance_thresholds": "TO_BE_ESTABLISHED_FROM_BASELINE",
        },
    }


def current_git_sha() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def render_markdown(report: Mapping[str, Any], provenance: Mapping[str, Any] | None = None) -> str:
    provenance = dict(provenance or {})
    coverage = report["category_coverage"]
    kis = report["kis"]["aggregate"]
    qa = report["qa"]["aggregate"]
    trake = report["trake"]["aggregate"]
    latency = report["latency"]

    def fmt(value: Any) -> str:
        if value is None:
            return "N/A"
        if isinstance(value, float):
            return f"{value:.4f}"
        return str(value)

    lines = [
        "# P0 Baseline Evaluation Report",
        "",
        "This report measures only the supplied ground-truth and prediction artifacts.",
        "",
        "## Provenance",
        "",
        f"- Measurement scope: `{report['measurement_scope']}`",
        f"- Git SHA: `{provenance.get('git_sha') or 'UNRESOLVED'}`",
        f"- Generation ID: `{provenance.get('generation_id') or 'UNRESOLVED'}`",
        f"- Runtime label: `{provenance.get('runtime_label') or 'UNRESOLVED'}`",
        "",
        "## Ground Truth Coverage",
        "",
        f"- Records: **{report['ground_truth_validation']['records']}**",
        f"- Categories covered: **{coverage['covered']}/{coverage['target']}**",
        f"- Complete 18-category coverage: **{'YES' if coverage['complete'] else 'NO'}**",
        "",
        "## KIS",
        "",
        f"- Positive queries: {kis['positive_queries']}",
        f"- Negative queries: {kis['negative_queries']}",
        f"- VR@1 / VR@5 / VR@20: {fmt(kis.get('vr@1'))} / {fmt(kis.get('vr@5'))} / {fmt(kis.get('vr@20'))}",
        f"- FIR@1 / FIR@5 / FIR@20: {fmt(kis.get('fir@1'))} / {fmt(kis.get('fir@5'))} / {fmt(kis.get('fir@20'))}",
        f"- MRR (positive): {fmt(kis['mrr_positive'])}",
        f"- MFD first interval hit: {fmt(kis['mfd_first_interval_hit'])}",
        f"- Negative abstention rate: {fmt(kis['negative_abstention_rate'])}",
        "",
        "## QA",
        "",
        f"- Answerable queries: {qa['answerable_queries']}",
        f"- Unanswerable queries: {qa['unanswerable_queries']}",
        f"- EFR@1 / EFR@5 / EFR@20: {fmt(qa.get('efr@1'))} / {fmt(qa.get('efr@5'))} / {fmt(qa.get('efr@20'))}",
        f"- Internal exact-match accuracy: {fmt(qa['internal_exact_match_accuracy_answerable'])}",
        f"- Evidence-grounded internal exact rate: {fmt(qa['evidence_grounded_internal_exact_rate'])}",
        f"- Negative QA abstention rate: {fmt(qa['negative_abstention_rate'])}",
        "- Official QA scoring semantics: **UNRESOLVED**; exact match here is an internal diagnostic only.",
        "",
        "## TRAKE",
        "",
        f"- Queries: {trake['queries']}",
        f"- Event Hit Recall: {fmt(trake['event_hit_recall'])}",
        f"- Complete Sequence Accuracy: {fmt(trake['complete_sequence_accuracy'])}",
        f"- Valid Monotonic Sequence Rate: {fmt(trake['valid_monotonic_sequence_rate'])}",
        f"- Mean Event Frame Error: {fmt(trake['mean_event_frame_error'])}",
        "",
        "## Latency",
        "",
        f"- Samples: {latency['samples']}",
        f"- p50: {fmt(latency['p50_ms'])} ms",
        f"- p95: {fmt(latency['p95_ms'])} ms",
        "",
        "## Missing Predictions",
        "",
        f"- Count: {len(report['missing_prediction_query_ids'])}",
    ]
    lines.extend(f"- `{query_id}`" for query_id in report["missing_prediction_query_ids"])
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate P0 KIS/QA/TRAKE prediction artifacts")
    parser.add_argument("--ground-truth", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md")
    parser.add_argument("--scope", default="UNSPECIFIED")
    parser.add_argument("--generation-id", default=None)
    parser.add_argument("--runtime-label", default=None)
    args = parser.parse_args(argv)

    started = time.time()
    records = load_ground_truth(args.ground_truth)
    predictions = load_predictions(args.predictions)
    report = evaluate(records, predictions, scope=args.scope)
    provenance = {
        "git_sha": current_git_sha(),
        "generation_id": args.generation_id,
        "runtime_label": args.runtime_label,
        "evaluated_at_unix": started,
        "ground_truth_path": str(Path(args.ground_truth)),
        "predictions_path": str(Path(args.predictions)),
    }
    payload = {"provenance": provenance, **report}

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.output_md:
        output_md = Path(args.output_md)
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_md.write_text(render_markdown(report, provenance), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

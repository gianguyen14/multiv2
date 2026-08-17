from __future__ import annotations

import argparse
import json
import math
import subprocess
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

CATEGORY_IDS = tuple(f"C{i:02d}" for i in range(1, 19))
TASK_TYPES = {"kis", "qa", "trake"}
DEFAULT_CUTOFFS = (1, 5, 20)
FAILURE_TAXONOMY = {
    "QUERY_PARSE_MISS", "TRANSLATION_MISS", "VISUAL_MODEL_MISS", "SAMPLING_MISS",
    "OCR_MISS", "ASR_MISS", "EXACT_TEXT_MISS", "FUSION_MISS", "RERANK_MISS",
    "DEDUP_MISS", "TEMPORAL_REFINEMENT_MISS", "TRAKE_ALIGNMENT_MISS", "FRAME_MAPPING_MISS",
    "RESULT_SERIALIZATION_MISS", "OPERATOR_UI_MISS", "DATASET_ANNOTATION_ISSUE", "AMBIGUOUS_QUERY",
}


class EvaluationDataError(ValueError):
    pass


@dataclass(frozen=True)
class ValidationReport:
    records: int
    by_task: dict[str, int]
    by_category: dict[str, int]
    errors: tuple[str, ...]

    @property
    def valid(self) -> bool:
        return not self.errors

    def as_dict(self) -> dict[str, Any]:
        return {
            "records": self.records,
            "by_task": dict(self.by_task),
            "by_category": dict(self.by_category),
            "errors": list(self.errors),
            "valid": self.valid,
        }


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise EvaluationDataError(f"invalid JSON at line {line_number}: {exc.msg}") from exc
            if not isinstance(value, dict):
                raise EvaluationDataError(f"line {line_number}: record must be a JSON object")
            records.append(value)
    return records


def load_ground_truth(path: str | Path) -> list[dict[str, Any]]:
    records = _read_jsonl(path)
    report = validate_ground_truth(records)
    if not report.valid:
        raise EvaluationDataError("invalid ground truth: " + "; ".join(report.errors))
    return records


def load_predictions(path: str | Path) -> dict[str, dict[str, Any]]:
    output = {}
    for index, record in enumerate(_read_jsonl(path), start=1):
        query_id = record.get("query_id")
        if not isinstance(query_id, str) or not query_id.strip():
            raise EvaluationDataError(f"prediction line {index}: query_id is required")
        if query_id in output:
            raise EvaluationDataError(f"duplicate prediction query_id: {query_id}")
        if not isinstance(record.get("results", []), list):
            raise EvaluationDataError(f"prediction {query_id}: results must be a list")
        output[query_id] = record
    return output


def validate_ground_truth(records: Sequence[Mapping[str, Any]]) -> ValidationReport:
    errors = []
    seen = set()
    by_task: dict[str, int] = defaultdict(int)
    by_category: dict[str, int] = defaultdict(int)
    for position, record in enumerate(records, start=1):
        prefix = f"record {position}"
        query_id = record.get("query_id")
        category_id = record.get("category_id")
        task_type = record.get("task_type")
        if not isinstance(query_id, str) or not query_id.strip():
            errors.append(f"{prefix}: query_id is required")
        elif query_id in seen:
            errors.append(f"{prefix}: duplicate query_id {query_id}")
        else:
            seen.add(query_id)
        if category_id not in CATEGORY_IDS:
            errors.append(f"{prefix}: category_id must be C01..C18")
        else:
            by_category[str(category_id)] += 1
        if task_type not in TASK_TYPES:
            errors.append(f"{prefix}: task_type must be one of {sorted(TASK_TYPES)}")
            continue
        by_task[str(task_type)] += 1
        if task_type in {"kis", "qa"}:
            if not isinstance(record.get("query"), str) or not record["query"].strip():
                errors.append(f"{prefix}: query is required for {task_type}")
            _validate_localization(record.get("ground_truth"), prefix, errors)
            if task_type == "qa":
                _validate_qa(record.get("qa_ground_truth"), prefix, errors)
        else:
            events = record.get("events")
            if not isinstance(events, list) or not events or not all(isinstance(v, str) and v.strip() for v in events):
                errors.append(f"{prefix}: non-empty events are required for trake")
            _validate_trake(record.get("trake_ground_truth"), events if isinstance(events, list) else [], prefix, errors)
        failure_category = record.get("failure_category")
        if failure_category is not None and failure_category not in FAILURE_TAXONOMY:
            errors.append(f"{prefix}: unknown failure_category {failure_category}")
    return ValidationReport(len(records), dict(sorted(by_task.items())), dict(sorted(by_category.items())), tuple(errors))


def _validate_localization(value: Any, prefix: str, errors: list[str]) -> None:
    if not isinstance(value, Mapping):
        errors.append(f"{prefix}: ground_truth object is required")
        return
    if not isinstance(value.get("video_id"), str) or not value["video_id"].strip():
        errors.append(f"{prefix}: ground_truth.video_id is required")
    interval = value.get("valid_frame_interval")
    if not _valid_interval(interval):
        errors.append(f"{prefix}: ground_truth.valid_frame_interval must be [start,end] with 0 <= start <= end")
        return
    exact_frame = value.get("exact_frame_id")
    if exact_frame is not None:
        if not isinstance(exact_frame, int) or isinstance(exact_frame, bool):
            errors.append(f"{prefix}: ground_truth.exact_frame_id must be an integer")
        elif not interval[0] <= exact_frame <= interval[1]:
            errors.append(f"{prefix}: exact_frame_id must lie inside valid_frame_interval")


def _validate_qa(value: Any, prefix: str, errors: list[str]) -> None:
    if not isinstance(value, Mapping):
        errors.append(f"{prefix}: qa_ground_truth object is required")
        return
    answerable = value.get("answerable", True)
    if not isinstance(answerable, bool):
        errors.append(f"{prefix}: qa_ground_truth.answerable must be boolean")
        return
    answers = value.get("accepted_answers", [])
    if not isinstance(answers, list) or not all(isinstance(v, str) for v in answers):
        errors.append(f"{prefix}: qa_ground_truth.accepted_answers must be a string list")
    elif answerable and not any(v.strip() for v in answers):
        errors.append(f"{prefix}: answerable QA requires at least one accepted answer")


def _validate_trake(value: Any, events: Sequence[Any], prefix: str, errors: list[str]) -> None:
    if not isinstance(value, Mapping):
        errors.append(f"{prefix}: trake_ground_truth object is required")
        return
    if not isinstance(value.get("video_id"), str) or not value["video_id"].strip():
        errors.append(f"{prefix}: trake_ground_truth.video_id is required")
    intervals = value.get("event_intervals")
    if not isinstance(intervals, list) or len(intervals) != len(events) or not intervals:
        errors.append(f"{prefix}: event_intervals must align one-to-one with events")
        return
    if not all(_valid_interval(v) for v in intervals):
        errors.append(f"{prefix}: every TRAKE event interval must be valid")
        return
    if any(a[0] >= b[0] for a, b in zip(intervals, intervals[1:])):
        errors.append(f"{prefix}: TRAKE event intervals must be chronologically ordered")
    event_frames = value.get("event_frames")
    if event_frames is not None:
        if not isinstance(event_frames, list) or len(event_frames) != len(intervals):
            errors.append(f"{prefix}: event_frames must align one-to-one with events")
        elif any(not isinstance(v, int) or isinstance(v, bool) for v in event_frames):
            errors.append(f"{prefix}: event_frames must contain integers")
        else:
            for frame, interval in zip(event_frames, intervals):
                if not interval[0] <= frame <= interval[1]:
                    errors.append(f"{prefix}: event_frame must lie inside its event interval")
                    break


def _valid_interval(value: Any) -> bool:
    return isinstance(value, (list, tuple)) and len(value) == 2 and all(isinstance(v, int) and not isinstance(v, bool) for v in value) and 0 <= value[0] <= value[1]


def _frame_hit(result: Mapping[str, Any], truth: Mapping[str, Any]) -> bool:
    interval = truth["valid_frame_interval"]
    return result.get("video_id") == truth["video_id"] and isinstance(result.get("frame_id"), int) and interval[0] <= result["frame_id"] <= interval[1]


def _video_hit(result: Mapping[str, Any], truth: Mapping[str, Any]) -> bool:
    return result.get("video_id") == truth["video_id"]


def _first_hit_rank(results: Sequence[Mapping[str, Any]], truth: Mapping[str, Any]) -> int | None:
    for rank, result in enumerate(results, start=1):
        if _frame_hit(result, truth):
            return rank
    return None


def _metric_mean(values: Iterable[float]) -> float:
    values = list(values)
    return float(sum(values) / len(values)) if values else 0.0


def evaluate_kis(records, predictions, cutoffs=DEFAULT_CUTOFFS):
    kis_records = [r for r in records if r["task_type"] == "kis"]
    per_query = []
    reciprocal_ranks = []
    frame_distances = []
    for record in kis_records:
        results = predictions.get(record["query_id"], {}).get("results", [])
        truth = record["ground_truth"]
        rank = _first_hit_rank(results, truth)
        reciprocal_ranks.append(0.0 if rank is None else 1.0 / rank)
        exact_frame = truth.get("exact_frame_id")
        if exact_frame is not None and rank is not None:
            frame_distances.append(abs(results[rank - 1]["frame_id"] - exact_frame))
        item = {"query_id": record["query_id"], "category_id": record["category_id"], "first_interval_hit_rank": rank}
        for cutoff in cutoffs:
            subset = results[:cutoff]
            item[f"vr@{cutoff}"] = float(any(_video_hit(v, truth) for v in subset))
            item[f"fir@{cutoff}"] = float(any(_frame_hit(v, truth) for v in subset))
        per_query.append(item)
    aggregate = {"queries": len(kis_records), "mrr": _metric_mean(reciprocal_ranks), "mfd": _metric_mean(frame_distances) if frame_distances else None}
    for cutoff in cutoffs:
        aggregate[f"vr@{cutoff}"] = _metric_mean(v[f"vr@{cutoff}"] for v in per_query)
        aggregate[f"fir@{cutoff}"] = _metric_mean(v[f"fir@{cutoff}"] for v in per_query)
    return {"aggregate": aggregate, "per_query": per_query}


def _normalize_answer(value: Any) -> str:
    return "" if value is None else " ".join(str(value).strip().casefold().split())


def evaluate_qa(records, predictions, cutoffs=DEFAULT_CUTOFFS):
    qa_records = [r for r in records if r["task_type"] == "qa"]
    per_query = []
    for record in qa_records:
        results = predictions.get(record["query_id"], {}).get("results", [])
        truth = record["ground_truth"]
        qa_truth = record["qa_ground_truth"]
        item = {"query_id": record["query_id"], "category_id": record["category_id"]}
        for cutoff in cutoffs:
            subset = results[:cutoff]
            item[f"evr@{cutoff}"] = float(any(_video_hit(v, truth) for v in subset))
            item[f"efr@{cutoff}"] = float(any(_frame_hit(v, truth) for v in subset))
        top = results[0] if results else {}
        answerable = qa_truth.get("answerable", True)
        accepted = {_normalize_answer(v) for v in qa_truth.get("accepted_answers", []) if _normalize_answer(v)}
        predicted_answer = _normalize_answer(top.get("answer"))
        answer_match = bool(predicted_answer and predicted_answer in accepted) if answerable else not bool(predicted_answer)
        evidence_hit = bool(top and _frame_hit(top, truth))
        item["answer_match_internal"] = float(answer_match)
        item["abstained"] = not bool(predicted_answer)
        item["evidence_grounded_answer"] = float(answer_match and evidence_hit)
        per_query.append(item)
    aggregate = {
        "queries": len(qa_records),
        "ema_internal": _metric_mean(v["answer_match_internal"] for v in per_query),
        "evidence_grounded_answer_rate": _metric_mean(v["evidence_grounded_answer"] for v in per_query),
        "official_qa_scoring_semantics": "UNRESOLVED",
    }
    for cutoff in cutoffs:
        aggregate[f"evr@{cutoff}"] = _metric_mean(v[f"evr@{cutoff}"] for v in per_query)
        aggregate[f"efr@{cutoff}"] = _metric_mean(v[f"efr@{cutoff}"] for v in per_query)
    return {"aggregate": aggregate, "per_query": per_query}


def _trake_sequence_metrics(result: Mapping[str, Any], truth: Mapping[str, Any]) -> dict[str, float]:
    frames = result.get("frame_ids", [])
    intervals = truth["event_intervals"]
    video_match = float(result.get("video_id") == truth["video_id"])
    if not isinstance(frames, list) or len(frames) != len(intervals) or not all(isinstance(v, int) for v in frames):
        return {"video_match": video_match, "ehr": 0.0, "csa": 0.0, "monotonic": 0.0, "mefe": math.nan}
    monotonic = float(all(a < b for a, b in zip(frames, frames[1:])))
    hits = [start <= frame <= end for frame, (start, end) in zip(frames, intervals)] if video_match else [False] * len(intervals)
    ehr = sum(hits) / len(intervals) if intervals else 0.0
    csa = float(bool(video_match and monotonic and all(hits)))
    exact = truth.get("event_frames")
    mefe = _metric_mean(abs(a - b) for a, b in zip(frames, exact)) if isinstance(exact, list) and len(exact) == len(frames) else math.nan
    return {"video_match": video_match, "ehr": ehr, "csa": csa, "monotonic": monotonic, "mefe": mefe}


def evaluate_trake(records, predictions):
    trake_records = [r for r in records if r["task_type"] == "trake"]
    per_query = []
    for record in trake_records:
        results = predictions.get(record["query_id"], {}).get("results", [])
        metrics = _trake_sequence_metrics(results[0] if results else {}, record["trake_ground_truth"])
        per_query.append({"query_id": record["query_id"], "category_id": record["category_id"], **metrics})
    mefe_values = [v["mefe"] for v in per_query if not math.isnan(v["mefe"])]
    aggregate = {
        "queries": len(trake_records),
        "video_match_rate": _metric_mean(v["video_match"] for v in per_query),
        "ehr": _metric_mean(v["ehr"] for v in per_query),
        "csa": _metric_mean(v["csa"] for v in per_query),
        "valid_monotonic_sequence_rate": _metric_mean(v["monotonic"] for v in per_query),
        "mefe": _metric_mean(mefe_values) if mefe_values else None,
    }
    return {"aggregate": aggregate, "per_query": per_query}


def category_coverage(records):
    counts = {category: 0 for category in CATEGORY_IDS}
    for record in records:
        if record.get("category_id") in counts:
            counts[record["category_id"]] += 1
    return {"counts": counts, "covered": sum(value > 0 for value in counts.values()), "target": len(CATEGORY_IDS), "complete": all(value > 0 for value in counts.values())}


def _percentile(values, quantile):
    if len(values) == 1:
        return float(values[0])
    position = (len(values) - 1) * quantile
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return float(values[lower])
    weight = position - lower
    return float(values[lower] * (1.0 - weight) + values[upper] * weight)


def latency_summary(predictions):
    values = sorted(float(v["latency_ms"]) for v in predictions.values() if isinstance(v.get("latency_ms"), (int, float)))
    return {"samples": len(values), "p50_ms": _percentile(values, 0.50) if values else None, "p95_ms": _percentile(values, 0.95) if values else None}


def evaluate(records, predictions, cutoffs=DEFAULT_CUTOFFS):
    validation = validate_ground_truth(records)
    if not validation.valid:
        raise EvaluationDataError("invalid ground truth: " + "; ".join(validation.errors))
    return {
        "measurement_status": "MEASURED_FROM_SUPPLIED_PREDICTIONS",
        "ground_truth_validation": validation.as_dict(),
        "category_coverage": category_coverage(records),
        "missing_prediction_query_ids": sorted(r["query_id"] for r in records if r["query_id"] not in predictions),
        "kis": evaluate_kis(records, predictions, cutoffs),
        "qa": evaluate_qa(records, predictions, cutoffs),
        "trake": evaluate_trake(records, predictions),
        "latency": latency_summary(predictions),
    }


def current_git_sha() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def render_markdown(report, provenance=None):
    provenance = dict(provenance or {})
    coverage, kis, qa, trake, latency = report["category_coverage"], report["kis"]["aggregate"], report["qa"]["aggregate"], report["trake"]["aggregate"], report["latency"]
    lines = [
        "# P0 Baseline Evaluation Report", "",
        "This report measures only the supplied ground truth and prediction artifacts. It does not infer unmeasured gains.", "",
        "## Provenance", "",
        f"- Git SHA: `{provenance.get('git_sha') or 'UNRESOLVED'}`",
        f"- Generation ID: `{provenance.get('generation_id') or 'UNRESOLVED'}`",
        f"- Device/runtime label: `{provenance.get('runtime_label') or 'UNRESOLVED'}`", "",
        "## Ground Truth Coverage", "",
        f"- Records: **{report['ground_truth_validation']['records']}**",
        f"- Categories covered: **{coverage['covered']}/{coverage['target']}**",
        f"- Complete 18-category coverage: **{'YES' if coverage['complete'] else 'NO'}**", "",
        "## KIS", "",
        f"- Queries: {kis['queries']}",
        f"- VR@1 / VR@5 / VR@20: {kis.get('vr@1', 0):.4f} / {kis.get('vr@5', 0):.4f} / {kis.get('vr@20', 0):.4f}",
        f"- FIR@1 / FIR@5 / FIR@20: {kis.get('fir@1', 0):.4f} / {kis.get('fir@5', 0):.4f} / {kis.get('fir@20', 0):.4f}",
        f"- MRR: {kis['mrr']:.4f}", f"- MFD: {kis['mfd'] if kis['mfd'] is not None else 'N/A'}", "",
        "## QA", "", f"- Queries: {qa['queries']}",
        f"- EFR@1 / EFR@5 / EFR@20: {qa.get('efr@1', 0):.4f} / {qa.get('efr@5', 0):.4f} / {qa.get('efr@20', 0):.4f}",
        f"- Internal EMA: {qa['ema_internal']:.4f}", f"- Evidence-grounded answer rate: {qa['evidence_grounded_answer_rate']:.4f}",
        "- Official QA scoring semantics: **UNRESOLVED**; EMA is an internal diagnostic metric only.", "",
        "## TRAKE", "", f"- Queries: {trake['queries']}", f"- Event Hit Recall: {trake['ehr']:.4f}",
        f"- Complete Sequence Accuracy: {trake['csa']:.4f}", f"- Valid Monotonic Sequence Rate: {trake['valid_monotonic_sequence_rate']:.4f}",
        f"- MEFE: {trake['mefe'] if trake['mefe'] is not None else 'N/A'}", "",
        "## Latency", "", f"- Samples: {latency['samples']}", f"- p50: {latency['p50_ms'] if latency['p50_ms'] is not None else 'N/A'} ms", f"- p95: {latency['p95_ms'] if latency['p95_ms'] is not None else 'N/A'} ms", "",
        "## Missing Predictions", "", f"- Count: {len(report['missing_prediction_query_ids'])}",
    ]
    if report["missing_prediction_query_ids"]:
        lines.extend(f"- `{query_id}`" for query_id in report["missing_prediction_query_ids"])
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate P0 multi-task retrieval baseline artifacts")
    parser.add_argument("--ground-truth", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md")
    parser.add_argument("--generation-id", default=None)
    parser.add_argument("--runtime-label", default=None)
    args = parser.parse_args(argv)
    started = time.time()
    records = load_ground_truth(args.ground_truth)
    predictions = load_predictions(args.predictions)
    report = evaluate(records, predictions)
    provenance = {"git_sha": current_git_sha(), "generation_id": args.generation_id, "runtime_label": args.runtime_label, "evaluated_at_unix": started, "ground_truth_path": str(Path(args.ground_truth)), "predictions_path": str(Path(args.predictions))}
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps({"provenance": provenance, **report}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.output_md:
        output_md = Path(args.output_md)
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_md.write_text(render_markdown(report, provenance), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

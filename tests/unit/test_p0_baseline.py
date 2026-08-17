import json

import pytest

from eval.p0_baseline import (
    EvaluationDataError,
    category_coverage,
    evaluate,
    evaluate_kis,
    evaluate_qa,
    evaluate_trake,
    load_ground_truth,
    load_predictions,
    render_markdown,
    validate_ground_truth,
)


def _records():
    return [
        {
            "query_id": "GT_C01_001", "category_id": "C01", "task_type": "kis", "query": "red truck",
            "ground_truth": {"video_id": "V1", "exact_frame_id": 110, "valid_frame_interval": [100, 120]},
        },
        {
            "query_id": "GT_C16_001", "category_id": "C16", "task_type": "qa", "query": "What color is the car?",
            "ground_truth": {"video_id": "V2", "exact_frame_id": 210, "valid_frame_interval": [200, 220]},
            "qa_ground_truth": {"answerable": True, "accepted_answers": ["red", "màu đỏ"]},
        },
        {
            "query_id": "GT_C13_001", "category_id": "C13", "task_type": "trake", "events": ["start", "middle", "end"],
            "trake_ground_truth": {"video_id": "V3", "event_frames": [10, 30, 50], "event_intervals": [[8, 12], [28, 32], [48, 52]]},
        },
    ]


def _predictions():
    return {
        "GT_C01_001": {"query_id": "GT_C01_001", "latency_ms": 10, "results": [{"video_id": "VX", "frame_id": 1}, {"video_id": "V1", "frame_id": 111}]},
        "GT_C16_001": {"query_id": "GT_C16_001", "latency_ms": 20, "results": [{"video_id": "V2", "frame_id": 211, "answer": " Red "}]},
        "GT_C13_001": {"query_id": "GT_C13_001", "latency_ms": 30, "results": [{"video_id": "V3", "frame_ids": [9, 30, 51]}]},
    }


def test_validate_ground_truth_and_coverage():
    report = validate_ground_truth(_records())
    assert report.valid is True
    assert report.by_task == {"kis": 1, "qa": 1, "trake": 1}
    coverage = category_coverage(_records())
    assert coverage["covered"] == 3
    assert coverage["target"] == 18
    assert coverage["complete"] is False


def test_duplicate_query_id_is_rejected():
    records = _records()
    records.append(dict(records[0]))
    report = validate_ground_truth(records)
    assert report.valid is False
    assert any("duplicate query_id" in error for error in report.errors)


def test_exact_frame_must_be_inside_interval():
    records = _records()
    records[0]["ground_truth"]["exact_frame_id"] = 121
    report = validate_ground_truth(records)
    assert report.valid is False
    assert any("exact_frame_id" in error for error in report.errors)


def test_trake_intervals_must_be_chronological():
    records = _records()
    records[2]["trake_ground_truth"]["event_intervals"] = [[28, 32], [8, 12], [48, 52]]
    report = validate_ground_truth(records)
    assert report.valid is False
    assert any("chronologically ordered" in error for error in report.errors)


def test_kis_interval_metrics_and_mrr():
    result = evaluate_kis(_records(), _predictions())
    aggregate = result["aggregate"]
    assert aggregate["vr@1"] == 0.0
    assert aggregate["vr@5"] == 1.0
    assert aggregate["fir@1"] == 0.0
    assert aggregate["fir@5"] == 1.0
    assert aggregate["mrr"] == pytest.approx(0.5)
    assert aggregate["mfd"] == 1.0


def test_qa_internal_exact_match_is_separate_from_evidence_metric():
    aggregate = evaluate_qa(_records(), _predictions())["aggregate"]
    assert aggregate["efr@1"] == 1.0
    assert aggregate["ema_internal"] == 1.0
    assert aggregate["evidence_grounded_answer_rate"] == 1.0
    assert aggregate["official_qa_scoring_semantics"] == "UNRESOLVED"


def test_negative_qa_requires_abstention_for_internal_metric():
    records = [_records()[1]]
    records[0]["qa_ground_truth"] = {"answerable": False, "accepted_answers": []}
    predictions = {records[0]["query_id"]: {"results": [{"video_id": "V2", "frame_id": 210, "answer": ""}]}}
    assert evaluate_qa(records, predictions)["aggregate"]["ema_internal"] == 1.0


def test_trake_metrics_require_monotonic_complete_sequence():
    aggregate = evaluate_trake(_records(), _predictions())["aggregate"]
    assert aggregate["video_match_rate"] == 1.0
    assert aggregate["ehr"] == 1.0
    assert aggregate["csa"] == 1.0
    assert aggregate["valid_monotonic_sequence_rate"] == 1.0
    assert aggregate["mefe"] == pytest.approx(2 / 3)


def test_trake_non_monotonic_sequence_cannot_be_complete():
    records = [_records()[2]]
    predictions = {records[0]["query_id"]: {"results": [{"video_id": "V3", "frame_ids": [50, 30, 10]}]}}
    aggregate = evaluate_trake(records, predictions)["aggregate"]
    assert aggregate["valid_monotonic_sequence_rate"] == 0.0
    assert aggregate["csa"] == 0.0


def test_evaluate_reports_missing_predictions_and_latency_percentiles():
    predictions = _predictions()
    predictions.pop("GT_C13_001")
    report = evaluate(_records(), predictions)
    assert report["missing_prediction_query_ids"] == ["GT_C13_001"]
    assert report["latency"]["samples"] == 2
    assert report["latency"]["p50_ms"] == 15.0
    assert report["latency"]["p95_ms"] == pytest.approx(19.5)


def test_jsonl_loaders_validate_input(tmp_path):
    gt = tmp_path / "gt.jsonl"
    pred = tmp_path / "pred.jsonl"
    gt.write_text("\n".join(json.dumps(v) for v in _records()) + "\n", encoding="utf-8")
    pred.write_text("\n".join(json.dumps(v) for v in _predictions().values()) + "\n", encoding="utf-8")
    assert len(load_ground_truth(gt)) == 3
    assert len(load_predictions(pred)) == 3
    pred.write_text(json.dumps({"query_id": "x", "results": []}) + "\n" + json.dumps({"query_id": "x", "results": []}), encoding="utf-8")
    with pytest.raises(EvaluationDataError, match="duplicate prediction query_id"):
        load_predictions(pred)


def test_render_markdown_labels_qa_metric_as_internal():
    markdown = render_markdown(evaluate(_records(), _predictions()), {"git_sha": "abc", "generation_id": "gen-1", "runtime_label": "cpu"})
    assert "Internal EMA" in markdown
    assert "Official QA scoring semantics: **UNRESOLVED**" in markdown
    assert "Categories covered: **3/18**" in markdown

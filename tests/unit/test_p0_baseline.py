import json

import pytest

from eval.p0_baseline import (
    EvaluationDataError,
    category_coverage,
    evaluate,
    evaluate_kis,
    evaluate_qa,
    evaluate_trake,
    load_predictions,
    render_markdown,
    validate_ground_truth,
)
from eval.p0_run_current import build_request, run_records


def _records():
    return [
        {
            "query_id": "GT_C01_001",
            "category_id": "C01",
            "task_type": "kis",
            "query": "red truck",
            "ground_truth": {
                "video_id": "V1",
                "exact_frame_id": 110,
                "valid_frame_intervals": [[100, 120]],
            },
        },
        {
            "query_id": "GT_C18_001",
            "category_id": "C18",
            "task_type": "kis",
            "query": "nonexistent object",
            "ground_truth": {"expect_no_result": True},
        },
        {
            "query_id": "GT_C16_001",
            "category_id": "C16",
            "task_type": "qa",
            "query": "What color is the car?",
            "ground_truth": {
                "video_id": "V2",
                "exact_frame_id": 210,
                "valid_frame_intervals": [[200, 220]],
            },
            "qa_ground_truth": {"answerable": True, "accepted_answers": ["red", "màu đỏ"]},
        },
        {
            "query_id": "GT_C18_QA",
            "category_id": "C18",
            "task_type": "qa",
            "query": "What unsupported fact is shown?",
            "qa_ground_truth": {"answerable": False, "accepted_answers": []},
        },
        {
            "query_id": "GT_C13_001",
            "category_id": "C13",
            "task_type": "trake",
            "events": ["start", "middle", "end"],
            "trake_ground_truth": {
                "video_id": "V3",
                "event_frames": [10, 30, 50],
                "event_intervals": [[8, 12], [28, 32], [48, 52]],
            },
        },
    ]


def _predictions():
    return {
        "GT_C01_001": {
            "query_id": "GT_C01_001",
            "latency_ms": 10,
            "results": [
                {"video_id": "VX", "frame_id": 1},
                {"video_id": "V1", "frame_id": 111},
            ],
        },
        "GT_C18_001": {"query_id": "GT_C18_001", "latency_ms": 20, "results": []},
        "GT_C16_001": {
            "query_id": "GT_C16_001",
            "latency_ms": 30,
            "results": [{"video_id": "V2", "frame_id": 211, "answer": " Red "}],
        },
        "GT_C18_QA": {
            "query_id": "GT_C18_QA",
            "latency_ms": 40,
            "results": [{"video_id": "VX", "frame_id": 999, "answer": ""}],
        },
        "GT_C13_001": {
            "query_id": "GT_C13_001",
            "latency_ms": 50,
            "results": [{"video_id": "V3", "frame_ids": [9, 30, 51]}],
        },
    }


def test_validate_ground_truth_and_category_coverage():
    report = validate_ground_truth(_records())
    assert report["valid"] is True
    assert report["by_task"] == {"kis": 2, "qa": 2, "trake": 1}
    coverage = category_coverage(_records())
    assert coverage["covered"] == 4
    assert coverage["target"] == 18
    assert coverage["complete"] is False


def test_kis_positive_metrics_and_negative_abstention_are_separate():
    aggregate = evaluate_kis(_records(), _predictions())["aggregate"]
    assert aggregate["positive_queries"] == 1
    assert aggregate["negative_queries"] == 1
    assert aggregate["vr@1"] == 0.0
    assert aggregate["vr@5"] == 1.0
    assert aggregate["fir@1"] == 0.0
    assert aggregate["fir@5"] == 1.0
    assert aggregate["mrr_positive"] == pytest.approx(0.5)
    assert aggregate["mfd_first_interval_hit"] == 1.0
    assert aggregate["negative_abstention_rate"] == 1.0
    assert aggregate["negative_false_positive_rate"] == 0.0


def test_negative_kis_false_positive_is_counted_without_polluting_recall_denominator():
    predictions = _predictions()
    predictions["GT_C18_001"] = {
        "query_id": "GT_C18_001",
        "results": [{"video_id": "V9", "frame_id": 5}],
    }
    aggregate = evaluate_kis(_records(), predictions)["aggregate"]
    assert aggregate["fir@5"] == 1.0
    assert aggregate["negative_abstention_rate"] == 0.0
    assert aggregate["negative_false_positive_rate"] == 1.0


def test_kis_accepts_multiple_annotated_intervals():
    record = _records()[0]
    record["ground_truth"]["valid_frame_intervals"] = [[100, 105], [200, 205]]
    record["ground_truth"]["exact_frame_id"] = 202
    predictions = {
        record["query_id"]: {
            "query_id": record["query_id"],
            "results": [{"video_id": "V1", "frame_id": 201}],
        }
    }
    aggregate = evaluate_kis([record], predictions)["aggregate"]
    assert aggregate["fir@1"] == 1.0
    assert aggregate["mfd_first_interval_hit"] == 1.0


def test_qa_internal_metric_is_not_claimed_as_official_scoring():
    aggregate = evaluate_qa(_records(), _predictions())["aggregate"]
    assert aggregate["answerable_queries"] == 1
    assert aggregate["unanswerable_queries"] == 1
    assert aggregate["localizable_queries"] == 1
    assert aggregate["efr@1"] == 1.0
    assert aggregate["internal_exact_match_accuracy_answerable"] == 1.0
    assert aggregate["evidence_grounded_internal_exact_rate"] == 1.0
    assert aggregate["negative_abstention_rate"] == 1.0
    assert aggregate["official_qa_scoring_semantics"] == "UNRESOLVED"


def test_trake_complete_sequence_requires_monotonic_order_and_interval_hits():
    aggregate = evaluate_trake(_records(), _predictions())["aggregate"]
    assert aggregate["video_match_rate"] == 1.0
    assert aggregate["event_hit_recall"] == 1.0
    assert aggregate["complete_sequence_accuracy"] == 1.0
    assert aggregate["valid_monotonic_sequence_rate"] == 1.0
    assert aggregate["mean_event_frame_error"] == pytest.approx(2 / 3)

    bad = dict(_predictions())
    bad["GT_C13_001"] = {
        "query_id": "GT_C13_001",
        "results": [{"video_id": "V3", "frame_ids": [50, 30, 10]}],
    }
    aggregate = evaluate_trake(_records(), bad)["aggregate"]
    assert aggregate["valid_monotonic_sequence_rate"] == 0.0
    assert aggregate["complete_sequence_accuracy"] == 0.0


def test_evaluate_reports_latency_missing_predictions_and_scope():
    predictions = _predictions()
    predictions.pop("GT_C13_001")
    report = evaluate(_records(), predictions, scope="SYNTHETIC_TEST")
    assert report["measurement_scope"] == "SYNTHETIC_TEST"
    assert report["missing_prediction_query_ids"] == ["GT_C13_001"]
    assert report["latency"]["samples"] == 4
    assert report["latency"]["p50_ms"] == 25.0
    assert report["latency"]["p95_ms"] == pytest.approx(38.5)
    assert report["notes"]["performance_thresholds"] == "TO_BE_ESTABLISHED_FROM_BASELINE"


def test_invalid_exact_frame_and_duplicate_prediction_ids_are_rejected(tmp_path):
    records = _records()
    records[0]["ground_truth"]["exact_frame_id"] = 121
    with pytest.raises(EvaluationDataError, match="exact_frame_id"):
        validate_ground_truth(records)

    pred = tmp_path / "pred.jsonl"
    pred.write_text(
        json.dumps({"query_id": "same", "results": []})
        + "\n"
        + json.dumps({"query_id": "same", "results": []})
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(EvaluationDataError, match="duplicate query_id"):
        load_predictions(pred)


def test_render_markdown_preserves_evidence_discipline():
    report = evaluate(_records(), _predictions(), scope="SYNTHETIC_TEST")
    markdown = render_markdown(
        report,
        {"git_sha": "abc", "generation_id": "gen-1", "runtime_label": "cpu"},
    )
    assert "Measurement scope: `SYNTHETIC_TEST`" in markdown
    assert "Official QA scoring semantics: **UNRESOLVED**" in markdown
    assert "Categories covered: **4/18**" in markdown


def test_runner_builds_task_aware_requests_and_captures_manifest():
    records = [_records()[0], _records()[2], _records()[4]]

    class FakeSearch:
        last_query_metrics = {"path": "query"}
        last_trake_metrics = {"path": "trake"}

        def __init__(self):
            self.requests = []

        def handle(self, request):
            self.requests.append(dict(request))
            if request["query_type"] == "trake":
                return [{"video_id": "V3", "frame_ids": [9, 30, 51]}]
            if request["query_type"] == "qa":
                return [{"video_id": "V2", "frame_id": 211, "answer": "red"}]
            return [{"video_id": "V1", "frame_id": 111}]

        def status(self):
            return {"configured": True}

        def readiness(self):
            return {"ready": True, "generation_id": "gen-test"}

    search = FakeSearch()
    predictions, manifest = run_records(
        records,
        search,
        top_k=20,
        query_refine=False,
        rerank=False,
        temporal_refine=False,
    )
    assert len(predictions) == 3
    assert manifest["task_counts"] == {"kis": 1, "qa": 1, "trake": 1}
    assert manifest["query_refine"] is False
    assert manifest["rerank"] is False
    assert manifest["temporal_refine"] is False
    assert all(request["top_k"] == 20 for request in search.requests)
    assert search.requests[0]["query"] == "red truck"
    assert search.requests[1]["query_type"] == "qa"
    assert search.requests[2]["events"] == ["start", "middle", "end"]
    assert predictions[2]["diagnostics"] == {"path": "trake"}


def test_build_request_always_records_temporal_refine_flag():
    request = build_request(
        _records()[0],
        top_k=5,
        query_refine=True,
        rerank=True,
        temporal_refine=False,
    )
    assert request["query_type"] == "kis"
    assert request["temporal_refine"] is False

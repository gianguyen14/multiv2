import json
import tempfile
import unittest
from pathlib import Path

from evaluation.p0_baseline import EvaluationInputError, _load_jsonl, evaluate


class P0BaselineEvaluatorTests(unittest.TestCase):
    def test_smoke_metrics_are_interval_based_and_deterministic(self):
        gt = [
            {
                "query_id": "kis",
                "category_id": "C02",
                "task_type": "kis",
                "ground_truth": {
                    "video_id": "V1",
                    "exact_frame_id": 105,
                    "valid_frame_intervals": [[100, 110]],
                },
            },
            {
                "query_id": "qa",
                "category_id": "C16",
                "task_type": "qa",
                "ground_truth": {
                    "answerable": True,
                    "video_id": "V2",
                    "exact_frame_id": 210,
                    "valid_frame_intervals": [[205, 215]],
                    "accepted_answers": ["40 độ c"],
                },
            },
            {
                "query_id": "qa-neg",
                "category_id": "C18",
                "task_type": "qa",
                "ground_truth": {"answerable": False, "accepted_answers": []},
            },
            {
                "query_id": "trake",
                "category_id": "C13",
                "task_type": "trake",
                "ground_truth": {
                    "video_id": "V3",
                    "event_intervals": [[10, 20], [30, 40], [50, 60]],
                    "exact_event_frames": [15, 35, 55],
                },
            },
        ]
        pred = [
            {
                "query_id": "kis",
                "latency_ms": 10,
                "results": [
                    {"video_id": "V9", "frame_id": 1},
                    {"video_id": "V1", "frame_id": 108},
                ],
            },
            {
                "query_id": "qa",
                "latency_ms": 20,
                "result": {"video_id": "V2", "frame_id": 209, "answer": "40 ĐỘ C"},
            },
            {"query_id": "qa-neg", "latency_ms": 30, "result": {"answer": ""}},
            {
                "query_id": "trake",
                "latency_ms": 40,
                "result": {"video_id": "V3", "frames": [15, 36, 55]},
            },
        ]

        report = evaluate(gt, pred, "SYNTHETIC_TEST")
        kis = report["metrics"]["kis"]
        self.assertEqual(kis["VR@1"], 0.0)
        self.assertEqual(kis["VR@5"], 1.0)
        self.assertEqual(kis["FIR@1"], 0.0)
        self.assertEqual(kis["FIR@5"], 1.0)
        self.assertEqual(kis["MRR"], 0.5)
        self.assertEqual(kis["MFD_first_interval_hit"], 3.0)

        qa = report["metrics"]["qa"]
        self.assertEqual(qa["evidence_frame_recall@1"], 1.0)
        self.assertEqual(qa["internal_exact_match_accuracy_answerable"], 1.0)
        self.assertEqual(qa["negative_abstention_rate"], 1.0)
        self.assertEqual(qa["official_qa_scoring_semantics"], "UNRESOLVED")

        trake = report["metrics"]["trake"]
        self.assertEqual(trake["valid_monotonic_sequence_rate"], 1.0)
        self.assertEqual(trake["event_hit_recall"], 1.0)
        self.assertEqual(trake["complete_sequence_accuracy"], 1.0)
        self.assertAlmostEqual(trake["mean_event_frame_error"], 1.0 / 3.0)

        runtime = report["metrics"]["runtime"]
        self.assertEqual(runtime["p50_latency_ms"], 25.0)
        self.assertEqual(runtime["p95_latency_ms"], 38.5)

    def test_interval_boundaries_are_inclusive(self):
        gt = [{
            "query_id": "q",
            "task_type": "kis",
            "ground_truth": {"video_id": "V", "valid_frame_intervals": [[10, 20]]},
        }]
        for frame_id in (10, 20):
            pred = [{"query_id": "q", "results": [{"video_id": "V", "frame_id": frame_id}]}]
            report = evaluate(gt, pred, "TEST")
            self.assertEqual(report["metrics"]["kis"]["FIR@1"], 1.0)

    def test_kis_supports_multiple_valid_intervals(self):
        gt = [{
            "query_id": "q",
            "task_type": "kis",
            "ground_truth": {
                "video_id": "V",
                "valid_frame_intervals": [[10, 20], [100, 110]],
            },
        }]
        pred = [{"query_id": "q", "results": [{"video_id": "V", "frame_id": 105}]}]
        report = evaluate(gt, pred, "TEST")
        self.assertEqual(report["metrics"]["kis"]["FIR@1"], 1.0)

    def test_missing_prediction_is_a_miss_not_an_exception(self):
        gt = [{
            "query_id": "q",
            "task_type": "kis",
            "ground_truth": {"video_id": "V", "valid_frame_intervals": [[10, 20]]},
        }]
        report = evaluate(gt, [{"query_id": "other"}], "TEST")
        self.assertEqual(report["metrics"]["kis"]["FIR@20"], 0.0)
        self.assertEqual(report["metrics"]["kis"]["MRR"], 0.0)

    def test_trake_complete_sequence_requires_monotonic_order(self):
        gt = [{
            "query_id": "q",
            "task_type": "trake",
            "ground_truth": {
                "video_id": "V",
                "event_intervals": [[10, 20], [30, 40], [50, 60]],
            },
        }]
        pred = [{
            "query_id": "q",
            "result": {"video_id": "V", "frames": [15, 35, 25]},
        }]
        report = evaluate(gt, pred, "TEST")
        self.assertEqual(report["metrics"]["trake"]["valid_monotonic_sequence_rate"], 0.0)
        self.assertEqual(report["metrics"]["trake"]["complete_sequence_accuracy"], 0.0)

    def test_invalid_ground_truth_interval_is_rejected(self):
        gt = [{
            "query_id": "q",
            "task_type": "kis",
            "ground_truth": {"video_id": "V", "valid_frame_intervals": [[20, 10]]},
        }]
        with self.assertRaises(EvaluationInputError):
            evaluate(gt, [{"query_id": "q"}], "TEST")

    def test_jsonl_loader_rejects_duplicate_query_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "dup.jsonl"
            path.write_text(
                json.dumps({"query_id": "same"}) + "\n" + json.dumps({"query_id": "same"}) + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(EvaluationInputError):
                _load_jsonl(path)


if __name__ == "__main__":
    unittest.main()

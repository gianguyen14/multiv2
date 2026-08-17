import json
import tempfile
import unittest
from pathlib import Path

from evaluation.p0_run_current import build_request, normalize_prediction, run_records, write_predictions


class FakeSearch:
    def __init__(self):
        self.requests = []
        self.last_query_metrics = {"channel": "fake"}
        self.last_trake_metrics = {"dp_ms": 1.0}

    def handle(self, request):
        self.requests.append(request)
        if request["query_type"] == "trake":
            return [{"video_id": "V3", "frame_ids": [10, 20, 30]}]
        if request["query_type"] == "qa":
            return [{"video_id": "V2", "frame_id": 20, "answer": "red"}]
        return [{"video_id": "V1", "frame_id": 10}]

    def readiness(self):
        return {"ready": True, "generation_id": "gen-test"}

    def status(self):
        return {"configured": True, "initialized": True}


def records():
    return [
        {"query_id": "k", "category_id": "C01", "task_type": "kis", "query": "q"},
        {"query_id": "q", "category_id": "C16", "task_type": "qa", "query": "question"},
        {"query_id": "t", "category_id": "C13", "task_type": "trake", "events": ["a", "b", "c"]},
    ]


class P0CurrentRunnerTests(unittest.TestCase):
    def test_build_request_routes_task_specific_fields(self):
        kis = build_request(records()[0], top_k=20, query_refine=True, rerank=False, temporal_refine=True)
        self.assertEqual(kis, {"query_type": "kis", "top_k": 20, "query_refine": True, "rerank": False, "query": "q"})
        trake = build_request(records()[2], top_k=5, query_refine=False, rerank=True, temporal_refine=False)
        self.assertEqual(trake["events"], ["a", "b", "c"])
        self.assertFalse(trake["temporal_refine"])
        self.assertNotIn("query", trake)

    def test_normalize_prediction_matches_evaluator_contracts(self):
        kis = normalize_prediction(records()[0], [{"video_id": "V1", "frame_id": 10}], 1.0)
        self.assertEqual(kis["results"][0]["frame_id"], 10)

        qa = normalize_prediction(records()[1], [{"video_id": "V2", "frame_id": 20, "answer": "red"}], 2.0)
        self.assertEqual(qa["result"], {"video_id": "V2", "frame_id": 20, "answer": "red"})

        trake = normalize_prediction(records()[2], [{"video_id": "V3", "frame_ids": [10, 20, 30]}], 3.0)
        self.assertEqual(trake["result"], {"video_id": "V3", "frames": [10, 20, 30]})

    def test_run_records_preserves_baseline_flags_and_diagnostics(self):
        search = FakeSearch()
        predictions, manifest = run_records(records(), search, top_k=7, query_refine=False, rerank=False, temporal_refine=False)
        self.assertEqual(len(predictions), 3)
        self.assertEqual(manifest["task_counts"], {"kis": 1, "qa": 1, "trake": 1})
        self.assertEqual(manifest["top_k"], 7)
        self.assertFalse(manifest["query_refine"])
        self.assertFalse(manifest["rerank"])
        self.assertFalse(manifest["temporal_refine"])
        self.assertEqual(manifest["search_readiness"]["generation_id"], "gen-test")
        self.assertEqual(search.requests[2]["query_type"], "trake")
        self.assertEqual(predictions[2]["diagnostics"], {"dp_ms": 1.0})

    def test_write_predictions_is_jsonl(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "predictions.jsonl"
            write_predictions(path, [{"query_id": "x", "results": []}, {"query_id": "y", "results": []}])
            lines = path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 2)
            self.assertEqual(json.loads(lines[0])["query_id"], "x")
            self.assertEqual(json.loads(lines[1])["query_id"], "y")


if __name__ == "__main__":
    unittest.main()

import json

from eval.p0_run_current import build_request, run_records, write_predictions


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


def _records():
    return [
        {"query_id": "k", "category_id": "C01", "task_type": "kis", "query": "q"},
        {"query_id": "q", "category_id": "C16", "task_type": "qa", "query": "question"},
        {"query_id": "t", "category_id": "C13", "task_type": "trake", "events": ["a", "b", "c"]},
    ]


def test_build_request_routes_task_specific_fields():
    kis = build_request(_records()[0], top_k=20, query_refine=True, rerank=False, temporal_refine=True)
    assert kis == {"query_type": "kis", "top_k": 20, "query_refine": True, "rerank": False, "query": "q"}
    trake = build_request(_records()[2], top_k=5, query_refine=False, rerank=True, temporal_refine=False)
    assert trake["events"] == ["a", "b", "c"]
    assert trake["temporal_refine"] is False
    assert "query" not in trake


def test_run_records_preserves_baseline_flags_and_results():
    search = FakeSearch()
    predictions, manifest = run_records(_records(), search, top_k=7, query_refine=False, rerank=False, temporal_refine=False)
    assert len(predictions) == 3
    assert manifest["task_counts"] == {"kis": 1, "qa": 1, "trake": 1}
    assert manifest["top_k"] == 7
    assert manifest["query_refine"] is False
    assert manifest["rerank"] is False
    assert manifest["temporal_refine"] is False
    assert manifest["search_readiness"]["generation_id"] == "gen-test"
    assert search.requests[2]["query_type"] == "trake"
    assert predictions[2]["diagnostics"] == {"dp_ms": 1.0}


def test_write_predictions_is_jsonl(tmp_path):
    destination = tmp_path / "predictions.jsonl"
    write_predictions(destination, [{"query_id": "x", "results": []}, {"query_id": "y", "results": []}])
    lines = destination.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["query_id"] == "x"
    assert json.loads(lines[1])["query_id"] == "y"

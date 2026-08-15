import json

import pytest

from backend.app.competition_data import load_jsonl


def test_loads_internal_ground_truth_contracts(tmp_path):
    values = {
        "kis": {"query_id": "k", "query": "event", "video_id": "v", "start_frame": 1, "end_frame": 2},
        "qa": {"query_id": "q", "query": "question", "video_id": "v", "start_frame": 1, "end_frame": 2, "answers": ["blue"]},
        "trake": {"query_id": "t", "events": ["one"], "video_id": "v", "windows": [{"start_frame": 1, "end_frame": 2}]},
    }
    for kind, value in values.items():
        path = tmp_path / f"{kind}.jsonl"
        path.write_text(json.dumps(value) + "\n")
        assert load_jsonl(path, kind)[0].query_id == value["query_id"]


def test_rejects_invalid_intervals_and_event_alignment(tmp_path):
    path = tmp_path / "kis.jsonl"
    path.write_text('{"query_id":"k","query":"x","video_id":"v","start_frame":2,"end_frame":1}\n')
    with pytest.raises(ValueError, match="interval"):
        load_jsonl(path, "kis")

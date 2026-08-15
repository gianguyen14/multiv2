import json

from backend.app.competition_evaluation import evaluate_competition


def test_competition_evaluator_reports_all_cutoffs(tmp_path):
    (tmp_path / "kis.jsonl").write_text(json.dumps({"query_id": "k", "query": "x", "video_id": "v", "start_frame": 1, "end_frame": 2}) + "\n")
    (tmp_path / "qa.jsonl").write_text(json.dumps({"query_id": "q", "query": "x", "video_id": "v", "start_frame": 1, "end_frame": 2, "answers": ["blue"]}) + "\n")
    (tmp_path / "trake.jsonl").write_text(json.dumps({"query_id": "t", "events": ["x"], "video_id": "v", "windows": [{"start_frame": 1, "end_frame": 2}]}) + "\n")
    runners = {
        "kis": lambda record: [{"video_id": "v", "frame_id": 1}],
        "qa": lambda record: [{"video_id": "v", "frame_id": 1, "answer": "BLUE"}],
        "trake": lambda record: [{"video_id": "v", "frame_ids": [2]}],
    }
    report = evaluate_competition(tmp_path, runners)
    assert report["final_score"] == 1
    assert all(report[kind][f"r@{cutoff}"] == 1 for kind in runners for cutoff in (1, 5, 20, 50, 100))

from backend.app.retrieval.competition_scoring import competition_score, diversify, kis_r_score, qa_r_score, trake_r_score


def test_competition_cutoff_score_rewards_early_hits():
    truth = {"video_id": "v", "start_frame": 10, "end_frame": 12}
    submissions = [{"video_id": "wrong", "frame_id": 0}] * 4 + [{"video_id": "v", "frame_id": 11}]
    score = competition_score(submissions, lambda item: kis_r_score(item, truth))
    assert score == {"final_score": 0.8, "r@1": 0.0, "r@5": 1.0, "r@20": 1.0, "r@50": 1.0, "r@100": 1.0}


def test_qa_and_trake_scores_apply_full_conditions():
    qa_truth = {"video_id": "v", "start_frame": 1, "end_frame": 2, "answer": "HCMC"}
    match = lambda actual, expected: actual.casefold() == expected.casefold()
    assert qa_r_score({"video_id": "v", "frame_id": 2, "answer": "hcmc"}, qa_truth, match) == 1
    trake_truth = {"video_id": "v", "event_windows": [(1, 2), (10, 12), (20, 21)]}
    assert trake_r_score({"video_id": "v", "frame_ids": [1, 9, 21]}, trake_truth) == 2 / 3
    assert trake_r_score({"video_id": "wrong", "frame_ids": [1, 11, 21]}, trake_truth) == 0


def test_diversification_removes_temporal_duplicates_and_caps_videos():
    candidates = [
        {"video_id": "a", "frame_id": 100}, {"video_id": "a", "frame_id": 101},
        {"video_id": "a", "frame_id": 150}, {"video_id": "b", "frame_id": 2},
    ]
    assert diversify(candidates, temporal_gap=5, per_video_limit=1) == [
        {"video_id": "a", "frame_id": 100}, {"video_id": "b", "frame_id": 2}]

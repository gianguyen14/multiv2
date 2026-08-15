CUTOFFS = (1, 5, 20, 50, 100)


def kis_r_score(submission, ground_truth):
    return float(submission["video_id"] == ground_truth["video_id"]
        and ground_truth["start_frame"] <= submission["frame_id"] <= ground_truth["end_frame"])


def qa_r_score(submission, ground_truth, answer_match):
    return kis_r_score(submission, ground_truth) * float(answer_match(submission["answer"], ground_truth["answer"]))


def trake_r_score(submission, ground_truth):
    if submission["video_id"] != ground_truth["video_id"]:
        return 0.0
    windows = ground_truth["event_windows"]
    frames = submission["frame_ids"]
    return sum(start <= frame <= end for frame, (start, end) in zip(frames, windows)) / len(windows) if windows else 0.0


def competition_score(submissions, scorer, cutoffs=CUTOFFS):
    values = []
    for cutoff in cutoffs:
        values.append(max((scorer(item) for item in submissions[:cutoff]), default=0.0))
    return {"final_score": sum(values) / len(values),
        **{f"r@{cutoff}": value for cutoff, value in zip(cutoffs, values)}}


def diversify(candidates, max_results=100, temporal_gap=10, per_video_limit=None):
    selected, counts = [], {}
    for candidate in candidates:
        video_id = candidate["video_id"]
        frame_id = candidate["frame_id"]
        if per_video_limit is not None and counts.get(video_id, 0) >= per_video_limit:
            continue
        if any(item["video_id"] == video_id and abs(item["frame_id"] - frame_id) <= temporal_gap
                for item in selected):
            continue
        selected.append(candidate)
        counts[video_id] = counts.get(video_id, 0) + 1
        if len(selected) == max_results:
            break
    return selected

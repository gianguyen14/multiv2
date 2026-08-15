from pathlib import Path

from backend.app.competition_data import load_jsonl
from backend.app.retrieval.competition_scoring import competition_score, kis_r_score, qa_r_score, trake_r_score
from backend.app.video.text_evidence import normalize_text


def evaluate_competition(ground_truth_root, runners):
    root = Path(ground_truth_root)
    paths = {kind: root / f"{kind}.jsonl" for kind in ("kis", "qa", "trake")}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing competition ground truth: " + ", ".join(missing))
    reports = {}
    for kind, path in paths.items():
        records = load_jsonl(path, kind)
        query_reports = []
        for record in records:
            submissions = runners[kind](record)
            if kind == "kis":
                truth = {"video_id": record.video_id, "start_frame": record.start_frame, "end_frame": record.end_frame}
                scorer = lambda item, truth=truth: kis_r_score(item, truth)
            elif kind == "qa":
                truth = {"video_id": record.video_id, "start_frame": record.start_frame,
                    "end_frame": record.end_frame, "answers": record.answers}
                scorer = lambda item, truth=truth: kis_r_score(item, truth) * float(
                    normalize_text(item.get("answer", "")) in {normalize_text(value) for value in truth["answers"]})
            else:
                truth = {"video_id": record.video_id,
                    "event_windows": [(value["start_frame"], value["end_frame"]) for value in record.windows]}
                scorer = lambda item, truth=truth: trake_r_score(item, truth)
            query_reports.append(competition_score(submissions, scorer))
        reports[kind] = _mean_report(query_reports)
    reports["final_score"] = sum(value["final_score"] for value in reports.values()) / len(reports)
    return reports


def _mean_report(reports):
    if not reports:
        return {key: 0.0 for key in ("final_score", "r@1", "r@5", "r@20", "r@50", "r@100")}
    return {key: sum(report[key] for report in reports) / len(reports) for key in reports[0]}

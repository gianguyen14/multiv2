"""Validation helpers for human-authored AIC ground-truth annotations."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ValidationReport:
    is_valid: bool = True
    total_queries: int = 0
    labeled_queries: int = 0
    unlabeled_queries: int = 0
    video_level_labeled: int = 0
    frame_level_labeled: int = 0
    qa_answers_labeled: int = 0
    errors: list[str] = field(default_factory=list)


def validate_ground_truth_file(
    path: str | Path, known_video_ids: Iterable[str] | None = None
) -> ValidationReport:
    report = ValidationReport()
    try:
        rows = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        report.errors.append(f"Unable to read ground truth: {type(exc).__name__}")
        report.is_valid = False
        return report
    if not isinstance(rows, list):
        report.errors.append("Ground truth root must be a list")
        report.is_valid = False
        return report

    known = {str(video_id) for video_id in known_video_ids} if known_video_ids else None
    seen_ids = set()
    report.total_queries = len(rows)
    for row_number, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            report.errors.append(f"Query #{row_number} is not an object")
            continue
        query_id = str(row.get("id", "")).strip()
        if not query_id:
            report.errors.append(f"Query #{row_number} has an empty query ID")
        elif query_id in seen_ids:
            report.errors.append(f"Duplicate query ID: {query_id}")
        seen_ids.add(query_id)

        ground_truth = row.get("ground_truth", {})
        if not isinstance(ground_truth, dict):
            report.errors.append(
                f"Query {query_id or row_number} ground_truth is malformed"
            )
            ground_truth = {}

        video_ids = ground_truth.get("video_ids", [])
        if not isinstance(video_ids, list):
            report.errors.append(
                f"Query {query_id or row_number} video_ids is malformed"
            )
            video_ids = []
        valid_video_ids = []
        for video_id in video_ids:
            if not isinstance(video_id, str) or not video_id.strip():
                report.errors.append(
                    f"Query {query_id or row_number} has a malformed video ID"
                )
                continue
            valid_video_ids.append(video_id)
            if known is not None and video_id not in known:
                report.errors.append(
                    "Query "
                    f"{query_id or row_number} references unknown video {video_id}"
                )

        ranges = ground_truth.get("frame_ranges", [])
        if not isinstance(ranges, list):
            report.errors.append(
                f"Query {query_id or row_number} frame_ranges is malformed"
            )
            ranges = []
        valid_ranges = []
        for range_index, frame_range in enumerate(ranges, start=1):
            label = f"Query {query_id or row_number} frame range #{range_index}"
            if not isinstance(frame_range, (list, tuple)) or len(frame_range) != 2:
                report.errors.append(f"{label} is malformed")
                continue
            start_frame, end_frame = frame_range
            if (
                not isinstance(start_frame, int)
                or isinstance(start_frame, bool)
                or not isinstance(end_frame, int)
                or isinstance(end_frame, bool)
            ):
                report.errors.append(f"{label} contains a non-integer frame index")
                continue
            if start_frame < 0 or end_frame < 0:
                report.errors.append(f"{label} contains a negative frame index")
                continue
            if end_frame < start_frame:
                report.errors.append(
                    f"{label} end_frame ({end_frame}) < start_frame ({start_frame})"
                )
                continue
            valid_ranges.append((start_frame, end_frame))

        answers = ground_truth.get("accepted_answers", [])
        if not isinstance(answers, list):
            report.errors.append(
                f"Query {query_id or row_number} accepted_answers is malformed"
            )
            answers = []
        valid_answers = [
            answer for answer in answers if isinstance(answer, str) and answer.strip()
        ]
        if valid_video_ids:
            report.video_level_labeled += 1
        if valid_ranges:
            report.frame_level_labeled += 1
        if valid_answers:
            report.qa_answers_labeled += 1
        if valid_video_ids or valid_ranges or valid_answers:
            report.labeled_queries += 1
        else:
            report.unlabeled_queries += 1

    report.is_valid = not report.errors
    return report

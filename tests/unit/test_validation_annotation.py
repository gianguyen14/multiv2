import json
from pathlib import Path
import pytest

from tools.prepare_validation_annotation import validate_ground_truth_file


def test_validate_ground_truth_template(tmp_path):
    valid_data = [
        {
            "id": "Q_KIS_01",
            "type": "kis",
            "query": "nguoi di xe may",
            "ground_truth": {
                "video_ids": ["L22_V001"],
                "frame_ranges": [[100, 200]]
            }
        },
        {
            "id": "Q_QA_01",
            "type": "qa",
            "query": "xe mau gi?",
            "ground_truth": {
                "video_ids": ["L22_V002"],
                "frame_ranges": [[500, 600]],
                "accepted_answers": ["mau do", "do"]
            }
        }
    ]
    gt_file = tmp_path / "valid_gt.json"
    gt_file.write_text(json.dumps(valid_data), encoding="utf-8")

    report = validate_ground_truth_file(gt_file, known_video_ids={"L22_V001", "L22_V002"})
    assert report.is_valid is True
    assert report.total_queries == 2
    assert report.labeled_queries == 2
    assert report.unlabeled_queries == 0
    assert report.video_level_labeled == 2
    assert report.frame_level_labeled == 2
    assert report.qa_answers_labeled == 1
    assert len(report.errors) == 0


def test_reject_duplicate_ids(tmp_path):
    dup_data = [
        {"id": "Q1", "type": "kis", "query": "test 1", "ground_truth": {}},
        {"id": "Q1", "type": "kis", "query": "test 2", "ground_truth": {}},
    ]
    gt_file = tmp_path / "dup_gt.json"
    gt_file.write_text(json.dumps(dup_data), encoding="utf-8")

    report = validate_ground_truth_file(gt_file)
    assert report.is_valid is False
    assert any("Duplicate query ID" in err for err in report.errors)


def test_reject_unknown_video_id(tmp_path):
    invalid_vid_data = [
        {
            "id": "Q1",
            "type": "kis",
            "query": "test",
            "ground_truth": {
                "video_ids": ["NONEXISTENT_VIDEO_999"]
            }
        }
    ]
    gt_file = tmp_path / "invalid_vid.json"
    gt_file.write_text(json.dumps(invalid_vid_data), encoding="utf-8")

    report = validate_ground_truth_file(gt_file, known_video_ids={"L22_V001", "L22_V002"})
    assert report.is_valid is False
    assert any("NONEXISTENT_VIDEO_999" in err for err in report.errors)


def test_reject_malformed_and_inverted_frame_ranges(tmp_path):
    bad_ranges_data = [
        {
            "id": "Q1",
            "type": "kis",
            "query": "test",
            "ground_truth": {
                "video_ids": ["L22_V001"],
                "frame_ranges": [
                    [500, 400],      # end < start
                    [-10, 100],      # negative start
                    [100, 200, 300], # length != 2
                    ["abc", 200],    # non-int
                ]
            }
        }
    ]
    gt_file = tmp_path / "bad_ranges.json"
    gt_file.write_text(json.dumps(bad_ranges_data), encoding="utf-8")

    report = validate_ground_truth_file(gt_file)
    assert report.is_valid is False
    assert len(report.errors) >= 4
    assert any("end_frame (400) < start_frame (500)" in err for err in report.errors)
    assert any("negative frame index" in err for err in report.errors)
    assert any("malformed" in err for err in report.errors)
    assert any("non-integer" in err for err in report.errors)


def test_unlabeled_template_validation(tmp_path):
    unlabeled_data = [
        {
            "id": "Q_KIS_01",
            "type": "kis",
            "query": "unlabeled query",
            "ground_truth": {
                "video_ids": [],
                "frame_ranges": []
            }
        }
    ]
    gt_file = tmp_path / "unlabeled.json"
    gt_file.write_text(json.dumps(unlabeled_data), encoding="utf-8")

    report = validate_ground_truth_file(gt_file)
    assert report.is_valid is True
    assert report.total_queries == 1
    assert report.labeled_queries == 0
    assert report.unlabeled_queries == 1

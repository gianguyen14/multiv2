import pytest

def apply_temporal_nms(results, min_distance=90, top_k=100):
    selected = []
    selected_by_vid = {}
    for r in results:
        vid = r["video_id"]
        fid = r["source_frame_index_zero_based"]
        too_close = False
        if vid in selected_by_vid:
            for sf in selected_by_vid[vid]:
                if abs(fid - sf) < min_distance:
                    too_close = True
                    break
        if not too_close:
            selected.append(r)
            selected_by_vid.setdefault(vid, []).append(fid)
            if len(selected) >= top_k:
                break
    return selected

def test_temporal_nms_dedup():
    rows = [
        {"video_id": "L22_V001", "source_frame_index_zero_based": 1000, "score": 1.0},
        {"video_id": "L22_V001", "source_frame_index_zero_based": 1030, "score": 0.95}, # within 90 -> suppress
        {"video_id": "L22_V001", "source_frame_index_zero_based": 1060, "score": 0.90}, # within 90 -> suppress
        {"video_id": "L22_V001", "source_frame_index_zero_based": 1200, "score": 0.85}, # gap 200 -> keep
        {"video_id": "L22_V002", "source_frame_index_zero_based": 1010, "score": 0.80}, # diff video -> keep
    ]
    out = apply_temporal_nms(rows, min_distance=90)
    assert len(out) == 3
    assert [r["source_frame_index_zero_based"] for r in out] == [1000, 1200, 1010]

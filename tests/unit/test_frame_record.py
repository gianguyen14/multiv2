from backend.app.video.frame_record import FrameRecord
import pytest


def test_frame_uid_is_deterministic_and_roundtrips():
    record = FrameRecord.create(video_id="L01_V001", source_frame_index_zero_based=1534,
        submission_frame_id=1534, timestamp_seconds=51.133, pts=1534,
        width=320, height=240, image_path="frame.jpg", sample_interval_seconds=1.0,
        ingestion_version="m15-v1")
    assert record.frame_uid == "L01_V001:000001534"
    assert FrameRecord.from_dict(record.to_dict()) == record


@pytest.mark.parametrize(
    "updates,error",
    [
        ({"sampling_reason": "periodic", "shot_id": 0}, "shot_id=None"),
        ({"sampling_reason": "shot", "shot_id": None}, "non-negative integer"),
        ({"sampling_reason": "unknown"}, "invalid sampling_reason"),
        ({"video_id": ""}, "video_id must be a non-empty string"),
        ({"frame_uid": "wrong"}, "canonical provenance"),
        ({"timestamp_seconds": float("nan")}, "non-negative and finite"),
    ],
)
def test_deserialization_rejects_invalid_provenance(updates, error):
    record = FrameRecord.create(
        video_id="L01_V001",
        source_frame_index_zero_based=1,
        submission_frame_id=1,
        timestamp_seconds=0.1,
        pts=1,
        width=320,
        height=240,
        image_path="frame.jpg",
        sample_interval_seconds=1.0,
        ingestion_version="m15-v1",
    )
    payload = {**record.to_dict(), **updates}

    with pytest.raises(ValueError, match=error):
        FrameRecord.from_dict(payload)

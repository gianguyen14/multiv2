from backend.app.video.frame_record import FrameRecord


def test_frame_uid_is_deterministic_and_roundtrips():
    record = FrameRecord.create(video_id="L01_V001", source_frame_index_zero_based=1534,
        submission_frame_id=1534, timestamp_seconds=51.133, pts=1534,
        width=320, height=240, image_path="frame.jpg", sample_interval_seconds=1.0,
        ingestion_version="m15-v1")
    assert record.frame_uid == "L01_V001:000001534"
    assert FrameRecord.from_dict(record.to_dict()) == record

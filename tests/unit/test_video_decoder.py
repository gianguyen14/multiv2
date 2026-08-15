from pathlib import Path

from backend.app.video.video_decoder import decode_frame_indices, inspect_video, iter_frames


FIXTURE = Path("tests/fixtures/test_5s.mp4")


def test_inspects_and_decodes_cfr_fixture_in_display_order():
    metadata = inspect_video(FIXTURE)
    frames = list(iter_frames(FIXTURE))
    assert metadata.video_id == "test_5s"
    assert metadata.reported_frame_count == 125
    assert [frame.source_frame_index_zero_based for frame in frames] == list(range(125))
    assert all(a.timestamp_seconds <= b.timestamp_seconds for a, b in zip(frames, frames[1:]))


def test_sequential_roundtrip_beginning_middle_and_end():
    frames = decode_frame_indices(FIXTURE, [0, 62, 124])
    assert [frame.source_frame_index_zero_based for frame in frames] == [0, 62, 124]
    assert all(frame.image.size == (320, 240) for frame in frames)

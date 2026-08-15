from PIL import Image

from backend.app.video.frame_sampler import iter_sample_frames, sample_frames
from backend.app.video.video_decoder import DecodedFrame


def frame(index, timestamp, pts=None):
    return DecodedFrame(index, index if pts is None else pts, timestamp, 2, 2, Image.new("RGB", (2, 2)))


def test_irregular_timestamps_preserve_original_source_indices():
    frames = [frame(0, 0.0), frame(1, 0.4), frame(2, 1.4), frame(3, 1.6), frame(4, 3.2)]
    sampled = sample_frames(frames, 1.0)
    assert [item.frame.source_frame_index_zero_based for item in sampled] == [0, 2, 3, 4]
    assert [item.frame.pts for item in sampled] == [0, 2, 3, 4]


def test_streaming_sampler_does_not_consume_ahead():
    consumed = []
    def frames():
        for index in range(100):
            consumed.append(index)
            yield frame(index, index * 0.1)
    sampled = iter_sample_frames(frames(), 1.0)
    first = next(sampled)
    assert first.frame.source_frame_index_zero_based == 0
    assert consumed == [0]
    second = next(sampled)
    assert second.frame.source_frame_index_zero_based == 10
    assert len(consumed) <= 12


def test_missing_pts_is_not_reconstructed_from_fps():
    frames = [frame(90, None, None), frame(777, 30.033, 700), frame(778, 30.067, 701)]
    sampled = sample_frames(frames, 30.0)
    assert sampled[0].frame.source_frame_index_zero_based == 777
    assert sampled[0].frame.source_frame_index_zero_based != round(sampled[0].frame.timestamp_seconds * 30)

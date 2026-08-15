from backend.app.video.video_decoder import decode_frame_indices, iter_frames


def get_frame_neighborhood(source_path, center_frame_id, radius_frames):
    if radius_frames < 0:
        raise ValueError("radius must be non-negative")
    start = max(0, center_frame_id - radius_frames)
    return decode_frame_indices(source_path, range(start, center_frame_id + radius_frames + 1))


def get_temporal_neighborhood(source_path, center_timestamp, before_seconds, after_seconds):
    if before_seconds < 0 or after_seconds < 0:
        raise ValueError("temporal bounds must be non-negative")
    start = center_timestamp - before_seconds
    end = center_timestamp + after_seconds
    return [frame for frame in iter_frames(source_path)
            if frame.timestamp_seconds is not None and start <= frame.timestamp_seconds <= end]

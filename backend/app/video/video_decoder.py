import json
import subprocess
from dataclasses import dataclass, replace
from fractions import Fraction
from pathlib import Path
from typing import Iterator, Optional

from PIL import Image

from backend.app.video.video_metadata import VideoMetadata


class VideoDecodeError(RuntimeError):
    pass


@dataclass(frozen=True)
class DecodedFrame:
    source_frame_index_zero_based: int
    pts: Optional[int]
    timestamp_seconds: Optional[float]
    width: int
    height: int
    image: Image.Image


def _optional_int(value):
    if value in (None, "", "N/A"):
        return None
    return int(value)


def _optional_float(value):
    if value in (None, "", "N/A"):
        return None
    return float(value)


def inspect_video(path, ingestion_version="m15-v1") -> VideoMetadata:
    path = Path(path)
    if not path.is_file():
        raise VideoDecodeError(f"video does not exist: {path}")
    command = ["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        payload = json.loads(result.stdout)
    except (OSError, subprocess.CalledProcessError, json.JSONDecodeError) as exc:
        raise VideoDecodeError(f"unable to inspect video: {path}") from exc
    stream = next((item for item in payload.get("streams", []) if item.get("codec_type") == "video"), None)
    if stream is None:
        raise VideoDecodeError("video stream is missing")
    avg = stream.get("avg_frame_rate")
    real = stream.get("r_frame_rate")
    vfr = None if not avg or not real else avg != real
    duration = _optional_float(stream.get("duration"))
    if duration is None:
        duration = _optional_float(payload.get("format", {}).get("duration"))
    return VideoMetadata(
        video_id=path.stem,
        source_path=str(path.resolve()),
        filename=path.name,
        width=int(stream.get("width") or 0),
        height=int(stream.get("height") or 0),
        duration_seconds=duration,
        avg_frame_rate=avg if avg not in (None, "0/0") else None,
        real_frame_rate=real if real not in (None, "0/0") else None,
        reported_frame_count=_optional_int(stream.get("nb_frames")),
        decoded_frame_count=None,
        time_base=stream.get("time_base"),
        start_time=_optional_float(stream.get("start_time", payload.get("format", {}).get("start_time"))),
        codec_name=stream.get("codec_name"),
        pixel_format=stream.get("pix_fmt"),
        variable_frame_rate_detected=vfr,
        ingestion_version=ingestion_version,
    )


def iter_frames(path) -> Iterator[DecodedFrame]:
    try:
        import av
    except ImportError as exc:
        raise VideoDecodeError("PyAV is required for video decoding") from exc
    try:
        with av.open(str(path)) as container:
            stream = next((item for item in container.streams if item.type == "video"), None)
            if stream is None:
                raise VideoDecodeError("video stream is missing")
            for index, frame in enumerate(container.decode(stream)):
                time_base = frame.time_base or stream.time_base
                timestamp = None
                if frame.pts is not None and time_base is not None:
                    timestamp = float(Fraction(frame.pts) * Fraction(time_base))
                yield DecodedFrame(index, frame.pts, timestamp, frame.width, frame.height, frame.to_image().convert("RGB"))
    except VideoDecodeError:
        raise
    except Exception as exc:
        raise VideoDecodeError(f"unable to decode video: {path}") from exc


def inspect_and_count_video(path, ingestion_version="m15-v1"):
    metadata = inspect_video(path, ingestion_version)
    count = sum(1 for _ in iter_frames(path))
    return replace(metadata, decoded_frame_count=count)


def decode_frame_indices(path, indices):
    targets = sorted(set(indices))
    if any(index < 0 for index in targets):
        raise ValueError("frame indices must be non-negative")
    wanted = set(targets)
    output = {}
    for frame in iter_frames(path):
        if frame.source_frame_index_zero_based in wanted:
            output[frame.source_frame_index_zero_based] = frame
        if targets and frame.source_frame_index_zero_based >= targets[-1]:
            break
    return [output[index] for index in targets if index in output]

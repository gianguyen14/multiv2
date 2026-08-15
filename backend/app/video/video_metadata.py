from dataclasses import asdict, dataclass
from typing import Optional


@dataclass(frozen=True)
class VideoMetadata:
    video_id: str
    source_path: str
    filename: str
    width: int
    height: int
    duration_seconds: Optional[float]
    avg_frame_rate: Optional[str]
    real_frame_rate: Optional[str]
    reported_frame_count: Optional[int]
    decoded_frame_count: Optional[int]
    time_base: Optional[str]
    start_time: Optional[float]
    codec_name: Optional[str]
    pixel_format: Optional[str]
    variable_frame_rate_detected: Optional[bool]
    ingestion_version: str

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        return cls(**value)

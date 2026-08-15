from dataclasses import asdict, dataclass
from typing import Optional


@dataclass(frozen=True)
class FrameRecord:
    frame_uid: str
    video_id: str
    source_frame_index_zero_based: int
    submission_frame_id: int
    timestamp_seconds: Optional[float]
    pts: Optional[int]
    width: int
    height: int
    image_path: str
    embedding_id: str
    sample_strategy: str
    sample_interval_seconds: float
    ingestion_version: str

    @classmethod
    def create(cls, *, video_id, source_frame_index_zero_based, submission_frame_id,
               timestamp_seconds, pts, width, height, image_path,
               sample_interval_seconds, ingestion_version):
        if source_frame_index_zero_based < 0:
            raise ValueError("source frame index must be non-negative")
        frame_uid = f"{video_id}:{source_frame_index_zero_based:09d}"
        return cls(frame_uid, video_id, source_frame_index_zero_based,
                   submission_frame_id, timestamp_seconds, pts, width, height,
                   str(image_path), frame_uid, "temporal_coarse",
                   sample_interval_seconds, ingestion_version)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        return cls(**value)

import math
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
    shot_id: Optional[int] = None
    sampling_reason: Optional[str] = "periodic"

    def __post_init__(self):
        if not isinstance(self.video_id, str) or not self.video_id.strip():
            raise ValueError("video_id must be a non-empty string")
        if (
            not isinstance(self.source_frame_index_zero_based, int)
            or isinstance(self.source_frame_index_zero_based, bool)
            or self.source_frame_index_zero_based < 0
        ):
            raise ValueError("source frame index must be a non-negative integer")
        expected_uid = f"{self.video_id}:{self.source_frame_index_zero_based:09d}"
        if self.frame_uid != expected_uid or self.embedding_id != expected_uid:
            raise ValueError("frame and embedding IDs must match canonical provenance")
        if self.timestamp_seconds is not None and (
            not isinstance(self.timestamp_seconds, (int, float))
            or isinstance(self.timestamp_seconds, bool)
            or not math.isfinite(self.timestamp_seconds)
            or self.timestamp_seconds < 0
        ):
            raise ValueError("timestamp_seconds must be non-negative and finite")
        if self.sampling_reason not in {None, "periodic", "shot", "periodic+shot"}:
            raise ValueError(f"invalid sampling_reason: {self.sampling_reason!r}")
        if self.shot_id is not None and (
            not isinstance(self.shot_id, int)
            or isinstance(self.shot_id, bool)
            or self.shot_id < 0
        ):
            raise ValueError(f"invalid shot_id: {self.shot_id!r}")
        if self.sampling_reason in {None, "periodic"} and self.shot_id is not None:
            raise ValueError(
                f"{self.sampling_reason or 'legacy'} sampling reason must have "
                f"shot_id=None, got {self.shot_id!r}"
            )
        if self.sampling_reason in {"shot", "periodic+shot"} and self.shot_id is None:
            raise ValueError(
                f"{self.sampling_reason} sampling reason must have a non-negative "
                f"integer shot_id, got {self.shot_id!r}"
            )

    @classmethod
    def create(cls, *, video_id, source_frame_index_zero_based, submission_frame_id,
               timestamp_seconds, pts, width, height, image_path,
               sample_interval_seconds, ingestion_version,
               shot_id: Optional[int] = None,
               sampling_reason: Optional[str] = "periodic"):
        frame_uid = f"{video_id}:{source_frame_index_zero_based:09d}"
        return cls(frame_uid, video_id, source_frame_index_zero_based,
                   submission_frame_id, timestamp_seconds, pts, width, height,
                   str(image_path), frame_uid, "temporal_coarse",
                   sample_interval_seconds, ingestion_version,
                   shot_id=shot_id, sampling_reason=sampling_reason)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        from dataclasses import fields
        valid_keys = {f.name for f in fields(cls)}
        data = {k: v for k, v in dict(value).items() if k in valid_keys}
        data.setdefault("shot_id", None)
        data.setdefault("sampling_reason", None)
        return cls(**data)

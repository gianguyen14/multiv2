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

    @classmethod
    def create(cls, *, video_id, source_frame_index_zero_based, submission_frame_id,
               timestamp_seconds, pts, width, height, image_path,
               sample_interval_seconds, ingestion_version,
               shot_id: Optional[int] = None,
               sampling_reason: Optional[str] = "periodic"):
        if source_frame_index_zero_based < 0:
            raise ValueError("source frame index must be non-negative")
        if sampling_reason is not None and sampling_reason not in {"periodic", "shot", "periodic+shot"}:
            raise ValueError(f"invalid sampling_reason: {sampling_reason!r}")
        if sampling_reason == "periodic":
            if shot_id is not None:
                raise ValueError(f"periodic sampling reason must have shot_id=None, got {shot_id!r}")
        elif sampling_reason in {"shot", "periodic+shot"}:
            if shot_id is None or not isinstance(shot_id, int) or isinstance(shot_id, bool) or shot_id < 0:
                raise ValueError(f"{sampling_reason} sampling reason must have a non-negative integer shot_id, got {shot_id!r}")
        if shot_id is not None:
            if not isinstance(shot_id, int) or isinstance(shot_id, bool) or shot_id < 0:
                raise ValueError(f"invalid shot_id: {shot_id!r}")
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

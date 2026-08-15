from dataclasses import asdict, dataclass, field
from typing import Optional


STATES = {"pending", "metadata_ready", "frames_ready", "embeddings_ready", "indexed", "failed"}
STAGES = {None, "metadata", "frames", "embeddings"}


@dataclass(frozen=True)
class IngestManifest:
    video_id: str
    source_path: str
    source_size: int
    source_mtime_ns: int
    source_hash: str
    ingestion_version: str
    schema_version: int = 2
    status: str = "pending"
    completed_stage: Optional[str] = None
    metadata_fingerprint: Optional[str] = None
    frames_fingerprint: Optional[str] = None
    embeddings_fingerprint: Optional[str] = None
    encoder_identity: dict = field(default_factory=dict)
    decoded_frame_count: Optional[int] = None
    sampled_frame_count: int = 0
    embedding_count: int = 0
    embedding_dim: Optional[int] = None
    failed_stage: Optional[str] = None
    error: Optional[str] = None

    def __post_init__(self):
        if self.status not in STATES or self.completed_stage not in STAGES:
            raise ValueError("invalid ingestion state")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        value = dict(value)
        if "schema_version" not in value:
            status = value.get("status", "pending")
            stage = {"metadata_ready": "metadata", "frames_ready": "frames",
                     "embeddings_ready": "embeddings", "indexed": "embeddings"}.get(status)
            value = {
                "video_id": value["video_id"], "source_path": value["source_path"],
                "source_size": value["source_size"], "source_mtime_ns": value["source_mtime_ns"],
                "source_hash": value["source_hash"], "ingestion_version": value["ingestion_version"],
                "status": status, "completed_stage": stage,
                "sampled_frame_count": value.get("sampled_frame_count", 0),
                "error": value.get("error"),
            }
        return cls(**value)

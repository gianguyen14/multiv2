import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path


def _fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True)
class VideoIngestConfig:
    enabled: bool = False
    processed_root: Path = Path("data/processed/videos")
    sample_interval_seconds: float = 1.0
    frame_id_policy: str = "zero_based"
    frame_format: str = "jpg"
    jpeg_quality: int = 90
    embed_batch_size: int | None = None
    device: str = "auto"
    resume: bool = True
    ingestion_version: str = "m15-v1"
    index_type: str = "flat"

    def __post_init__(self):
        if self.sample_interval_seconds <= 0:
            raise ValueError("sample interval must be positive")
        if self.frame_id_policy not in {"zero_based", "one_based"}:
            raise ValueError("unsupported frame ID policy")
        if self.frame_format not in {"jpg", "webp", "png"}:
            raise ValueError("unsupported frame format")
        if not 1 <= self.jpeg_quality <= 100 or (self.embed_batch_size is not None and self.embed_batch_size <= 0):
            raise ValueError("invalid video ingest configuration")
        if self.index_type not in {"flat", "hnsw"}:
            raise ValueError("unsupported video index type")

    @classmethod
    def from_env(cls):
        return cls(
            enabled=os.getenv("VIDEO_INGEST_ENABLED", "false").lower() == "true",
            processed_root=Path(os.getenv("VIDEO_PROCESSED_ROOT", "data/processed/videos")),
            sample_interval_seconds=float(os.getenv("VIDEO_SAMPLE_INTERVAL_SECONDS", "1.0")),
            frame_id_policy=os.getenv("VIDEO_FRAME_ID_POLICY", "zero_based"),
            frame_format=os.getenv("VIDEO_FRAME_FORMAT", "jpg"),
            jpeg_quality=int(os.getenv("VIDEO_JPEG_QUALITY", "90")),
            embed_batch_size=int(os.environ["VIDEO_EMBED_BATCH_SIZE"]) if "VIDEO_EMBED_BATCH_SIZE" in os.environ else None,
            device=os.getenv("VIDEO_INGEST_DEVICE", os.getenv("COMPUTE_DEVICE", "auto")),
            resume=os.getenv("VIDEO_RESUME", "true").lower() == "true",
            index_type=os.getenv("VIDEO_INDEX_TYPE", "flat"),
        )

    def metadata_fingerprint(self):
        return _fingerprint({"ingestion_version": self.ingestion_version, "metadata_schema": 2})

    def frames_fingerprint(self):
        return _fingerprint({
            "metadata_fingerprint": self.metadata_fingerprint(),
            "sample_interval_seconds": self.sample_interval_seconds,
            "frame_id_policy": self.frame_id_policy,
            "frame_format": self.frame_format,
            "jpeg_quality": self.jpeg_quality if self.frame_format in {"jpg", "webp"} else None,
            "frame_schema": 2,
            "sampling_policy": "nearest-observed-timestamp-earlier-tie-v1",
        })

    def embeddings_fingerprint(self, encoder_identity):
        return _fingerprint({
            "frames_fingerprint": self.frames_fingerprint(),
            "encoder_identity": encoder_identity,
            "dtype": "float32",
            "normalization": "l2",
            "embedding_schema": 2,
        })

    def index_fingerprint(self, corpus_fingerprint, embedding_dim):
        return _fingerprint({
            "corpus_fingerprint": corpus_fingerprint,
            "embedding_dim": embedding_dim,
            "index_type": self.index_type,
            "index_schema": 2,
            "payload_schema": 2,
        })

    def fingerprint(self):
        return self.frames_fingerprint()

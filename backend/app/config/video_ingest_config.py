import hashlib
import json
import math
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
    visual_sampling_mode: str = "legacy"
    visual_global_sample_seconds: float = 5.0
    visual_dedup_enabled: bool = False
    visual_dedup_threshold: float = 0.97

    @property
    def effective_sample_interval_seconds(self) -> float:
        """Return the active sample interval in seconds based on visual_sampling_mode."""
        if self.visual_sampling_mode == "sparse_shot":
            return self.visual_global_sample_seconds
        return self.sample_interval_seconds

    def __post_init__(self):
        if self.visual_sampling_mode not in {"legacy", "sparse_shot"}:
            raise ValueError(f"unsupported visual sampling mode: {self.visual_sampling_mode!r}")
        if (
            self.sample_interval_seconds is None
            or not isinstance(self.sample_interval_seconds, (int, float))
            or math.isnan(self.sample_interval_seconds)
            or math.isinf(self.sample_interval_seconds)
            or self.sample_interval_seconds <= 0
        ):
            raise ValueError("sample interval must be a positive and finite number")
        if (
            self.visual_global_sample_seconds is None
            or not isinstance(self.visual_global_sample_seconds, (int, float))
            or math.isnan(self.visual_global_sample_seconds)
            or math.isinf(self.visual_global_sample_seconds)
            or self.visual_global_sample_seconds <= 0
        ):
            raise ValueError("visual global sample seconds must be a positive and finite number")
        if (
            self.visual_dedup_threshold is None
            or not isinstance(self.visual_dedup_threshold, (int, float))
            or math.isnan(self.visual_dedup_threshold)
            or math.isinf(self.visual_dedup_threshold)
            or self.visual_dedup_threshold < 0.0
            or self.visual_dedup_threshold > 1.0
        ):
            raise ValueError("visual dedup threshold must be a finite number between 0.0 and 1.0")
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
        raw_global_sec = os.getenv("VISUAL_GLOBAL_SAMPLE_SECONDS", "5.0")
        try:
            visual_global_sample_seconds = float(raw_global_sec)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"invalid VISUAL_GLOBAL_SAMPLE_SECONDS: {raw_global_sec!r}") from exc

        raw_dedup_thresh = os.getenv("VISUAL_DEDUP_THRESHOLD", "0.97")
        try:
            visual_dedup_threshold = float(raw_dedup_thresh)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"invalid VISUAL_DEDUP_THRESHOLD: {raw_dedup_thresh!r}") from exc

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
            visual_sampling_mode=os.getenv("VISUAL_SAMPLING_MODE", "legacy").lower(),
            visual_global_sample_seconds=visual_global_sample_seconds,
            visual_dedup_enabled=os.getenv("VISUAL_DEDUP_ENABLED", "false").lower() in ("true", "1", "yes"),
            visual_dedup_threshold=visual_dedup_threshold,
        )

    def metadata_fingerprint(self):
        return _fingerprint({"ingestion_version": self.ingestion_version, "metadata_schema": 2})

    def frames_fingerprint(self):
        return _fingerprint({
            "metadata_fingerprint": self.metadata_fingerprint(),
            "sample_interval_seconds": self.effective_sample_interval_seconds,
            "visual_sampling_mode": self.visual_sampling_mode,
            "visual_dedup_enabled": self.visual_dedup_enabled,
            "visual_dedup_threshold": self.visual_dedup_threshold if self.visual_dedup_enabled else None,
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

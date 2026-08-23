import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from backend.app.config.video_ingest_config import VideoIngestConfig
from backend.app.shot_detection.base import get_shot_detector
from backend.app.video.frame_dedup import filter_near_duplicate_frames
from backend.app.video.frame_id_policy import FrameIdPolicy
from backend.app.video.frame_record import FrameRecord
from backend.app.video.frame_sampler import iter_sample_frames, sample_sparse_shot_frames_with_protection
from backend.app.video.frame_store import FrameStore, source_hash
from backend.app.video.ingest_manifest import IngestManifest
from backend.app.video.video_decoder import inspect_video, iter_frames


class VideoIngestionPipeline:
    def __init__(self, encoder, config=None, failpoint=None, shot_detector=None):
        self.config = config or VideoIngestConfig.from_env()
        self.encoder = encoder
        self.store = FrameStore(self.config.processed_root)
        self.policy = FrameIdPolicy(self.config.frame_id_policy)
        self.failpoint = failpoint or (lambda name, context: None)
        self.shot_detector = shot_detector if shot_detector is not None else (
            get_shot_detector() if self.config.visual_sampling_mode == "sparse_shot" else None
        )
        self._protected_indices_by_video: dict[str, set[int]] = {}

    def _encoder_identity(self):
        identity = self.encoder.identity() if hasattr(self.encoder, "identity") else {
            "provider": "python", "model_name": type(self.encoder).__name__,
            "embedding_dim": self.encoder.embedding_dim, "normalization": "l2",
            "contract_version": "m15.1-v1"}
        identity = dict(identity)
        if "embedding_dim" not in identity:
            identity["embedding_dim"] = self.encoder.embedding_dim
        return identity

    def ingest_video(self, path, force=False):
        path = Path(path)
        stat = path.stat()
        video_id = path.stem
        digest = source_hash(path)
        identity = self._encoder_identity()
        current = self.store.load_manifest(video_id)
        plan = self.store.plan_resume(current, digest, self.config, identity, force)
        base = IngestManifest(video_id, str(path.resolve()), stat.st_size, stat.st_mtime_ns,
            digest, self.config.ingestion_version, encoder_identity=identity)
        checkpoint = current if current and current.source_hash == digest else base
        started = time.perf_counter()
        metadata_ms = extraction_ms = embedding_ms = 0.0
        decoded_count = checkpoint.decoded_frame_count
        try:
            if plan.start_stage == "complete":
                metadata = self.store.load_metadata(video_id)
                records = self.store.load_records(video_id)
                if (current.status == "failed" or current.failed_stage is not None
                        or current.error is not None):
                    current = replace(current, status="embeddings_ready",
                        completed_stage="embeddings", failed_stage=None, error=None)
                    self.store.save_manifest(current)
                return self._result(metadata, records, plan, started, 0, 0, 0)
            if plan.start_stage == "metadata":
                stage_started = time.perf_counter()
                metadata = inspect_video(path, self.config.ingestion_version)
                self.store.save_metadata(metadata)
                metadata_ms = (time.perf_counter() - stage_started) * 1000
                checkpoint = replace(base, status="metadata_ready", completed_stage="metadata",
                    metadata_fingerprint=self.config.metadata_fingerprint(), failed_stage=None, error=None)
                self.store.save_manifest(checkpoint)
                self.failpoint("after_metadata", {"video_id": video_id})
            else:
                metadata = self.store.load_metadata(video_id)
            if plan.start_stage in {"metadata", "frames"}:
                stage_started = time.perf_counter()
                decoded_count = 0
                def counted_frames():
                    nonlocal decoded_count
                    for decoded_frame in iter_frames(path):
                        decoded_count = decoded_frame.source_frame_index_zero_based + 1
                        yield decoded_frame
                frame_dir = self.store.video_dir(video_id) / "frames"
                if frame_dir.exists():
                    for managed in frame_dir.glob("*.*"):
                        managed.unlink()
                records = []
                if self.config.visual_sampling_mode == "sparse_shot":
                    shot_boundaries = None
                    if self.shot_detector is not None:
                        try:
                            raw_shots = self.shot_detector.detect_shots(path)
                            shot_boundaries = [
                                (s / 1000.0, e / 1000.0)
                                for s, e in raw_shots
                                if isinstance(s, (int, float)) and isinstance(e, (int, float)) and e >= s and s >= 0
                            ]
                        except Exception:
                            shot_boundaries = None
                    sampled_items, protected_set = sample_sparse_shot_frames_with_protection(
                        counted_frames(),
                        self.config.effective_sample_interval_seconds,
                        shot_boundaries=shot_boundaries,
                    )
                    self._protected_indices_by_video[video_id] = protected_set
                    sampler_iter = iter(sampled_items)
                else:
                    self._protected_indices_by_video[video_id] = set()
                    sampler_iter = iter_sample_frames(
                        counted_frames(),
                        self.config.effective_sample_interval_seconds,
                    )

                for item in sampler_iter:
                    frame = item.frame
                    image_path = self.store.image_path(video_id, frame.source_frame_index_zero_based, self.config.frame_format)
                    self.store.save_image(image_path, frame.image, self.config.frame_format, self.config.jpeg_quality)
                    records.append(FrameRecord.create(video_id=video_id,
                        source_frame_index_zero_based=frame.source_frame_index_zero_based,
                        submission_frame_id=self.policy.to_submission_frame_id(frame.source_frame_index_zero_based),
                        timestamp_seconds=frame.timestamp_seconds, pts=frame.pts, width=frame.width,
                        height=frame.height, image_path=image_path,
                        sample_interval_seconds=self.config.effective_sample_interval_seconds,
                        ingestion_version=self.config.ingestion_version,
                        shot_id=getattr(item, "shot_id", None),
                        sampling_reason=getattr(item, "sampling_reason", "periodic")))
                metadata = replace(metadata, decoded_frame_count=decoded_count)
                self.store.save_metadata(metadata)
                self.store.save_records(video_id, records)
                extraction_ms = (time.perf_counter() - stage_started) * 1000
                checkpoint = replace(checkpoint, status="frames_ready", completed_stage="frames",
                    metadata_fingerprint=self.config.metadata_fingerprint(), frames_fingerprint=self.config.frames_fingerprint(),
                    decoded_frame_count=decoded_count, sampled_frame_count=len(records),
                    embeddings_fingerprint=None, embedding_count=0, embedding_dim=None,
                    failed_stage=None, error=None)
                self.store.save_manifest(checkpoint)
                self.failpoint("after_frames", {"video_id": video_id})
            else:
                records = self.store.load_records(video_id)
            if plan.start_stage in {"metadata", "frames", "embeddings"}:
                stage_started = time.perf_counter()
                embeddings = np.asarray(self.encoder.encode_image([record.image_path for record in records],
                    batch_size=self.config.embed_batch_size, normalize=True), dtype=np.float32)
                norms = np.linalg.norm(embeddings, axis=1) if embeddings.ndim == 2 else np.array([])
                if (embeddings.shape != (len(records), identity["embedding_dim"]) or not np.isfinite(embeddings).all()
                        or not np.allclose(norms, 1.0, atol=1e-5)):
                    raise ValueError("invalid frame embeddings")
                if self.config.visual_dedup_enabled:
                    protected = self._protected_indices_by_video.get(video_id, set())
                    records, embeddings, _ = filter_near_duplicate_frames(
                        records,
                        embeddings,
                        protected_source_frame_indices=protected,
                        threshold=self.config.visual_dedup_threshold,
                        enabled=True,
                    )
                    self.store.save_records(video_id, records)
                self.store.save_embeddings(video_id, embeddings)
                embedding_ms = (time.perf_counter() - stage_started) * 1000
                checkpoint = replace(checkpoint, status="embeddings_ready", completed_stage="embeddings",
                    embeddings_fingerprint=self.config.embeddings_fingerprint(identity), encoder_identity=identity,
                    sampled_frame_count=len(records),
                    embedding_count=len(records), embedding_dim=identity["embedding_dim"], failed_stage=None, error=None)
                self.store.save_manifest(checkpoint)
                self.failpoint("after_embeddings", {"video_id": video_id})
            return self._result(metadata, records, plan, started, metadata_ms, extraction_ms, embedding_ms)
        except Exception as exc:
            latest = self.store.load_manifest(video_id) or checkpoint
            self.store.save_manifest(replace(latest, status="failed", failed_stage={"metadata": "metadata", "frames": "frames", "embeddings": "embeddings", "complete": None}[plan.start_stage], error=f"{type(exc).__name__}: {exc}"))
            raise

    def _result(self, metadata, records, plan, started, metadata_ms, extraction_ms, embedding_ms):
        total_ms = (time.perf_counter() - started) * 1000
        return {"video_id": metadata.video_id, "status": "resumed" if plan.start_stage == "complete" else "embeddings_ready",
            "start_stage": plan.start_stage, "reused_metadata": plan.reused_metadata,
            "reused_frames": plan.reused_frames, "reused_embeddings": plan.reused_embeddings,
            "duration_seconds": metadata.duration_seconds, "reported_frame_count": metadata.reported_frame_count,
            "decoded_frame_count": metadata.decoded_frame_count, "sampled_frame_count": len(records),
            "metadata_ms": metadata_ms, "extraction_ms": extraction_ms, "embedding_ms": embedding_ms,
            "indexing_ms": 0.0, "total_ms": total_ms,
            "frames_per_second_processed": (metadata.decoded_frame_count or 0) / max(total_ms / 1000, 1e-12)}

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from backend.app.video.atomic_io import write_json_atomic, write_numpy_atomic
from backend.app.video.frame_record import FrameRecord
from backend.app.video.ingest_manifest import IngestManifest
from backend.app.video.video_metadata import VideoMetadata


def source_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class ResumePlan:
    start_stage: str
    reused_metadata: bool
    reused_frames: bool
    reused_embeddings: bool
    reason: str


class FrameStore:
    def __init__(self, root):
        self.root = Path(root)

    def video_dir(self, video_id):
        return self.root / video_id

    def image_path(self, video_id, source_index, extension):
        return self.video_dir(video_id) / "frames" / f"{source_index:09d}.{extension}"

    def save_image(self, path, image, frame_format="jpg", jpeg_quality=90):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.tmp")
        options = {"quality": jpeg_quality} if frame_format in {"jpg", "webp"} else {}
        image.save(temporary, format={"jpg": "JPEG", "webp": "WEBP", "png": "PNG"}[frame_format], **options)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)

    def save_metadata(self, metadata):
        write_json_atomic(self.video_dir(metadata.video_id) / "metadata.json", metadata.to_dict())

    def load_metadata(self, video_id):
        return VideoMetadata.from_dict(json.loads((self.video_dir(video_id) / "metadata.json").read_text()))

    def save_records(self, video_id, records):
        write_json_atomic(self.video_dir(video_id) / "frames.json", [record.to_dict() for record in records])

    def load_records(self, video_id):
        values = json.loads((self.video_dir(video_id) / "frames.json").read_text())
        return [FrameRecord.from_dict(value) for value in values]

    def save_embeddings(self, video_id, embeddings):
        write_numpy_atomic(self.video_dir(video_id) / "embeddings.npy", np.asarray(embeddings, dtype=np.float32))

    def load_embeddings(self, video_id):
        return np.load(self.video_dir(video_id) / "embeddings.npy", allow_pickle=False)

    def save_manifest(self, manifest):
        write_json_atomic(self.video_dir(manifest.video_id) / "manifest.json", manifest.to_dict())

    def load_manifest(self, video_id):
        path = self.video_dir(video_id) / "manifest.json"
        return IngestManifest.from_dict(json.loads(path.read_text())) if path.is_file() else None

    def manifests(self):
        for path in sorted(self.root.glob("*/manifest.json")):
            if path.parent.name == "index" or path.parent.name.startswith("."):
                continue
            try:
                yield IngestManifest.from_dict(json.loads(path.read_text()))
            except (ValueError, TypeError, KeyError, json.JSONDecodeError):
                continue

    def validate_metadata(self, manifest, fingerprint):
        try:
            metadata = self.load_metadata(manifest.video_id)
            return (manifest.metadata_fingerprint == fingerprint and metadata.video_id == manifest.video_id
                    and metadata.width > 0 and metadata.height > 0)
        except Exception:
            return False

    def validate_frames(self, manifest, fingerprint, policy, extension):
        if manifest.frames_fingerprint != fingerprint:
            return False
        try:
            records = self.load_records(manifest.video_id)
            indices = [record.source_frame_index_zero_based for record in records]
            if len(records) != manifest.sampled_frame_count or indices != sorted(set(indices)):
                return False
            for record in records:
                if record.video_id != manifest.video_id or record.frame_uid != f"{manifest.video_id}:{record.source_frame_index_zero_based:09d}":
                    return False
                if record.submission_frame_id != policy.to_submission_frame_id(record.source_frame_index_zero_based):
                    return False
                path = Path(record.image_path)
                if path.suffix != f".{extension}" or not path.is_file():
                    return False
                with Image.open(path) as image:
                    if image.size != (record.width, record.height):
                        return False
            return True
        except Exception:
            return False

    def validate_embeddings(self, manifest, fingerprint, embedding_dim, legacy_fingerprint=None):
        if manifest.embedding_dim != embedding_dim:
            return False
        if manifest.embeddings_fingerprint != fingerprint:
            if legacy_fingerprint is None or manifest.embeddings_fingerprint != legacy_fingerprint:
                return False
        try:
            records = self.load_records(manifest.video_id)
            values = self.load_embeddings(manifest.video_id)
            norms = np.linalg.norm(values, axis=1)
            return (values.dtype == np.float32 and values.ndim == 2 and values.shape == (len(records), embedding_dim)
                    and len(records) == manifest.embedding_count and np.isfinite(values).all()
                    and np.allclose(norms, 1.0, atol=1e-5))
        except Exception:
            return False

    def plan_resume(self, manifest, source_digest, config, encoder_identity, force=False):
        if force or not config.resume or manifest is None or manifest.source_hash != source_digest:
            return ResumePlan("metadata", False, False, False, "new, forced, or changed source")
        if not self.validate_metadata(manifest, config.metadata_fingerprint()):
            return ResumePlan("metadata", False, False, False, "metadata invalid")
        if not self.validate_frames(manifest, config.frames_fingerprint(), __import__("backend.app.video.frame_id_policy", fromlist=["FrameIdPolicy"]).FrameIdPolicy(config.frame_id_policy), config.frame_format):
            return ResumePlan("frames", True, False, False, "frames invalid or changed")
        fingerprint = config.embeddings_fingerprint(encoder_identity)
        legacy_fingerprint = config.embeddings_fingerprint({**encoder_identity, "revision": "default"}) if encoder_identity.get("revision") != "default" else None
        if not self.validate_embeddings(manifest, fingerprint, encoder_identity["embedding_dim"], legacy_fingerprint):
            return ResumePlan("embeddings", True, True, False, "embeddings invalid or changed")
        return ResumePlan("complete", True, True, True, "all artifacts valid")


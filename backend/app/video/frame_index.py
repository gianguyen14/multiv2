import hashlib
import json
import os
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from backend.app.indexes.faiss_siglip_index import FaissSigLIPIndex
from backend.app.retrieval.candidate_resolver import PersistentCandidateResolver
from backend.app.video.atomic_io import fsync_directory, write_json_atomic


@dataclass(frozen=True)
class FrameIndexBundle:
    generation_id: str
    index: FaissSigLIPIndex
    resolver: PersistentCandidateResolver
    metadata: dict
    generation_path: Path


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def current_generation_id(output_root):
    path = Path(output_root) / "CURRENT"
    if not path.is_file():
        return None
    value = json.loads(path.read_text())
    return value.get("generation_id")


def cleanup_stale_staging(output_root):
    staging_root = Path(output_root) / ".staging"
    if not staging_root.is_dir():
        return
    for child in staging_root.iterdir():
        if child.is_dir() and child.name.startswith("gen-") and not child.is_symlink():
            shutil.rmtree(child)


def _mapping_ids(path):
    value = json.loads(Path(path).read_text())
    mapping = value.get("frame_id_mapping", value)
    keys = sorted(int(key) for key in mapping)
    if keys != list(range(len(mapping))):
        raise ValueError("invalid frame ID mapping positions")
    identifiers = [str(mapping[str(index)] if str(index) in mapping else mapping[index]) for index in keys]
    if len(identifiers) != len(set(identifiers)) or any(not identifier for identifier in identifiers):
        raise ValueError("duplicate or empty mapped frame ID")
    return identifiers


def validate_generation(path, expected_generation_id=None):
    path = Path(path)
    required = [path / name for name in ("frames.faiss", "mapping.json", "payloads.json", "generation.json")]
    if not all(item.is_file() for item in required):
        raise ValueError("index generation is incomplete")
    metadata = json.loads((path / "generation.json").read_text())
    if metadata.get("schema_version") != 1 or metadata.get("generation_id") != path.name:
        raise ValueError("invalid index generation metadata")
    if expected_generation_id is not None and path.name != expected_generation_id:
        raise ValueError("index generation identity mismatch")
    for name, digest in metadata.get("artifact_sha256", {}).items():
        if _sha256(path / name) != digest:
            raise ValueError("index generation artifact checksum mismatch")
    index = FaissSigLIPIndex.load(path / "frames.faiss", path / "mapping.json")
    identifiers = _mapping_ids(path / "mapping.json")
    resolver = PersistentCandidateResolver(path / "payloads.json")
    payload_ids = set(resolver.payloads)
    if (index.index.ntotal != metadata.get("vector_count") or len(identifiers) != len(payload_ids)
            or set(identifiers) != payload_ids):
        raise ValueError("index generation artifact identities differ")
    for identifier in identifiers:
        payload = resolver.resolve(identifier)
        if payload.get("candidate_id") != identifier or payload.get("frame_uid") != identifier:
            raise ValueError("resolver payload identity mismatch")
        expected = f"{payload['video_id']}:{payload['source_frame_index_zero_based']:09d}"
        if identifier != expected:
            raise ValueError("non-canonical indexed frame identity")
    return FrameIndexBundle(path.name, index, resolver, metadata, path)


def load_current_frame_index(output_root):
    output_root = Path(output_root)
    pointer = json.loads((output_root / "CURRENT").read_text())
    generation_id = pointer.get("generation_id")
    if (not isinstance(generation_id, str) or not generation_id.startswith("gen-")
            or "/" in generation_id or "\\" in generation_id or ".." in generation_id):
        raise ValueError("invalid active generation pointer")
    return validate_generation(output_root / "generations" / generation_id, generation_id)


def build_frame_index(store, manifests, output_root, embedding_dim, index_type="flat", failpoint=None):
    failpoint = failpoint or (lambda name, context: None)
    output_root = Path(output_root)
    staging_root = output_root / ".staging"
    generations_root = output_root / "generations"
    staging_root.mkdir(parents=True, exist_ok=True)
    generations_root.mkdir(parents=True, exist_ok=True)
    cleanup_stale_staging(output_root)
    generation_id = f"gen-{uuid.uuid4().hex}"
    staging = staging_root / generation_id
    staging.mkdir()
    vectors, frame_ids, payloads = [], [], {}
    for manifest in sorted(manifests, key=lambda item: item.video_id):
        records = store.load_records(manifest.video_id)
        embeddings = store.load_embeddings(manifest.video_id)
        if embeddings.shape != (len(records), embedding_dim) or not np.isfinite(embeddings).all():
            raise ValueError(f"invalid frame embeddings for {manifest.video_id}")
        vectors.append(embeddings)
        for record in records:
            if record.frame_uid in payloads:
                raise ValueError("duplicate frame UID")
            frame_ids.append(record.frame_uid)
            payloads[record.frame_uid] = {"candidate_id": record.frame_uid, **record.to_dict()}
    if not vectors:
        raise ValueError("no completed frame embeddings to index")
    matrix = np.concatenate(vectors).astype(np.float32, copy=False)
    index = FaissSigLIPIndex(embedding_dim, index_type)
    index.add(matrix, frame_ids)
    index.save(staging / "frames.faiss", staging / "mapping.json")
    failpoint("after_staging_write", {"generation_id": generation_id})
    write_json_atomic(staging / "payloads.json", {"schema_version": 1, "payloads": payloads})
    metadata = {"schema_version": 1, "generation_id": generation_id, "index_type": index_type,
        "embedding_dim": embedding_dim, "vector_count": len(frame_ids),
        "video_ids": [manifest.video_id for manifest in sorted(manifests, key=lambda item: item.video_id)]}
    metadata["artifact_sha256"] = {name: _sha256(staging / name) for name in ("frames.faiss", "mapping.json", "payloads.json")}
    write_json_atomic(staging / "generation.json", metadata)
    failpoint("before_validation", {"generation_id": generation_id})
    validate_generation(staging, generation_id)
    failpoint("after_validation", {"generation_id": generation_id})
    final = generations_root / generation_id
    os.replace(staging, final)
    fsync_directory(generations_root)
    failpoint("index_pre_publish", {"generation_id": generation_id})
    write_json_atomic(output_root / "CURRENT", {"schema_version": 1, "generation_id": generation_id})
    failpoint("after_current_publish", {"generation_id": generation_id})
    return load_current_frame_index(output_root)

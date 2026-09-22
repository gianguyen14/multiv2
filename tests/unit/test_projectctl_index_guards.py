from argparse import Namespace
import json
from pathlib import Path

import projectctl
from backend.app.video.ingest_manifest import IngestManifest


def test_projectctl_index_rejects_identityless_manifest(monkeypatch, tmp_path, capsys):
    root = tmp_path / "processed"
    manifest_path = root / "video" / "manifest.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(json.dumps(IngestManifest(
        video_id="video", source_path="/tmp/video.mp4", source_size=1,
        source_mtime_ns=1, source_hash="x", ingestion_version="m15-v1",
        status="embeddings_ready", completed_stage="embeddings",
    ).to_dict()))
    monkeypatch.setenv("VIDEO_PROCESSED_ROOT", str(root))
    args = Namespace(index_type="flat", json=True, verbose=False, output=None)
    try:
        projectctl.command_index(args)
    except RuntimeError as exc:
        assert "encoder identity" in str(exc)
    else:
        raise AssertionError("identityless manifest was accepted")

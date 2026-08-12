# Test metadata loader

import json
from pathlib import Path

from backend.app.loaders.metadata_loader import load_metadata, MetadataError


def test_load_metadata_fixture():
    # Use the provided fixture video (now a real video - ffprobe will work)
    video_path = Path(__file__).parent.parent / "fixtures" / "test_5s.mp4"
    manifest = load_metadata(video_path)

    # Check key fields from ffprobe (video_id is the full stem)
    assert manifest["video_id"] == "test_5s"
    assert manifest["duration_ms"] == 5000
    assert manifest["fps"] == 25
    assert len(manifest["shots"]) == 1
    assert manifest["shots"][0]["shot_id"] == "shot_0"
    assert manifest["shots"][0]["start_ms"] == 0
    assert manifest["shots"][0]["end_ms"] == 5000
    assert len(manifest["shots"][0]["frames"]) == 1
    assert manifest["shots"][0]["frames"][0]["frame_id"] == "frame_0"
    assert manifest["shots"][0]["frames"][0]["timestamp_ms"] == 0


def test_missing_file_raises():
    missing = Path("nonexistent.mp4")
    try:
        load_metadata(missing)
    except FileNotFoundError:
        assert True
    else:
        assert False, "Expected FileNotFoundError"


def test_invalid_name_raises_without_fallback():
    # Create a temporary file with a non-matching name
    tmp_path = Path(__file__).parent / "tmp_invalid.mp4"
    tmp_path.touch()
    try:
        load_metadata(tmp_path)  # allow_fallback defaults to False
    except MetadataError:
        assert True
    else:
        assert False, "Expected MetadataError for unknown filename pattern"
    finally:
        tmp_path.unlink()


def test_fallback_only_when_allowed():
    # Create a temporary file with a non-matching name
    tmp_path = Path(__file__).parent / "tmp_fallback.mp4"
    tmp_path.touch()
    try:
        # Should fail without allow_fallback
        try:
            load_metadata(tmp_path)
        except MetadataError:
            pass
        else:
            assert False, "Expected MetadataError without allow_fallback"
        
        # Should succeed with allow_fallback=True (but only for matching pattern)
        # This will still fail because the filename doesn't match the pattern
        try:
            load_metadata(tmp_path, allow_fallback=True)
        except MetadataError:
            pass
        else:
            assert False, "Expected MetadataError for non-matching filename even with allow_fallback"
    finally:
        tmp_path.unlink()

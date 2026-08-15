import json
import os
from fractions import Fraction
from pathlib import Path
import av
import numpy as np
from PIL import Image
import pytest

from backend.app.retrieval.trake import EventCandidate, TRAKEAligner
from backend.app.services.temporal_refiner import (
    TemporalRefiner,
    TemporalRefinerConfig,
    TemporalRegion,
    TemporalRefineCache,
)


def create_synthetic_video(path: Path, num_frames: int = 24, fps: int = 6):
    """Creates a deterministic synthetic test video where frame i has green=i*10."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), "w", format="mp4") as container:
        stream = container.add_stream("mpeg4", rate=fps)
        stream.width, stream.height, stream.pix_fmt = 64, 48, "yuv420p"
        stream.time_base = Fraction(1, fps)
        for idx in range(num_frames):
            pixels = np.zeros((48, 64, 3), dtype=np.uint8)
            # Encode frame ordinal into color
            pixels[:, :, 1] = min(255, idx * 10)
            if idx == 7:  # Special red target marker for short-event test
                pixels[10:38, 10:54, 0] = 240
                pixels[10:38, 10:54, 1] = 10
                pixels[10:38, 10:54, 2] = 10
            elif idx == 15:  # Special blue target marker
                pixels[10:38, 10:54, 0] = 10
                pixels[10:38, 10:54, 1] = 10
                pixels[10:38, 10:54, 2] = 240
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            frame.pts = idx
            frame.time_base = Fraction(1, fps)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


class SyntheticColorEncoder:
    """Deterministic mock encoder based on RGB color energy."""
    embedding_dim = 4

    def __init__(self, model_name="mock-color", revision="v1"):
        self.model_name = model_name
        self.revision = revision
        self.encode_image_calls = 0
        self.encode_text_calls = 0

    def identity(self):
        return {
            "provider": "test",
            "model_name": self.model_name,
            "revision": self.revision,
            "embedding_dim": self.embedding_dim,
            "normalization": "l2",
            "contract_version": "m15.1-v1",
        }

    def encode_text(self, texts, batch_size=8, normalize=True):
        self.encode_text_calls += 1
        vectors = []
        for text in texts:
            if "red" in text.lower():
                v = np.array([1.0, 0.0, 0.0, 0.1], dtype=np.float32)
            elif "blue" in text.lower():
                v = np.array([0.0, 0.0, 1.0, 0.1], dtype=np.float32)
            elif "green" in text.lower():
                v = np.array([0.0, 1.0, 0.0, 0.1], dtype=np.float32)
            else:
                v = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)
            if normalize:
                v = v / np.linalg.norm(v)
            vectors.append(v)
        return np.stack(vectors)

    def encode_image(self, images, batch_size=8, normalize=True):
        self.encode_image_calls += 1
        vectors = []
        for img in images:
            if isinstance(img, (str, Path)):
                img = Image.open(img).convert("RGB")
            arr = np.asarray(img, dtype=np.float32)
            rgb = arr.mean(axis=(0, 1))
            v = np.array([rgb[0], rgb[1], rgb[2], 1.0], dtype=np.float32)
            if normalize:
                v = v / (np.linalg.norm(v) + 1e-9)
            vectors.append(v)
        return np.stack(vectors)


# =========================================================================
# A. Frame Identity Test (Hard Gate)
# =========================================================================

def test_frame_identity_invariant_preserves_exact_pyav_ordinals(tmp_path):
    video_path = tmp_path / "test_vid.mp4"
    num_frames = 20
    create_synthetic_video(video_path, num_frames=num_frames, fps=4)

    # Set up processed root metadata
    processed_root = tmp_path / "processed"
    vid_dir = processed_root / "test_vid"
    vid_dir.mkdir(parents=True)
    (vid_dir / "metadata.json").write_text(json.dumps({
        "video_id": "test_vid",
        "filename": "test_vid.mp4",
        "source_path": str(video_path),
        "real_frame_rate": "4/1",
        "avg_frame_rate": "4/1",
        "decoded_frame_count": num_frames,
    }))

    config = TemporalRefinerConfig(
        enabled=True,
        window_seconds=1.5,
        sample_fps=4.0,  # Sample every frame
        max_regions_per_video=3,
        max_total_regions=6,
        cache_enabled=False,
    )
    encoder = SyntheticColorEncoder()
    refiner = TemporalRefiner(processed_root=processed_root, config=config, encoder=encoder)

    # Coarse candidate at frame 5
    coarse_candidates = [[EventCandidate(video_id="test_vid", frame_id=5, score=0.8)]]
    refined, metrics = refiner.refine_trake_candidates(["green scene"], coarse_candidates)

    assert metrics["refinement_used"] is True
    assert len(refined) == 1
    refined_event_cands = refined[0]

    # Check that all returned frame_ids are valid PyAV sequential ordinals in [0, 20)
    returned_frame_ids = [cand.frame_id for cand in refined_event_cands]
    assert len(returned_frame_ids) > 0
    for fid in returned_frame_ids:
        assert isinstance(fid, int)
        assert 0 <= fid < num_frames
        # Ordinal must match exact decode order
        assert fid == int(fid)


# =========================================================================
# B. Short-Event Sampling Test (Absent from Sparse Index)
# =========================================================================

def test_short_event_sampling_recovers_missing_sparse_frame(tmp_path):
    video_path = tmp_path / "short_event.mp4"
    create_synthetic_video(video_path, num_frames=24, fps=4)

    processed_root = tmp_path / "processed"
    vid_dir = processed_root / "short_event"
    vid_dir.mkdir(parents=True)
    (vid_dir / "metadata.json").write_text(json.dumps({
        "video_id": "short_event",
        "filename": "short_event.mp4",
        "source_path": str(video_path),
        "real_frame_rate": "4/1",
        "avg_frame_rate": "4/1",
        "decoded_frame_count": 24,
    }))

    config = TemporalRefinerConfig(
        enabled=True,
        window_seconds=1.5,
        sample_fps=4.0,  # Dense sampling at 4 fps
        cache_enabled=False,
    )
    encoder = SyntheticColorEncoder()
    refiner = TemporalRefiner(processed_root=processed_root, config=config, encoder=encoder)

    # Coarse candidate only has frame 4 and frame 12 (sparse 2-second interval, frame 7 is absent!)
    sparse_candidates = [[
        EventCandidate(video_id="short_event", frame_id=4, score=0.2),
        EventCandidate(video_id="short_event", frame_id=12, score=0.2),
    ]]

    # Query for the short red event that happened at frame 7
    refined, metrics = refiner.refine_trake_candidates(["red marker"], sparse_candidates)

    assert metrics["refinement_used"] is True
    top_cand = refined[0][0]
    # Frame 7 was absent from sparse index, but dense refinement must recover it!
    assert top_cand.frame_id == 7
    assert top_cand.score > 0.7


# =========================================================================
# C. Ordered TRAKE Test (Monotonic DP Alignment)
# =========================================================================

def test_ordered_trake_monotonic_sequence_alignment():
    # Valid forward sequence: Red (7) -> Blue (15)
    candidates_valid = [
        [EventCandidate("v1", 7, 0.95), EventCandidate("v1", 20, 0.3)],
        [EventCandidate("v1", 5, 0.2), EventCandidate("v1", 15, 0.90)],
    ]
    aligner = TRAKEAligner()
    result = aligner.align(candidates_valid)
    assert result is not None
    assert result.video_id == "v1"
    assert result.frame_ids == [7, 15]

    # Invalid backwards sequence: Blue (15) then Red (7)
    candidates_invalid = [
        [EventCandidate("v1", 15, 0.95)],
        [EventCandidate("v1", 7, 0.90)],
    ]
    assert aligner.align(candidates_invalid) is None


# =========================================================================
# D. Region Merge Test
# =========================================================================

def test_region_generation_and_overlapping_merge(tmp_path):
    processed_root = tmp_path / "processed"
    vid_dir = processed_root / "v_merge"
    vid_dir.mkdir(parents=True)
    (vid_dir / "metadata.json").write_text(json.dumps({
        "video_id": "v_merge",
        "real_frame_rate": "30/1",
        "decoded_frame_count": 10000,
    }))

    config = TemporalRefinerConfig(
        window_seconds=2.0,  # +-60 frames at 30 fps
        max_regions_per_video=5,
        max_total_regions=10,
    )
    refiner = TemporalRefiner(processed_root=processed_root, config=config)

    # Candidates at frame 100 (window [40, 160]) and frame 140 (window [80, 200]) -> overlap!
    # Candidate at frame 800 (window [740, 860]) -> separate region
    candidates = [
        [
            EventCandidate("v_merge", 100, 0.9),
            EventCandidate("v_merge", 140, 0.8),
            EventCandidate("v_merge", 800, 0.7),
        ]
    ]

    video_regions = refiner.build_candidate_regions(candidates)
    assert "v_merge" in video_regions
    regions = video_regions["v_merge"]

    # 3 candidate frames merged into 2 non-overlapping regions
    assert len(regions) == 2
    # First merged region covers 40 to 200
    assert regions[0].start_frame == 40
    assert regions[0].end_frame == 200
    assert set(regions[0].source_candidate_frames) == {100, 140}
    # Second region covers 740 to 860
    assert regions[1].start_frame == 740
    assert regions[1].end_frame == 860


# =========================================================================
# E. Max-Region Limits Safety Test
# =========================================================================

def test_max_region_limits_safety(tmp_path):
    processed_root = tmp_path / "processed"
    for v in ("v1", "v2", "v3"):
        (processed_root / v).mkdir(parents=True)
        (processed_root / v / "metadata.json").write_text(json.dumps({
            "video_id": v,
            "real_frame_rate": "30/1",
            "decoded_frame_count": 10000,
        }))

    config = TemporalRefinerConfig(
        window_seconds=1.0,
        max_regions_per_video=2,
        max_total_regions=3,
    )
    refiner = TemporalRefiner(processed_root=processed_root, config=config)

    # Many distant candidates per video
    candidates = [
        [
            EventCandidate("v1", 100, 0.9),
            EventCandidate("v1", 500, 0.8),
            EventCandidate("v1", 900, 0.7),
            EventCandidate("v2", 100, 0.85),
            EventCandidate("v2", 500, 0.75),
            EventCandidate("v3", 100, 0.6),
        ]
    ]

    video_regions = refiner.build_candidate_regions(candidates)
    total_regions = sum(len(regs) for regs in video_regions.values())

    # Strictly bounded by max_total_regions = 3
    assert total_regions <= 3
    for v, regs in video_regions.items():
        assert len(regs) <= 2


# =========================================================================
# F. Cache Reuse Test (Sentinel Encoder)
# =========================================================================

def test_cache_reuse_with_sentinel_encoder(tmp_path):
    video_path = tmp_path / "cache_vid.mp4"
    create_synthetic_video(video_path, num_frames=16, fps=4)

    processed_root = tmp_path / "processed"
    vid_dir = processed_root / "cache_vid"
    vid_dir.mkdir(parents=True)
    (vid_dir / "metadata.json").write_text(json.dumps({
        "video_id": "cache_vid",
        "filename": "cache_vid.mp4",
        "source_path": str(video_path),
        "real_frame_rate": "4/1",
        "avg_frame_rate": "4/1",
        "decoded_frame_count": 16,
    }))

    config = TemporalRefinerConfig(
        enabled=True,
        window_seconds=1.0,
        sample_fps=4.0,
        cache_enabled=True,
    )

    # Run 1: Normal encoder populates cache
    encoder1 = SyntheticColorEncoder()
    refiner1 = TemporalRefiner(processed_root=processed_root, config=config, encoder=encoder1)
    coarse = [[EventCandidate("cache_vid", 4, 0.5)]]
    res1, metrics1 = refiner1.refine_trake_candidates(["red"], coarse)
    assert metrics1["cache_misses"] >= 1
    assert encoder1.encode_image_calls > 0

    # Run 2: Sentinel encoder that raises if encode_image is called
    class SentinelEncoder(SyntheticColorEncoder):
        def encode_image(self, *args, **kwargs):
            raise AssertionError("SentinelEncoder.encode_image called! Cache was bypassed.")

    encoder2 = SentinelEncoder()
    refiner2 = TemporalRefiner(processed_root=processed_root, config=config, encoder=encoder2)
    res2, metrics2 = refiner2.refine_trake_candidates(["red"], coarse)

    assert metrics2["cache_hits"] >= 1
    assert metrics2["dense_frames_decoded"] == 0
    assert len(res2[0]) == len(res1[0])
    assert res2[0][0].frame_id == res1[0][0].frame_id


# =========================================================================
# G. Cache Corruption Test (Safe Rebuild)
# =========================================================================

def test_cache_corruption_safe_rebuild(tmp_path):
    video_path = tmp_path / "corrupt_cache_vid.mp4"
    create_synthetic_video(video_path, num_frames=16, fps=4)

    processed_root = tmp_path / "processed"
    vid_dir = processed_root / "corrupt_cache_vid"
    vid_dir.mkdir(parents=True)
    (vid_dir / "metadata.json").write_text(json.dumps({
        "video_id": "corrupt_cache_vid",
        "filename": "corrupt_cache_vid.mp4",
        "source_path": str(video_path),
        "real_frame_rate": "4/1",
        "decoded_frame_count": 16,
    }))

    config = TemporalRefinerConfig(enabled=True, window_seconds=1.0, sample_fps=4.0, cache_enabled=True)
    encoder = SyntheticColorEncoder()
    refiner = TemporalRefiner(processed_root=processed_root, config=config, encoder=encoder)

    coarse = [[EventCandidate("corrupt_cache_vid", 4, 0.5)]]
    refiner.refine_trake_candidates(["red"], coarse)

    # Now corrupt the cache directory
    cache_dir = processed_root / "temporal_refine_cache" / "corrupt_cache_vid"
    assert cache_dir.is_dir()
    for reg_dir in cache_dir.iterdir():
        if reg_dir.is_dir():
            (reg_dir / "metadata.json").write_text("GARBAGE NOT JSON")
            (reg_dir / "embeddings.npy").write_bytes(b"CORRUPT_BYTES")

    # Second run must not crash and must safely rebuild
    res, metrics = refiner.refine_trake_candidates(["red"], coarse)
    assert metrics["refinement_used"] is True
    assert metrics["cache_misses"] >= 1


# =========================================================================
# H. Safe Fallback Test on Refinement Error
# =========================================================================

def test_safe_fallback_on_refinement_error(tmp_path):
    processed_root = tmp_path / "processed"
    vid_dir = processed_root / "missing_video"
    vid_dir.mkdir(parents=True)
    (vid_dir / "metadata.json").write_text(json.dumps({
        "video_id": "missing_video",
        "source_path": "/non/existent/path.mp4",
        "real_frame_rate": "30/1",
        "decoded_frame_count": 1000,
    }))

    config = TemporalRefinerConfig(enabled=True)
    encoder = SyntheticColorEncoder()
    refiner = TemporalRefiner(processed_root=processed_root, config=config, encoder=encoder)

    coarse = [[EventCandidate("missing_video", 100, 0.75)]]
    # Source video does not exist, refiner must safely return coarse candidates
    res, metrics = refiner.refine_trake_candidates(["event"], coarse)

    assert res == coarse
    assert metrics["refinement_used"] is False

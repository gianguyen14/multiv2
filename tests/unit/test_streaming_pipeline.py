"""Focused unit tests for bounded streaming producer-consumer pipeline.

Covers:
- Ordering: preservation of chronological/source index order, 1:1 records/embeddings alignment.
- Backpressure: producer throttling on bounded queue depth.
- Exceptions & Clean Termination: producer failures, consumer failures, save failures, no hanging threads.
- Selected-only Materialization: non-selected frames are NEVER converted to PIL RGB.
- Vector Contract Validation: shape, dimension, count, finiteness, and L2 normalization.
- Batch 1 Compatibility, path payload, and async persistence hooks.
- Real PyAV end-to-end decoding.
"""

from __future__ import annotations

import math
import tempfile
import threading
import time
from pathlib import Path
from typing import List
from unittest.mock import MagicMock

import numpy as np
import pytest
from PIL import Image

from backend.app.video.frame_id_policy import FrameIdPolicy
from backend.app.video.frame_record import FrameRecord
from backend.app.video.frame_sampler import FrameSamplingError
from backend.app.video.streaming_pipeline import (
    BoundedStreamingPipeline,
    SelectedFrame,
    StreamingPipelineConfig,
    StreamingProgress,
    UnmaterializedFrame,
    VectorContractError,
    iter_unmaterialized_frames,
    stream_sample_frames,
    validate_vector_contract,
)
from backend.app.video.video_decoder import VideoDecodeError

FIXTURE_5S = Path("tests/fixtures/test_5s.mp4")


def _make_mock_frame(
    index: int,
    timestamp: float,
    pts: int | None = None,
    width: int = 10,
    height: int = 10,
    call_tracker: dict[int, int] | None = None,
) -> UnmaterializedFrame:
    """Create an UnmaterializedFrame with an invocation-tracking loader."""
    def loader():
        if call_tracker is not None:
            call_tracker[index] = call_tracker.get(index, 0) + 1
        return Image.new("RGB", (width, height), color=(index % 255, 0, 0))

    return UnmaterializedFrame(
        source_frame_index_zero_based=index,
        pts=index if pts is None else pts,
        timestamp_seconds=timestamp,
        width=width,
        height=height,
        materialize_fn=loader,
    )


def _normalized_vector(dim: int, seed: int = 0) -> np.ndarray:
    """Generate a reproducible L2-normalized float32 vector."""
    rng = np.random.default_rng(seed)
    vec = rng.standard_normal(dim).astype(np.float32)
    norm = np.linalg.norm(vec)
    return vec / norm


# ============================================================================
# 1. ORDERING TESTS
# ============================================================================

def test_ordering_display_order_and_records_embeddings_alignment():
    """Verify that records and embeddings maintain strict display order and 1:1 alignment."""
    # 7 frames spanning 3 seconds at 0.5s spacing
    frames = [_make_mock_frame(i, i * 0.5) for i in range(7)]
    dim = 64

    def mock_encoder(batch):
        # Return distinct normalized embeddings for each frame
        embs = []
        for item in batch:
            # item is PIL Image
            embs.append(_normalized_vector(dim, seed=len(embs) + 1))
        return np.vstack(embs)

    pipeline = BoundedStreamingPipeline(
        config=StreamingPipelineConfig(
            sample_interval_seconds=1.0,
            batch_size=1,
            queue_depth=4,
        ),
        decoder_fn=lambda _: frames,
        encoder_fn=mock_encoder,
        embedding_dim=dim,
    )

    result = pipeline.run("dummy.mp4")

    # Sample interval 1.0s on [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    # Targets: 0.0 -> frame 0 (t=0.0)
    #          1.0 -> frame 2 (t=1.0)
    #          2.0 -> frame 4 (t=2.0)
    #          3.0 -> frame 6 (t=3.0)
    assert len(result.records) == 4
    assert result.embeddings.shape == (4, dim)

    indices = [r.source_frame_index_zero_based for r in result.records]
    assert indices == [0, 2, 4, 6]
    assert indices == sorted(indices)

    # Submission IDs in zero_based mode
    assert [r.submission_frame_id for r in result.records] == [0, 2, 4, 6]

    # Verify timestamps strictly ascending
    timestamps = [r.timestamp_seconds for r in result.records]
    assert timestamps == [0.0, 1.0, 2.0, 3.0]
    assert timestamps == sorted(timestamps)

    # Frame UID and Embedding ID match
    for r in result.records:
        assert r.embedding_id == r.frame_uid
        assert r.frame_uid == f"dummy:{r.source_frame_index_zero_based:09d}"


def test_ordering_with_multi_item_batches_and_async_persistence():
    """Verify ordering is strictly preserved across multi-item batches with async persistence."""
    frames = [_make_mock_frame(i, i * 0.2) for i in range(25)]  # 0.0 to 4.8s
    dim = 32

    def mock_encoder(batch):
        return np.vstack([_normalized_vector(dim, seed=i) for i in range(len(batch))])

    def mock_save(frame: SelectedFrame) -> str:
        time.sleep(0.01)  # small simulated I/O latency
        return f"/tmp/storage/{frame.video_id}_{frame.source_frame_index_zero_based:09d}.jpg"

    pipeline = BoundedStreamingPipeline(
        config=StreamingPipelineConfig(
            sample_interval_seconds=1.0,
            batch_size=3,
            queue_depth=5,
            async_persistence=True,
            persistence_workers=2,
        ),
        decoder_fn=lambda _: frames,
        encoder_fn=mock_encoder,
        save_fn=mock_save,
        embedding_dim=dim,
    )

    result = pipeline.run("ordered_test")

    # Sample interval 1.0s -> frames at t=0.0 (idx 0), 1.0 (idx 5), 2.0 (idx 10), 3.0 (idx 15), 4.0 (idx 20)
    expected_indices = [0, 5, 10, 15, 20]
    actual_indices = [r.source_frame_index_zero_based for r in result.records]
    assert actual_indices == expected_indices
    assert len(result.records) == 5
    assert result.embeddings.shape == (5, dim)

    # Verify each record has correctly populated saved path
    for r in result.records:
        assert r.image_path == f"/tmp/storage/ordered_test_{r.source_frame_index_zero_based:09d}.jpg"


def test_ordering_earlier_tie_preservation():
    """Verify that ties in candidate distances preserve the earlier frame."""
    # Target 1.0s. Frame 0 at 0.9s (diff 0.1), Frame 1 at 1.1s (diff 0.1) -> Frame 0 selected!
    frames = [
        _make_mock_frame(0, 0.9),
        _make_mock_frame(1, 1.1),
    ]
    pipeline = BoundedStreamingPipeline(
        config=StreamingPipelineConfig(sample_interval_seconds=1.0),
        decoder_fn=lambda _: frames,
    )
    result = pipeline.run("tie_test")
    # Candidate for target 0.0 is frame 0 (since it's first)
    # Candidate for target 1.0: frame 0 (diff |0.9-1.0|=0.1) vs frame 1 (diff |1.1-1.0|=0.1).
    # Since diffs are equal, strict '<' keeps earlier candidate (frame 0).
    assert len(result.records) == 1
    assert result.records[0].source_frame_index_zero_based == 0


# ============================================================================
# 2. BACKPRESSURE TESTS
# ============================================================================

def test_backpressure_throttles_producer_on_bounded_queue():
    """Verify that the producer blocks when the queue reaches configured queue_depth."""
    total_frames = 20
    frames = [_make_mock_frame(i, float(i)) for i in range(total_frames)]
    queue_depth = 2

    consumer_ready_event = threading.Event()
    consumed_count = 0

    def blocking_encoder(batch):
        nonlocal consumed_count
        # Wait until test signals consumer to proceed
        consumer_ready_event.wait(timeout=2.0)
        consumed_count += len(batch)
        dim = 16
        return np.vstack([_normalized_vector(dim) for _ in batch])

    pipeline = BoundedStreamingPipeline(
        config=StreamingPipelineConfig(
            sample_interval_seconds=1.0,
            batch_size=1,
            queue_depth=queue_depth,
        ),
        decoder_fn=lambda _: frames,
        encoder_fn=blocking_encoder,
        embedding_dim=16,
    )

    # Run in a separate thread so we can inspect producer state while blocked
    run_thread = threading.Thread(target=pipeline.run, args=("bp_video",))
    run_thread.start()

    # Give producer time to run and fill the queue (queue_depth items)
    time.sleep(0.15)

    # Consumer was paused on batch 1.
    # The queue can hold at most `queue_depth` items.
    # The producer should be blocked waiting for the queue to drain!
    # Sampled frames should not have reached all 20 frames!
    # Unblock the consumer
    consumer_ready_event.set()
    run_thread.join(timeout=3.0)
    assert not run_thread.is_alive(), "Pipeline thread should have terminated cleanly"
    assert consumed_count == total_frames


def test_backpressure_with_queue_depth_one():
    """Verify execution succeeds with minimal queue depth of 1 (lockstep backpressure)."""
    frames = [_make_mock_frame(i, float(i)) for i in range(6)]
    dim = 16

    pipeline = BoundedStreamingPipeline(
        config=StreamingPipelineConfig(
            sample_interval_seconds=1.0,
            batch_size=1,
            queue_depth=1,
        ),
        decoder_fn=lambda _: frames,
        encoder_fn=lambda batch: np.vstack([_normalized_vector(dim) for _ in batch]),
        embedding_dim=dim,
    )

    result = pipeline.run("depth_one_video")
    assert len(result.records) == 6
    assert result.embeddings.shape == (6, dim)


# ============================================================================
# 3. EXCEPTION PROPAGATION & CLEAN TERMINATION
# ============================================================================

def test_exception_propagation_producer_failure():
    """Verify producer decode exception terminates the consumer and re-raises cleanly."""
    def faulty_decoder(_):
        for i in range(3):
            yield _make_mock_frame(i, float(i))
        raise VideoDecodeError("Corrupted bitstream at frame 3")

    pipeline = BoundedStreamingPipeline(
        config=StreamingPipelineConfig(queue_depth=2, batch_size=1),
        decoder_fn=faulty_decoder,
    )

    with pytest.raises(VideoDecodeError, match="Corrupted bitstream"):
        pipeline.run("corrupt_video")


def test_exception_propagation_consumer_encoder_failure():
    """Verify consumer encoder failure stops the producer without hanging or leaking."""
    frames = [_make_mock_frame(i, float(i)) for i in range(50)]

    def failing_encoder(batch):
        raise RuntimeError("GPU CUDA Out Of Memory simulated")

    pipeline = BoundedStreamingPipeline(
        config=StreamingPipelineConfig(queue_depth=2, batch_size=1),
        decoder_fn=lambda _: frames,
        encoder_fn=failing_encoder,
    )

    with pytest.raises(RuntimeError, match="GPU CUDA Out Of Memory"):
        pipeline.run("oom_video")


def test_exception_propagation_persistence_failure():
    """Verify async persistence exception propagates to caller and cleans up threads."""
    frames = [_make_mock_frame(i, float(i)) for i in range(10)]

    def failing_save(frame: SelectedFrame) -> str:
        if frame.source_frame_index_zero_based >= 2:
            raise IOError("No space left on device")
        return f"/path/{frame.source_frame_index_zero_based}.jpg"

    pipeline = BoundedStreamingPipeline(
        config=StreamingPipelineConfig(
            sample_interval_seconds=1.0,
            batch_size=1,
            queue_depth=2,
            async_persistence=True,
        ),
        decoder_fn=lambda _: frames,
        save_fn=failing_save,
    )

    with pytest.raises(IOError, match="No space left on device"):
        pipeline.run("disk_full_video")


def test_exception_propagation_no_usable_timestamps():
    """Verify exception when all video frames lack valid timestamps."""
    frames = [_make_mock_frame(i, timestamp=None) for i in range(5)]

    pipeline = BoundedStreamingPipeline(
        decoder_fn=lambda _: frames,
    )

    with pytest.raises(FrameSamplingError, match="no usable frame timestamps"):
        pipeline.run("notime_video")


# ============================================================================
# 4. SELECTED-ONLY MATERIALIZATION TESTS
# ============================================================================

def test_selected_only_materialization_non_selected_never_materialized():
    """Verify that non-selected frames are NEVER converted to PIL RGB images."""
    call_tracker: dict[int, int] = {}
    total_frames = 50  # 50 frames at 10 fps (0.0 to 4.9s)
    frames = [
        _make_mock_frame(i, timestamp=i * 0.1, call_tracker=call_tracker)
        for i in range(total_frames)
    ]

    pipeline = BoundedStreamingPipeline(
        config=StreamingPipelineConfig(sample_interval_seconds=1.0),
        decoder_fn=lambda _: frames,
    )

    result = pipeline.run("mat_test")

    # Sample targets: 0.0, 1.0, 2.0, 3.0, 4.0
    # Expected selected frames: 0, 10, 20, 30, 40 (5 frames)
    selected_indices = [r.source_frame_index_zero_based for r in result.records]
    assert selected_indices == [0, 10, 20, 30, 40]
    assert len(selected_indices) == 5

    # Check call tracker: exactly 5 frames materialized, each exactly once!
    assert len(call_tracker) == 5
    for idx in selected_indices:
        assert call_tracker[idx] == 1, f"Selected frame {idx} should be materialized once"

    # All 45 non-selected frames MUST NOT have been materialized!
    for i in range(total_frames):
        if i not in selected_indices:
            assert i not in call_tracker, f"Non-selected frame {i} was materialized!"


def test_selected_only_materialization_closer_candidate_materialized():
    """Verify that when a subsequent frame is closer to target, earlier is dropped without materializing."""
    call_tracker: dict[int, int] = {}
    # Target 1.0s:
    # Frame 0: t=0.0 (selected for target 0.0)
    # Frame 1: t=0.7 (diff |0.7-1.0|=0.3)
    # Frame 2: t=1.1 (diff |1.1-1.0|=0.1) -> closer! Frame 2 wins target 1.0
    frames = [
        _make_mock_frame(0, 0.0, call_tracker=call_tracker),
        _make_mock_frame(1, 0.7, call_tracker=call_tracker),
        _make_mock_frame(2, 1.1, call_tracker=call_tracker),
    ]

    pipeline = BoundedStreamingPipeline(
        config=StreamingPipelineConfig(sample_interval_seconds=1.0),
        decoder_fn=lambda _: frames,
    )

    result = pipeline.run("closer_test")
    selected_indices = [r.source_frame_index_zero_based for r in result.records]
    assert selected_indices == [0, 2]

    # Frame 1 was evaluated as candidate but lost to Frame 2: it must NEVER be materialized!
    assert 1 not in call_tracker
    assert call_tracker[0] == 1
    assert call_tracker[2] == 1


# ============================================================================
# 5. VECTOR CONTRACT VALIDATION TESTS
# ============================================================================

def test_vector_contract_valid_embeddings():
    """Verify that valid float32 L2-normalized 2D vectors pass contract validation."""
    dim = 64
    valid = np.vstack([_normalized_vector(dim, seed=i) for i in range(5)])
    out = validate_vector_contract(valid, expected_count=5, expected_dim=dim)
    assert out.shape == (5, dim)
    assert out.dtype == np.float32


def test_vector_contract_dimension_mismatch():
    """Verify contract error on dimension mismatch."""
    embs = np.vstack([_normalized_vector(32) for _ in range(3)])
    with pytest.raises(VectorContractError, match="dimension mismatch"):
        validate_vector_contract(embs, expected_count=3, expected_dim=64)


def test_vector_contract_count_mismatch():
    """Verify contract error on row count mismatch."""
    embs = np.vstack([_normalized_vector(32) for _ in range(3)])
    with pytest.raises(VectorContractError, match="count mismatch"):
        validate_vector_contract(embs, expected_count=4, expected_dim=32)


def test_vector_contract_non_finite_values():
    """Verify contract error on NaN or Inf."""
    embs = np.vstack([_normalized_vector(32) for _ in range(2)])
    embs[0, 0] = np.nan
    with pytest.raises(VectorContractError, match="non-finite vector"):
        validate_vector_contract(embs, expected_count=2, expected_dim=32)

    embs[0, 0] = np.inf
    with pytest.raises(VectorContractError, match="non-finite vector"):
        validate_vector_contract(embs, expected_count=2, expected_dim=32)


def test_vector_contract_unnormalized_vectors():
    """Verify contract error when vectors are not L2 normalized."""
    embs = np.ones((3, 32), dtype=np.float32)  # norm is sqrt(32) != 1.0
    with pytest.raises(VectorContractError, match="not L2 normalized"):
        validate_vector_contract(embs, expected_count=3, expected_dim=32)


def test_vector_contract_non_2d():
    """Verify contract error on 1D or 3D inputs."""
    with pytest.raises(VectorContractError, match="must be a 2D array"):
        validate_vector_contract(np.zeros(32), expected_count=1, expected_dim=32)


def test_pipeline_enforces_vector_contract_on_bad_encoder():
    """Verify that BoundedStreamingPipeline rejects unnormalized encoder output."""
    frames = [_make_mock_frame(0, 0.0)]

    def bad_encoder(batch):
        # Returns unnormalized vectors
        return np.ones((len(batch), 16), dtype=np.float32) * 5.0

    pipeline = BoundedStreamingPipeline(
        config=StreamingPipelineConfig(sample_interval_seconds=1.0),
        decoder_fn=lambda _: frames,
        encoder_fn=bad_encoder,
        embedding_dim=16,
    )

    with pytest.raises(VectorContractError, match="not L2 normalized"):
        pipeline.run("bad_encoder_video")


# ============================================================================
# 6. BATCH 1 COMPATIBILITY, PATH PAYLOAD & PROGRESS TESTS
# ============================================================================

def test_batch_1_compatibility_strictly_invokes_encoder_with_batch_size_one():
    """Verify batch_size=1 passes batches of exactly size 1 to encoder."""
    frames = [_make_mock_frame(i, float(i)) for i in range(4)]
    dim = 16
    batch_sizes_seen = []

    def tracking_encoder(batch):
        batch_sizes_seen.append(len(batch))
        return np.vstack([_normalized_vector(dim) for _ in batch])

    pipeline = BoundedStreamingPipeline(
        config=StreamingPipelineConfig(
            sample_interval_seconds=1.0,
            batch_size=1,
        ),
        decoder_fn=lambda _: frames,
        encoder_fn=tracking_encoder,
        embedding_dim=dim,
    )

    result = pipeline.run("b1_video")
    assert len(result.records) == 4
    assert batch_sizes_seen == [1, 1, 1, 1]


def test_path_payload_preservation():
    """Verify that frames with pre-existing image_path payload preserve it without re-saving."""
    def path_payload_decoder(_):
        for i in range(3):
            frame = _make_mock_frame(i, float(i))
            # Set pre-existing image_path payload
            frame.image_path = f"/pre_extracted/frame_{i:04d}.png"
            yield frame

    save_mock = MagicMock()
    pipeline = BoundedStreamingPipeline(
        config=StreamingPipelineConfig(sample_interval_seconds=1.0),
        decoder_fn=path_payload_decoder,
        save_fn=save_mock,
    )

    result = pipeline.run("pre_extracted_video")
    assert len(result.records) == 3
    # save_fn should not have been called because path payload was already present
    assert save_mock.call_count == 0
    assert [r.image_path for r in result.records] == [
        "/pre_extracted/frame_0000.png",
        "/pre_extracted/frame_0001.png",
        "/pre_extracted/frame_0002.png",
    ]


def test_progress_counters_and_interval_callback():
    """Verify progress counters increment correctly and callback is invoked."""
    frames = [_make_mock_frame(i, i * 0.5) for i in range(10)]
    progress_snapshots: list[StreamingProgress] = []

    def on_progress(p: StreamingProgress):
        progress_snapshots.append(p)

    pipeline = BoundedStreamingPipeline(
        config=StreamingPipelineConfig(
            sample_interval_seconds=1.0,
            batch_size=1,
            progress_interval=2,
        ),
        decoder_fn=lambda _: frames,
        progress_callback=on_progress,
    )

    result = pipeline.run("prog_video")
    assert result.progress.decoded_frames == 10
    assert result.progress.sampled_frames == 5
    assert result.progress.persisted_frames == 5
    assert result.progress.encoded_frames == 5
    assert len(progress_snapshots) > 0


def test_frame_id_policy_one_based():
    """Verify FrameIdPolicy one_based maps submission ID correctly."""
    frames = [_make_mock_frame(0, 0.0), _make_mock_frame(5, 1.0)]
    pipeline = BoundedStreamingPipeline(
        config=StreamingPipelineConfig(
            sample_interval_seconds=1.0,
            frame_id_policy="one_based",
        ),
        decoder_fn=lambda _: frames,
    )
    result = pipeline.run("one_based_video")
    assert [r.source_frame_index_zero_based for r in result.records] == [0, 5]
    assert [r.submission_frame_id for r in result.records] == [1, 6]


# ============================================================================
# 7. REAL PYAV END-TO-END DECODING FIXTURE TEST
# ============================================================================

def test_real_pyav_fixture_unmaterialized_and_sampled():
    """Test with real video fixture tests/fixtures/test_5s.mp4 using PyAV decoder."""
    assert FIXTURE_5S.is_file(), f"Fixture missing: {FIXTURE_5S}"
    dim = 32

    pipeline = BoundedStreamingPipeline(
        config=StreamingPipelineConfig(
            sample_interval_seconds=1.0,
            batch_size=2,
            queue_depth=4,
        ),
        encoder_fn=lambda batch: np.vstack([_normalized_vector(dim, seed=i) for i in range(len(batch))]),
        embedding_dim=dim,
    )

    result = pipeline.run(FIXTURE_5S)

    # 5-second video sampled at 1.0s -> ~5 or 6 frames
    assert len(result.records) >= 5
    assert result.embeddings.shape == (len(result.records), dim)

    # Ordering check on real frames
    indices = [r.source_frame_index_zero_based for r in result.records]
    assert indices == sorted(indices)
    assert len(indices) == len(set(indices))

    # All frames should have valid dimensions
    for r in result.records:
        assert r.width == 896
        assert r.height == 672
        assert r.timestamp_seconds is not None


def test_selected_1920x1080_resized_and_persisted_at_configured_width(tmp_path):
    tracker = {}
    frames = [_make_mock_frame(0, 0.0, width=1920, height=1080, call_tracker=tracker),
              _make_mock_frame(1, 0.5, width=1920, height=1080, call_tracker=tracker),
              _make_mock_frame(2, 1.0, width=1920, height=1080, call_tracker=tracker)]
    received = []
    saved = []
    dim = 8
    def encoder(payloads):
        received.extend(image.size for image in payloads)
        value = np.zeros((len(payloads), dim), dtype=np.float32)
        value[:, 0] = 1.0
        return value
    def save(frame):
        saved.append(frame.image.size)
        path = tmp_path / f"{frame.source_frame_index_zero_based}.jpg"
        frame.image.save(path)
        return str(path)
    pipeline = BoundedStreamingPipeline(
        config=StreamingPipelineConfig(sample_interval_seconds=1.0, image_width=896),
        decoder_fn=lambda _: frames, encoder_fn=encoder, save_fn=save, embedding_dim=dim,
    )
    result = pipeline.run("resize-test")
    assert received == [(896, 504), (896, 504)]
    assert saved == [(896, 504), (896, 504)]
    assert [(r.width, r.height) for r in result.records] == [(896, 504), (896, 504)]
    assert tracker == {0: 1, 2: 1}

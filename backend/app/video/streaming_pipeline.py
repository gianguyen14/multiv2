"""Bounded streaming producer-consumer video ingestion pipeline.

Decodes, samples, persists, and embeds video frames using a bounded-queue
producer-consumer architecture. Eliminates unnecessary memory pressure by
ensuring non-selected frames are NEVER materialized into PIL RGB images.
"""

from __future__ import annotations

import math
import os
import queue
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image

from backend.app.video.frame_id_policy import FrameIdPolicy
from backend.app.video.frame_record import FrameRecord
from backend.app.video.frame_sampler import FrameSamplingError
from backend.app.video.video_decoder import VideoDecodeError


class StreamingPipelineError(RuntimeError):
    """Base exception for streaming pipeline errors."""
    pass


class VectorContractError(StreamingPipelineError, ValueError):
    """Raised when embeddings violate the vector contract (dimension, finiteness, L2 normalization)."""
    pass


# Internal sentinel tokens for producer-consumer queue coordination
_SENTINEL_EOF = object()
_SENTINEL_ERROR = object()


@dataclass
class UnmaterializedFrame:
    """Lightweight video frame container holding metadata and a lazy image loader.

    The underlying image is only materialized to a PIL RGB Image when `materialize()`
    is called on selected candidate frames. Non-selected frames can simply be discarded
    without incurring image conversion overhead.
    """
    source_frame_index_zero_based: int
    pts: Optional[int]
    timestamp_seconds: Optional[float]
    width: int
    height: int
    materialize_fn: Optional[Callable[[], Image.Image]] = None
    _materialized_image: Optional[Image.Image] = None

    def materialize(self) -> Image.Image:
        """Materialize and return the PIL RGB Image, caching it on this instance."""
        if self._materialized_image is None:
            if self.materialize_fn is not None:
                self._materialized_image = self.materialize_fn()
            else:
                raise RuntimeError(
                    f"Frame {self.source_frame_index_zero_based} has no image or materialize_fn"
                )
        return self._materialized_image

    @property
    def is_materialized(self) -> bool:
        """True if the image has already been materialized into memory."""
        return self._materialized_image is not None

    @property
    def image(self) -> Image.Image:
        """Access materialized image, materializing on demand if necessary."""
        return self.materialize()


@dataclass
class SelectedFrame:
    """A sampled frame chosen for ingestion, preserving source provenance."""
    source_frame_index_zero_based: int
    submission_frame_id: int
    pts: Optional[int]
    timestamp_seconds: Optional[float]
    width: int
    height: int
    image: Optional[Image.Image] = None
    image_path: Optional[str] = None
    target_timestamp_seconds: Optional[float] = None
    sampling_reason: Optional[str] = "periodic"
    shot_id: Optional[int] = None
    video_id: str = ""

    def to_frame_record(
        self,
        sample_interval_seconds: float,
        ingestion_version: str,
        fallback_image_path: Optional[str] = None,
    ) -> FrameRecord:
        """Construct a validated FrameRecord instance preserving all provenance."""
        resolved_path = self.image_path or fallback_image_path
        if resolved_path is None:
            resolved_path = f"{self.video_id}/frames/{self.source_frame_index_zero_based:09d}.jpg"

        return FrameRecord.create(
            video_id=self.video_id,
            source_frame_index_zero_based=self.source_frame_index_zero_based,
            submission_frame_id=self.submission_frame_id,
            timestamp_seconds=self.timestamp_seconds,
            pts=self.pts,
            width=self.width,
            height=self.height,
            image_path=resolved_path,
            sample_interval_seconds=sample_interval_seconds,
            ingestion_version=ingestion_version,
            shot_id=self.shot_id,
            sampling_reason=self.sampling_reason,
        )


@dataclass
class StreamingPipelineConfig:
    """Configuration for bounded streaming producer-consumer ingestion."""
    queue_depth: int = 8
    batch_size: int = 1
    sample_interval_seconds: float = 1.0
    frame_id_policy: str = "zero_based"
    ingestion_version: str = "m15-v1"
    async_persistence: bool = True
    persistence_workers: int = 2
    validate_contract: bool = True
    progress_interval: int = 1
    progress_interval_seconds: float = 3.0
    image_width: int = 896
    decode_threads: int = 8

    def __post_init__(self):
        if self.queue_depth < 1 or self.batch_size < 1 or self.persistence_workers < 1:
            raise ValueError("queue_depth, batch_size, and persistence_workers must be positive")
        if not math.isfinite(self.sample_interval_seconds) or self.sample_interval_seconds <= 0:
            raise ValueError("sample_interval_seconds must be positive and finite")
        if self.image_width < 2 or self.image_width % 2:
            raise ValueError("image_width must be an even integer >= 2")
        if self.decode_threads < 1:
            raise ValueError("decode_threads must be positive")


@dataclass
class StreamingProgress:
    """Real-time progress counters for pipeline stages."""
    decoded_frames: int = 0
    sampled_frames: int = 0
    persisted_frames: int = 0
    encoded_frames: int = 0
    total_frames: Optional[int] = None
    queue_depth: int = 0
    decode_fps: float = 0.0
    embed_fps: float = 0.0
    effective_batch_size: int = 1
    elapsed_seconds: float = 0.0
    embedding_active_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "decoded_frames": self.decoded_frames,
            "sampled_frames": self.sampled_frames,
            "persisted_frames": self.persisted_frames,
            "encoded_frames": self.encoded_frames,
            "embedded_frames": self.encoded_frames,
            "selected_frames": self.sampled_frames,
            "total_frames": self.total_frames,
            "queue_depth": self.queue_depth,
            "decode_fps": self.decode_fps,
            "embed_fps": self.embed_fps,
            "effective_batch_size": self.effective_batch_size,
            "elapsed_seconds": self.elapsed_seconds,
            "embedding_active_seconds": self.embedding_active_seconds,
        }


@dataclass
class StreamingPipelineResult:
    """Result of a streaming ingestion execution."""
    records: list[FrameRecord]
    embeddings: np.ndarray
    progress: StreamingProgress
    video_id: str
    duration_seconds: Optional[float] = None


def _materialize_frame_image(frame: Any) -> Image.Image:
    """Helper to safely materialize PIL RGB Image from an unmaterialized or decoded frame."""
    if hasattr(frame, "materialize") and callable(frame.materialize):
        return frame.materialize()
    if hasattr(frame, "load_image") and callable(frame.load_image):
        return frame.load_image()
    if hasattr(frame, "to_image") and callable(frame.to_image):
        return frame.to_image()
    if hasattr(frame, "image"):
        img = frame.image
        if callable(img):
            return img()
        if img is not None:
            return img
    if hasattr(frame, "image_path") and frame.image_path:
        return Image.open(frame.image_path).convert("RGB")
    raise RuntimeError(f"Cannot materialize image from frame candidate: {frame!r}")


def validate_vector_contract(
    embeddings: Any,
    expected_count: Optional[int] = None,
    expected_dim: Optional[int] = None,
) -> np.ndarray:
    """Validate 2D shape, count, dimension, finiteness, and L2 normalization of vectors."""
    arr = np.asarray(embeddings, dtype=np.float32)
    if arr.ndim != 2:
        raise VectorContractError(f"embeddings must be a 2D array, got shape {arr.shape}")
    if expected_count is not None and arr.shape[0] != expected_count:
        raise VectorContractError(
            f"embedding count mismatch: expected {expected_count}, got {arr.shape[0]}"
        )
    if expected_dim is not None and arr.shape[1] != expected_dim:
        raise VectorContractError(
            f"embedding dimension mismatch: expected {expected_dim}, got {arr.shape[1]}"
        )
    if not np.isfinite(arr).all():
        raise VectorContractError("embedding contract mismatch: non-finite vector")
    if arr.shape[0] > 0:
        norms = np.linalg.norm(arr, axis=1)
        if not np.allclose(norms, 1.0, atol=1e-5):
            raise VectorContractError(
                f"embedding contract mismatch: vectors are not L2 normalized (min={norms.min():.6f}, max={norms.max():.6f})"
            )
    return arr


def iter_unmaterialized_frames(path: Union[str, Path], decode_threads: int = 8) -> Iterator[UnmaterializedFrame]:
    """Decode video frames as UnmaterializedFrame objects without converting to PIL RGB.

    Each yielded frame holds lightweight metadata and a closure to materialize
    the frame on demand. Non-selected frames will be garbage-collected without
    incurring conversion overhead.
    """
    try:
        import av
    except ImportError as exc:
        raise VideoDecodeError("PyAV is required for video decoding") from exc

    resolved_path = Path(path)
    if not resolved_path.is_file():
        raise VideoDecodeError(f"video does not exist: {resolved_path}")

    previous_av_level = None
    try:
        import av.logging
        previous_av_level = av.logging.get_level()
        av.logging.set_level(av.logging.ERROR)
        container = av.open(str(resolved_path))
    except Exception as exc:
        raise VideoDecodeError(f"unable to open video: {resolved_path}") from exc

    try:
        stream = next((item for item in container.streams if item.type == "video"), None)
        if stream is None:
            raise VideoDecodeError("video stream is missing")
        stream.thread_type = "AUTO"
        if decode_threads:
            stream.thread_count = int(decode_threads)

        for index, frame in enumerate(container.decode(stream)):
            time_base = frame.time_base or stream.time_base
            timestamp = None
            if frame.pts is not None and time_base is not None:
                timestamp = float(Fraction(frame.pts) * Fraction(time_base))

            # Store closure binding current raw PyAV frame
            raw_frame = frame
            loader = lambda f=raw_frame: f.to_image().convert("RGB")

            yield UnmaterializedFrame(
                source_frame_index_zero_based=index,
                pts=frame.pts,
                timestamp_seconds=timestamp,
                width=frame.width,
                height=frame.height,
                materialize_fn=loader,
            )
    except VideoDecodeError:
        raise
    except Exception as exc:
        raise VideoDecodeError(f"unable to decode video: {resolved_path}") from exc
    finally:
        container.close()
        if previous_av_level is not None:
            try:
                av.logging.set_level(previous_av_level)
            except Exception:
                pass



def resize_selected_image(image: Image.Image, target_width: int) -> Image.Image:
    """Resize only a selected RGB image, preserving aspect ratio and even height."""
    image = image.convert("RGB")
    if target_width < 2:
        raise ValueError("target image width must be >= 2")
    height = max(2, round(image.height * target_width / image.width))
    if height % 2:
        height += 1
    return image.resize((target_width, height), Image.Resampling.LANCZOS)


def stream_sample_frames(
    frames: Iterable[Any],
    interval_seconds: float = 1.0,
    policy: Optional[FrameIdPolicy] = None,
    video_id: str = "",
    image_width: int = 896,
) -> Iterator[SelectedFrame]:
    """Sample frames at regular intervals without materializing non-selected frames.

    Preserves exact earlier-tie behavior: when target falls equidistant between
    previous and current frame timestamps, previous (earlier) frame is selected.
    Materialization is deferred until a candidate is confirmed selected.
    """
    if (
        interval_seconds is None
        or not isinstance(interval_seconds, (int, float))
        or math.isnan(interval_seconds)
        or math.isinf(interval_seconds)
        or interval_seconds <= 0
    ):
        raise ValueError("sample interval must be a positive and finite number")

    policy = policy or FrameIdPolicy("zero_based")
    previous = None
    target = 0.0
    last_selected_index = None
    last_timed = None

    for current in frames:
        if current.timestamp_seconds is None:
            continue
        last_timed = current
        if previous is None:
            previous = current

        while target <= current.timestamp_seconds:
            candidate = previous
            # Strict inequality preserves earlier-tie behavior (candidate stays previous on equality)
            if abs(current.timestamp_seconds - target) < abs(previous.timestamp_seconds - target):
                candidate = current

            if candidate.source_frame_index_zero_based != last_selected_index:
                # Materialize ONLY selected candidate
                materialized_img = resize_selected_image(
                    _materialize_frame_image(candidate), image_width
                )
                sub_id = policy.to_submission_frame_id(candidate.source_frame_index_zero_based)

                yield SelectedFrame(
                    source_frame_index_zero_based=candidate.source_frame_index_zero_based,
                    submission_frame_id=sub_id,
                    pts=candidate.pts,
                    timestamp_seconds=candidate.timestamp_seconds,
                    width=materialized_img.width,
                    height=materialized_img.height,
                    image=materialized_img,
                    image_path=getattr(candidate, "image_path", None),
                    target_timestamp_seconds=target,
                    sampling_reason=getattr(candidate, "sampling_reason", "periodic"),
                    shot_id=getattr(candidate, "shot_id", None),
                    video_id=video_id,
                )
                last_selected_index = candidate.source_frame_index_zero_based

            target += interval_seconds

        previous = current

    if last_timed is None:
        raise FrameSamplingError("video has no usable frame timestamps")


class BoundedStreamingPipeline:
    """Bounded streaming producer-consumer pipeline for video frame ingestion.

    Features:
    - Bounded queue depth with configurable backpressure.
    - Single consumer compatible with batch_size=1 or batch_size=N.
    - Non-selected frames are NEVER materialized to PIL RGB.
    - Async persistence hook or pre-existing path payload.
    - Ordered records and embeddings output.
    - Comprehensive producer/consumer exception propagation and clean termination.
    - Vector contract validation.
    - Generic testable API accepting decoder, sampler, encoder, and save callbacks.
    """

    def __init__(
        self,
        *,
        encoder: Optional[Any] = None,
        config: Optional[StreamingPipelineConfig] = None,
        decoder_fn: Optional[Callable[[Any], Iterable[Any]]] = None,
        sampler_fn: Optional[Callable[[Iterable[Any]], Iterable[SelectedFrame]]] = None,
        encoder_fn: Optional[Callable[[list[Any]], Union[np.ndarray, list[Any]]]] = None,
        save_fn: Optional[Callable[[SelectedFrame], str]] = None,
        async_persistence_hook: Optional[Callable[[SelectedFrame], Any]] = None,
        progress_callback: Optional[Callable[[StreamingProgress], None]] = None,
        policy: Optional[FrameIdPolicy] = None,
        embedding_dim: Optional[int] = None,
    ):
        self.encoder = encoder
        self.config = config or StreamingPipelineConfig()
        self.decoder_fn = decoder_fn
        self.sampler_fn = sampler_fn
        self.encoder_fn = encoder_fn
        self.save_fn = save_fn
        self.async_persistence_hook = async_persistence_hook
        self.progress_callback = progress_callback
        self.policy = policy or FrameIdPolicy(self.config.frame_id_policy)
        self._embedding_dim = embedding_dim

    def _resolve_embedding_dim(self) -> Optional[int]:
        if self._embedding_dim is not None:
            return self._embedding_dim
        if self.encoder is not None:
            if hasattr(self.encoder, "embedding_dim"):
                return int(self.encoder.embedding_dim)
            if hasattr(self.encoder, "identity"):
                ident = self.encoder.identity()
                if "embedding_dim" in ident:
                    return int(ident["embedding_dim"])
        return None

    def run(
        self,
        video_source: Any,
        video_id: Optional[str] = None,
    ) -> StreamingPipelineResult:
        """Run the bounded streaming pipeline on the specified video source."""
        if video_id is None:
            if isinstance(video_source, (str, Path)):
                video_id = Path(video_source).stem
            else:
                video_id = "video"

        expected_dim = self._resolve_embedding_dim()
        work_queue: queue.Queue = queue.Queue(maxsize=max(1, self.config.queue_depth))
        stop_event = threading.Event()
        progress = StreamingProgress(effective_batch_size=self.config.batch_size)
        started = time.monotonic()
        last_progress_report = [0.0]
        def report_progress(force=False):
            if not self.progress_callback:
                return
            now = time.monotonic()
            if force or now - last_progress_report[0] >= self.config.progress_interval_seconds:
                last_progress_report[0] = now
                progress.queue_depth = work_queue.qsize()
                progress.elapsed_seconds = now - started
                progress.decode_fps = progress.decoded_frames / progress.elapsed_seconds if progress.elapsed_seconds else 0.0
                progress.embed_fps = progress.encoded_frames / progress.elapsed_seconds if progress.elapsed_seconds else 0.0
                self.progress_elapsed = now
                self.progress_callback(StreamingProgress(**progress.__dict__))

        producer_error: list[Optional[Exception]] = [None]
        consumer_error: list[Optional[Exception]] = [None]

        # Async persistence executor
        persistence_executor: Optional[ThreadPoolExecutor] = None
        has_save_hook = (self.save_fn is not None) or (self.async_persistence_hook is not None)
        if has_save_hook:
            persistence_executor = ThreadPoolExecutor(
                max_workers=max(1, self.config.persistence_workers),
                thread_name_prefix="streaming-persist",
            )

        def _run_producer():
            try:
                # 1. Obtain raw frames
                if self.decoder_fn is not None:
                    raw_frames = self.decoder_fn(video_source)
                else:
                    raw_frames = iter_unmaterialized_frames(video_source, self.config.decode_threads)

                # Counting and cancellation wrapper
                def _tracked_stream():
                    for frame in raw_frames:
                        if stop_event.is_set():
                            break
                        progress.decoded_frames += 1
                        report_progress()
                        yield frame

                # 2. Run sampler
                if self.sampler_fn is not None:
                    sampled_stream = self.sampler_fn(_tracked_stream())
                else:
                    sampled_stream = stream_sample_frames(
                        _tracked_stream(),
                        interval_seconds=self.config.sample_interval_seconds,
                        policy=self.policy,
                        video_id=video_id,
                        image_width=self.config.image_width,
                    )

                # 3. Put into bounded queue (blocks when queue is full -> backpressure)
                for selected in sampled_stream:
                    if stop_event.is_set():
                        break
                    selected.video_id = selected.video_id or video_id
                    progress.sampled_frames += 1

                    while not stop_event.is_set():
                        try:
                            work_queue.put(selected, timeout=0.05)
                            break
                        except queue.Full:
                            continue

                if not stop_event.is_set():
                    while not stop_event.is_set():
                        try:
                            work_queue.put(_SENTINEL_EOF, timeout=0.05)
                            break
                        except queue.Full:
                            continue

            except Exception as exc:
                producer_error[0] = exc
                stop_event.set()
                # Place error sentinel without deadlocking
                try:
                    work_queue.put_nowait(_SENTINEL_ERROR)
                except Exception:
                    pass

        producer_thread = threading.Thread(
            target=_run_producer,
            name="streaming-producer",
            daemon=True,
        )
        producer_thread.start()

        # Consumer loop in the calling thread
        all_records: list[FrameRecord] = []
        all_embeddings: list[np.ndarray] = []
        pending_batch: list[Tuple[SelectedFrame, Optional[Union[Future, Any]]]] = []

        def _flush_batch(batch_items: list[Tuple[SelectedFrame, Optional[Union[Future, Any]]]]):
            if not batch_items:
                return

            frames: list[SelectedFrame] = [item[0] for item in batch_items]

            # Encode selected PIL images directly when the encoder supports image
            # payloads. Persistence futures continue while the GPU is embedding.
            payloads = [f.image if f.image is not None else f.image_path for f in frames]
            embed_started = time.perf_counter()
            if self.encoder_fn is not None:
                raw_embs = self.encoder_fn(payloads)
            elif self.encoder is not None:
                raw_embs = self.encoder.encode_image(
                    payloads,
                    batch_size=len(frames),
                    normalize=True,
                )
            else:
                # Default zero vectors if no encoder provided
                dim = expected_dim or 128
                eye = np.zeros((len(frames), dim), dtype=np.float32)
                eye[:, 0] = 1.0  # L2 normalized dummy
                raw_embs = eye

            progress.embedding_active_seconds = getattr(progress, "embedding_active_seconds", 0.0) + (time.perf_counter() - embed_started)
            # Validate vector contract
            if self.config.validate_contract:
                validated_embs = validate_vector_contract(
                    raw_embs,
                    expected_count=len(frames),
                    expected_dim=expected_dim,
                )
            else:
                validated_embs = np.asarray(raw_embs, dtype=np.float32)

            all_embeddings.append(validated_embs)
            progress.encoded_frames += len(frames)

            # Only after embedding completes do we await the asynchronous JPEG
            # persistence, so decode/save and GPU embedding overlap.
            for frame_item, save_fut in batch_items:
                if save_fut is not None:
                    if isinstance(save_fut, Future):
                        frame_item.image_path = str(save_fut.result())
                    elif callable(save_fut):
                        frame_item.image_path = str(save_fut())
                elif frame_item.image_path is None and self.save_fn is not None:
                    frame_item.image_path = str(self.save_fn(frame_item))
                progress.persisted_frames += 1

            for f in frames:
                record = f.to_frame_record(
                    sample_interval_seconds=self.config.sample_interval_seconds,
                    ingestion_version=self.config.ingestion_version,
                )
                all_records.append(record)

            report_progress()

        try:
            while True:
                if stop_event.is_set() and producer_error[0] is not None:
                    raise producer_error[0]

                try:
                    item = work_queue.get(timeout=0.05)
                except queue.Empty:
                    if producer_error[0] is not None:
                        raise producer_error[0]
                    if not producer_thread.is_alive() and work_queue.empty():
                        break
                    continue

                if item is _SENTINEL_ERROR:
                    work_queue.task_done()
                    if producer_error[0] is not None:
                        raise producer_error[0]
                    raise StreamingPipelineError("Producer encountered an unspecified failure")

                if item is _SENTINEL_EOF:
                    work_queue.task_done()
                    if pending_batch:
                        _flush_batch(pending_batch)
                        pending_batch = []
                    break

                # Item is SelectedFrame
                selected_item: SelectedFrame = item
                work_queue.task_done()

                # Dispatch persistence
                save_future = None
                if selected_item.image_path is None:
                    hook = self.save_fn or self.async_persistence_hook
                    if hook is not None:
                        if persistence_executor is not None:
                            save_future = persistence_executor.submit(hook, selected_item)
                        else:
                            selected_item.image_path = hook(selected_item)

                pending_batch.append((selected_item, save_future))

                # Batch 1 compatibility or batch_size flush
                if len(pending_batch) >= max(1, self.config.batch_size):
                    _flush_batch(pending_batch)
                    pending_batch = []

        except Exception as exc:
            consumer_error[0] = exc
            stop_event.set()
            # Drain work queue so producer thread never hangs on full queue
            while not work_queue.empty():
                try:
                    work_queue.get_nowait()
                    work_queue.task_done()
                except Exception:
                    break
            raise
        finally:
            stop_event.set()
            # Clean termination: join producer thread
            producer_thread.join(timeout=5.0)
            if persistence_executor is not None:
                persistence_executor.shutdown(wait=False, cancel_futures=True)

        final_embeddings = (
            np.vstack(all_embeddings)
            if all_embeddings
            else np.zeros((0, expected_dim or 0), dtype=np.float32)
        )

        return StreamingPipelineResult(
            records=all_records,
            embeddings=final_embeddings,
            progress=progress,
            video_id=video_id,
        )

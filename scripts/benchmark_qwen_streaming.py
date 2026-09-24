#!/usr/bin/env python3
"""Benchmark bounded Qwen streaming ingest for one video.

Example RTX 4050 profile:
  python scripts/benchmark_qwen_streaming.py --input /video/N001-V001.mov \
    --output /tmp/qwen-bench --sample-interval 10 --width 896 \
    --dtype float16 --batch-size 1 --queue-depth 8 --decode-threads 8
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.config.video_ingest_config import VideoIngestConfig
from backend.app.embeddings.ingest_encoder import create_ingest_encoder
from backend.app.video.ingest import ingest_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=os.getenv("QWEN_BENCHMARK_VIDEO", "N001-V001.mov"))
    parser.add_argument("--output", default="data/benchmarks/qwen-streaming")
    parser.add_argument("--sample-interval", type=float, default=10.0)
    parser.add_argument("--width", type=int, default=896)
    parser.add_argument("--dtype", default="float16", choices=("auto", "bfloat16", "float16", "float32"))
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--queue-depth", type=int, default=8)
    parser.add_argument("--decode-threads", type=int, default=8)
    parser.add_argument("--device", default=os.getenv("VIDEO_INGEST_DEVICE", "auto"))
    parser.add_argument("--model-dir")
    args = parser.parse_args()
    started = time.perf_counter()
    config = VideoIngestConfig(
        processed_root=Path(args.output), sample_interval_seconds=args.sample_interval,
        embed_batch_size=args.batch_size, device=args.device, ingest_backend="qwen3_vl",
        qwen_dtype=args.dtype, qwen_batch_min=1, qwen_batch_max=1,
        qwen_image_width=args.width, decode_threads=args.decode_threads,
        ingest_queue_depth=args.queue_depth,
    )
    encoder = create_ingest_encoder(config, model_dir=args.model_dir)
    encoder.load_model()
    try:
        report = ingest_path(args.input, encoder, config, limit=1, force=True, fail_fast=True)
    finally:
        encoder.clear_cache()
    wall_seconds = time.perf_counter() - started
    item = (report.get("results") or [{}])[0]
    duration = float(item.get("duration_seconds") or 0.0)
    sampled = int(item.get("sampled_frame_count") or 0)
    report["benchmark"] = {
        "input": str(args.input), "sample_interval_seconds": args.sample_interval,
        "width": args.width, "dtype": args.dtype, "batch_size": args.batch_size,
        "queue_depth": args.queue_depth, "decode_threads": args.decode_threads,
        "wall_seconds": round(wall_seconds, 3),
        "sampled_frames": sampled,
        "indexed_frames": report.get("indexed_frames", 0),
        "extraction_seconds": round(float(item.get("extraction_ms") or 0.0) / 1000.0, 3),
        "embedding_active_seconds": round(float(item.get("embedding_ms") or 0.0) / 1000.0, 3),
        "overlapped_wall_seconds": round(float(item.get("overlapped_wall_ms") or 0.0) / 1000.0, 3),
        "qwen_seconds_per_frame": item.get("qwen_seconds_per_frame", 0.0),
        "estimated_hours_for_200h": round((200.0 * wall_seconds / duration), 3) if duration > 0 else None,
        "realtime_factor": round(duration / wall_seconds, 3) if wall_seconds > 0 else None,
        "failures": report.get("videos_failed", 0),
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 1 if report.get("videos_failed") else 0


if __name__ == "__main__":
    raise SystemExit(main())

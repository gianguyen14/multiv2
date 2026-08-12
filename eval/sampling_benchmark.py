"""Sampling benchmark for M2 evaluation.

Reports for each sampling strategy:
- frame count
- processing time
- output storage size (estimated)
- first/last timestamp
"""

import json
import time
from pathlib import Path
from typing import Dict, List

from backend.app.loaders.aic_loader import create_loader, AICLoader
from backend.app.samplers.base import (
    FixedFPSStrategy,
    ShotPlusFixedFPSStrategy,
    AdaptiveDenseStrategy,
)
from backend.app.shot_detection.base import NullShotDetector
from backend.app.schemas.frame import FrameData


def estimate_storage_size(frames: List[FrameData]) -> int:
    """Estimate storage size in bytes for frames (FrameData only, not images)."""
    # Rough estimate: each FrameData ~ 200 bytes when serialized
    return len(frames) * 200


def run_benchmark(
    loader: AICLoader,
    video_path: Path,
    strategy_name: str,
    allow_metadata_fallback: bool = True,
) -> Dict:
    """Run benchmark for a single strategy."""
    start_time = time.perf_counter()
    frames = loader.process_video(video_path, allow_metadata_fallback=allow_metadata_fallback)
    elapsed_ms = (time.perf_counter() - start_time) * 1000

    if not frames:
        return {
            "strategy": strategy_name,
            "frame_count": 0,
            "processing_time_ms": elapsed_ms,
            "storage_size_bytes": 0,
            "first_timestamp_ms": None,
            "last_timestamp_ms": None,
        }

    return {
        "strategy": strategy_name,
        "frame_count": len(frames),
        "processing_time_ms": round(elapsed_ms, 2),
        "storage_size_bytes": estimate_storage_size(frames),
        "first_timestamp_ms": frames[0].timestamp_ms,
        "last_timestamp_ms": frames[-1].timestamp_ms,
    }


def main():
    video_path = Path(__file__).parent.parent / "tests" / "fixtures" / "test_5s.mp4"

    if not video_path.exists():
        print(f"Error: Test video not found at {video_path}")
        return

    # Define strategies to benchmark
    strategies = [
        ("fixed_0.25fps", create_loader(sampling_type="fixed", fps=0.25, shot_detector_type="none")),
        ("fixed_0.5fps", create_loader(sampling_type="fixed", fps=0.5, shot_detector_type="none")),
        ("fixed_1fps", create_loader(sampling_type="fixed", fps=1.0, shot_detector_type="none")),
        ("fixed_2fps", create_loader(sampling_type="fixed", fps=2.0, shot_detector_type="none")),
        ("shot_fixed_1fps", create_loader(sampling_type="shot_fixed", fps=1.0, shot_detector_type="none")),
        ("adaptive_dense", create_loader(sampling_type="adaptive", shot_detector_type="none")),
    ]

    results = []
    for name, loader in strategies:
        print(f"Benchmarking {name}...")
        result = run_benchmark(loader, video_path, name)
        results.append(result)
        print(f"  Frames: {result['frame_count']}, Time: {result['processing_time_ms']}ms, "
              f"Storage: {result['storage_size_bytes']} bytes, "
              f"Range: {result['first_timestamp_ms']}-{result['last_timestamp_ms']}ms")

    # Save results
    output_path = Path(__file__).parent / "sampling_benchmark_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_path}")

    # Print summary table
    print("\n" + "=" * 80)
    print(f"{'Strategy':<25} {'Frames':>8} {'Time(ms)':>10} {'Storage(B)':>12} {'First':>8} {'Last':>8}")
    print("-" * 80)
    for r in results:
        print(f"{r['strategy']:<25} {r['frame_count']:>8} {r['processing_time_ms']:>10.2f} "
              f"{r['storage_size_bytes']:>12} {r['first_timestamp_ms']:>8} {r['last_timestamp_ms']:>8}")


if __name__ == "__main__":
    main()
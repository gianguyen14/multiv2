#!/usr/bin/env python3
"""Synthetic parity/performance benchmark for the optional C++ native core.

This benchmark intentionally avoids model downloads and dataset artifacts. It measures
only CPU-side kernels that have equivalent Python and C++ implementations.
"""

from __future__ import annotations

import argparse
import os
import statistics
import time
from contextlib import contextmanager

import numpy as np

from backend.app.native import (
    align_trake_events,
    merge_temporal_regions,
    native_available,
    native_status,
    smooth_scores,
    temporal_nms_indices,
)


@contextmanager
def native_mode(mode: str):
    old = os.environ.get("UVR_NATIVE_CORE")
    os.environ["UVR_NATIVE_CORE"] = mode
    try:
        yield
    finally:
        if old is None:
            os.environ.pop("UVR_NATIVE_CORE", None)
        else:
            os.environ["UVR_NATIVE_CORE"] = old


def median_ms(fn, iterations: int) -> float:
    samples = []
    for _ in range(iterations):
        start = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - start) * 1000.0)
    return statistics.median(samples)


def compare(name, python_fn, cpp_fn, iterations: int):
    python_ms = median_ms(python_fn, iterations)
    cpp_ms = median_ms(cpp_fn, iterations)
    speedup = python_ms / cpp_ms if cpp_ms > 0 else float("inf")
    print(f"{name:24s} python={python_ms:9.3f} ms  cpp={cpp_ms:9.3f} ms  speedup={speedup:7.2f}x")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=7)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    if not native_available():
        print("C++ extension is not available. Build with: UVR_NATIVE_STRICT_BUILD=1 python setup.py build_ext --inplace")
        return 2

    print("native status:", native_status())
    rng = np.random.default_rng(args.seed)

    raw_scores = rng.normal(size=50_000).astype(np.float32)

    video_ids = [f"v{i % 24:02d}" for i in range(20_000)]
    frame_ids = rng.integers(0, 200_000, size=20_000, dtype=np.int64).tolist()

    region_frames = rng.integers(0, 200_000, size=8_000, dtype=np.int64).tolist()
    region_scores = rng.random(8_000).tolist()

    trake_frames = []
    trake_scores = []
    for event_idx in range(5):
        frames = np.sort(rng.choice(np.arange(event_idx, 20_000, 5), size=350, replace=False)).tolist()
        scores = rng.random(350).tolist()
        trake_frames.append(frames)
        trake_scores.append(scores)

    def py_smooth():
        with native_mode("python"):
            return smooth_scores(raw_scores, 0.8, 0.2, 2)

    def cpp_smooth():
        with native_mode("cpp"):
            return smooth_scores(raw_scores, 0.8, 0.2, 2)

    def py_nms():
        with native_mode("python"):
            return temporal_nms_indices(video_ids, frame_ids, 60, 100)

    def cpp_nms():
        with native_mode("cpp"):
            return temporal_nms_indices(video_ids, frame_ids, 60, 100)

    def py_merge():
        with native_mode("python"):
            return merge_temporal_regions(region_frames, region_scores, 120, 200_000, 32)

    def cpp_merge():
        with native_mode("cpp"):
            return merge_temporal_regions(region_frames, region_scores, 120, 200_000, 32)

    def py_trake():
        with native_mode("python"):
            return align_trake_events(trake_frames, trake_scores, 0.0001, 5000)

    def cpp_trake():
        with native_mode("cpp"):
            return align_trake_events(trake_frames, trake_scores, 0.0001, 5000)

    np.testing.assert_allclose(py_smooth(), cpp_smooth(), rtol=1e-6, atol=1e-6)
    assert py_nms() == cpp_nms()
    assert py_merge() == cpp_merge()
    py_dp = py_trake()
    cpp_dp = cpp_trake()
    assert py_dp is not None and cpp_dp is not None
    assert py_dp[1] == cpp_dp[1]
    assert abs(py_dp[0] - cpp_dp[0]) < 1e-10

    print("\nParity: PASS")
    print("Benchmark (median):")
    compare("temporal smoothing", py_smooth, cpp_smooth, args.iterations)
    compare("temporal NMS", py_nms, cpp_nms, args.iterations)
    compare("region merging", py_merge, cpp_merge, args.iterations)
    compare("TRAKE DP", py_trake, cpp_trake, args.iterations)
    print("\nNo speedup threshold is enforced; use representative workloads before promoting a kernel.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Benchmark for FaissSigLIPIndex.

Usage:
    python -m eval.faiss_siglip_benchmark [--num-vectors N] [--search-reps N]
    python eval/faiss_siglip_benchmark.py  [--num-vectors N] [--search-reps N]
"""

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import List

import numpy as np

# Ensure the project root is on sys.path regardless of how the script is
# invoked (python -m eval... vs python eval/...py).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from backend.app.indexes.faiss_siglip_index import FaissSigLIPIndex  # noqa: E402


def generate_synthetic_vectors(num_vectors: int, dim: int = 768) -> np.ndarray:
    """Generate synthetic normalized vectors."""
    np.random.seed(42)
    vectors = np.random.rand(num_vectors, dim).astype(np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / norms


def generate_frame_ids(num_ids: int) -> list:
    """Generate synthetic frame IDs."""
    return [f"frame_{i}" for i in range(num_ids)]


def percentile_95(values: List[float]) -> float:
    """Return the p95 of a list of float values."""
    if not values:
        return 0.0
    return float(np.percentile(values, 95))


def benchmark_faiss_sigclip_index(
    num_vectors: int = 1000,
    search_reps: int = 10,
):
    """Run benchmark for FaissSigLIPIndex."""
    import tempfile

    dim = 768
    top_k_values = [1, 10, 100]
    tmpdir = tempfile.mkdtemp(prefix="faiss_siglip_bench_")

    index_path = Path(tmpdir) / "faiss_siglip_index.faiss"
    mapping_path = Path(tmpdir) / "faiss_siglip_mapping.json"

    # Generate data
    print(f"Generating {num_vectors} synthetic vectors...")
    vectors = generate_synthetic_vectors(num_vectors, dim)
    frame_ids = generate_frame_ids(num_vectors)

    # Create index
    print("Creating index...")
    start_time = time.time()
    index = FaissSigLIPIndex()
    index.add(vectors, frame_ids)
    build_time = time.time() - start_time

    # Save index
    print("Saving index...")
    start_time = time.time()
    index.save(index_path, mapping_path)
    save_time = time.time() - start_time

    # Get file sizes
    index_size = index_path.stat().st_size / (1024 * 1024)  # MB
    mapping_size = mapping_path.stat().st_size / (1024 * 1024)  # MB

    # Load index
    print("Loading index...")
    start_time = time.time()
    loaded_index = FaissSigLIPIndex.load(index_path, mapping_path)
    load_time = time.time() - start_time

    # Search benchmark — multiple repetitions for stable timings
    print(f"Running search benchmarks ({search_reps} reps per top_k)...")
    query_vector = vectors[0]
    search_results = {}

    for k in top_k_values:
        timings = []
        for _ in range(search_reps):
            start_time = time.perf_counter()
            loaded_index.search(query_vector, k)
            elapsed = time.perf_counter() - start_time
            timings.append(elapsed)
        search_results[k] = {
            "avg_time_s": statistics.mean(timings),
            "p95_time_s": percentile_95(timings),
        }

    # Clean up temp dir
    import shutil
    shutil.rmtree(tmpdir, ignore_errors=True)

    results = {
        "num_vectors": num_vectors,
        "embedding_dim": dim,
        "search_reps": search_reps,
        "build_time_s": round(build_time, 6),
        "save_time_s": round(save_time, 6),
        "load_time_s": round(load_time, 6),
        "index_size_mb": round(index_size, 6),
        "mapping_size_mb": round(mapping_size, 6),
        "search_times": {
            str(k): {
                "avg_time_s": round(v["avg_time_s"], 8),
                "p95_time_s": round(v["p95_time_s"], 8),
            }
            for k, v in search_results.items()
        },
    }

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark FaissSigLIPIndex with synthetic data."
    )
    parser.add_argument(
        "--num-vectors",
        type=int,
        default=1000,
        help="Number of synthetic vectors to index (default: 1000).",
    )
    parser.add_argument(
        "--search-reps",
        type=int,
        default=10,
        help="Number of search repetitions per top_k value (default: 10).",
    )
    args = parser.parse_args()

    if args.num_vectors <= 0:
        parser.error("--num-vectors must be positive")
    if args.search_reps <= 0:
        parser.error("--search-reps must be positive")

    print(f"Num vectors: {args.num_vectors}, Search reps: {args.search_reps}")
    results = benchmark_faiss_sigclip_index(
        num_vectors=args.num_vectors,
        search_reps=args.search_reps,
    )
    print("\nBenchmark results:")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

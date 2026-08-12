"""SigLIP2 encoder benchmark for M3 evaluation.

Reports:
- model load time
- text encoding latency (batch sizes)
- image encoding latency (batch sizes)
- RAM/VRAM usage
- embedding dimension verification
"""

import gc
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
from PIL import Image

try:
    import torch
    import psutil
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def get_memory_usage_mb() -> float:
    """Get current process memory usage in MB."""
    if TORCH_AVAILABLE:
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    return 0.0


def get_gpu_memory_mb() -> float:
    """Get GPU memory usage in MB."""
    if TORCH_AVAILABLE and torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0.0


def create_test_images(count: int) -> List[Image.Image]:
    """Create test images for benchmarking."""
    images = []
    for i in range(count):
        # Vary colors slightly
        color = (i * 20 % 256, (i * 50) % 256, (i * 80) % 256)
        img = Image.new("RGB", (224, 224), color=color)
        images.append(img)
    return images


def run_benchmark():
    """Run SigLIP2 benchmark."""
    if not TORCH_AVAILABLE:
        print("torch/transformers not available - skipping benchmark")
        return

    from backend.app.embeddings.siglip2 import SigLIP2Encoder
    import backend.app.core.config as config

    config.SIGLIP_ENABLED = True

    print("=" * 80)
    print("SigLIP2 Encoder Benchmark")
    print("=" * 80)

    # Test configurations
    batch_sizes = [1, 2, 4, 8, 16]
    text_counts = [1, 10, 50, 100]
    image_counts = [1, 10, 50, 100]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Initialize encoder
    print("\n--- Model Loading ---")
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    mem_before = get_memory_usage_mb()
    start = time.perf_counter()

    encoder = SigLIP2Encoder(device="cuda" if torch.cuda.is_available() else "cpu")

    load_start = time.perf_counter()
    _ = encoder.embedding_dim  # triggers model load
    load_time = time.perf_counter() - load_start

    mem_after = get_memory_usage_mb()
    gpu_mem = get_gpu_memory_mb()

    print(f"Model load time: {load_time:.3f}s")
    print(f"RAM delta: {mem_after - mem_before:.1f} MB")
    if device == "cuda":
        print(f"GPU memory: {gpu_mem:.1f} MB")
    print(f"Embedding dim: {encoder.embedding_dim}")

    results = []

    # Text encoding benchmarks
    print("\n--- Text Encoding Benchmarks ---")
    for count in text_counts:
        texts = [f"query number {i} for benchmark testing" for i in range(count)]

        # Warmup
        _ = encoder.encode_text(texts[:min(4, count)], batch_size=4)

        for batch_size in batch_sizes:
            if batch_size > count:
                continue

            gc.collect()
            if device == "cuda":
                torch.cuda.synchronize()

            mem_start = get_memory_usage_mb()
            start = time.perf_counter()

            embeddings = encoder.encode_text(texts, batch_size=batch_size)

            if device == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            mem_end = get_memory_usage_mb()

            latency_per_item = (elapsed / count) * 1000  # ms per item
            throughput = count / elapsed

            result = {
                "type": "text",
                "count": count,
                "batch_size": batch_size,
                "latency_ms": round(elapsed * 1000, 2),
                "latency_per_item_ms": round(latency_per_item, 2),
                "throughput_items_sec": round(throughput, 1),
                "ram_delta_mb": round(get_memory_usage_mb() - mem_start, 1),
                "embedding_shape": list(embeddings.shape),
                "normalized": bool(np.allclose(np.linalg.norm(embeddings, axis=1), 1.0, atol=1e-5)),
            }
            results.append(result)
            print(f"  Text: {count} items, batch={batch_size}, "
                  f"{result['latency_ms']:.1f}ms total, "
                  f"{result['latency_per_item_ms']:.2f}ms/item, "
                  f"{result['throughput_items_sec']:.1f} items/s")

    # Image encoding benchmarks
    print("\n--- Image Encoding Benchmarks ---")
    for count in image_counts:
        images = create_test_images(count)

        # Warmup
        _ = encoder.encode_image(images[:min(4, count)], batch_size=4)

        for batch_size in batch_sizes:
            if batch_size > count:
                continue

            gc.collect()
            if device == "cuda":
                torch.cuda.synchronize()

            mem_start = get_memory_usage_mb()
            start = time.perf_counter()

            embeddings = encoder.encode_image(images, batch_size=batch_size)

            if device == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            mem_end = get_memory_usage_mb()

            latency_per_item = (elapsed / count) * 1000
            throughput = count / elapsed

            result = {
                "type": "image",
                "count": count,
                "batch_size": batch_size,
                "latency_ms": round(elapsed * 1000, 2),
                "latency_per_item_ms": round(latency_per_item, 2),
                "throughput_items_sec": round(throughput, 1),
                "ram_delta_mb": round(get_memory_usage_mb() - mem_start, 1),
                "embedding_shape": list(embeddings.shape),
                "normalized": bool(np.allclose(np.linalg.norm(embeddings, axis=1), 1.0, atol=1e-5)),
            }
            results.append(result)
            print(f"  Image: {count} items, batch={batch_size}, "
                  f"{result['latency_ms']:.1f}ms total, "
                  f"{result['latency_per_item_ms']:.2f}ms/item, "
                  f"{result['throughput_items_sec']:.1f} items/s")

    # Cross-modal similarity benchmark
    print("\n--- Cross-Modal Similarity ---")
    texts = ["query " + str(i) for i in range(10)]
    images = create_test_images(10)

    text_emb = encoder.encode_text(texts)
    img_emb = encoder.encode_image(images)

    start = time.perf_counter()
    sim = text_emb @ img_emb.T
    elapsed = time.perf_counter() - start

    print(f"  10x10 similarity matrix: {elapsed*1000:.2f}ms")
    print(f"  Shape: {sim.shape}, range: [{sim.min():.4f}, {sim.max():.4f}]")

    # Summary table
    print("\n" + "=" * 100)
    print(f"{'Type':<8} {'Count':>6} {'Batch':>6} {'Total(ms)':>10} {'PerItem(ms)':>12} {'Throughput':>12} {'Norm?':>6}")
    print("-" * 100)
    for r in results:
        print(f"{r['type']:<8} {r['count']:>6} {r['batch_size']:>6} "
              f"{r['latency_ms']:>10.1f} {r['latency_per_item_ms']:>12.2f} "
              f"{r['throughput_items_sec']:>12.1f} {str(r['normalized']):>6}")

    # Save results
    output = {
        "device": device,
        "embedding_dim": 768,
        "model_load_time_s": round(load_time, 3),
        "ram_usage_mb": round(mem_after - mem_before, 1),
        "gpu_memory_mb": round(gpu_mem, 1) if device == "cuda" else 0,
        "results": results,
    }

    output_path = Path(__file__).parent / "siglip2_benchmark_results.json"
    import json
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    run_benchmark()
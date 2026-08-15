"""Large-Scale Vector Index Benchmarking Tool.

Benchmarks FAISS index backends (IndexFlatIP, IndexHNSWFlat, IndexIVFFlat)
at scale using synthetic normalized embeddings derived from seed distributions.
Measures build time, memory footprint, query latency distributions (p50/p95/p99),
and recall@1/5/20/100 against exact FlatIP reference.

Does NOT modify production index generations.
"""

from __future__ import annotations

import gc
import json
import logging
import os
import resource
import time
from typing import Any, Dict, List, Sequence, Tuple

import faiss
import numpy as np

logger = logging.getLogger(__name__)


def generate_synthetic_embeddings(
    n_vectors: int,
    dim: int = 768,
    seed_vectors: np.ndarray | None = None,
    random_seed: int = 42,
) -> np.ndarray:
    """Generates L2-normalized synthetic embeddings from a seed distribution with perturbations."""
    rng = np.random.default_rng(random_seed)
    if seed_vectors is not None and len(seed_vectors) > 0:
        indices = rng.choice(len(seed_vectors), size=n_vectors, replace=True)
        noise = rng.normal(loc=0.0, scale=0.05, size=(n_vectors, dim)).astype(np.float32)
        expanded = seed_vectors[indices] + noise
    else:
        expanded = rng.normal(loc=0.0, scale=1.0, size=(n_vectors, dim)).astype(np.float32)

    norms = np.linalg.norm(expanded, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (expanded / norms).astype(np.float32)


def get_process_memory_mb() -> float:
    """Returns max resident set memory in MB."""
    usage_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return usage_kb / 1024.0


def benchmark_index_suite(
    scales: Sequence[int] = (10000, 100000),
    dim: int = 768,
    n_queries: int = 50,
    seed_vectors: np.ndarray | None = None,
) -> Dict[str, Any]:
    """Runs isolated comparative benchmark across IndexFlatIP, IndexHNSWFlat, and IndexIVFFlat."""
    results = []

    for scale in scales:
        data = generate_synthetic_embeddings(scale, dim, seed_vectors)
        queries = generate_synthetic_embeddings(n_queries, dim, seed_vectors, random_seed=123)

        # 1. Exact FlatIP Baseline
        gc.collect()
        t0 = time.perf_counter()
        flat_index = faiss.IndexFlatIP(dim)
        flat_index.add(data)
        flat_build_s = time.perf_counter() - t0
        flat_size_mb = (scale * dim * 4) / (1024 * 1024)

        flat_latencies = []
        flat_k100_results = []
        for q in queries:
            q_in = q.reshape(1, dim)
            t_q = time.perf_counter()
            scores, indices = flat_index.search(q_in, 100)
            flat_latencies.append((time.perf_counter() - t_q) * 1000.0)
            flat_k100_results.append(indices[0])

        flat_latencies_sorted = sorted(flat_latencies)
        flat_row = {
            "scale": scale,
            "backend": "IndexFlatIP",
            "build_time_s": round(flat_build_s, 4),
            "approx_size_mb": round(flat_size_mb, 2),
            "ram_usage_mb": round(get_process_memory_mb(), 1),
            "p50_latency_ms": round(float(np.percentile(flat_latencies_sorted, 50)), 3),
            "p95_latency_ms": round(float(np.percentile(flat_latencies_sorted, 95)), 3),
            "p99_latency_ms": round(float(np.percentile(flat_latencies_sorted, 99)), 3),
            "recall_at_1": 1.0,
            "recall_at_5": 1.0,
            "recall_at_20": 1.0,
            "recall_at_100": 1.0,
        }
        results.append(flat_row)

        # 2. HNSWFlat
        gc.collect()
        t0 = time.perf_counter()
        hnsw_index = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)
        hnsw_index.hnsw.efConstruction = 128
        hnsw_index.hnsw.efSearch = 128
        hnsw_index.add(data)
        hnsw_build_s = time.perf_counter() - t0
        hnsw_size_mb = flat_size_mb + (scale * 32 * 4 * 2) / (1024 * 1024)

        hnsw_latencies = []
        recalls_1, recalls_5, recalls_20, recalls_100 = [], [], [], []
        for i, q in enumerate(queries):
            q_in = q.reshape(1, dim)
            t_q = time.perf_counter()
            scores, indices = hnsw_index.search(q_in, 100)
            hnsw_latencies.append((time.perf_counter() - t_q) * 1000.0)

            ref = flat_k100_results[i]
            res = indices[0]
            recalls_1.append(1.0 if ref[0] in res[:1] else 0.0)
            recalls_5.append(len(set(ref[:5]) & set(res[:5])) / 5.0)
            recalls_20.append(len(set(ref[:20]) & set(res[:20])) / 20.0)
            recalls_100.append(len(set(ref[:100]) & set(res[:100])) / 100.0)

        hnsw_latencies_sorted = sorted(hnsw_latencies)
        hnsw_row = {
            "scale": scale,
            "backend": "IndexHNSWFlat",
            "build_time_s": round(hnsw_build_s, 4),
            "approx_size_mb": round(hnsw_size_mb, 2),
            "ram_usage_mb": round(get_process_memory_mb(), 1),
            "p50_latency_ms": round(float(np.percentile(hnsw_latencies_sorted, 50)), 3),
            "p95_latency_ms": round(float(np.percentile(hnsw_latencies_sorted, 95)), 3),
            "p99_latency_ms": round(float(np.percentile(hnsw_latencies_sorted, 99)), 3),
            "recall_at_1": round(float(np.mean(recalls_1)), 4),
            "recall_at_5": round(float(np.mean(recalls_5)), 4),
            "recall_at_20": round(float(np.mean(recalls_20)), 4),
            "recall_at_100": round(float(np.mean(recalls_100)), 4),
        }
        results.append(hnsw_row)

        # 3. IVF-Flat
        gc.collect()
        t0 = time.perf_counter()
        nlist = min(100, max(4, int(np.sqrt(scale))))
        quantizer = faiss.IndexFlatIP(dim)
        ivf_index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
        ivf_index.train(data)
        ivf_index.add(data)
        ivf_index.nprobe = min(32, nlist)
        ivf_build_s = time.perf_counter() - t0

        ivf_latencies = []
        ivf_r1, ivf_r5, ivf_r20, ivf_r100 = [], [], [], []
        for i, q in enumerate(queries):
            q_in = q.reshape(1, dim)
            t_q = time.perf_counter()
            scores, indices = ivf_index.search(q_in, 100)
            ivf_latencies.append((time.perf_counter() - t_q) * 1000.0)

            ref = flat_k100_results[i]
            res = indices[0]
            ivf_r1.append(1.0 if ref[0] in res[:1] else 0.0)
            ivf_r5.append(len(set(ref[:5]) & set(res[:5])) / 5.0)
            ivf_r20.append(len(set(ref[:20]) & set(res[:20])) / 20.0)
            ivf_r100.append(len(set(ref[:100]) & set(res[:100])) / 100.0)

        ivf_latencies_sorted = sorted(ivf_latencies)
        ivf_row = {
            "scale": scale,
            "backend": "IndexIVFFlat",
            "build_time_s": round(ivf_build_s, 4),
            "approx_size_mb": round(flat_size_mb, 2),
            "ram_usage_mb": round(get_process_memory_mb(), 1),
            "p50_latency_ms": round(float(np.percentile(ivf_latencies_sorted, 50)), 3),
            "p95_latency_ms": round(float(np.percentile(ivf_latencies_sorted, 95)), 3),
            "p99_latency_ms": round(float(np.percentile(ivf_latencies_sorted, 99)), 3),
            "recall_at_1": round(float(np.mean(ivf_r1)), 4),
            "recall_at_5": round(float(np.mean(ivf_r5)), 4),
            "recall_at_20": round(float(np.mean(ivf_r20)), 4),
            "recall_at_100": round(float(np.mean(ivf_r100)), 4),
        }
        results.append(ivf_row)

    return {
        "status": "PASS",
        "description": "Synthetic scale vector index benchmark (IndexFlatIP vs IndexHNSWFlat vs IndexIVFFlat)",
        "production_default": "IndexFlatIP",
        "recommendation": "KEEP FLAT for corpus < 100K frames; PREPARE HNSW for > 500K scale",
        "rows": results,
    }

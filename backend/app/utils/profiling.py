import time
from contextlib import contextmanager
from functools import wraps
from typing import Callable

import numpy as np


def timer(func: Callable) -> Callable:
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        wrapper.last_elapsed = time.perf_counter() - start
        return result

    wrapper.last_elapsed = 0.0
    return wrapper


@contextmanager
def timing():
    measurement = {"elapsed": 0.0}
    start = time.perf_counter()
    try:
        yield measurement
    finally:
        measurement["elapsed"] = time.perf_counter() - start


def measure_search_time(search_fn: Callable, query, top_k: int) -> float:
    start = time.perf_counter()
    search_fn(query, top_k)
    return time.perf_counter() - start


def measure_batch_latency(search_fn: Callable, embeddings: np.ndarray, top_k: int) -> float:
    start = time.perf_counter()
    search_fn(embeddings, top_k)
    return time.perf_counter() - start


def memory_usage_estimate(num_vectors: int, embedding_dim: int, dtype=np.float32) -> int:
    return int(num_vectors * embedding_dim * np.dtype(dtype).itemsize)

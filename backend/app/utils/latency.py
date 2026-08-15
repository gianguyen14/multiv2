import math
import time


class LatencyRecorder:
    def __init__(self):
        self.values = []

    def measure(self, function, *args, **kwargs):
        started = time.perf_counter()
        result = function(*args, **kwargs)
        self.values.append((time.perf_counter() - started) * 1000)
        return result

    def summary(self):
        values = sorted(self.values)
        if not values:
            return {"mean_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0, "p99_ms": 0.0}

        def percentile(value):
            return values[min(len(values) - 1, math.ceil(value * len(values)) - 1)]

        return {"mean_ms": sum(values) / len(values), "p50_ms": percentile(0.5),
            "p95_ms": percentile(0.95), "p99_ms": percentile(0.99)}

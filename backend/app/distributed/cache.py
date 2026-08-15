from collections import OrderedDict
import hashlib
import threading
import time

import numpy as np


class QueryCache:
    def __init__(self, max_size=1024, ttl=None):
        if max_size <= 0:
            raise ValueError("max_size must be positive")
        self.max_size = max_size
        self.ttl = ttl
        self._values = OrderedDict()
        self._lock = threading.RLock()
        self.hits = 0
        self.misses = 0

    def _key(self, query, top_k, top_n=1):
        if isinstance(query, np.ndarray):
            payload = np.asarray(query).tobytes()
        elif isinstance(query, bytes):
            payload = query
        else:
            payload = str(query).encode("utf-8")
        return hashlib.sha256(payload + f"{top_k}:{top_n}".encode("ascii")).hexdigest()

    def get(self, query, top_k, top_n=1):
        with self._lock:
            key = self._key(query, top_k, top_n)
            entry = self._values.get(key)
            if entry is None or self.ttl is not None and time.monotonic() - entry[0] >= self.ttl:
                self._values.pop(key, None)
                self.misses += 1
                return None
            self.hits += 1
            self._values.move_to_end(key)
            return [dict(item) for item in entry[1]]

    def set(self, query, top_k, results, top_n=1):
        with self._lock:
            key = self._key(query, top_k, top_n)
            self._values[key] = (time.monotonic(), [dict(item) for item in results])
            self._values.move_to_end(key)
            while len(self._values) > self.max_size:
                self._values.popitem(last=False)

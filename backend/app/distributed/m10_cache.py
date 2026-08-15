from collections import OrderedDict
import hashlib
import threading
import time

import numpy as np


class M10QueryCache:
    def __init__(self, max_size=1024, ttl=None):
        self.max_size = max_size
        self.ttl = ttl
        self._values = OrderedDict()
        self._lock = threading.RLock()
        self.hits = 0
        self.misses = 0

    def _key(self, embedding, top_k, top_n, index_type, shard_count):
        payload = np.asarray(embedding, dtype=np.float32).tobytes()
        config = f"{top_k}:{top_n}:{index_type}:{shard_count}".encode("ascii")
        return hashlib.sha256(payload + config).hexdigest()

    def get(self, embedding, top_k, top_n, index_type, shard_count):
        with self._lock:
            key = self._key(embedding, top_k, top_n, index_type, shard_count)
            entry = self._values.get(key)
            if entry is None or self.ttl is not None and time.monotonic() - entry[0] >= self.ttl:
                self._values.pop(key, None)
                self.misses += 1
                return None
            self.hits += 1
            self._values.move_to_end(key)
            return [dict(result) for result in entry[1]]

    def set(self, embedding, top_k, top_n, index_type, shard_count, results):
        with self._lock:
            key = self._key(embedding, top_k, top_n, index_type, shard_count)
            self._values[key] = (time.monotonic(), [dict(result) for result in results])
            self._values.move_to_end(key)
            while len(self._values) > self.max_size:
                self._values.popitem(last=False)

from collections import OrderedDict
import hashlib
import threading
from typing import Optional

import numpy as np
from PIL import Image


class EmbeddingCache:
    def __init__(self, max_size: int = 1024):
        if max_size <= 0:
            raise ValueError("max_size must be positive")
        self.max_size = max_size
        self._values = OrderedDict()
        self._lock = threading.RLock()

    @staticmethod
    def text_key(text: str) -> str:
        return "text:" + hashlib.sha256(text.encode("utf-8")).hexdigest()

    @staticmethod
    def image_key(image: Image.Image) -> str:
        rgb = image.convert("RGB")
        return "image:" + hashlib.sha256(rgb.tobytes()).hexdigest()

    def get(self, key: str) -> Optional[np.ndarray]:
        with self._lock:
            value = self._values.get(key)
            if value is None:
                return None
            self._values.move_to_end(key)
            return value.copy()

    def set(self, key: str, embedding: np.ndarray) -> None:
        with self._lock:
            self._values[key] = np.asarray(embedding).copy()
            self._values.move_to_end(key)
            while len(self._values) > self.max_size:
                self._values.popitem(last=False)

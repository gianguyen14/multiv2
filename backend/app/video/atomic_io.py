import json
import os
import tempfile
from pathlib import Path

import numpy as np


def fsync_directory(path):
    descriptor = os.open(Path(path), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _replace(path, writer):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            writer(handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        fsync_directory(path.parent)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def write_json_atomic(path, value):
    data = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    _replace(path, lambda handle: handle.write(data))


def write_numpy_atomic(path, array):
    _replace(path, lambda handle: np.save(handle, array, allow_pickle=False))

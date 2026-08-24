"""Availability preflight for the historical M22 end-to-end benchmark."""

from __future__ import annotations

import os
from pathlib import Path


def run_benchmark(processed_root: str | Path | None = None) -> dict[str, str]:
    value = processed_root or os.getenv("VIDEO_PROCESSED_ROOT")
    if not value:
        return {
            "end_to_end_kis": "unavailable",
            "reason": "VIDEO_PROCESSED_ROOT is not configured",
        }
    root = Path(value)
    if not (root / "index" / "CURRENT").is_file():
        return {
            "end_to_end_kis": "unavailable",
            "reason": "a published frame index is not available",
        }
    return {
        "end_to_end_kis": "unavailable",
        "reason": "benchmark queries and trusted ground truth were not provided",
    }

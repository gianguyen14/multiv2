"""Shared search-provider construction and request dispatch.

The HTTP API and operator CLI keep separate transport/output adapters, but they
must use the same backend selector and provider ``handle`` contract. This module
contains only that shared boundary; it does not start a server or parse CLI
arguments.
"""

from __future__ import annotations

import os
from typing import Any


def build_search_provider(processed_root=None):
    """Construct the configured search provider using the canonical selector."""
    backend = (os.getenv("SEARCH_BACKEND") or os.getenv("SEARCH_ENCODER") or "qwen3_vl")
    backend = backend.strip().lower()
    if backend in ("qwen3_vl", "qwen3-vl", "qwen"):
        from backend.app.services.qwen_runtime_search import QwenRuntimeSearch

        return QwenRuntimeSearch(processed_root=processed_root)
    if backend == "siglip2":
        from backend.app.services.configured_search import ConfiguredSearch

        return ConfiguredSearch(processed_root)
    raise RuntimeError(
        f"Unknown SEARCH_BACKEND={backend!r}; supported values: qwen3_vl (default), siglip2 (legacy)"
    )


def dispatch_search(provider, request: dict[str, Any]):
    """Dispatch one normalized search request through the provider contract."""
    return provider.handle(request)


def make_search_request(
    query_type: str,
    *,
    query: str = "",
    top_k: int = 100,
    events: list[str] | None = None,
    query_refine: bool = True,
    temporal_refine: bool = True,
    rerank: bool = True,
) -> dict[str, Any]:
    """Build the common request shape consumed by API and CLI adapters."""
    request: dict[str, Any] = {
        "query_type": query_type,
        "query": query,
        "top_k": top_k,
        "query_refine": query_refine,
        "temporal_refine": temporal_refine,
        "rerank": rerank,
    }
    if events is not None:
        request["events"] = events
    return request

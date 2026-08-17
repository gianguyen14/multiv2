from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from eval.p0_baseline import current_git_sha, load_ground_truth

SAFE_ENV_KEYS = (
    "SEARCH_ENABLE_OCR",
    "SEARCH_ENABLE_ASR",
    "QUERY_REFINER_ENABLED",
    "RERANKER_ENABLED",
    "TRAKE_TEMPORAL_REFINE_ENABLED",
    "VISUAL_DEVICE",
    "SEARCH_MODEL_DEVICE",
    "HF_HUB_OFFLINE",
    "TRANSFORMERS_OFFLINE",
)


def build_request(
    record: Mapping[str, Any],
    *,
    top_k: int,
    query_refine: bool,
    rerank: bool,
    temporal_refine: bool,
) -> dict[str, Any]:
    task_type = record["task_type"]
    request: dict[str, Any] = {
        "query_type": task_type,
        "top_k": top_k,
        "query_refine": query_refine,
        "rerank": rerank,
        "temporal_refine": temporal_refine,
    }
    if task_type == "trake":
        request["events"] = list(record["events"])
    else:
        request["query"] = record["query"]
    return request


def _safe_diagnostics(search: Any, task_type: str) -> dict[str, Any]:
    if task_type == "trake":
        value = getattr(search, "last_trake_metrics", None)
    else:
        value = getattr(search, "last_query_metrics", None)
    return dict(value) if isinstance(value, Mapping) else {}


def run_records(
    records: Sequence[Mapping[str, Any]],
    search: Any,
    *,
    top_k: int = 20,
    query_refine: bool = True,
    rerank: bool = True,
    temporal_refine: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    predictions: list[dict[str, Any]] = []
    task_counts = {"kis": 0, "qa": 0, "trake": 0}
    started = time.perf_counter()

    for record in records:
        request = build_request(
            record,
            top_k=top_k,
            query_refine=query_refine,
            rerank=rerank,
            temporal_refine=temporal_refine,
        )
        t0 = time.perf_counter()
        results = search.handle(request)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        if not isinstance(results, list):
            raise TypeError(f"search result for {record['query_id']} must be a list")
        task_counts[record["task_type"]] += 1
        predictions.append(
            {
                "query_id": record["query_id"],
                "category_id": record["category_id"],
                "task_type": record["task_type"],
                "latency_ms": latency_ms,
                "results": results,
                "diagnostics": _safe_diagnostics(search, record["task_type"]),
            }
        )

    manifest: dict[str, Any] = {
        "git_sha": current_git_sha(),
        "records": len(records),
        "task_counts": task_counts,
        "top_k": top_k,
        "query_refine": query_refine,
        "rerank": rerank,
        "temporal_refine": temporal_refine,
        "total_wall_ms": (time.perf_counter() - started) * 1000.0,
        "safe_environment": {key: os.getenv(key) for key in SAFE_ENV_KEYS if os.getenv(key) is not None},
    }
    status = getattr(search, "status", None)
    if callable(status):
        try:
            manifest["search_status"] = status()
        except Exception as exc:  # diagnostic only
            manifest["search_status"] = {"error_type": type(exc).__name__}
    readiness = getattr(search, "readiness", None)
    if callable(readiness):
        try:
            manifest["search_readiness"] = readiness()
        except Exception as exc:  # diagnostic only
            manifest["search_readiness"] = {"ready": False, "error_type": type(exc).__name__}
    return predictions, manifest


def write_predictions(path: str | Path, predictions: Sequence[Mapping[str, Any]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for item in predictions:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the frozen CurrentSystem over P0 ground truth")
    parser.add_argument("--ground-truth", required=True)
    parser.add_argument("--processed-root", required=True)
    parser.add_argument("--output", required=True, help="Predictions JSONL")
    parser.add_argument("--manifest", required=True, help="Run manifest JSON")
    parser.add_argument("--device", default=None)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--no-query-refine", action="store_true")
    parser.add_argument("--no-rerank", action="store_true")
    parser.add_argument("--no-temporal-refine", action="store_true")
    args = parser.parse_args(argv)

    records = load_ground_truth(args.ground_truth)
    from backend.app.services.configured_search import ConfiguredSearch

    search = ConfiguredSearch(processed_root=args.processed_root, device=args.device)
    readiness = search.readiness()
    if not readiness.get("ready"):
        reason = readiness.get("reason", "unknown readiness failure")
        raise RuntimeError(f"CurrentSystem baseline is not ready: {reason}")

    predictions, manifest = run_records(
        records,
        search,
        top_k=args.top_k,
        query_refine=not args.no_query_refine,
        rerank=not args.no_rerank,
        temporal_refine=not args.no_temporal_refine,
    )
    manifest.update(
        {
            "processed_root": str(Path(args.processed_root)),
            "device_argument": args.device,
            "readiness_before_run": readiness,
        }
    )
    write_predictions(args.output, predictions)
    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

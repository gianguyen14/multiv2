#!/usr/bin/env python3
"""Run the existing CurrentSystem retrieval path against P0 ground truth.

This module does not change retrieval configuration or scoring. It only routes
annotated KIS / QA / TRAKE queries through ConfiguredSearch.handle(), records
wall-clock latency, normalizes outputs for evaluation/p0_baseline.py, and writes
an explicit provenance manifest.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from evaluation.p0_baseline import _load_jsonl, _validate_ground_truth


def _git_sha() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def build_request(
    record: Mapping[str, Any],
    *,
    top_k: int,
    query_refine: bool,
    rerank: bool,
    temporal_refine: bool,
) -> dict[str, Any]:
    task = record["task_type"]
    request: dict[str, Any] = {
        "query_type": task,
        "top_k": top_k,
        "query_refine": query_refine,
        "rerank": rerank,
    }
    if task == "trake":
        request["events"] = list(record.get("events", []))
        request["temporal_refine"] = temporal_refine
    else:
        request["query"] = record.get("query", "")
    return request


def normalize_prediction(record: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], latency_ms: float) -> dict[str, Any]:
    task = record["task_type"]
    base: dict[str, Any] = {
        "query_id": record["query_id"],
        "category_id": record.get("category_id"),
        "task_type": task,
        "latency_ms": float(latency_ms),
    }
    if task == "kis":
        base["results"] = [dict(row) for row in rows]
        return base

    top = dict(rows[0]) if rows else {}
    if task == "qa":
        base["result"] = {
            "video_id": top.get("video_id"),
            "frame_id": top.get("frame_id"),
            "answer": top.get("answer", ""),
        }
        return base

    frame_ids = top.get("frame_ids")
    if not isinstance(frame_ids, list):
        frame_ids = [event.get("frame_id") for event in top.get("events", []) if isinstance(event, Mapping)]
    base["result"] = {
        "video_id": top.get("video_id"),
        "frames": frame_ids if isinstance(frame_ids, list) else [],
    }
    return base


def run_records(
    records: Sequence[Mapping[str, Any]],
    search: Any,
    *,
    top_k: int = 20,
    query_refine: bool = True,
    rerank: bool = True,
    temporal_refine: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
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
        rows = search.handle(request)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        if not isinstance(rows, list):
            raise TypeError(f"search result for {record['query_id']} must be a list")
        task_counts[record["task_type"]] += 1
        prediction = normalize_prediction(record, rows, latency_ms)
        if record["task_type"] == "trake" and hasattr(search, "last_trake_metrics"):
            prediction["diagnostics"] = dict(getattr(search, "last_trake_metrics") or {})
        elif hasattr(search, "last_query_metrics"):
            prediction["diagnostics"] = dict(getattr(search, "last_query_metrics") or {})
        predictions.append(prediction)

    manifest: dict[str, Any] = {
        "measurement_scope": "REAL_BASELINE_CANDIDATE",
        "git_sha": _git_sha(),
        "records": len(records),
        "task_counts": task_counts,
        "top_k": top_k,
        "query_refine": query_refine,
        "rerank": rerank,
        "temporal_refine": temporal_refine,
        "total_wall_ms": (time.perf_counter() - started) * 1000.0,
    }
    readiness = getattr(search, "readiness", None)
    if callable(readiness):
        try:
            manifest["search_readiness"] = readiness()
        except Exception as exc:
            manifest["search_readiness"] = {"ready": False, "error_type": type(exc).__name__}
    status = getattr(search, "status", None)
    if callable(status):
        try:
            manifest["search_status"] = status()
        except Exception as exc:
            manifest["search_status"] = {"error_type": type(exc).__name__}
    return predictions, manifest


def write_predictions(path: Path, predictions: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for prediction in predictions:
            handle.write(json.dumps(prediction, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--processed-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--no-query-refine", action="store_true")
    parser.add_argument("--no-rerank", action="store_true")
    parser.add_argument("--no-temporal-refine", action="store_true")
    args = parser.parse_args()
    if args.top_k <= 0:
        raise ValueError("top_k must be positive")

    records = _load_jsonl(args.ground_truth)
    _validate_ground_truth(records)

    from backend.app.services.configured_search import ConfiguredSearch

    search = ConfiguredSearch(processed_root=args.processed_root, device=args.device)
    predictions, manifest = run_records(
        records,
        search,
        top_k=args.top_k,
        query_refine=not args.no_query_refine,
        rerank=not args.no_rerank,
        temporal_refine=not args.no_temporal_refine,
    )
    manifest.update({
        "processed_root": str(args.processed_root),
        "device_argument": args.device,
    })
    write_predictions(args.output, predictions)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

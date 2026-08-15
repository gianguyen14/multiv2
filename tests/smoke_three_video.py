"""End-to-End Sanity Smoke Test on Three-Video Final Corpus."""

import json
import time
from backend.app.services.configured_search import ConfiguredSearch


def run_smoke():
    search = ConfiguredSearch(processed_root="data/processed-validation/three-video-final")
    print(f"ConfiguredSearch initialized: {search.configured}")

    queries = [
        {"type": "kis", "query": "người đi xe máy mặc áo mưa màu đỏ"},
        {"type": "kis", "query": "xe lam 79H-6072"},
        {"type": "qa", "query": "biển số xe lam màu gì?"},
        {"type": "trake", "events": ["người đi bộ", "xe ô tô rẽ", "người dừng lại"]},
    ]

    for q in queries:
        t0 = time.perf_counter()
        if q["type"] == "kis":
            res = search.search(q["query"], top_k=5)
            metrics = getattr(search, "last_query_metrics", {})
        elif q["type"] == "qa":
            res = search.handle({"query_type": "qa", "query": q["query"], "top_k": 5})
            metrics = getattr(search, "last_query_metrics", {})
        elif q["type"] == "trake":
            res = search.search_trake(q["events"], top_k=5)
            metrics = getattr(search, "last_trake_metrics", {})

        dt = (time.perf_counter() - t0) * 1000.0
        print("=" * 60)
        print(f"QUERY TYPE: {q['type'].upper()} | QUERY/EVENTS: {q.get('query') or q.get('events')}")
        print(f"Results Count: {len(res)} | Total Execution Time: {dt:.2f} ms")
        if res:
            top1 = res[0]
            print(f"Top-1 Video: {top1.get('video_id')}")
            print(f"Top-1 Frame (zero-based): {top1.get('source_frame_index_zero_based') or top1.get('frame_id')}")
            print(f"Top-1 Submission Frame: {top1.get('submission_frame_id')}")
            print(f"Top-1 Score: {float(top1.get('score', 0.0)):.6f}")
            if q["type"] == "qa":
                print(f"Top-1 QA Answer: {top1.get('answer')} (confidence: {top1.get('confidence')})")
            if q["type"] == "trake":
                print(f"TRAKE Aligned Frame IDs: {top1.get('frame_ids')}")
                print(f"TRAKE Coherence: {top1.get('coherence')}")
        print(f"Timings & Metrics: {json.dumps(metrics, indent=2, default=str)}")


if __name__ == "__main__":
    run_smoke()

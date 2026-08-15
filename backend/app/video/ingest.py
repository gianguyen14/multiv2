import argparse
import json
from pathlib import Path

from backend.app.config.video_ingest_config import VideoIngestConfig
from backend.app.embeddings.siglip2 import SigLIP2Encoder
from backend.app.video.frame_index import build_frame_index
from backend.app.video.m15_ingestion_pipeline import VideoIngestionPipeline


SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm", ".avi"}


def discover_videos(input_path):
    input_path = Path(input_path)
    if input_path.is_file():
        return [input_path] if input_path.suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS else []
    return sorted(path for path in input_path.rglob("*") if path.is_file() and path.suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS)


def ingest_path(input_path, encoder, config, limit=None, force=False, fail_fast=False, stage_failpoint=None, index_failpoint=None):
    videos = discover_videos(input_path)
    if limit is not None:
        videos = videos[:limit]
    identifiers = {}
    for video in videos:
        if video.stem in identifiers and video.resolve() != identifiers[video.stem].resolve():
            raise ValueError(f"duplicate video_id: {video.stem}")
        identifiers[video.stem] = video
    pipeline = VideoIngestionPipeline(encoder, config, failpoint=stage_failpoint)
    results, failures = [], []
    for video in videos:
        try:
            results.append(pipeline.ingest_video(video, force=force))
        except Exception as exc:
            failures.append({"source_path": str(video), "error": f"{type(exc).__name__}: {exc}"})
            if fail_fast:
                break
    identity = pipeline._encoder_identity()
    manifests = [manifest for manifest in pipeline.store.manifests()
        if manifest.status in {"embeddings_ready", "indexed"}
        and manifest.completed_stage == "embeddings"
        and manifest.failed_stage is None and manifest.error is None
        and pipeline.store.validate_metadata(manifest, config.metadata_fingerprint())
        and pipeline.store.validate_frames(manifest, config.frames_fingerprint(), pipeline.policy, config.frame_format)
        and pipeline.store.validate_embeddings(manifest, config.embeddings_fingerprint(identity), identity["embedding_dim"])]
    indexed = 0
    indexing_ms = 0.0
    publication_error = None
    active_generation = None
    if manifests:
        import time
        started = time.perf_counter()
        try:
            bundle = build_frame_index(pipeline.store, manifests, config.processed_root / "index",
                identity["embedding_dim"], config.index_type, failpoint=index_failpoint)
            indexed = bundle.index.index.ntotal
            active_generation = bundle.generation_id
        except Exception as exc:
            publication_error = f"{type(exc).__name__}: {exc}"
            from backend.app.video.frame_index import current_generation_id, load_current_frame_index
            active_generation = current_generation_id(config.processed_root / "index")
            if active_generation:
                indexed = load_current_frame_index(config.processed_root / "index").index.index.ntotal
        indexing_ms = (time.perf_counter() - started) * 1000
    return {"videos_discovered": len(videos), "videos_succeeded": len(results),
        "videos_failed": len(failures), "videos_resumed": sum(result["status"] == "resumed" for result in results),
        "indexed_frames": indexed, "active_generation": active_generation,
        "publication_error": publication_error, "indexing_ms": indexing_ms,
        "results": results, "failures": failures}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="data/processed/videos")
    parser.add_argument("--sample-interval", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()
    config = VideoIngestConfig(processed_root=Path(args.output), sample_interval_seconds=args.sample_interval,
        embed_batch_size=args.batch_size, device=args.device, resume=args.resume)
    encoder = SigLIP2Encoder(device=args.device, force_download=False)
    report = ingest_path(args.input, encoder, config, args.limit, args.force, args.fail_fast)
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(1 if report["videos_failed"] else 0)


if __name__ == "__main__":
    main()

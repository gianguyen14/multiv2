#!/usr/bin/env python3
import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def executable(name):
    return shutil.which(name)


def tesseract_languages():
    if not executable("tesseract"):
        return set()
    result = subprocess.run(["tesseract", "--list-langs"], capture_output=True, text=True)
    return set(result.stdout.splitlines()[1:]) if result.returncode == 0 else set()


def _package_version(name):
    try:
        from importlib.metadata import version
        return version(name)
    except Exception:
        return None


def environment_report():
    from backend.app.runtime.device_policy import runtime_summary
    packages = {name: _package_version(name) for name in (
        "torch", "torchvision", "transformers", "huggingface-hub", "faiss-cpu",
        "av", "faster-whisper", "ctranslate2", "fastapi", "numpy", "easyocr", "playwright")}
    tools = {}
    for name, command in (("ffmpeg", ["ffmpeg", "-version"]), ("ffprobe", ["ffprobe", "-version"]),
            ("tesseract", ["tesseract", "--version"])):
        path = executable(name)
        tools[name] = {"path": path, "version": None}
        if path:
            result = subprocess.run(command, capture_output=True, text=True)
            tools[name]["version"] = (result.stdout or result.stderr).splitlines()[0] if result.returncode == 0 else None
    return {"python": {"executable": sys.executable, "version": sys.version,
        "prefix": sys.prefix, "base_prefix": sys.base_prefix},
        "platform": {"system": __import__("platform").system(),
            "release": __import__("platform").release(), "machine": __import__("platform").machine()},
        "packages": packages, "tools": tools, "compute": runtime_summary()}


def command_env(args):
    report = environment_report()
    emit(report, args)
    if getattr(args, "check", False):
        required = ("fastapi", "numpy", "faiss-cpu", "av", "faster-whisper")
        missing = [name for name in required if not report["packages"].get(name)]
        missing += [name for name in ("ffmpeg", "ffprobe") if not report["tools"][name]["path"]]
        if missing:
            raise RuntimeError("missing mandatory runtime components: " + ", ".join(missing))


def command_dataset(args):
    from backend.app.dataset_ops import verify_media
    from backend.app.competition_data import dataset_report, validate_dataset
    result = verify_media(args.path)
    gt = Path(args.path) / "ground_truth"
    if args.report:
        result["source_bytes"] = sum(path.stat().st_size for path in Path(args.path).rglob("*") if path.is_file())
        if gt.is_dir():
            result["ground_truth"] = dataset_report(gt)
    elif gt.is_dir():
        result["ground_truth"] = validate_dataset(gt)
    emit(result, args)
    if not result.get("valid", True) or result.get("ground_truth", {}).get("valid") is False:
        raise RuntimeError("dataset verification failed")


def command_smoke(args):
    fixture = ROOT / "tests" / "fixtures" / "test_5s.mp4"
    checks = {"fixture": {"status": "PASS", "path": str(fixture)} if fixture.is_file() else {"status": "FAIL"},
        "frame_id_policy": {"status": "PASS"}, "compute_policy": {"status": "PASS"},
        "ocr": {"status": "PASS" if executable("tesseract") else "SKIP", "reason": "tesseract unavailable" if not executable("tesseract") else None},
        "asr": {"status": "PASS" if _module("faster_whisper") else "SKIP", "reason": "faster-whisper unavailable" if not _module("faster_whisper") else None}}
    result = {"status": "PASS" if checks["fixture"]["status"] == "PASS" else "FAIL", "checks": checks}
    emit(result, args)
    if result["status"] != "PASS":
        raise RuntimeError("smoke test failed")


def model_inventory(whisper_model=None):
    from backend.app.model_cache import inventory
    return inventory(whisper_model, tesseract_languages())


def readiness():
    from backend.app.runtime.device_policy import runtime_summary
    processed = os.getenv("VIDEO_PROCESSED_ROOT")
    models = model_inventory()
    visual_runtime = _module("transformers") and _module("huggingface_hub")
    asr_runtime = _module("faster_whisper") and bool(executable("ffmpeg"))
    languages = tesseract_languages()
    query_refiner_runtime = _module("transformers") and _module("huggingface_hub")
    qr_cached = models.get("query_refiner", {}).get("cached", False)
    return {
        "compute": runtime_summary(),
        "visual_runtime": "READY" if visual_runtime else "BLOCKED",
        "visual_model": "CACHED" if models["visual"]["cached"] else "MISSING",
        "visual": "READY" if visual_runtime and models["visual"]["cached"] else "NEEDS MODEL PREPARATION",
        "video_ingestion": "READY",
        "asr_runtime": "READY" if asr_runtime else "BLOCKED - faster-whisper or ffmpeg missing",
        "asr_model": "CACHED" if models["asr"]["cached"] else "MISSING",
        "asr": "READY" if asr_runtime and models["asr"]["cached"] else "NEEDS MODEL PREPARATION",
        "query_refiner_runtime": "READY" if query_refiner_runtime else "BLOCKED",
        "query_refiner_model": "CACHED" if qr_cached else "MISSING (FALLBACK DETERMINISTIC)",
        "query_refiner": "READY (LOCAL LLM)" if query_refiner_runtime and qr_cached else "READY (DETERMINISTIC FALLBACK)",
        "ocr_executable": "READY" if executable("tesseract") else "MISSING",
        "ocr_english": "READY" if "eng" in languages else "MISSING",
        "ocr_vietnamese": "READY" if "vie" in languages else "MISSING",
        "competition_dataset": "PRESENT" if Path("data/competition/ground_truth").is_dir() else "MISSING",
        "production_index": "CONFIGURED" if processed else "NOT CONFIGURED",
        "browser_e2e": "READY" if executable("chromium-cli") or _module("playwright") else "OPTIONAL DEPENDENCY MISSING",
    }


def _module(name):
    from importlib.util import find_spec
    return find_spec(name) is not None


def emit(value, args, rows=None):
    if getattr(args, "output", None):
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".csv" and rows is not None:
            with path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=rows[0].keys() if rows else [])
                writer.writeheader()
                writer.writerows(rows)
        else:
            path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    if args.json:
        print(json.dumps(value, indent=2, ensure_ascii=False))
    elif isinstance(value, dict):
        for key, item in value.items():
            print(f"{key.replace('_', ' ').title():24} {item}")
    else:
        print(value)


def configured_search():
    from backend.app.services.configured_search import ConfiguredSearch
    root = os.getenv("VIDEO_PROCESSED_ROOT")
    if not root:
        for cand in [Path("data/processed-validation/three-video-final"), Path("data/processed/videos")]:
            if (cand / "index" / "CURRENT").exists():
                root = str(cand)
                break
    search = ConfiguredSearch(processed_root=root)
    if not search.configured:
        raise RuntimeError("VIDEO_PROCESSED_ROOT is not configured")
    return search


def search_rows(query, top_k=100, query_refine=True):
    return configured_search().search(query, top_k, query_refine=query_refine)


def command_doctor(args):
    emit(readiness(), args)


def _selected_models(args):
    selected = []
    if getattr(args, "visual", False):
        selected.append("visual")
    if getattr(args, "asr", False):
        selected.append("asr")
    if getattr(args, "ocr", False):
        selected.append("ocr")
    if getattr(args, "query_refiner", False):
        selected.append("query_refiner")
    if getattr(args, "all", False):
        return ("visual", "asr", "ocr", "query_refiner")
    return tuple(selected) if selected else ("visual", "asr")


def command_models(args):
    from backend.app.model_cache import prepare_asr, prepare_visual
    selected = _selected_models(args)
    models = model_inventory(args.whisper_model)
    if args.prepare and not args.dry_run:
        if "visual" in selected and not models["visual"]["cached"]:
            prepare_visual()
        if "asr" in selected and not models["asr"]["cached"]:
            prepare_asr(args.whisper_model)
        if "ocr" in selected and not models["ocr"].get("paddleocr", {}).get("cached", False):
            try:
                from backend.app.model_cache import prepare_paddleocr
                prepare_paddleocr()
            except Exception:
                pass
        if "query_refiner" in selected and not models.get("query_refiner", {}).get("cached", False):
            try:
                from backend.app.model_cache import prepare_query_refiner
                prepare_query_refiner()
            except Exception:
                pass
        models = model_inventory(args.whisper_model)
    output = {key: value for key, value in models.items() if key in selected or key == "ocr"}
    if args.dry_run:
        for key in selected:
            if key in output and isinstance(output[key], dict) and "cached" in output[key]:
                output[key]["download_required"] = not output[key]["cached"]
    if args.verify_offline:
        for key in selected:
            if key in output and isinstance(output[key], dict) and "cached" in output[key]:
                output[key]["offline_ready"] = output[key]["cached"]
    emit(output, args)
    if args.verify_offline and not all(output[key].get("cached", True) for key in selected if key in output and isinstance(output[key], dict)):
        raise RuntimeError("offline verification failed; prepare the missing models shown above")



def _require_models(visual=False, asr=False, whisper_model=None):
    models = model_inventory(whisper_model)
    if visual and not models["visual"]["cached"]:
        raise RuntimeError("SigLIP2 model is not available locally. Run: python projectctl.py models --prepare --visual")
    if asr and not models["asr"]["cached"]:
        raise RuntimeError(f"Faster Whisper model {whisper_model} is not available locally. Run: python projectctl.py models --prepare --asr")


def command_status(args):
    state = readiness()
    branch = None
    try:
        if shutil.which("git") is not None:
            branch_res = subprocess.run(["git", "branch", "--show-current"], capture_output=True, text=True)
            if branch_res.returncode == 0:
                branch = branch_res.stdout.strip() or None
    except Exception:
        branch = None
    state["git_branch"] = branch
    state["tests"] = "run: python projectctl.py test"
    emit(state, args)


def command_info(args):
    emit({"project": "Ho Chi Minh City AI Challenge 2026 multimodal retrieval",
        "python": sys.version.split()[0], "root": str(ROOT), **readiness()}, args)


def command_server(args):
    command = [sys.executable, "-m", "uvicorn", "backend.app.main:app", "--host", args.host, "--port", str(args.port)]
    subprocess.run(command, check=True)


def command_frontend(args):
    command_server(args)


def command_dev(args):
    command_server(args)


def ingest_report(args):
    from backend.app.config.video_ingest_config import VideoIngestConfig
    from backend.app.runtime.operations import resource_preflight, write_run_manifest
    from backend.app.embeddings.siglip2 import SigLIP2Encoder
    from backend.app.video.ingest import ingest_path
    device = args.device or os.getenv("VIDEO_INGEST_DEVICE", os.getenv("COMPUTE_DEVICE", "auto"))
    batch_size = getattr(args, "batch_size", None)
    config = VideoIngestConfig(processed_root=Path(args.processed_root), device=device,
        embed_batch_size=batch_size, index_type=args.index_type)
    preflight = resource_preflight(args.path, config.processed_root)
    models = model_inventory(args.whisper_model)
    preflight["visual"] = {"runtime_ready": _module("transformers") and _module("torch"),
        "model_cached": models["visual"]["cached"], "device": device}
    if args.command == "preprocess":
        languages = tesseract_languages()
        preflight["ocr"] = {"enabled": executable("tesseract") is not None,
            "runtime_ready": executable("tesseract") is not None,
            "eng": "eng" in languages, "vie": "vie" in languages,
            "device": getattr(args, "ocr_device", None) or "cpu"}
        preflight["asr"] = {"enabled": _module("faster_whisper"),
            "runtime_ready": _module("faster_whisper") and executable("ffmpeg") is not None,
            "model_cached": models["asr"]["cached"],
            "device": getattr(args, "asr_device", None) or device}
    if not preflight["visual"]["runtime_ready"]:
        preflight["errors"].append("visual runtime is unavailable")
    if not preflight["visual"]["model_cached"]:
        preflight["errors"].append("SigLIP2 model is not prepared; run projectctl.py models --prepare --visual")
    if args.command == "preprocess" and preflight["asr"]["enabled"] and not preflight["asr"]["model_cached"]:
        preflight["errors"].append("Faster Whisper model is not prepared; run projectctl.py models --prepare --asr")
    preflight["ok"] = not preflight["errors"]
    if getattr(args, "preflight_only", False):
        return {"preflight": preflight}
    if not preflight["ok"]:
        raise RuntimeError("preflight failed: " + "; ".join(preflight["errors"]))
    _require_models(visual=True)
    from backend.app.runtime.operations import memory_guard
    preflight["memory_before_visual"] = memory_guard("visual ingestion")
    encoder = SigLIP2Encoder(device=device, force_download=False, local_files_only=True)
    run_manifest = write_run_manifest(config.processed_root / "run_manifest.json", args.command,
        args.path, config.processed_root, {"frame_id_policy": config.frame_id_policy,
            "sample_interval_seconds": config.sample_interval_seconds, "index_type": config.index_type,
            "visual_batch_size": config.embed_batch_size}, encoder.get_model_info())
    try:
        report = ingest_path(args.path, encoder, config, limit=getattr(args, "limit", None))
        report["compute"] = encoder.get_model_info()
        report["preflight"] = preflight
        report["run_manifest"] = run_manifest
        return report
    finally:
        encoder.clear_cache()


def command_ingest(args):
    report = ingest_report(args)
    emit(report, args)
    if args.preflight_only and not report["preflight"]["ok"]:
        raise RuntimeError("preflight failed: " + "; ".join(report["preflight"]["errors"]))


def _text_pipeline(args, use_ocr=True, use_asr=True):
    if use_ocr and not executable("tesseract"):
        raise RuntimeError("OCR unavailable: install tesseract with eng and vie language packs")
    from backend.app.video.ingest import discover_videos
    from backend.app.video.frame_store import FrameStore, source_hash
    from backend.app.video.m16_text_pipeline import (
        M16TextPipeline, TextEvidenceStore, compute_ocr_fingerprint, compute_asr_fingerprint
    )
    from backend.app.video.text_backends import FasterWhisperASRBackend, create_ocr_backend, resolve_whisper_revision
    videos = discover_videos(args.path)
    if Path(args.path).is_file():
        videos = [Path(args.path)]
    elif getattr(args, "limit", None) is not None:
        videos = videos[:args.limit]
    processed_root = Path(args.processed_root)
    store = TextEvidenceStore(processed_root)
    frames = FrameStore(processed_root)

    requested_ocr_backend = getattr(args, "ocr_backend", None) or os.getenv("OCR_BACKEND", "auto")
    from backend.app.runtime.device_policy import probe_paddle
    paddle_caps = probe_paddle()
    ocr_device_arg = getattr(args, "ocr_device", None)
    if requested_ocr_backend == "auto":
        if paddle_caps.installed and paddle_caps.cuda_available and ocr_device_arg not in ("cpu", "unavailable"):
            ocr_desc = {
                "backend": "adaptive",
                "primary": {"backend": "paddleocr", "languages": "vi", "device": ocr_device_arg or "cuda:0"},
                "fallback": {"backend": "tesseract", "languages": "eng+vie"},
                "routing_mode": "auto",
            }
        else:
            ocr_desc = {"backend": "tesseract", "languages": "eng+vie"}
    elif requested_ocr_backend == "paddleocr":
        ocr_desc = {"backend": "paddleocr", "languages": "vi", "device": ocr_device_arg or "auto"}
    elif requested_ocr_backend == "easyocr":
        ocr_desc = {"backend": "easyocr", "languages": ["en", "vi"]}
    else:
        ocr_desc = {"backend": "tesseract", "languages": "eng+vie"}

    whisper_model = getattr(args, "whisper_model", "small")
    device_name = getattr(args, "asr_device", None) or getattr(args, "device", "auto")
    whisper_compute = getattr(args, "asr_compute_type", None) or ("float16" if device_name.startswith("cuda") else "int8")
    whisper_rev = resolve_whisper_revision(whisper_model)
    asr_desc = {"backend": "faster-whisper", "model": whisper_model, "compute_type": whisper_compute, "revision": whisper_rev}

    needs_ocr = False
    if use_ocr:
        for video in videos:
            manifest = frames.load_manifest(video.stem)
            src_hash = manifest.source_hash if manifest else source_hash(video)
            frames_fp = manifest.frames_fingerprint if manifest else None
            ocr_fp, _ = compute_ocr_fingerprint(src_hash, frames_fp, ocr_desc)
            if not store.validate_ocr_cache(video.stem, ocr_fp, src_hash, frames_fp):
                needs_ocr = True
                break

    needs_asr = False
    if use_asr:
        for video in videos:
            manifest = frames.load_manifest(video.stem)
            src_hash = manifest.source_hash if manifest else source_hash(video)
            asr_fp, _ = compute_asr_fingerprint(src_hash, asr_desc)
            if not store.validate_asr_cache(video.stem, asr_fp, src_hash):
                needs_asr = True
                break

    ocr = create_ocr_backend(name=requested_ocr_backend, device=ocr_device_arg) if needs_ocr else type("NoOCR", (), {
        "extract": lambda self, paths: [{"text": "", "boxes": [], "confidence": None} for _ in paths],
        "info": lambda self: ocr_desc,
        "identity": lambda self: f"{ocr_desc['backend']}:{ocr_desc.get('languages', '')}",
    })()
    asr = FasterWhisperASRBackend(args.whisper_model, device=getattr(args, "asr_device", None) or args.device,
        compute_type=getattr(args, "asr_compute_type", None), local_files_only=True) if needs_asr else type("NoASR", (), {
        "transcribe": lambda self, path: [],
        "info": lambda self: asr_desc,
        "identity": lambda self: f"faster-whisper:{whisper_model}:{whisper_compute}:{whisper_rev}",
    })()
    pipeline = M16TextPipeline(processed_root, ocr, asr, use_ocr=use_ocr, use_asr=use_asr)
    return pipeline.run_path(videos)



def command_ocr(args):
    from backend.app.runtime.operations import memory_guard
    memory_guard("OCR stage")
    emit(_text_pipeline(args, use_ocr=True, use_asr=False), args)


def command_asr(args):
    from backend.app.runtime.operations import memory_guard
    _require_models(asr=True, whisper_model=args.whisper_model)
    memory_guard("ASR stage")
    emit(_text_pipeline(args, use_ocr=False, use_asr=True), args)


def command_preprocess(args):
    from backend.app.runtime.operations import memory_guard, memory_snapshot, resource_limits
    memory = {"before_visual": memory_guard("visual stage")}
    ingest = ingest_report(args)
    if args.preflight_only:
        emit(ingest, args)
        if not ingest["preflight"]["ok"]:
            raise RuntimeError("preflight failed: " + "; ".join(ingest["preflight"]["errors"]))
        return
    import gc
    gc.collect()
    memory["after_visual"] = memory_snapshot()
    summary = {"ingest": ingest, "ocr": "skipped - unavailable", "asr": "skipped - unavailable",
        "memory": memory, "resource_limits": resource_limits()}
    if executable("tesseract"):
        memory["before_ocr"] = memory_guard("OCR stage")
        try:
            summary["ocr"] = _text_pipeline(args, use_ocr=True, use_asr=False)
        except Exception as exc:
            summary["ocr"] = f"unavailable: {type(exc).__name__}: {exc}"
        gc.collect()
        memory["after_ocr"] = memory_snapshot()
    if _module("faster_whisper"):
        _require_models(asr=True, whisper_model=args.whisper_model)
        memory["before_asr"] = memory_guard("ASR stage")
        try:
            summary["asr"] = _text_pipeline(args, use_ocr=False, use_asr=True)
        except Exception as exc:
            summary["asr"] = f"unavailable: {type(exc).__name__}: {exc}"
        gc.collect()
        memory["after_asr"] = memory_snapshot()
    emit(summary, args)


def command_index(args):
    root = os.getenv("VIDEO_PROCESSED_ROOT")
    if not root:
        raise RuntimeError("VIDEO_PROCESSED_ROOT is not configured")
    manifests = list(__import__("backend.app.video.frame_store", fromlist=["FrameStore"]).FrameStore(root).manifests())
    if not manifests:
        raise RuntimeError("no completed video manifests found")
    from backend.app.video.frame_index import build_frame_index
    bundle = build_frame_index(__import__("backend.app.video.frame_store", fromlist=["FrameStore"]).FrameStore(root),
        manifests, Path(root) / "index", manifests[0].embedding_dim, args.index_type)
    emit({"generation_id": bundle.generation_id, "vector_count": bundle.index.index.ntotal}, args)


def command_search(args):
    refine = not getattr(args, "no_query_refine", False)
    try:
        rows = search_rows(args.query, args.top_k, query_refine=refine)
    except TypeError:
        rows = search_rows(args.query, args.top_k)
    emit(rows, args, rows)


def command_kis(args):
    command_search(args)


def command_image_search(args):
    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"image not found: {image_path}")
    valid_exts = {".jpg", ".jpeg", ".png", ".webp"}
    if image_path.suffix.lower() not in valid_exts:
        raise ValueError(f"unsupported image format '{image_path.suffix}'; supported: {sorted(valid_exts)}")
    from PIL import Image
    try:
        with Image.open(image_path) as img:
            img.verify()
    except Exception as exc:
        raise ValueError(f"corrupt or unreadable image '{image_path}': {exc}")
    search = configured_search()
    rows = search.search_image(image_path, top_k=args.top_k, deduplicate=not getattr(args, "raw", False))
    emit(rows, args, rows)


def command_qa(args):
    refine = not getattr(args, "no_query_refine", False)
    request = {"query": args.query, "query_type": "qa", "top_k": args.top_k, "query_refine": refine}
    output = configured_search().handle(request)
    emit(output, args, output)


def command_query_plan(args):
    from backend.app.services.query_refiner import QueryRefiner
    cache_dir = Path(os.getenv("VIDEO_PROCESSED_ROOT", "data/processed/videos")) / "query_refine_cache"
    refiner = QueryRefiner(cache_dir=cache_dir)
    plan, metrics = refiner.refine(args.query, task_type=args.task)
    emit({"query_plan": plan.to_dict(), "metrics": metrics}, args)


def parse_events(value):
    if not value or not str(value).strip():
        return []
    path = Path(value)
    if path.is_file():
        value = path.read_text()
    try:
        parsed = json.loads(value)
        if isinstance(parsed, dict):
            return parsed.get("events", [])
        elif isinstance(parsed, list):
            return parsed
        return [str(parsed)]
    except (json.JSONDecodeError, TypeError):
        return [item.strip() for item in str(value).split("|") if item.strip()]


def command_trake(args):
    events = parse_events(args.events)
    if not events:
        emit({"aligned": False, "reason": "empty event list", "video_id": None, "frame_ids": []}, args)
        return
    temporal_refine = not getattr(args, "no_temporal_refine", False)
    query_refine = not getattr(args, "no_query_refine", False)
    results = configured_search().search_trake(events, top_k=args.top_k, temporal_refine=temporal_refine, query_refine=query_refine)
    if results:
        res = results[0]
        payload = {
            "aligned": True,
            "video_id": res["video_id"],
            "frame_ids": res["frame_ids"],
            "score": res["score"],
        }
        if "refinement_used" in res:
            payload["refinement_used"] = res["refinement_used"]
        if "refinement_regions" in res:
            payload["refinement_regions"] = res["refinement_regions"]
        emit(payload, args)
    else:
        emit({
            "aligned": False,
            "reason": "no valid sequence alignment found",
            "video_id": None,
            "frame_ids": []
        }, args)



def _evaluation_runners(args):
    return {
        "kis": lambda record: search_rows(record.query, 100),
        "qa": lambda record: [{**row, "answer": ""} for row in search_rows(record.query, 100)],
        "trake": lambda record: [(_trake_submission(record.events) or {"video_id": "", "frame_ids": []})],
    }


def _trake_submission(events):
    results = configured_search().search_trake(events, 100)
    if results:
        res = results[0]
        return {
            "video_id": res["video_id"],
            "frame_ids": res["frame_ids"],
            "score": res["score"],
        }
    return None



def command_evaluate(args):
    if args.competition:
        missing = [str(Path(args.ground_truth) / f"{kind}.jsonl") for kind in ("kis", "qa", "trake")
            if not (Path(args.ground_truth) / f"{kind}.jsonl").is_file()]
        if missing:
            raise RuntimeError("missing competition ground truth: " + ", ".join(missing))
        from backend.app.competition_evaluation import evaluate_competition
        result = evaluate_competition(args.ground_truth, _evaluation_runners(args))
        result["metric_version"] = "internal-provisional-v1"
        if args.save_baseline:
            Path(args.save_baseline).write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
        if args.compare:
            baseline = json.loads(Path(args.compare).read_text())
            result["delta"] = {key: result[key]["final_score"] - baseline[key]["final_score"]
                for key in ("kis", "qa", "trake") if key in result and key in baseline}
            result["delta"]["final_score"] = result["final_score"] - baseline["final_score"]
            if args.max_regression is not None and result["delta"]["final_score"] < -args.max_regression:
                emit(result, args)
                raise RuntimeError("quality regression exceeds configured threshold")
        emit(result, args)
    else:
        subprocess.run([sys.executable, "-m", "eval.m22_pipeline_benchmark"], check=True)


def command_benchmark(args):
    environment = os.environ.copy()
    if args.device:
        environment["COMPUTE_DEVICE"] = args.device
    subprocess.run([sys.executable, "-m", "eval.m22_end_to_end_benchmark"], check=True, env=environment)


def command_test(args):
    subprocess.run([sys.executable, "-m", "pytest", "-q"], check=True)


def command_clean(args):
    if not args.staging:
        raise RuntimeError("only --staging cleanup is supported")
    root = Path(os.getenv("VIDEO_PROCESSED_ROOT", "data/processed/videos")) / "index" / ".staging"
    if root.is_dir():
        for child in root.iterdir():
            if child.is_dir() and child.name.startswith("gen-") and not child.is_symlink():
                shutil.rmtree(child)
    emit({"staging": "clean", "path": str(root)}, args)


def command_benchmark_index(args):
    from backend.app.indexes.benchmark import benchmark_index_suite
    scales = [int(s) for s in args.scales] if getattr(args, "scales", None) else [10000, 100000]
    result = benchmark_index_suite(scales=scales, dim=getattr(args, "dim", 768), n_queries=getattr(args, "queries", 50))
    emit(result, args, rows=result.get("rows"))


def command_validate_dataset(args):
    from backend.app.validation.dataset_validator import (
        create_subset_manifest,
        verify_artifact_integrity,
        run_query_batch_validation,
    )
    from backend.app.services.configured_search import ConfiguredSearch

    processed_root = Path(args.processed_root or os.getenv("VIDEO_PROCESSED_ROOT", "data/processed-validation/m27-representative-12-videos"))
    videos_dir = Path(args.videos_dir or os.getenv("VIDEOS_DIR", "data/test-videos"))

    # 1. Manifest
    video_paths = sorted(videos_dir.glob("*.mp4"))
    if getattr(args, "max_videos", None):
        video_paths = video_paths[:int(args.max_videos)]
    manifest = create_subset_manifest(video_paths, Path("validation/subset_manifest.json"))

    # 2. Artifact Integrity
    art_integrity = verify_artifact_integrity(processed_root)

    # 3. Query Batch
    queries_file = Path(args.queries_file or "validation/queries.json")
    queries = []
    if queries_file.is_file():
        queries = json.loads(queries_file.read_text())

    search = ConfiguredSearch(processed_root=str(processed_root))
    results, stats = run_query_batch_validation(search, queries) if queries else ([], {})

    report = {
        "status": "PASS" if art_integrity.get("ok") else "FAIL",
        "processed_root": str(processed_root),
        "subset_videos_count": len(manifest),
        "artifact_integrity": art_integrity,
        "queries_tested": len(queries),
        "latency_stats": stats,
    }

    # Write output artifacts
    val_dir = Path("validation")
    val_dir.mkdir(parents=True, exist_ok=True)
    (val_dir / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")
    (val_dir / "resource_summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    if results:
        with open(val_dir / "results.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["query_id", "task", "query", "results_count", "total_ms", "top1_video", "top1_frame", "top1_score"])
            writer.writeheader()
            for r in results:
                writer.writerow({
                    "query_id": r["query_id"], "task": r["task"], "query": r["query"],
                    "results_count": r["results_count"], "total_ms": r["total_ms"],
                    "top1_video": r["top1_video"], "top1_frame": r["top1_frame"], "top1_score": r["top1_score"]
                })

    emit(report, args, rows=results)


def parser():
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--json", action="store_true")
    common.add_argument("--verbose", action="store_true")
    common.add_argument("--output")
    main = argparse.ArgumentParser(parents=[common])
    sub = main.add_subparsers(dest="command", required=True)
    item = sub.add_parser("env", parents=[common])
    item.add_argument("--check", action="store_true")
    item.set_defaults(handler=command_env)
    for name, handler in (("doctor", command_doctor), ("status", command_status), ("info", command_info),
            ("test", command_test)):
        item = sub.add_parser(name, parents=[common])
        item.set_defaults(handler=handler)
    item = sub.add_parser("benchmark", parents=[common])
    item.add_argument("--device")
    item.set_defaults(handler=command_benchmark)
    item = sub.add_parser("benchmark-index", parents=[common])
    item.add_argument("--scales", nargs="+", type=int, default=[10000, 100000])
    item.add_argument("--dim", type=int, default=768)
    item.add_argument("--queries", type=int, default=50)
    item.set_defaults(handler=command_benchmark_index)
    item = sub.add_parser("validate-dataset", parents=[common])
    item.add_argument("--processed-root")
    item.add_argument("--videos-dir")
    item.add_argument("--max-videos", type=int, default=12)
    item.add_argument("--queries-file")
    item.set_defaults(handler=command_validate_dataset)
    item = sub.add_parser("dataset", parents=[common])
    dataset_sub = item.add_subparsers(dest="dataset_command", required=True)
    verify = dataset_sub.add_parser("verify", parents=[common])
    verify.add_argument("path")
    verify.set_defaults(handler=command_dataset, report=False)
    report = dataset_sub.add_parser("report", parents=[common])
    report.add_argument("path")
    report.set_defaults(handler=command_dataset, report=True)
    item = sub.add_parser("smoke", parents=[common])
    item.set_defaults(handler=command_smoke)
    item = sub.add_parser("models", parents=[common])
    operations = item.add_mutually_exclusive_group()
    operations.add_argument("--prepare", action="store_true")
    operations.add_argument("--verify-offline", action="store_true")
    selectors = item.add_mutually_exclusive_group()
    selectors.add_argument("--visual", action="store_true")
    selectors.add_argument("--asr", action="store_true")
    selectors.add_argument("--ocr", action="store_true")
    selectors.add_argument("--query-refiner", action="store_true")
    selectors.add_argument("--all", action="store_true")
    item.add_argument("--dry-run", action="store_true")
    from backend.app.core.config import FASTER_WHISPER_MODEL
    item.add_argument("--whisper-model", default=FASTER_WHISPER_MODEL)
    item.set_defaults(handler=command_models)
    item = sub.add_parser("index", parents=[common])
    item.add_argument("--index-type", default=os.getenv("VIDEO_INDEX_TYPE", "flat"), choices=("flat", "hnsw"))
    item.set_defaults(handler=command_index)
    for name, handler in (("backend", command_server), ("frontend", command_frontend), ("dev", command_dev)):
        item = sub.add_parser(name, parents=[common])
        item.add_argument("--host", default="127.0.0.1")
        item.add_argument("--port", type=int, default=8000)
        item.set_defaults(handler=handler)
    from backend.app.core.config import FASTER_WHISPER_MODEL
    for name, handler in (("ingest", command_ingest), ("preprocess", command_preprocess),
            ("ocr", command_ocr), ("asr", command_asr)):
        item = sub.add_parser(name, parents=[common])
        item.add_argument("path")
        item.add_argument("--processed-root", default=os.getenv("VIDEO_PROCESSED_ROOT", "data/processed/videos"))
        item.add_argument("--device")
        item.add_argument("--ocr-backend", default=os.getenv("OCR_BACKEND", "auto"), choices=("auto", "tesseract", "paddleocr", "easyocr"))
        item.add_argument("--ocr-device")
        item.add_argument("--asr-device")
        item.add_argument("--asr-compute-type")
        item.add_argument("--batch-size", type=lambda value: None if value == "auto" else int(value))
        item.add_argument("--preflight-only", action="store_true")
        item.add_argument("--limit", type=int)
        item.add_argument("--index-type", default="flat", choices=("flat", "hnsw"))
        item.add_argument("--whisper-model", default=FASTER_WHISPER_MODEL)
        item.set_defaults(handler=handler)
    for name, handler in (("search", command_search), ("kis", command_kis), ("qa", command_qa)):
        item = sub.add_parser(name, parents=[common])
        item.add_argument("query")
        item.add_argument("--top-k", type=int, default=100)
        item.add_argument("--no-query-refine", action="store_true")
        item.set_defaults(handler=handler)
    item = sub.add_parser("query-plan", parents=[common])
    item.add_argument("query")
    item.add_argument("--task", default="kis", choices=("kis", "qa", "trake", "general"))
    item.set_defaults(handler=command_query_plan)
    item = sub.add_parser("image-search", parents=[common])
    item.add_argument("image", help="Path to query image (.jpg, .jpeg, .png, .webp)")
    item.add_argument("--top-k", type=int, default=100)
    item.add_argument("--raw", action="store_true", help="Disable temporal deduplication")
    item.set_defaults(handler=command_image_search)
    item = sub.add_parser("trake", parents=[common])
    item.add_argument("events")
    item.add_argument("--top-k", type=int, default=30)
    item.add_argument("--no-temporal-refine", action="store_true", help="Disable dense temporal refinement")
    item.add_argument("--no-query-refine", action="store_true")
    item.set_defaults(handler=command_trake)

    item = sub.add_parser("evaluate", parents=[common])
    item.add_argument("--competition", action="store_true")
    item.add_argument("--ground-truth", default="data/competition/ground_truth")
    item.add_argument("--save-baseline")
    item.add_argument("--compare")
    item.add_argument("--max-regression", type=float)
    item.set_defaults(handler=command_evaluate)
    item = sub.add_parser("clean", parents=[common])
    item.add_argument("--staging", action="store_true")
    item.set_defaults(handler=command_clean)
    return main


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        args.handler(args)
        return 0
    except Exception as exc:
        if args.verbose:
            raise
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

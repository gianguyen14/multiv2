import json
import os
import resource
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


def memory_snapshot():
    values = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, value = line.split(":", 1)
        values[key] = int(value.strip().split()[0]) * 1024
    return {"process_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "total_ram_bytes": values["MemTotal"], "available_ram_bytes": values["MemAvailable"],
        "swap_total_bytes": values["SwapTotal"], "swap_free_bytes": values["SwapFree"]}


def memory_guard(stage):
    snapshot = memory_snapshot()
    fraction = float(os.getenv("MAX_MEMORY_FRACTION", "0.75"))
    minimum = int(os.getenv("MIN_AVAILABLE_MEMORY_BYTES", str(2 * 1024 ** 3)))
    used_fraction = 1 - snapshot["available_ram_bytes"] / snapshot["total_ram_bytes"]
    if snapshot["available_ram_bytes"] < minimum or used_fraction > fraction:
        raise RuntimeError(f"insufficient available memory for {stage}; existing artifacts were preserved")
    return {"stage": stage, **snapshot}


def resource_limits():
    return {"max_memory_fraction": float(os.getenv("MAX_MEMORY_FRACTION", "0.75")),
        "max_workers": int(os.getenv("MAX_WORKERS", "1")),
        "decode_workers": int(os.getenv("DECODE_WORKERS", "1")),
        "ocr_workers": int(os.getenv("OCR_WORKERS", "1")),
        "asr_concurrency": int(os.getenv("ASR_CONCURRENCY", "1")),
        "visual_batch_size": os.getenv("VISUAL_BATCH_SIZE", "auto")}


def resource_preflight(source, processed_root):
    source, processed_root = Path(source), Path(processed_root)
    errors, warnings = [], []
    if not source.exists():
        errors.append(f"input does not exist: {source}")
    processed_root.mkdir(parents=True, exist_ok=True)
    if not processed_root.is_dir() or not processed_root.stat():
        errors.append(f"processed root is unavailable: {processed_root}")
    free = shutil.disk_usage(processed_root).free
    source_bytes = source.stat().st_size if source.is_file() else sum(
        path.stat().st_size for path in source.rglob("*") if path.is_file()) if source.exists() else 0
    if source_bytes and free < source_bytes:
        errors.append("free disk is smaller than source data size")
    elif source_bytes and free < source_bytes * 2:
        warnings.append("free disk is less than twice the source data size")
    for tool in ("ffmpeg", "ffprobe"):
        if shutil.which(tool) is None:
            errors.append(f"missing executable: {tool}")
    return {"ok": not errors, "errors": errors, "warnings": warnings,
        "free_disk_bytes": free, "source_bytes": source_bytes,
        "processed_root": str(processed_root)}


def _get_git_provenance():
    try:
        if shutil.which("git") is None:
            return {"git_commit": None, "git_dirty": False, "provenance_source": "unavailable"}
        commit_res = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
        commit = commit_res.stdout.strip() if commit_res.returncode == 0 else None
        dirty_res = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)
        dirty = bool(dirty_res.stdout.strip()) if dirty_res.returncode == 0 else False
        return {
            "git_commit": commit or None,
            "git_dirty": dirty,
            "provenance_source": "git" if commit else "unavailable",
        }
    except Exception:
        return {"git_commit": None, "git_dirty": False, "provenance_source": "unavailable"}


def write_run_manifest(path, command, source, processed_root, config, compute):
    git_info = _get_git_provenance()
    manifest = {
        "run_id": uuid4().hex,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_info["git_commit"],
        "git_dirty": git_info["git_dirty"],
        "provenance_source": git_info["provenance_source"],
        "python_executable": sys.executable,
        "python_version": sys.version.split()[0],
        "command": command,
        "source_path": str(Path(source).resolve()),
        "processed_root": str(Path(processed_root).resolve()),
        "config": config,
        "compute": compute,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)
    return manifest

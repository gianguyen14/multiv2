import json
from pathlib import Path

from backend.app.competition_data import dataset_report, validate_dataset


def verify_media(root):
    root = Path(root)
    videos = [path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in {".mp4", ".mov", ".mkv", ".webm", ".avi"}]
    errors = []
    for path in videos:
        if not path.stat().st_size:
            errors.append(f"empty video: {path}")
    return {"path": str(root), "video_count": len(videos), "errors": errors, "valid": not errors}

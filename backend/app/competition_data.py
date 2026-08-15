import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class KISGroundTruth:
    query_id: str
    query: str
    video_id: str
    start_frame: int
    end_frame: int


@dataclass(frozen=True)
class QAGroundTruth:
    query_id: str
    query: str
    video_id: str
    start_frame: int
    end_frame: int
    answers: list[str]


@dataclass(frozen=True)
class TRAKEGroundTruth:
    query_id: str
    events: list[str]
    video_id: str
    windows: list[dict]


RECORD_TYPES = {"kis": KISGroundTruth, "qa": QAGroundTruth, "trake": TRAKEGroundTruth}


def validate_dataset(root, require_media=False):
    root = Path(root)
    errors, counts = [], {}
    for record_type in ("kis", "qa", "trake"):
        path = root / f"{record_type}.jsonl"
        if not path.is_file():
            errors.append(f"missing {path}")
            counts[record_type] = 0
            continue
        try:
            records = load_jsonl(path, record_type)
            ids = [record.query_id for record in records]
            duplicates = sorted({item for item in ids if ids.count(item) > 1})
            if duplicates:
                errors.append(f"duplicate {record_type} query_id: {', '.join(duplicates)}")
            counts[record_type] = len(records)
            if require_media:
                media_root = root.parent.parent / "videos"
                for record in records:
                    if not (media_root / record.video_id).exists():
                        errors.append(f"missing media for {record.query_id}: {record.video_id}")
        except (OSError, ValueError) as exc:
            errors.append(f"{record_type}: {exc}")
            counts[record_type] = 0
    return {"path": str(root), "counts": counts, "errors": errors, "valid": not errors}


def dataset_report(root):
    result = validate_dataset(root)
    result["source_bytes"] = sum(path.stat().st_size for path in Path(root).rglob("*") if path.is_file())
    return result


def load_jsonl(path, record_type):
    cls = RECORD_TYPES[record_type]
    records = []
    with Path(path).open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = cls(**json.loads(line))
            except (TypeError, json.JSONDecodeError) as exc:
                raise ValueError(f"invalid {record_type} record at line {line_number}") from exc
            _validate(record)
            records.append(record)
    return records


def _validate(record):
    if not record.query_id or not record.video_id:
        raise ValueError("query_id and video_id are required")
    if isinstance(record, (KISGroundTruth, QAGroundTruth)):
        if record.start_frame < 0 or record.end_frame < record.start_frame:
            raise ValueError("invalid frame interval")
    if isinstance(record, QAGroundTruth) and not record.answers:
        raise ValueError("QA answers are required")
    if isinstance(record, TRAKEGroundTruth):
        if len(record.events) != len(record.windows) or not record.events:
            raise ValueError("TRAKE events and windows must be non-empty and aligned")
        for window in record.windows:
            if window["start_frame"] < 0 or window["end_frame"] < window["start_frame"]:
                raise ValueError("invalid TRAKE frame interval")


class OrganizerAdapter:
    def convert(self, source_path, destination_root):
        raise NotImplementedError("implement conversion from the organizer export to internal JSONL schemas")

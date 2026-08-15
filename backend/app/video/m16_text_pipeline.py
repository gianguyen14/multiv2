import hashlib
import json
import time
from bisect import bisect_left
from pathlib import Path

from backend.app.video.atomic_io import write_json_atomic
from backend.app.video.frame_store import FrameStore, source_hash
from backend.app.video.text_evidence import ASRSegment, OCRRecord
from backend.app.video.video_decoder import iter_frames


def _fingerprint_digest(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def compute_ocr_fingerprint(source_hash, frames_fingerprint, ocr_backend_info, repeated_text_gap_seconds=2.0, schema_version=1):
    payload = {
        "schema_version": schema_version,
        "source_hash": source_hash,
        "frames_fingerprint": frames_fingerprint,
        "ocr_backend": ocr_backend_info,
        "repeated_text_gap_seconds": repeated_text_gap_seconds,
    }
    return _fingerprint_digest(payload), payload


def compute_asr_fingerprint(source_hash, asr_backend_info, schema_version=1):
    payload = {
        "schema_version": schema_version,
        "source_hash": source_hash,
        "asr_backend": asr_backend_info,
    }
    return _fingerprint_digest(payload), payload


class TextEvidenceStore:
    def __init__(self, root):
        self.root = Path(root)

    def _path(self, video_id, name):
        return self.root / video_id / name

    def save_ocr(self, video_id, records, meta=None):
        write_json_atomic(self._path(video_id, "ocr.json"), [record.to_dict() for record in records])
        if meta is not None:
            write_json_atomic(self._path(video_id, "ocr_meta.json"), meta)

    def load_ocr(self, video_id):
        raw = json.loads(self._path(video_id, "ocr.json").read_text())
        if not isinstance(raw, list):
            raise ValueError(f"Corrupt OCR data for {video_id}: expected list")
        return [OCRRecord.from_dict(value) for value in raw]

    def load_ocr_meta(self, video_id):
        path = self._path(video_id, "ocr_meta.json")
        if not path.is_file():
            return None
        try:
            return json.loads(path.read_text())
        except Exception:
            return None

    def save_asr(self, video_id, segments, meta=None):
        write_json_atomic(self._path(video_id, "asr.json"), [segment.to_dict() for segment in segments])
        if meta is not None:
            write_json_atomic(self._path(video_id, "asr_meta.json"), meta)

    def load_asr(self, video_id):
        raw = json.loads(self._path(video_id, "asr.json").read_text())
        if not isinstance(raw, list):
            raise ValueError(f"Corrupt ASR data for {video_id}: expected list")
        return [ASRSegment.from_dict(value) for value in raw]

    def load_asr_meta(self, video_id):
        path = self._path(video_id, "asr_meta.json")
        if not path.is_file():
            return None
        try:
            return json.loads(path.read_text())
        except Exception:
            return None

    def validate_ocr_cache(self, video_id, expected_fingerprint=None, source_hash=None, frames_fingerprint=None):
        path = self._path(video_id, "ocr.json")
        if not path.is_file():
            return False
        try:
            raw = json.loads(path.read_text())
            if not isinstance(raw, list):
                return False
        except Exception:
            return False
        if expected_fingerprint is None:
            return True
        meta = self.load_ocr_meta(video_id)
        if meta is not None:
            return meta.get("fingerprint") == expected_fingerprint
        # Safe legacy fallback: if meta is absent, check manifest agreement
        manifest_path = self.root / video_id / "manifest.json"
        if manifest_path.is_file():
            try:
                manifest_data = json.loads(manifest_path.read_text())
                if source_hash and manifest_data.get("source_hash") != source_hash:
                    return False
                if frames_fingerprint and manifest_data.get("frames_fingerprint") != frames_fingerprint:
                    return False
                return True
            except Exception:
                return False
        return True

    def validate_asr_cache(self, video_id, expected_fingerprint=None, source_hash=None):
        path = self._path(video_id, "asr.json")
        if not path.is_file():
            return False
        try:
            raw = json.loads(path.read_text())
            if not isinstance(raw, list):
                return False
        except Exception:
            return False
        if expected_fingerprint is None:
            return True
        meta = self.load_asr_meta(video_id)
        if meta is not None:
            return meta.get("fingerprint") == expected_fingerprint
        # Safe legacy fallback: if meta is absent, check manifest agreement
        manifest_path = self.root / video_id / "manifest.json"
        if manifest_path.is_file():
            try:
                manifest_data = json.loads(manifest_path.read_text())
                if source_hash and manifest_data.get("source_hash") != source_hash:
                    return False
                return True
            except Exception:
                return False
        return True


class M16TextPipeline:
    def __init__(self, processed_root, ocr_backend, asr_backend, repeated_text_gap_seconds=2.0,
            use_ocr=True, use_asr=True):
        self.root = Path(processed_root)
        self.frames = FrameStore(self.root)
        self.store = TextEvidenceStore(self.root)
        self.ocr_backend = ocr_backend
        self.asr_backend = asr_backend
        self.repeated_text_gap_seconds = repeated_text_gap_seconds
        self.use_ocr = use_ocr
        self.use_asr = use_asr

    def _frame_timeline(self, source_path):
        return [(frame.timestamp_seconds, frame.source_frame_index_zero_based)
            for frame in iter_frames(source_path) if frame.timestamp_seconds is not None]

    @staticmethod
    def _nearest_frame(timeline, seconds):
        if not timeline:
            return None
        timestamps = [item[0] for item in timeline]
        position = bisect_left(timestamps, seconds)
        choices = timeline[max(0, position - 1):position + 1]
        return min(choices, key=lambda item: (abs(item[0] - seconds), item[0]))[1]

    def run_video(self, source_path, force=False, timings=None):
        video_id = Path(source_path).stem
        manifest = self.frames.load_manifest(video_id)
        source_digest = manifest.source_hash if manifest else (source_hash(source_path) if Path(source_path).is_file() else "")
        frames_fp = manifest.frames_fingerprint if manifest else None

        ocr_info = self.ocr_backend.info() if hasattr(self.ocr_backend, "info") else {"backend": type(self.ocr_backend).__name__}
        asr_info = self.asr_backend.info() if hasattr(self.asr_backend, "info") else {"backend": type(self.asr_backend).__name__}

        ocr_fp, ocr_payload = compute_ocr_fingerprint(source_digest, frames_fp, ocr_info, self.repeated_text_gap_seconds)
        asr_fp, asr_payload = compute_asr_fingerprint(source_digest, asr_info)

        ocr_valid = self.store.validate_ocr_cache(video_id, ocr_fp, source_digest, frames_fp) if self.use_ocr else True
        asr_valid = self.store.validate_asr_cache(video_id, asr_fp, source_digest) if self.use_asr else True

        if not force and ocr_valid and asr_valid:
            return {
                "video_id": video_id,
                "status": "resumed",
                "ocr_count": len(self.store.load_ocr(video_id)) if self.use_ocr and self.store._path(video_id, "ocr.json").is_file() else 0,
                "asr_count": len(self.store.load_asr(video_id)) if self.use_asr and self.store._path(video_id, "asr.json").is_file() else 0,
            }

        timings = timings if timings is not None else {}
        ocr_records = self.store.load_ocr(video_id) if ocr_valid and self.store._path(video_id, "ocr.json").is_file() else []
        if self.use_ocr and (force or not ocr_valid):
            records = self.frames.load_records(video_id)
            started = time.perf_counter()
            extracted = self.ocr_backend.extract([record.image_path for record in records])
            timings["ocr_seconds"] = time.perf_counter() - started
            ocr_records, previous = [], None
            for record, result in zip(records, extracted, strict=True):
                item = OCRRecord.create(record, result.get("text", ""), result.get("boxes"), result.get("confidence"))
                if not item.normalized_text:
                    continue
                if (previous and item.normalized_text == previous.normalized_text
                        and item.timestamp_seconds is not None and previous.timestamp_seconds is not None
                        and item.timestamp_seconds - previous.timestamp_seconds <= self.repeated_text_gap_seconds):
                    continue
                ocr_records.append(item)
                previous = item
            self.store.save_ocr(video_id, ocr_records, {**ocr_payload, "fingerprint": ocr_fp, "record_count": len(ocr_records)})

        asr_segments = self.store.load_asr(video_id) if asr_valid and self.store._path(video_id, "asr.json").is_file() else []
        if self.use_asr and (force or not asr_valid):
            timeline = self._frame_timeline(source_path)
            asr_segments = []
            started = time.perf_counter()
            transcript = self.asr_backend.transcribe(source_path)
            timings["asr_seconds"] = time.perf_counter() - started
            for index, value in enumerate(transcript):
                text = value.get("text", "")
                if not text.strip():
                    continue
                start = float(value["start_seconds"])
                end = float(value["end_seconds"])
                asr_segments.append(ASRSegment.create(video_id, index, start, end,
                    self._nearest_frame(timeline, start), self._nearest_frame(timeline, end), text,
                    value.get("language"), value.get("confidence")))
            self.store.save_asr(video_id, asr_segments, {**asr_payload, "fingerprint": asr_fp, "segment_count": len(asr_segments)})

        return {
            "video_id": video_id,
            "status": "completed",
            "ocr_count": len(ocr_records),
            "asr_count": len(asr_segments),
        }

    def run_path(self, source_paths, force=False):
        results, failures = [], []
        for source_path in source_paths:
            try:
                results.append(self.run_video(source_path, force))
            except Exception as exc:
                failures.append({"source_path": str(source_path), "error": f"{type(exc).__name__}: {exc}"})
        return {"succeeded": len(results), "failed": len(failures), "results": results, "failures": failures}


"""Qwen3-VL packed-DB search provider for the operator API.

This provider serves ``/api/search`` from the *verified* AIC Qwen runtime DB:
a published FAISS generation (47,430 x 1024-d, normalized inner product) with
frame payloads plus released OCR/ASR spool evidence.  It mirrors the exact
retrieval semantics of the verified Hermes query runtime (``aic_query.py``):

- encode the query with Qwen3-VL-Embedding-2B (instruction + MRL truncation +
  float32 L2 normalization);
- FAISS inner-product recall of ``max(top_k * 2, 200)``;
- visual-score min-max normalization over the recalled set;
- lexical OCR/ASR hits mapped to their nearest indexed frame per video;
- deterministic fusion ``0.70 * visual + 0.18 * ocr + 0.12 * asr``;
- sort by ``(-score, video_id, frame_id)`` and rank the top-k.

The DB is opened read-only and is never copied, moved, rewritten or ingested.
The SigLIP2 search path is untouched and remains the default; this provider is
selected explicitly with ``SEARCH_ENCODER=qwen3_vl``.
"""

from __future__ import annotations

import bisect
import os
import re
import threading
import time
import unicodedata
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Optional

from backend.app.embeddings.qwen3_vl import (
    DEFAULT_INSTRUCTION,
    Qwen3VlLocalEmbedder,
    resolve_model_dir,
    weights_available,
)
from backend.app.video.frame_index import load_current_frame_index

# Deterministic fusion weights from the verified query runtime.
VISUAL_FUSION_WEIGHT = 0.70
OCR_FUSION_WEIGHT = 0.18
ASR_FUSION_WEIGHT = 0.12

_STOPWORDS = frozenset({
    "có", "là", "và", "của", "trong", "ở", "một", "những", "cho", "để", "với",
    "không", "đến", "các", "thì", "mà", "như", "the", "a", "an", "of", "in", "on", "to",
})

QWEN_BACKEND_NAMES = {"qwen3_vl", "qwen3_vl_embedding_2b", "qwen3-vl-embedding-2b"}


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", value or "")).strip().casefold()


def content_tokens(value: str) -> list[str]:
    return [
        token for token in re.findall(r"[\w]+", normalize_text(value), flags=re.UNICODE)
        if token not in _STOPWORDS
    ]


def lexical_score(query: str, text: str) -> float:
    """Dice-coefficient / phrase lexical overlap used by the verified runtime."""
    q = content_tokens(query)
    t = content_tokens(text)
    qs, ts = set(q), set(t)
    if not qs or not ts:
        return 0.0
    overlap = qs & ts
    if not overlap:
        return 0.0
    dice = 2.0 * len(overlap) / (len(qs) + len(ts))
    phrase = 1.0 if len(q) >= 2 and " ".join(q) in " ".join(t) else 0.0
    return min(1.0, max(dice, phrase))


def build_timeline(payloads: Mapping[str, dict[str, Any]]):
    rows: dict[str, list[tuple[float, str]]] = defaultdict(list)
    for uid, payload in payloads.items():
        timestamp = payload.get("timestamp_seconds")
        if timestamp is not None:
            rows[payload["video_id"]].append((float(timestamp), uid))
    for values in rows.values():
        values.sort()
    return rows


def nearest_uid(timeline, video_id: str, timestamp: float) -> Optional[str]:
    values = timeline.get(video_id)
    if not values:
        return None
    position = bisect.bisect_left(values, (float(timestamp), ""))
    options = []
    if position < len(values):
        options.append(values[position])
    if position:
        options.append(values[position - 1])
    return min(options, key=lambda item: abs(item[0] - float(timestamp)))[1]


class QwenRuntimeSearch:
    """Serves text queries against the Qwen3-VL packed DB using runtime semantics."""

    def __init__(
        self,
        processed_root=None,
        *,
        model_dir=None,
        encoder_factory=None,
        instruction: str = DEFAULT_INSTRUCTION,
        enable_ocr: bool = True,
        enable_asr: bool = True,
    ):
        value = processed_root or os.getenv("VIDEO_PROCESSED_ROOT")
        self.processed_root = Path(value) if value else None
        self.model_dir = Path(resolve_model_dir(model_dir))
        self.instruction = instruction
        self.enable_ocr = enable_ocr and os.getenv("SEARCH_ENABLE_OCR", "true").lower() == "true"
        self.enable_asr = enable_asr and os.getenv("SEARCH_ENABLE_ASR", "true").lower() == "true"
        self.encoder_factory = encoder_factory or (
            lambda: Qwen3VlLocalEmbedder(model_dir=self.model_dir, instruction=self.instruction)
        )
        self._lock = threading.Lock()
        self._bundle = None
        self._encoder = None
        self._ocr_records = []
        self._asr_records = []
        self._timeline = {}
        self._dimension = None
        self._generation_id = None
        self._validated_generation_id = None
        # Attributes read by the operator API layer.
        self.last_query_plan = None
        self.last_query_metrics = {}

    # -- configuration -----------------------------------------------------

    @property
    def configured(self) -> bool:
        return self.processed_root is not None

    def _index_root(self) -> Path:
        return self.processed_root / "index"

    def _ocr_root(self) -> Path:
        return self.processed_root / "ocr"

    def _asr_root(self) -> Path:
        return self.processed_root / "asr"

    # -- loading (read-only) ----------------------------------------------

    def _load_evidence_rows(self, root: Path, kind: str) -> list[dict[str, Any]]:
        rows = []
        if not root.is_dir():
            return rows
        for path in sorted(root.glob("*.json")):
            try:
                records = path.read_text(encoding="utf-8")
            except OSError:
                continue
            import json

            try:
                parsed = json.loads(records)
            except ValueError:
                continue
            if not isinstance(parsed, list):
                continue
            for row in parsed:
                if not isinstance(row, dict):
                    continue
                if kind == "ocr":
                    text = row.get("normalized_text") or row.get("raw_text", "")
                    timestamp = float(row.get("timestamp_seconds") or 0.0)
                    raw = row.get("raw_text", "")
                else:
                    text = (
                        row.get("normalized_transcript")
                        or row.get("normalized_text")
                        or row.get("raw_transcript", "")
                    )
                    start = float(row.get("start_seconds") or 0.0)
                    end = float(row.get("end_seconds") or 0.0)
                    timestamp = (start + end) / 2.0
                    raw = row.get("raw_transcript") or row.get("raw_text", "")
                rows.append({
                    "video_id": str(row.get("video_id", "")),
                    "timestamp_seconds": timestamp,
                    "text": normalize_text(text),
                    "raw_text": raw,
                })
        return rows

    def _initialize(self) -> None:
        if self._bundle is not None:
            return
        if not self.configured:
            raise RuntimeError("VIDEO_PROCESSED_ROOT is not configured")
        with self._lock:
            if self._bundle is not None:
                return
            bundle = load_current_frame_index(self._index_root())
            encoder_identity = bundle.metadata.get("encoder_identity") or {}
            backend_name = str(encoder_identity.get("backend", "")).lower()
            if backend_name not in QWEN_BACKEND_NAMES:
                raise RuntimeError(
                    "refusing to serve a non-Qwen index with the qwen3_vl encoder: "
                    f"index encoder backend is {backend_name!r}"
                )
            payloads = bundle.resolver.payloads
            self._bundle = bundle
            self._timeline = build_timeline(payloads)
            self._dimension = int(bundle.metadata["embedding_dim"])
            self._generation_id = bundle.metadata["generation_id"]
            self._encoder = self.encoder_factory()
            if self.enable_ocr:
                self._ocr_records = self._load_evidence_rows(self._ocr_root(), "ocr")
            if self.enable_asr:
                self._asr_records = self._load_evidence_rows(self._asr_root(), "asr")

    # -- introspection -----------------------------------------------------

    def status(self) -> dict[str, Any]:
        initialized = self._bundle is not None
        return {
            "backend": "qwen3_vl",
            "configured": self.configured,
            "initialized": initialized,
            "processed_root": str(self.processed_root) if self.processed_root else None,
            "model_dir": str(self.model_dir),
            "generation_id": self._generation_id if initialized else None,
            "dimension": self._dimension if initialized else None,
            "ocr_records": len(self._ocr_records) if initialized else 0,
            "asr_records": len(self._asr_records) if initialized else 0,
            "weights_present": weights_available(self.model_dir),
            "capabilities": {
                "kis": True,
                "qa": True,
                "trake": True,
                "image": False,
                "thumbnails": False,
                "raw_video_preview": False,
            },
        }

    def readiness(self) -> dict[str, Any]:
        if not self.configured:
            return {"ready": False, "reason": "VIDEO_PROCESSED_ROOT is not configured"}
        try:
            from backend.app.video.frame_index import current_generation_id

            generation_id = current_generation_id(self._index_root())
            if not generation_id:
                return {"ready": False, "reason": "CURRENT index generation is missing"}
            if generation_id == self._validated_generation_id:
                return {"ready": True, "generation_id": generation_id}
            bundle = load_current_frame_index(self._index_root())
            encoder_identity = bundle.metadata.get("encoder_identity") or {}
            if str(encoder_identity.get("backend", "")).lower() not in QWEN_BACKEND_NAMES:
                return {"ready": False, "reason": "active generation is not a Qwen3-VL index"}
            if int(bundle.metadata["embedding_dim"]) <= 0:
                return {"ready": False, "reason": "invalid embedding dimension"}
            if not weights_available(self.model_dir):
                return {
                    "ready": False,
                    "reason": f"Qwen3-VL model weights are not available at {self.model_dir}",
                }
            # Cache the validated generation so readiness healthchecks do not
            # re-hash the artifact bundle on every poll (mirrors ConfiguredSearch).
            self._validated_generation_id = generation_id
            return {"ready": True, "generation_id": generation_id}
        except Exception as exc:  # mirror ConfiguredSearch.readiness behavior
            return {"ready": False, "reason": f"invalid search artifacts: {type(exc).__name__}"}

    # -- search core (mirrors the verified runtime) --------------------------

    def _top_text_hits(self, query: str, records: list[dict[str, Any]], limit: int = 500):
        best: dict[str, tuple[float, str]] = {}
        for row in records:
            score = lexical_score(query, row["text"])
            if score <= 0.0:
                continue
            uid = nearest_uid(self._timeline, row["video_id"], row["timestamp_seconds"])
            if uid is not None and (uid not in best or score > best[uid][0]):
                best[uid] = (score, row["raw_text"])
        return sorted(best.items(), key=lambda item: (-item[1][0], item[0]))[:limit]

    @staticmethod
    def _minmax(values: list[float]) -> list[float]:
        if not values:
            return []
        lo, hi = min(values), max(values)
        if hi == lo:
            return [1.0 if hi > 0 else 0.0 for _ in values]
        return [(value - lo) / (hi - lo) for value in values]

    def search_single(self, query: str, top_k: int = 100) -> list[dict[str, Any]]:
        """Runs one text query with the exact verified runtime semantics."""
        self._initialize()
        bundle = self._bundle
        assert bundle is not None and self._encoder is not None
        dimension = self._dimension
        assert dimension is not None

        vector = self._encoder.encode_query(query, dimension)
        recall_k = min(max(top_k * 2, 200), int(bundle.index.index.ntotal))
        hits = bundle.index.search(vector, recall_k)

        payloads = bundle.resolver.payloads
        visual = []
        for hit in hits:
            uid = str(hit["frame_id"])
            payload = payloads.get(uid)
            if payload is not None:
                visual.append((uid, float(hit["score"])))

        ocr = self._top_text_hits(query, self._ocr_records) if self.enable_ocr else []
        asr = self._top_text_hits(query, self._asr_records) if self.enable_asr else []

        all_uids = {uid for uid, _ in visual} | {uid for uid, _ in ocr} | {uid for uid, _ in asr}
        visual_map = dict(visual)
        ocr_map = {uid: value for uid, value in ocr}
        asr_map = {uid: value for uid, value in asr}
        vnorm = dict(
            zip(
                (uid for uid, _ in visual),
                self._minmax([score for _, score in visual]),
            )
        )

        rows = []
        for uid in all_uids:
            payload = payloads[uid]
            ocr_score, ocr_text = ocr_map.get(uid, (0.0, ""))
            asr_score, asr_text = asr_map.get(uid, (0.0, ""))
            fused = (
                VISUAL_FUSION_WEIGHT * vnorm.get(uid, 0.0)
                + OCR_FUSION_WEIGHT * ocr_score
                + ASR_FUSION_WEIGHT * asr_score
            )
            source_index = int(payload["source_frame_index_zero_based"])
            rows.append({
                "video_id": payload["video_id"],
                "frame_id": int(payload.get("submission_frame_id", source_index)),
                "source_frame_index_zero_based": source_index,
                "frame_uid": uid,
                "timestamp_seconds": payload.get("timestamp_seconds"),
                "score": float(fused),
                "visual_score": float(visual_map.get(uid, 0.0)),
                "ocr_score": float(ocr_score),
                "asr_score": float(asr_score),
                "ocr_evidence": ocr_text,
                "asr_evidence": asr_text,
            })
        rows.sort(key=lambda row: (-row["score"], row["video_id"], row["frame_id"]))
        rows = rows[: int(top_k)]
        for rank, row in enumerate(rows, 1):
            row["rank"] = rank
        return rows

    # -- API entry points ---------------------------------------------------

    @staticmethod
    def _qa_answer(row: dict[str, Any]) -> str:
        """Answer a Q&A row from its top OCR/ASR evidence (100-char cap)."""
        evidence = (row.get("ocr_evidence") or row.get("asr_evidence") or "").strip()
        if not evidence:
            return ""
        return evidence[:100]

    def search_trake(self, events: list[str], top_k: int = 100) -> list[dict[str, Any]]:
        """Ordered TRAKE: query each event, then enforce one video + increasing frames.

        Mirrors the verified runtime guidance: split the ordered events, retrieve
        candidates for each, and keep the single video whose frames can cover the
        full event sequence in strictly increasing frame order. If no single video
        covers every event, an explicit error is raised instead of returning a
        wrong-space or cross-video fallback.
        """
        self._initialize()
        events = [event for event in events if isinstance(event, str) and event.strip()]
        if not events:
            raise ValueError("TRAKE requires a non-empty ordered events list")

        per_event = [
            self.search_single(event, top_k=max(10, top_k)) for event in events
        ]

        # Candidate videos: union of videos present in the first event's results
        # (a video cannot lead the sequence if it has no first-event candidate).
        videos = {row["video_id"] for row in per_event[0]}
        best_video = None
        best_sequence = []
        for video in sorted(videos):
            sequence = []
            previous = -1
            for event_rows in per_event:
                chosen = next(
                    (
                        row for row in event_rows
                        if row["video_id"] == video
                        and row["source_frame_index_zero_based"] > previous
                    ),
                    None,
                )
                if chosen is None:
                    sequence = []
                    break
                sequence.append(chosen)
                previous = chosen["source_frame_index_zero_based"]
            if len(sequence) > len(best_sequence):
                best_video = video
                best_sequence = sequence

        if best_video is None or len(best_sequence) != len(events):
            raise ValueError(
                "TRAKE: no single video covers every event in increasing frame order"
            )

        frame_ids = [int(row["source_frame_index_zero_based"]) for row in best_sequence]
        return [{
            "video_id": best_video,
            "frame_id": frame_ids[0],
            "frame_ids": frame_ids,
            "source_frame_index_zero_based": frame_ids[0],
            "frame_uid": f"{best_video}:{str(frame_ids[0]).zfill(9)}",
            "timestamp_seconds": best_sequence[0].get("timestamp_seconds"),
            "score": float(sum(row["score"] for row in best_sequence) / len(best_sequence)),
            "visual_score": float(sum(row["visual_score"] for row in best_sequence) / len(best_sequence)),
            "ocr_score": float(sum(row["ocr_score"] for row in best_sequence) / len(best_sequence)),
            "asr_score": float(sum(row["asr_score"] for row in best_sequence) / len(best_sequence)),
            "events": [{"frame_id": frame_id} for frame_id in frame_ids],
        }]

    def handle(self, request: dict[str, Any]) -> list[dict[str, Any]]:
        """Serves ``POST /api/search`` request payloads with explicit capability errors."""
        query_type = request.get("query_type", "kis")
        top_k = int(request.get("top_k", 100))
        started = time.perf_counter()

        if query_type == "trake":
            events = request.get("events")
            if not isinstance(events, list) or not events:
                raise ValueError("TRAKE requires a non-empty ordered events list")
            results = self.search_trake(events, top_k=top_k)
        elif query_type in ("kis", "qa"):
            query = request.get("query", "")
            if not isinstance(query, str) or not query.strip():
                raise ValueError("query is required")
            results = self.search_single(query, top_k=top_k)
            if query_type == "qa":
                for row in results:
                    row["answer"] = self._qa_answer(row)
        elif query_type == "image":
            raise RuntimeError(
                "image search is not supported by the qwen3_vl backend; "
                "the packed DB is a text-query frame index"
            )
        else:
            raise ValueError(f"unsupported query_type: {query_type}")

        self.last_query_metrics = {
            "backend": "qwen3_vl",
            "generation_id": self._generation_id,
            "total_query_ms": round((time.perf_counter() - started) * 1000.0, 2),
        }
        return results

    def search_image(self, image_or_path, top_k=100, deduplicate=True):
        """Image queries are outside the verified text runtime contract."""
        raise RuntimeError(
            "image search is not supported by the qwen3_vl backend; "
            "the packed DB is a text-query frame index"
        )

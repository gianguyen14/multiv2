"""Load and validate the historical M13.5 retrieval evaluation corpus."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CorpusCandidate:
    candidate_id: str
    image_path: Path
    metadata: dict[str, Any]


@dataclass(frozen=True)
class CorpusQuery:
    query_id: str
    text: str
    relevance: dict[str, int]


@dataclass(frozen=True)
class EvaluationCorpus:
    root: Path
    candidates: tuple[CorpusCandidate, ...]
    queries: tuple[CorpusQuery, ...]
    fingerprint: str


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON on {path.name}:{line_number}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"rows in {path.name} must be objects")
        rows.append(value)
    return rows


def load_corpus(root: str | Path) -> EvaluationCorpus:
    root = Path(root).resolve()
    candidate_rows = _read_jsonl(root / "candidates.jsonl")
    query_rows = _read_jsonl(root / "queries.jsonl")

    candidates = []
    candidate_ids = set()
    for row in candidate_rows:
        candidate_id = str(row.get("candidate_id", "")).strip()
        if not candidate_id:
            raise ValueError("candidate ID must be non-empty")
        if candidate_id in candidate_ids:
            raise ValueError(f"duplicate candidate ID: {candidate_id}")
        candidate_ids.add(candidate_id)
        relative_path = row.get("image_path")
        if not isinstance(relative_path, str) or not relative_path.strip():
            raise ValueError(f"missing image path for candidate {candidate_id}")
        image_path = (root / relative_path).resolve()
        if not image_path.is_relative_to(root) or not image_path.is_file():
            raise ValueError(f"missing image path for candidate {candidate_id}")
        metadata = row.get("metadata", {})
        if not isinstance(metadata, dict):
            raise ValueError(f"metadata for candidate {candidate_id} must be an object")
        candidates.append(CorpusCandidate(candidate_id, image_path, dict(metadata)))

    queries = []
    query_ids = set()
    for row in query_rows:
        query_id = str(row.get("query_id", "")).strip()
        if not query_id:
            raise ValueError("query ID must be non-empty")
        if query_id in query_ids:
            raise ValueError(f"duplicate query ID: {query_id}")
        query_ids.add(query_id)
        text = str(row.get("text", "")).strip()
        relevance = row.get("relevance")
        if not isinstance(relevance, dict):
            raise ValueError(f"query {query_id} relevance must be an object")
        normalized_relevance = {}
        for candidate_id, grade in relevance.items():
            if candidate_id not in candidate_ids:
                raise ValueError(
                    f"query {query_id} references missing candidate {candidate_id}"
                )
            if (
                not isinstance(grade, int)
                or isinstance(grade, bool)
                or grade < 0
                or grade > 3
            ):
                raise ValueError(f"invalid relevance grade for query {query_id}")
            normalized_relevance[str(candidate_id)] = grade
        if not any(grade > 0 for grade in normalized_relevance.values()):
            raise ValueError(f"query {query_id} has no relevant candidate")
        queries.append(CorpusQuery(query_id, text, normalized_relevance))

    canonical = {
        "candidates": [
            {
                "candidate_id": candidate.candidate_id,
                "image_path": candidate.image_path.relative_to(root).as_posix(),
                "metadata": candidate.metadata,
            }
            for candidate in candidates
        ],
        "queries": [
            {
                "query_id": query.query_id,
                "text": query.text,
                "relevance": query.relevance,
            }
            for query in queries
        ],
    }
    fingerprint = hashlib.sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return EvaluationCorpus(root, tuple(candidates), tuple(queries), fingerprint)


def corpus_statistics(corpus: EvaluationCorpus) -> dict[str, Any]:
    grades = [grade for query in corpus.queries for grade in query.relevance.values()]
    return {
        "query_count": len(corpus.queries),
        "candidate_count": len(corpus.candidates),
        "relevance_judgment_count": len(grades),
        "relevant_judgment_count": sum(grade > 0 for grade in grades),
        "fingerprint": corpus.fingerprint,
    }

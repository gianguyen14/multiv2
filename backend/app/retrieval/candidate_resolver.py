import json
from abc import ABC, abstractmethod
from collections.abc import Mapping
from pathlib import Path
from typing import Optional


class CandidateResolver(ABC):
    @abstractmethod
    def resolve(self, candidate_id: str) -> Optional[dict]:
        raise NotImplementedError

    def resolve_batch(self, candidate_ids: list[str]) -> list[Optional[dict]]:
        return [self.resolve(candidate_id) for candidate_id in candidate_ids]


class MappingCandidateResolver(CandidateResolver):
    def __init__(self, payloads: Mapping[str, dict]):
        self.payloads = payloads

    def resolve(self, candidate_id: str) -> Optional[dict]:
        payload = self.payloads.get(str(candidate_id))
        return dict(payload) if payload is not None else None


class PersistentCandidateResolver(MappingCandidateResolver):
    def __init__(self, path):
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("schema_version") != 1 or not isinstance(payload.get("payloads"), dict):
            raise ValueError("invalid persistent candidate payloads")
        super().__init__(payload["payloads"])


def candidate_id(candidate: dict) -> str:
    return str(candidate.get("candidate_id", candidate.get("document_id", candidate.get("frame_id", ""))))


def resolve_candidates(candidates: list[dict], resolver: CandidateResolver) -> tuple[list[dict], list[str]]:
    ids = [candidate_id(candidate) for candidate in candidates]
    payloads = resolver.resolve_batch(ids)
    if len(payloads) != len(candidates):
        raise ValueError("resolver returned the wrong number of payloads")
    resolved = []
    missing = []
    for candidate, identifier, payload in zip(candidates, ids, payloads):
        if payload is None:
            missing.append(identifier)
            continue
        merged = dict(payload)
        merged.update(candidate)
        merged.setdefault("candidate_id", identifier)
        resolved.append(merged)
    return resolved, missing

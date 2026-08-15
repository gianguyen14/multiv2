"""Evidence-Aware Candidate Reranker.

Provides deterministic, evidence-aware reranking on top of first-stage RRF candidate unions
without GT tuning or arbitrary linear model weights.

Reranking criteria:
1. Exact string evidence (license plates, codes, quoted phrases) with distinction between
   full exact match, separator-normalized match, and partial/fuzzy match.
2. Multi-channel agreement (corroborating evidence across independent modalities: visual, OCR, ASR).
3. Preserves underlying RRF score and visual semantic rank.
4. Deterministic tie-breaking by (video_id, source_frame_index_zero_based).
"""

from __future__ import annotations

import os
import time
import re
import logging
from typing import Any, Dict, List, Optional, Tuple

from backend.app.core.config import RERANKER_ENABLED
from backend.app.services.query_refiner import QueryPlan
from backend.app.video.text_evidence import normalize_text

logger = logging.getLogger(__name__)


def classify_exact_match(term: str, raw_text: str, norm_text: str) -> str:
    """Classifies match strength between an exact search term and candidate text evidence.

    Returns:
        'full_exact': exact raw case-insensitive substring or normalized exact substring.
        'normalized_separator': match after stripping punctuation/separators (e.g. 50H-052.03 vs 50H05203).
        'partial': partial token overlap only (not full identifier).
        'none': no match.
    """
    if not term or not raw_text:
        return "none"

    term_clean = term.strip()
    if not term_clean:
        return "none"

    # 1. Full exact match (raw or normalized)
    if term_clean.lower() in raw_text.lower():
        return "full_exact"

    norm_term = normalize_text(term_clean)
    if norm_term and norm_term in norm_text:
        return "full_exact"

    # 2. Normalized separator match (e.g., license plates or hyphenated codes)
    compact_term = re.sub(r"[\s.-]", "", norm_term)
    if len(compact_term) >= 4:
        compact_norm = re.sub(r"[\s.-]", "", norm_text)
        if compact_term in compact_norm:
            return "normalized_separator"

    # 3. Partial match (distinct from full exact)
    term_tokens = set(re.findall(r"\w+", norm_term))
    text_tokens = set(re.findall(r"\w+", norm_text))
    if term_tokens and (term_tokens & text_tokens):
        return "partial"

    # Sub-identifier overlap (e.g. 50h-052 in 50h-052.03)
    compact_text_tokens = [re.sub(r"[\s.-]", "", t) for t in text_tokens if len(t) >= 3]
    if any(ct in compact_term or (len(ct) >= 4 and compact_term in ct) for ct in compact_text_tokens):
        return "partial"

    return "none"


class CandidateReranker:
    """Deterministic, evidence-aware candidate reranker."""

    def __init__(self, enabled: Optional[bool] = None):
        self.enabled = RERANKER_ENABLED if enabled is None else enabled

    def rerank(
        self,
        candidates: List[Dict[str, Any]],
        plan: QueryPlan,
        ocr_evidence: Optional[List[Any]] = None,
        asr_evidence: Optional[List[Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Reranks candidate list safely. Returns reranked list with diagnostic evidence metadata."""
        if not self.enabled or not candidates or not plan:
            return candidates

        try:
            return self._rerank_internal(candidates, plan, ocr_evidence, asr_evidence)
        except Exception as exc:
            logger.warning("CandidateReranker failed (%s); falling back to base RRF ranking", exc)
            return candidates

    def _rerank_internal(
        self,
        candidates: List[Dict[str, Any]],
        plan: QueryPlan,
        ocr_evidence: Optional[List[Any]] = None,
        asr_evidence: Optional[List[Any]] = None,
    ) -> List[Dict[str, Any]]:
        # Optional profiling (enabled via RERANKER_PROFILE=1)
        profiling_enabled = os.getenv("RERANKER_PROFILE") == "1"
        if profiling_enabled:
            prof = {
                "start_time": time.perf_counter(),
                "candidate_count": len(candidates),
                "exact_term_count": len([ex for ex in plan.exact_strings if ex.strip()]),
                "ocr_evidence_count": len(ocr_evidence) if ocr_evidence else 0,
                "asr_evidence_count": len(asr_evidence) if asr_evidence else 0,
            }

        exact_terms = [ex.strip() for ex in plan.exact_strings if ex.strip()]
        has_exact_requirement = len(exact_terms) > 0

        # Fast lookup indices for candidate frames only: (video_id, frame_id) -> list of (raw_text, norm_text)
        cand_keys = {
            (c["video_id"], c.get("source_frame_index_zero_based", c.get("frame_id", 0)))
            for c in candidates
        }
        ocr_by_cand: Dict[Tuple[str, int], List[Tuple[str, str]]] = {}
        if has_exact_requirement and ocr_evidence:
            for o in ocr_evidence:
                key = (o.video_id, o.source_frame_index_zero_based)
                if key in cand_keys:
                    ocr_by_cand.setdefault(key, []).append(
                        (getattr(o, "raw_text", ""), getattr(o, "normalized_text", ""))
                    )

        asr_by_cand: Dict[Tuple[str, int], List[Tuple[str, str]]] = {}
        if has_exact_requirement and asr_evidence:
            for a in asr_evidence:
                vid = a.video_id
                sf = getattr(a, "start_frame", None)
                ef = getattr(a, "end_frame", None) or sf
                if sf is not None:
                    for cvid, cfid in cand_keys:
                        if cvid == vid and sf <= cfid <= ef:
                            asr_by_cand.setdefault((cvid, cfid), []).append(
                                (getattr(a, "raw_text", ""), getattr(a, "normalized_text", ""))
                            )

        scored_candidates = []
        for cand in candidates:
            vid = cand["video_id"]
            fid = cand.get("source_frame_index_zero_based", cand.get("frame_id", 0))
            matched_by = set(cand.get("matched_by", []))
            base_rrf = float(cand.get("score", 0.0))

            # Group visual variants as a single independent modality.
            has_visual = any(
                m.startswith("visual") or m.startswith("vi_") or m.startswith("en_")
                for m in matched_by
            )
            has_ocr = "ocr" in matched_by
            has_asr = "asr" in matched_by
            independent_channels = (
                (1 if has_visual else 0) + (1 if has_ocr else 0) + (1 if has_asr else 0)
            )
            if independent_channels == 0 and matched_by:
                independent_channels = len(matched_by)

            exact_tier = 0
            matched_exact_terms = []
            if has_exact_requirement:
                cand_texts = ocr_by_cand.get((vid, fid), []) + asr_by_cand.get((vid, fid), [])
                max_match_type = "none"
                for ex in exact_terms:
                    for raw_t, norm_t in cand_texts:
                        m_type = classify_exact_match(ex, raw_t, norm_t)
                        if m_type == "full_exact":
                            max_match_type = "full_exact"
                            matched_exact_terms.append(ex)
                            break
                        if m_type == "normalized_separator" and max_match_type != "full_exact":
                            max_match_type = "normalized_separator"
                            matched_exact_terms.append(ex)
                        elif m_type == "partial" and max_match_type not in (
                            "full_exact",
                            "normalized_separator",
                        ):
                            max_match_type = "partial"

                if max_match_type == "full_exact":
                    exact_tier = 3
                elif max_match_type == "normalized_separator":
                    exact_tier = 2
                elif max_match_type == "partial":
                    exact_tier = 1

            sort_key = (
                -exact_tier,
                -independent_channels,
                -base_rrf,
                str(vid),
                int(fid),
            )

            cand_copy = dict(cand)
            cand_copy["rerank_metadata"] = {
                "exact_tier": exact_tier,
                "independent_channels": independent_channels,
                "matched_exact_terms": list(dict.fromkeys(matched_exact_terms)),
                "base_rrf_score": base_rrf,
                "reranked": True,
            }
            scored_candidates.append((sort_key, cand_copy))

        if profiling_enabled:
            duration_ms = (time.perf_counter() - prof["start_time"]) * 1000.0
            prof["duration_ms"] = duration_ms
            logger.info("candidate_reranker_profile", extra=prof)

        # The tuple was intentionally constructed as an ordering key; apply it.
        scored_candidates.sort(key=lambda item: item[0])
        return [candidate for _, candidate in scored_candidates]

"""TRAKE Temporal Coherence Analyzer.

Provides temporal gap diagnostics, metrics, and non-destructive coherence analysis
for multi-stage TRAKE alignment while preserving the authoritative monotonic DP validity
contract (same video, f1 < f2 < ... < fn) and PyAV display-order frame identities.
"""

from __future__ import annotations

import logging
import statistics
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

from backend.app.core.config import TRAKE_COHERENCE_MODE

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CoherenceDiagnostics:
    video_id: str
    frame_ids: List[int]
    is_single_video: bool
    is_monotonic: bool
    frame_gaps: List[int]
    max_gap: int
    min_gap: int
    mean_gap: float
    median_gap: float
    total_frame_span: int
    timestamp_gaps: List[float] = field(default_factory=list)
    total_time_span: float = 0.0
    normalized_dispersion: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class TRAKECoherenceAnalyzer:
    """Analyzes temporal gaps and coherence of aligned event sequences."""

    def __init__(self, mode: Optional[str] = None):
        self.mode: str = mode.lower() if mode else TRAKE_COHERENCE_MODE

    def analyze(
        self,
        video_id: str,
        frame_ids: Sequence[int],
        timestamps: Optional[Sequence[float]] = None,
    ) -> CoherenceDiagnostics:
        """Computes comprehensive temporal gap and dispersion diagnostics."""
        fids = list(frame_ids)
        if not fids:
            return CoherenceDiagnostics(
                video_id=video_id,
                frame_ids=[],
                is_single_video=True,
                is_monotonic=True,
                frame_gaps=[],
                max_gap=0,
                min_gap=0,
                mean_gap=0.0,
                median_gap=0.0,
                total_frame_span=0,
                timestamp_gaps=[],
                total_time_span=0.0,
                normalized_dispersion=0.0,
            )

        # Monotonicity check
        frame_gaps = [fids[i + 1] - fids[i] for i in range(len(fids) - 1)]
        is_monotonic = all(g > 0 for g in frame_gaps) if frame_gaps else True

        max_g = max(frame_gaps) if frame_gaps else 0
        min_g = min(frame_gaps) if frame_gaps else 0
        mean_g = float(statistics.mean(frame_gaps)) if frame_gaps else 0.0
        median_g = float(statistics.median(frame_gaps)) if frame_gaps else 0.0
        total_span = (fids[-1] - fids[0]) if len(fids) > 1 else 0

        # Timestamp gaps if available
        ts_gaps: List[float] = []
        total_time = 0.0
        if timestamps and len(timestamps) == len(fids):
            ts_gaps = [float(timestamps[i + 1] - timestamps[i]) for i in range(len(timestamps) - 1)]
            total_time = float(timestamps[-1] - timestamps[0]) if len(timestamps) > 1 else 0.0

        # Dispersion metric (coefficient of variation of gaps)
        dispersion = 0.0
        if len(frame_gaps) > 1 and mean_g > 0:
            stdev = statistics.stdev(frame_gaps)
            dispersion = float(stdev / mean_g)

        return CoherenceDiagnostics(
            video_id=video_id,
            frame_ids=fids,
            is_single_video=bool(video_id),
            is_monotonic=is_monotonic,
            frame_gaps=frame_gaps,
            max_gap=max_g,
            min_gap=min_g,
            mean_gap=mean_g,
            median_gap=median_g,
            total_frame_span=total_span,
            timestamp_gaps=ts_gaps,
            total_time_span=total_time,
            normalized_dispersion=dispersion,
        )

    def select_best_alignment(
        self,
        candidate_results: List[Any],
        tolerance: float = 1e-6,
    ) -> Optional[Any]:
        """Selects best alignment among competing valid sequences.

        In 'diagnostic' or 'off' mode: selects highest score strictly.
        In 'tie_break' mode: if scores are equal within tolerance, prefers lower max_gap then lower span.
        """
        if not candidate_results:
            return None

        if self.mode != "tie_break" or len(candidate_results) <= 1:
            # Authoritative score ordering
            return max(candidate_results, key=lambda item: (item.score, item.video_id))

        # Tie-break mode
        best_score = max(r.score for r in candidate_results)
        top_tier = [r for r in candidate_results if abs(r.score - best_score) <= tolerance]
        if len(top_tier) == 1:
            return top_tier[0]

        # Tie-break by temporal coherence: smallest max_gap, smallest total span, then video_id
        def tie_break_key(res):
            diag = self.analyze(res.video_id, res.frame_ids)
            return (diag.max_gap, diag.total_frame_span, res.video_id)

        return min(top_tier, key=tie_break_key)

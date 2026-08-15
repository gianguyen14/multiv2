"""Unit tests for TRAKECoherenceAnalyzer."""

import pytest
from dataclasses import dataclass

from backend.app.services.trake_coherence import CoherenceDiagnostics, TRAKECoherenceAnalyzer


@dataclass
class DummyResult:
    video_id: str
    frame_ids: list
    score: float


def test_close_valid_sequence():
    analyzer = TRAKECoherenceAnalyzer(mode="diagnostic")
    diag = analyzer.analyze("L22_V001", [100, 130, 160], [4.0, 5.2, 6.4])

    assert diag.is_single_video is True
    assert diag.is_monotonic is True
    assert diag.frame_gaps == [30, 30]
    assert diag.max_gap == 30
    assert diag.min_gap == 30
    assert diag.mean_gap == 30.0
    assert diag.median_gap == 30.0
    assert diag.total_frame_span == 60
    assert diag.timestamp_gaps == pytest.approx([1.2, 1.2])
    assert diag.total_time_span == pytest.approx(2.4)


def test_large_gap_valid_sequence_remains_valid():
    analyzer = TRAKECoherenceAnalyzer(mode="diagnostic")
    # Observed real smoke pattern: [9, 134, 1284]
    diag = analyzer.analyze("L22_V002", [9, 134, 1284])

    # In diagnostic mode, large gap sequences remain strictly valid under monotonic invariant
    assert diag.is_monotonic is True
    assert diag.frame_gaps == [125, 1150]
    assert diag.max_gap == 1150
    assert diag.total_frame_span == 1275


def test_non_monotonic_sequence():
    analyzer = TRAKECoherenceAnalyzer(mode="diagnostic")
    diag = analyzer.analyze("L22_V001", [200, 150, 300])

    assert diag.is_monotonic is False
    assert diag.frame_gaps == [-50, 150]


def test_empty_sequence():
    analyzer = TRAKECoherenceAnalyzer(mode="diagnostic")
    diag = analyzer.analyze("L22_V001", [])

    assert diag.frame_gaps == []
    assert diag.total_frame_span == 0
    assert diag.max_gap == 0


def test_single_frame_sequence():
    analyzer = TRAKECoherenceAnalyzer(mode="diagnostic")
    diag = analyzer.analyze("L22_V001", [100])

    assert diag.is_monotonic is True
    assert diag.frame_gaps == []
    assert diag.total_frame_span == 0
    assert diag.max_gap == 0


def test_tie_break_mode_determinism():
    analyzer_tie = TRAKECoherenceAnalyzer(mode="tie_break")

    # Two competing sequences with EXACT same alignment score:
    # Seq A: score 0.90, frame_ids [100, 200, 300] (max gap 100, total span 200)
    # Seq B: score 0.90, frame_ids [100, 1100, 1200] (max gap 1000, total span 1100)
    seq_a = DummyResult("L22_V001", [100, 200, 300], 0.90)
    seq_b = DummyResult("L22_V002", [100, 1100, 1200], 0.90)

    selected = analyzer_tie.select_best_alignment([seq_b, seq_a])
    # Tie-break selects seq_a because of lower max gap and lower span
    assert selected.video_id == "L22_V001"
    assert selected.frame_ids == [100, 200, 300]


def test_diagnostic_mode_preserves_higher_semantic_score_despite_larger_gap():
    analyzer_diag = TRAKECoherenceAnalyzer(mode="diagnostic")

    # Seq A: score 0.95, large gap [100, 1200]
    # Seq B: score 0.85, small gap [100, 130]
    seq_a = DummyResult("L22_V001", [100, 1200], 0.95)
    seq_b = DummyResult("L22_V002", [100, 130], 0.85)

    selected = analyzer_diag.select_best_alignment([seq_b, seq_a])
    # Must NOT override a clearly higher semantic score solely because it spans more time
    assert selected.video_id == "L22_V001"
    assert selected.score == 0.95

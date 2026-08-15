from backend.app.retrieval.trake import EventCandidate, TRAKEAligner
import projectctl


def test_trake_dynamic_programming_enforces_order():
    events = [
        [EventCandidate("v", 30, 1.0), EventCandidate("v", 10, 0.9), EventCandidate("other", 1, 1)],
        [EventCandidate("v", 20, 0.8), EventCandidate("v", 5, 1.0), EventCandidate("other", 2, 1)],
        [EventCandidate("v", 40, 0.7), EventCandidate("other", 3, 1)],
    ]
    result = TRAKEAligner().align(events)
    assert result.video_id == "other"
    assert result.frame_ids == [1, 2, 3]
    assert result.score == 3


def test_trake_rejects_wrong_order_and_impossible_gap():
    reversed_only = [[EventCandidate("v", 20, 1)], [EventCandidate("v", 10, 1)]]
    assert TRAKEAligner().align(reversed_only) is None
    wide_gap = [[EventCandidate("v", 1, 1)], [EventCandidate("v", 100, 1)]]
    assert TRAKEAligner(max_gap=10).align(wide_gap) is None


def test_trake_impossible_intermediate_transition_no_spurious_path():
    # Event 0: frame 100
    # Event 1: frame 50 (impossible transition from 100)
    # Event 2: frame 200 (could transition from 50, but whole chain is invalid)
    cands = [
        [EventCandidate("V001", 100, 1.0)],
        [EventCandidate("V001", 50, 1.0)],
        [EventCandidate("V001", 200, 1.0)],
    ]
    assert TRAKEAligner().align(cands) is None



def test_trake_same_video_enforcement():
    # Candidates for event 1 and event 2 from disjoint videos must not align
    disjoint_events = [
        [EventCandidate("video_A", 100, 0.95), EventCandidate("video_B", 100, 0.90)],
        [EventCandidate("video_C", 200, 0.95), EventCandidate("video_D", 200, 0.90)],
    ]
    assert TRAKEAligner().align(disjoint_events) is None


def test_trake_small_top_k_disjoint_candidates():
    # Simulates top_k=3 where the top candidates for each event happen to be in different videos
    top_3_event_1 = [
        EventCandidate("L22_V002", 14350, 0.9),
        EventCandidate("L22_V003", 12725, 0.8),
        EventCandidate("L22_V002", 13900, 0.7),
    ]
    top_3_event_2 = [
        EventCandidate("L22_V001", 6120, 0.9),
        EventCandidate("L22_V001", 21785, 0.8),
        EventCandidate("L22_V001", 14970, 0.7),
    ]
    assert TRAKEAligner().align([top_3_event_1, top_3_event_2]) is None

    # Expanding candidate depth (e.g. top_k=10) introduces L22_V002 for event 2
    expanded_event_2 = top_3_event_2 + [EventCandidate("L22_V002", 20550, 0.85)]
    result = TRAKEAligner().align([top_3_event_1, expanded_event_2])
    assert result is not None
    assert result.video_id == "L22_V002"
    assert result.frame_ids == [14350, 20550]


def test_trake_malformed_and_edge_case_events():
    aligner = TRAKEAligner()
    # Empty candidates
    assert aligner.align([]) is None
    assert aligner.align([[]]) is None

    # Single event sequence
    single = [[EventCandidate("v1", 100, 0.8), EventCandidate("v1", 200, 0.9)]]
    res = aligner.align(single)
    assert res is not None
    assert res.video_id == "v1"
    assert res.frame_ids == [200]

    # CLI parse_events edge cases
    assert projectctl.parse_events("") == []
    assert projectctl.parse_events("   ") == []
    assert projectctl.parse_events("single_event") == ["single_event"]
    assert projectctl.parse_events("e1 | e2 | e3") == ["e1", "e2", "e3"]
    assert projectctl.parse_events('["e1", "e2"]') == ["e1", "e2"]
    assert projectctl.parse_events('{"events": ["e1", "e2"]}') == ["e1", "e2"]


def test_trake_ties_are_deterministic():
    events = [[EventCandidate("v", 1, 1), EventCandidate("v", 2, 1)],
        [EventCandidate("v", 3, 1), EventCandidate("v", 4, 1)]]
    assert TRAKEAligner().align(events).frame_ids == [1, 3]


def test_trake_cli_and_configured_search_consistency(monkeypatch):
    class StubSearch:
        def __init__(self, *args, **kwargs):
            self.configured = True

        def _initialize(self):
            pass

        def search(self, query, top_k):
            if "forest" in query:
                return [{"video_id": "v1", "source_frame_index_zero_based": 100, "score": 0.9, "frame_id": 100}]
            elif "firefighter" in query:
                return [{"video_id": "v1", "source_frame_index_zero_based": 200, "score": 0.85, "frame_id": 200}]
            return []

    from backend.app.services.configured_search import ConfiguredSearch
    search = ConfiguredSearch(processed_root="/fake/path")
    search._initialize = lambda: None
    search.search = StubSearch().search

    # API handle with string events
    res_str = search.handle({"query_type": "trake", "events": "forest | firefighter", "top_k": 10})
    assert len(res_str) == 1
    assert res_str[0]["video_id"] == "v1"
    assert res_str[0]["frame_ids"] == [100, 200]
    assert res_str[0]["frame_id"] == 100

    # API handle with list events
    res_list = search.handle({"query_type": "trake", "events": ["forest", "firefighter"], "top_k": 10})
    assert res_list == res_str

    # API handle when no alignment exists
    res_none = search.handle({"query_type": "trake", "events": ["forest", "unmatched_query"], "top_k": 10})
    assert res_none == []


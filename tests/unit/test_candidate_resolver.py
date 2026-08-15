from backend.app.retrieval.candidate_resolver import MappingCandidateResolver, resolve_candidates


def test_single_and_missing_resolution():
    resolver = MappingCandidateResolver({"a": {"caption": "first"}})
    assert resolver.resolve("a") == {"caption": "first"}
    assert resolver.resolve("missing") is None


def test_batch_preserves_order_and_metadata():
    resolver = MappingCandidateResolver({"a": {"caption": "first"}, "b": {"caption": "second"}})
    assert resolver.resolve_batch(["b", "missing", "a"]) == [{"caption": "second"}, None, {"caption": "first"}]


def test_resolve_candidates_preserves_retrieval_fields():
    resolver = MappingCandidateResolver({"a": {"caption": "first", "metadata": {"source": "fixture"}}})
    resolved, missing = resolve_candidates([{"frame_id": "a", "score": 0.9}], resolver)
    assert missing == []
    assert resolved == [{"caption": "first", "metadata": {"source": "fixture"}, "frame_id": "a", "score": 0.9, "candidate_id": "a"}]

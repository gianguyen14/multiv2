import pytest

from backend.app.services.search_dispatch import (
    build_search_provider,
    dispatch_search,
    make_search_request,
)


def test_make_search_request_preserves_common_contract():
    request = make_search_request(
        "trake", events=["a", "b"], top_k=7,
        query_refine=False, temporal_refine=False, rerank=False,
    )
    assert request == {
        "query_type": "trake",
        "query": "",
        "top_k": 7,
        "query_refine": False,
        "temporal_refine": False,
        "rerank": False,
        "events": ["a", "b"],
    }


def test_dispatch_search_uses_provider_handle():
    class Provider:
        def handle(self, request):
            return request

    request = make_search_request("qa", query="question", top_k=3)
    assert dispatch_search(Provider(), request) is request


def test_build_search_provider_uses_canonical_backend_selector(monkeypatch):
    class FakeProvider:
        def __init__(self, processed_root=None):
            self.processed_root = processed_root

    monkeypatch.setenv("SEARCH_BACKEND", "qwen3_vl")
    monkeypatch.setattr(
        "backend.app.services.qwen_runtime_search.QwenRuntimeSearch", FakeProvider,
    )
    provider = build_search_provider("/tmp/processed")
    assert isinstance(provider, FakeProvider)
    assert provider.processed_root == "/tmp/processed"


def test_build_search_provider_rejects_unknown_backend(monkeypatch):
    monkeypatch.setenv("SEARCH_BACKEND", "not-a-backend")
    with pytest.raises(RuntimeError, match="Unknown SEARCH_BACKEND"):
        build_search_provider()

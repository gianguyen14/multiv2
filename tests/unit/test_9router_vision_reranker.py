import json

import pytest

from backend.app.retrieval.ninerouter_vision_scorer import (
    NineRouterError,
    NineRouterHTTPError,
    NineRouterVisionScorer,
)


@pytest.fixture
def image_path(tmp_path):
    """Create the tiny local image payload required by scorer unit tests."""
    path = tmp_path / "img_001.png"
    path.write_bytes(b"\x89PNG\r\n\x1a\n")
    return path


def response(score=87, model="cx/gpt-5.6-sol"):
    return {
        "model": model,
        "choices": [
            {
                "message": {"content": json.dumps({"score": score})},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 3},
    }


def test_constructs_pairwise_multimodal_request_without_retrieval_signals(image_path):
    calls = []

    def transport(url, headers, payload, timeout):
        calls.append((url, headers, payload, timeout))
        return 200, response()

    scorer = NineRouterVisionScorer(
        api_key="secret", transport=transport, sleep=lambda _: None
    )
    candidate = {
        "candidate_id": "a",
        "image_path": image_path,
        "embedding": [1],
        "score": 0.99,
        "relevance": 3,
        "metadata": {"description": "fixture truth"},
    }
    assert scorer.score_batch("red circle", [candidate]) == [87.0]
    url, headers, payload, timeout = calls[0]
    assert url == "http://localhost:20128/v1/chat/completions"
    assert headers["Authorization"] == "Bearer secret"
    serialized = json.dumps(payload)
    assert "red circle" in serialized and "data:image/png;base64," in serialized
    assert (
        "fixture truth" not in serialized
        and "embedding" not in serialized
        and "0.99" not in serialized
        and '"relevance"' not in serialized
    )
    assert payload["model"] == "cx/gpt-5.6-sol"
    assert "secret" not in json.dumps(scorer.diagnostics)


@pytest.mark.parametrize(
    "content",
    [
        "not json",
        "{}",
        '{"score":"87"}',
        '{"score":-1}',
        '{"score":101}',
        '{"score":NaN}',
        '{"score":Infinity}',
        '{"score":87,"other":1}',
    ],
)
def test_rejects_invalid_model_output(content, image_path):
    scorer = NineRouterVisionScorer(
        api_key="x",
        transport=lambda *args: (
            200,
            {
                "model": "cx/gpt-5.6-sol",
                "choices": [{"message": {"content": content}}],
            },
        ),
        sleep=lambda _: None,
    )
    with pytest.raises(NineRouterError, match="candidate scoring failed: ValueError"):
        scorer.score_batch("query", [{"candidate_id": "a", "image_path": image_path}])


def test_retries_only_transient_failures(image_path):
    attempts = []

    def transport(*args):
        attempts.append(1)
        if len(attempts) < 3:
            raise NineRouterHTTPError(503)
        return 200, response()

    scorer = NineRouterVisionScorer(
        api_key="x", transport=transport, max_retries=2, sleep=lambda _: None
    )
    assert scorer.score_batch(
        "query", [{"candidate_id": "a", "image_path": image_path}]
    ) == [87]
    assert len(attempts) == 3
    assert scorer.diagnostics["retry_count"] == 2


def test_does_not_retry_auth_failure(image_path):
    attempts = []

    def transport(*args):
        attempts.append(1)
        raise NineRouterHTTPError(401)

    scorer = NineRouterVisionScorer(
        api_key="x", transport=transport, sleep=lambda _: None
    )
    with pytest.raises(NineRouterError):
        scorer.score_batch("query", [{"candidate_id": "a", "image_path": image_path}])
    assert len(attempts) == 1


def test_accepts_observed_route_to_canonical_model_normalization(image_path):
    scorer = NineRouterVisionScorer(
        api_key="x",
        transport=lambda *args: (200, response(model="gpt-5.6-sol")),
        sleep=lambda _: None,
    )

    assert scorer.score_batch(
        "query", [{"candidate_id": "a", "image_path": image_path}]
    ) == [87]
    assert scorer.diagnostics["requested_route_model"] == "cx/gpt-5.6-sol"
    assert scorer.diagnostics["response_models"] == ["gpt-5.6-sol"]
    assert scorer.diagnostics["normalized_model_family"] == "gpt-5.6-sol"
    assert scorer.diagnostics["normalized_response_models"] == ["gpt-5.6-sol"]
    assert scorer.diagnostics["fixed_model_preserved"] is True


@pytest.mark.parametrize(
    "model",
    [
        "gpt-5.6-terra",
        "gpt-5.6-luna",
        "gpt-5.5",
        "claude-sonnet-5",
        "gemini-vision",
        "fallback-model",
    ],
)
def test_rejects_upstream_model_switch(model, image_path):
    scorer = NineRouterVisionScorer(
        api_key="x",
        transport=lambda *args: (200, response(model=model)),
        sleep=lambda _: None,
    )
    with pytest.raises(NineRouterError, match="fixed-model"):
        scorer.score_batch("query", [{"candidate_id": "a", "image_path": image_path}])
    assert scorer.diagnostics["fixed_model_preserved"] is False

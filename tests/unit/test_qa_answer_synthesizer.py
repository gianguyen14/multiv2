import json
import urllib.error

from backend.app.services.qa_answer_synthesizer import QAAnswerSynthesizer

_ABSTAIN = "Không đủ bằng chứng."


class _Response:
    def __init__(self, content="Tây Bắc"):
        self.content = content

    def __enter__(self): return self
    def __exit__(self, *args): pass

    def read(self, limit):
        return json.dumps({"choices": [{"message": {"content": self.content}}]}).encode()


def _synth(content="Tây Bắc", api_key="secret", **kwargs):
    calls = {"n": 0}

    def opener(request, timeout):
        calls["n"] += 1
        return _Response(content)

    synth = QAAnswerSynthesizer(
        backend="remote_llm", base_url="http://nas/v1", model="agent-lite",
        api_key=api_key, opener=opener, **kwargs,
    )
    synth._test_calls = calls  # type: ignore[attr-defined]
    return synth


def test_extractive_is_default_and_preserves_fallback():
    result = QAAnswerSynthesizer(backend="extractive").synthesize(
        "Tác giả gắn bó với đề tài nào?", [{"id": "ocr", "text": "Tây Bắc"}], fallback="Tây Bắc"
    )
    assert result.answer == "Tây Bắc"
    assert result.backend == "extractive"
    assert result.status == "disabled"


def test_max_tokens_uses_configurable_budget_and_sends_it():
    seen = {}

    class LocalResponse(_Response):
        def read(self, limit):
            return super().read(limit)

    def opener(request, timeout):
        seen["body"] = json.loads(request.data)
        return LocalResponse()

    synth = QAAnswerSynthesizer(
        backend="remote_llm", base_url="http://nas/v1", api_key="secret",
        max_tokens=1024, opener=opener,
    )
    result = synth.synthesize("Q", [{"id": "ocr", "text": "evidence"}], fallback="")
    assert result.backend == "remote_llm" and result.status == "ok"
    assert synth.max_tokens == 1024
    assert seen["body"]["max_tokens"] == 1024


def test_max_tokens_floor_and_env_default():
    # explicit tiny value is floored to 64
    assert QAAnswerSynthesizer(backend="remote_llm", max_tokens=1).max_tokens == 64
    # default comes from config QA_ANSWER_MAX_TOKENS (env override respected)
    assert QAAnswerSynthesizer(backend="remote_llm").max_tokens >= 64


def test_remote_sends_only_bounded_evidence_and_parses_answer():
    seen = {}

    class LocalResponse(_Response):
        def read(self, limit):
            seen["bytes"] = limit
            return super().read(limit)

    def opener(request, timeout):
        seen["timeout"] = timeout
        seen["body"] = json.loads(request.data)
        seen["url"] = request.full_url
        return LocalResponse()

    result = QAAnswerSynthesizer(
        backend="remote_llm", base_url="http://nas/v1", model="agent-lite", api_key="secret",
        timeout_seconds=3, opener=opener,
    ).synthesize("Đề tài miền núi nào?", [{"id": "ocr", "text": "Tây Bắc"}], fallback="")
    assert result.answer == "Tây Bắc" and result.backend == "remote_llm" and result.status == "ok"
    assert seen["timeout"] == 3
    assert seen["body"]["model"] == "agent-lite"
    assert "Tây Bắc" in seen["body"]["messages"][1]["content"]
    assert seen["url"].endswith("/v1/chat/completions")


def test_remote_failure_falls_back_without_raising():
    def opener(request, timeout):
        raise urllib.error.URLError("offline")
    result = QAAnswerSynthesizer(
        backend="remote_llm", base_url="http://nas/v1", api_key="secret", opener=opener
    ).synthesize("Q", [{"id": "asr", "text": "evidence"}], fallback="fallback")
    assert result.answer == "fallback"
    assert result.backend == "extractive"
    assert result.status == "remote_error_fallback"


def test_remote_abstention_keeps_remote_provenance():
    synth = _synth(content=_ABSTAIN)
    result = synth.synthesize("Q", [{"id": "ocr", "text": "evidence"}], fallback="fallback")
    assert result.answer == _ABSTAIN
    assert result.backend == "remote_llm"
    assert result.model == "agent-lite"
    assert result.status == "abstained"


def test_remote_empty_response_uses_extractive_fallback():
    synth = _synth(content="   ")
    result = synth.synthesize("Q", [{"id": "ocr", "text": "evidence"}], fallback="fallback")
    assert result.answer == "fallback"
    assert result.backend == "extractive"
    assert result.status == "empty_remote_fallback"


def test_default_remote_top_n_is_five():
    assert QAAnswerSynthesizer(backend="extractive").remote_top_n == 5


def test_remote_top_n_clamped_to_safe_range():
    assert QAAnswerSynthesizer(remote_top_n=0).remote_top_n == 1
    assert QAAnswerSynthesizer(remote_top_n=-7).remote_top_n == 1
    assert QAAnswerSynthesizer(remote_top_n=50).remote_top_n == 10
    assert QAAnswerSynthesizer(remote_top_n=3).remote_top_n == 3


def test_no_remote_call_when_not_configured():
    synth = QAAnswerSynthesizer(backend="remote_llm", base_url="", api_key="")
    result = synth.synthesize("Q", [{"id": "ocr", "text": "evidence"}], fallback="fallback")
    assert result.answer == "fallback"
    assert result.status == "not_configured"
    assert synth.remote_calls == 0
import json
import urllib.error

from backend.app.services.qa_answer_synthesizer import QAAnswerSynthesizer


def test_extractive_is_default_and_preserves_fallback():
    result = QAAnswerSynthesizer(backend="extractive").synthesize(
        "Tác giả gắn bó với đề tài nào?", [{"id": "ocr", "text": "Tây Bắc"}], fallback="Tây Bắc"
    )
    assert result.answer == "Tây Bắc"
    assert result.backend == "extractive"
    assert result.status == "disabled"


def test_remote_sends_only_bounded_evidence_and_parses_answer():
    seen = {}

    class Response:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read(self, limit):
            return json.dumps({"choices": [{"message": {"content": "Tây Bắc"}}]}).encode()

    def opener(request, timeout):
        seen["timeout"] = timeout
        seen["body"] = json.loads(request.data)
        return Response()

    result = QAAnswerSynthesizer(
        backend="remote_llm", base_url="http://nas/v1", model="agent-lite", api_key="secret",
        timeout_seconds=3, opener=opener,
    ).synthesize("Đề tài miền núi nào?", [{"id": "ocr", "text": "Tây Bắc"}], fallback="")
    assert result.answer == "Tây Bắc" and result.backend == "remote_llm" and result.status == "ok"
    assert seen["timeout"] == 3
    assert seen["body"]["model"] == "agent-lite"
    assert "Tây Bắc" in seen["body"]["messages"][1]["content"]


def test_remote_failure_falls_back_without_raising():
    def opener(request, timeout):
        raise urllib.error.URLError("offline")
    result = QAAnswerSynthesizer(
        backend="remote_llm", base_url="http://nas/v1", api_key="secret", opener=opener
    ).synthesize("Q", [{"id": "asr", "text": "evidence"}], fallback="fallback")
    assert result.answer == "fallback"
    assert result.backend == "extractive"
    assert result.status == "remote_error_fallback"

from collections import namedtuple
import pytest
from projectctl import command_qa

class MockArgs:
    def __init__(self, query, top_k=1):
        self.query = query
        self.top_k = top_k

def test_command_qa_decomposition(monkeypatch):
    handled_request = None
    
    class MockConfiguredSearch:
        def handle(self, request):
            nonlocal handled_request
            handled_request = request
            return []

    def mock_configured_search():
        return MockConfiguredSearch()
    
    def mock_emit(results, args, output):
        pass

    import projectctl
    monkeypatch.setattr(projectctl, "configured_search", mock_configured_search)
    monkeypatch.setattr(projectctl, "emit", mock_emit)
    
    # Test decomposition is handled in configured search now, so command_qa just passes the request
    command_qa(MockArgs("Dự án cải tạo đền thờ nào đang được khởi công?"))
    assert handled_request["query_type"] == "qa"
    assert handled_request["query"] == "Dự án cải tạo đền thờ nào đang được khởi công?"

def test_qa_query_decomposer():
    from backend.app.retrieval.qa_query_decomposition import QAQueryDecomposer
    decomposer = QAQueryDecomposer()
    
    assert decomposer.decompose("Dự án cải tạo đền thờ nào đang được khởi công?")["retrieval_query"] == "Dự án cải tạo đền thờ đang được khởi công"
    assert decomposer.decompose("Cháy rừng xảy ra ở đâu?")["retrieval_query"] == "Cháy rừng xảy ra"
    assert decomposer.decompose("Thiếu niên nghiện smartphone dễ bị bệnh gì?")["retrieval_query"] == "Thiếu niên nghiện smartphone dễ bị bệnh"
    assert decomposer.decompose("What is the temperature?")["retrieval_query"] == "the temperature"
    
    # Empty after decomposition
    assert decomposer.decompose("Ai?")["retrieval_query"] == "Ai?"

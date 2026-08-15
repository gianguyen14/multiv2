import re

class QAQueryDecomposer:
    def __init__(self):
        # We target specific WH words to neutralize unknown answer slots
        # while preserving localization constraints.
        self.wh_patterns = [
            r"\b(?:how many|what color|what is|what|where|when|who|why|holding|doing|which)\b",
            r"\b(?:như thế nào|thế nào|ở đâu|khi nào|bao nhiêu|tại sao|có bao nhiêu|đang làm gì|màu gì|cầm gì|mặc gì|ai|gì|nào|mấy)\b",
            r"\?$"
        ]
        self.compiled_patterns = [re.compile(p, re.IGNORECASE) for p in self.wh_patterns]

    def decompose(self, question: str) -> dict:
        q_clean = question
        for pat in self.compiled_patterns:
            q_clean = pat.sub("", q_clean)
        
        q_clean = re.sub(r"\s+", " ", q_clean).strip()
        
        # If the question was entirely stripped (e.g. just WH words), fallback to original
        if not q_clean:
            q_clean = question
            
        return {
            "original_question": question,
            "retrieval_query": q_clean,
            "answer_type": "unknown",
            "unknown_slot": "unknown",
            "known_constraints": {}
        }

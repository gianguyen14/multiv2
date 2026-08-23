import re


class QAQueryDecomposer:
    def __init__(self):
        # Remove only the unknown answer slot. Action words such as
        # "holding"/"cầm" must remain useful retrieval evidence.
        self.wh_patterns = (
            r"\b(?:đang\s+)?làm\s+gì\b",
            r"\b(?:đang\s+)?(?:nói|phát\s+biểu)\s+về\s+(?:(?:chủ\s+đề|nội\s+dung)\s+)?gì\b",
            r"\b(?:how\s+many|what\s+color|what\s+is)\b",
            r"\b(?:where|when|who|why|which|what)\b",
            r"\b(?:như\s+thế\s+nào|thế\s+nào|ở\s+đâu|khi\s+nào|có\s+bao\s+nhiêu|bao\s+nhiêu|tại\s+sao|màu\s+gì)\b",
        )
        self.compiled_patterns = [re.compile(p, re.IGNORECASE) for p in self.wh_patterns]
        self.single_vietnamese_wh_pattern = re.compile(
            r"\b(?:ai|gì|nào|mấy)\b",
            re.IGNORECASE,
        )

    @staticmethod
    def _starts_capitalized_name(text: str, match: re.Match) -> bool:
        """Return whether a WH-shaped token starts a capitalized proper name."""
        token = match.group(0)
        if not token[0].isupper():
            return False

        following_word = re.match(r"\s+([^\W\d_]+)", text[match.end():], re.UNICODE)
        return bool(following_word and following_word.group(1)[0].isupper())

    def _remove_single_vietnamese_wh(self, text: str) -> tuple[str, int]:
        removed = 0

        def replace(match: re.Match) -> str:
            nonlocal removed
            if self._starts_capitalized_name(text, match):
                return match.group(0)
            removed += 1
            return ""

        return self.single_vietnamese_wh_pattern.sub(replace, text), removed

    @staticmethod
    def _clean_removed_slot(text: str) -> str:
        """Repair grammar fragments introduced by removing an answer slot."""
        text = re.sub(r"[?？]+\s*$", "", text)
        text = re.sub(r"\s+", " ", text).strip(" ,;:")

        # English WH-fronting leaves an auxiliary at the start after the WH
        # phrase is removed: "Where did the car stop?" -> "did the car stop".
        text = re.sub(
            r"^(?:is|are|was|were|do|does|did)\b\s*",
            "",
            text,
            flags=re.IGNORECASE,
        )

        # Copulas at the end point to the removed answer and carry no visual,
        # OCR, or ASR evidence: "Dòng chữ ... là gì?" -> "Dòng chữ ...".
        previous = None
        while text and text != previous:
            previous = text
            text = re.sub(
                r"(?:\s+|^)(?:là|is|are|was|were)\s*$",
                "",
                text,
                flags=re.IGNORECASE,
            ).strip(" ,;:")

        return re.sub(r"\s+", " ", text).strip()

    def decompose(self, question: str) -> dict:
        q_clean = question
        removed_slot = False
        for pat in self.compiled_patterns:
            q_clean, substitutions = pat.subn("", q_clean)
            removed_slot = removed_slot or substitutions > 0

        # In an English "what ... doing" question, "doing" names the
        # unknown action rather than a known visual constraint.  Only remove
        # it after a WH slot was found so yes/no descriptions stay intact.
        if removed_slot:
            q_clean = re.sub(r"\bdoing\b", "", q_clean, flags=re.IGNORECASE)

        q_clean, substitutions = self._remove_single_vietnamese_wh(q_clean)
        removed_slot = removed_slot or substitutions > 0

        if removed_slot:
            q_clean = self._clean_removed_slot(q_clean)
        else:
            q_clean = re.sub(r"[?？]+\s*$", "", q_clean)
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

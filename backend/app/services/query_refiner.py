"""Query Intelligence and Multi-Path Retrieval Service.

Transforms natural language queries into structured QueryPlans containing multi-lingual
visual queries (VI + EN), exact text/OCR terms, preserved Vietnamese cultural terms,
and structured TRAKE stages, enabling rank-safe multi-path retrieval without GT tuning.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, Field

from backend.app.core.config import (
    QUERY_REFINER_BACKEND,
    QUERY_REFINER_CACHE_ENABLED,
    QUERY_REFINER_ENABLED,
    QUERY_REFINER_MAX_VISUAL_VARIANTS,
    QUERY_REFINER_MODEL,
)
from backend.app.video.atomic_io import write_json_atomic
from backend.app.video.text_evidence import normalize_text

logger = logging.getLogger(__name__)


# =========================================================================
# 1. QueryPlan Data Models
# =========================================================================

class VisualQuery(BaseModel):
    language: Literal["vi", "en"]
    text: str
    weight: float = 1.0
    channel: str = "visual"


class TRAKEStagePlan(BaseModel):
    stage_index: int
    visual_vi: str
    visual_en: str
    original_text: str = ""
    exact_strings: List[str] = Field(default_factory=list)
    kept_vi_terms: List[str] = Field(default_factory=list)


class QueryPlan(BaseModel):
    task_type: Literal["kis", "qa", "trake", "image", "general"] = "kis"
    original_query: str

    visual_queries: List[VisualQuery] = Field(default_factory=list)
    lexical_terms: List[str] = Field(default_factory=list)
    exact_strings: List[str] = Field(default_factory=list)
    kept_vi_terms: List[str] = Field(default_factory=list)

    objects: List[str] = Field(default_factory=list)
    attributes: List[str] = Field(default_factory=list)

    trake_stages: List[TRAKEStagePlan] = Field(default_factory=list)
    strict_order: bool = False

    refinement_used: bool = False
    refinement_backend: str = "deterministic"
    warnings: List[str] = Field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> QueryPlan:
        return cls.model_validate(data)


class LLMRefinementResponse(BaseModel):
    """Schema for validating constrained JSON output from local instruction models."""
    visual_caption_vi: Optional[str] = None
    visual_caption_en: Optional[str] = None
    kept_vi_terms: List[str] = Field(default_factory=list)
    objects: List[str] = Field(default_factory=list)
    attributes: List[str] = Field(default_factory=list)
    query_variants_vi: List[str] = Field(default_factory=list)
    query_variants_en: List[str] = Field(default_factory=list)
    trake_stages: List[Dict[str, Any]] = Field(default_factory=list)


# =========================================================================
# 2. Deterministic Knowledge Tables (Layer A Generic Helpers)
# =========================================================================

KNOWN_VI_CULTURAL_TERMS = {
    "áo dài", "áo bà ba", "áo tứ thân", "nón lá", "xe lam", "xe xích lô", "xích lô",
    "bánh mì", "phở", "bún bò", "bánh bèo", "bánh xèo", "chè", "cà phê sữa đá",
    "đền thờ", "chùa", "đình", "miếu", "lăng", "di tích",
    "vũ công", "nhà rông", "cồng chiêng", "múa lân", "đờn ca tài tử", "quan họ",
    "bão số 3", "thời sự", "đài truyền hình", "tp.hcm", "sài gòn", "hà nội", "đà nẵng", "huế",
}

COLOR_TERMS = {
    "đỏ": "red", "xanh lá": "green", "xanh dương": "blue", "xanh": "blue",
    "vàng": "yellow", "trắng": "white", "đen": "black", "tím": "purple",
    "hồng": "pink", "cam": "orange", "nâu": "brown", "xám": "gray", "bạc": "silver",
}

OBJECT_TERMS = {
    "người": "person", "phụ nữ": "woman", "đàn ông": "man", "trẻ em": "children",
    "bé gái": "girl", "bé trai": "boy", "xe": "vehicle", "xe lam": "auto rickshaw",
    "xe máy": "motorcycle", "xe tải": "truck", "xe ô tô": "car", "xe buýt": "bus",
    "xe xích lô": "rickshaw", "tàu hỏa": "train", "máy bay": "airplane", "nhà": "building",
    "đền thờ": "temple", "chùa": "pagoda", "núi": "mountain", "sông": "river",
    "biển": "sea", "rùa": "turtle", "cứu hộ": "rescue workers", "lính cứu hỏa": "firefighters",
}

VIETNAMESE_DIACRITICS_PATTERN = re.compile(
    r"[àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ"
    r"ÀÁẢÃẠĂẰẮẲẴẶÂẦẤẨẪẬÈÉẺẼẸÊỀẾỂỄỆÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴĐ]"
)

COMMON_VIETNAMESE_TOKENS = {
    "người", "phụ", "nữ", "đàn", "ông", "trẻ", "em", "mặc", "đứng", "ngồi", "chạy",
    "cạnh", "trong", "tại", "biển", "số", "bản", "tin", "thời", "sự", "cẩu", "xe",
    "của", "và", "với", "cho", "được", "có", "là", "các", "những", "một", "nhiều"
}


def validate_english_caption(caption: Optional[str]) -> bool:
    """Validates that a visual caption is genuine English and free of mixed Vietnamese fragments."""
    if not caption or not isinstance(caption, str):
        return False
    text = caption.strip()
    if not text or len(text) < 3:
        return False
    # Reject if containing Vietnamese diacritics
    if VIETNAMESE_DIACRITICS_PATTERN.search(text):
        return False
    # Reject if containing obvious untranslated Vietnamese tokens
    tokens = set(re.findall(r"\b[a-z]+\b", text.lower()))
    if tokens & COMMON_VIETNAMESE_TOKENS:
        return False
    return True


# =========================================================================
# 3. Layer A: Deterministic Query Parser
# =========================================================================

class DeterministicQueryParser:
    """Extracts exact alphanumeric codes, OCR strings, cultural terms, and builds visual captions."""

    # Generic Regex Patterns (No Ground Truth specific hardcoding)
    PATTERN_QUOTES = re.compile(r"['\"«“]([^'\"»”\n]{2,40})['\"»”]")
    PATTERN_LICENSE_PLATE = re.compile(r"\b\d{2}[A-ZĐ][\s-]?\d{3,5}(?:\.\d{2})?\b", re.I)
    PATTERN_ALPHANUMERIC_CODE = re.compile(r"\b[A-Z0-9]{2,10}(?:-[A-Z0-9]{2,10})+\b", re.I)
    PATTERN_UPPERCASE_CODE = re.compile(r"\b[A-Z0-9]{3,12}\b")
    PATTERN_TEMPERATURE = re.compile(r"\b\d+(?:[.,]\d+)?\s*(?:°C|°c|°F|độ\s*C|độ\s*c|độ)", re.I)
    PATTERN_PERCENTAGE = re.compile(r"\b\d+(?:[.,]\d+)?\s*%")
    PATTERN_CURRENCY = re.compile(r"\b\d+(?:[.,]\d+)?\s*(?:USD|VND|tỷ\s*đồng|triệu\s*đồng|nghìn\s*đồng|ngàn\s*đồng|bảng\s*Anh|đồng/lít)\b", re.I)
    PATTERN_DATE_TIME = re.compile(r"\b\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?\b")
    PATTERN_PROPER_NAME = re.compile(
        r"\b[A-ZĐÀÁẢÃẠĂẰẮẲẴẶÂẦẤẨẪẬÈÉẺẼẸÊỀẾỂỄỆÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴ][a-zđàáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵ]*"
        r"(?:\s+[A-ZĐÀÁẢÃẠĂẰẮẲẴẶÂẦẤẨẪẬÈÉẺẼẸÊỀẾỂỄỆÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴ][a-zđàáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵ]*)+\b"
    )

    def parse(self, query: str, task_type: str = "kis") -> QueryPlan:
        q_raw = query.strip()
        q_norm = normalize_text(q_raw)

        exact_strings: List[str] = []
        lexical_terms: List[str] = []
        kept_vi_terms: List[str] = []
        objects: List[str] = []
        attributes: List[str] = []
        warnings: List[str] = []

        # 1. Extract Quoted Strings
        for match in self.PATTERN_QUOTES.finditer(q_raw):
            val = match.group(1).strip()
            if val and val not in exact_strings:
                exact_strings.append(val)
                lexical_terms.append(val)

        # 2. Extract License Plates
        for match in self.PATTERN_LICENSE_PLATE.finditer(q_raw):
            val = match.group(0).strip()
            if val not in exact_strings:
                exact_strings.append(val)
                lexical_terms.append(val)

        # 3. Extract Alphanumeric Codes (e.g. 79H-6072, F-16, ZE300)
        for match in self.PATTERN_ALPHANUMERIC_CODE.finditer(q_raw):
            val = match.group(0).strip()
            if val not in exact_strings:
                exact_strings.append(val)
                lexical_terms.append(val)

        # 4. Extract Temperatures, Percentages, Currencies, Dates
        for pat in (self.PATTERN_TEMPERATURE, self.PATTERN_PERCENTAGE, self.PATTERN_CURRENCY, self.PATTERN_DATE_TIME):
            for match in pat.finditer(q_raw):
                val = match.group(0).strip()
                if val not in exact_strings:
                    exact_strings.append(val)
                    lexical_terms.append(val)

        # 5. Extract Upper-case codes / tokens (e.g. TADANO, RON95)
        for token in q_raw.split():
            clean_tok = re.sub(r"[^\w-]", "", token)
            if self.PATTERN_UPPERCASE_CODE.match(clean_tok):
                # Filter out pure Vietnamese capitalized words unless they contain digits
                if re.search(r"\d", clean_tok) or (clean_tok.isupper() and len(clean_tok) >= 3):
                    if clean_tok not in exact_strings:
                        exact_strings.append(clean_tok)
                        lexical_terms.append(clean_tok)

        # 6. Extract Generic Proper Names (e.g. Nguyễn Hữu Cảnh, Hải Vân Quan, Barcelona, VinFast, etc.)
        for match in self.PATTERN_PROPER_NAME.finditer(q_raw):
            val = match.group(0).strip()
            if val and val not in kept_vi_terms:
                kept_vi_terms.append(val)
            if val and val not in lexical_terms:
                lexical_terms.append(val)

        # Single capitalized non-stopword tokens (e.g. Barcelona, Tokyo, VinFast)
        for token in q_raw.split()[1:]:  # skip first word to avoid capitalization due to sentence start
            clean_tok = re.sub(r"[^\w-]", "", token)
            if clean_tok and clean_tok[0].isupper() and len(clean_tok) >= 3:
                if clean_tok.lower() not in COMMON_VIETNAMESE_TOKENS and clean_tok not in kept_vi_terms:
                    kept_vi_terms.append(clean_tok)
                if clean_tok.lower() not in COMMON_VIETNAMESE_TOKENS and clean_tok not in lexical_terms:
                    lexical_terms.append(clean_tok)

        # 7. Extract Known Cultural Terms
        for cult in KNOWN_VI_CULTURAL_TERMS:
            if cult in q_norm:
                if cult not in kept_vi_terms:
                    kept_vi_terms.append(cult)
                if cult not in lexical_terms:
                    lexical_terms.append(cult)

        # 8. Extract Colors & Attributes
        for vi_color, en_color in COLOR_TERMS.items():
            if re.search(r"\b" + re.escape(vi_color) + r"\b", q_norm):
                if en_color not in attributes:
                    attributes.append(en_color)

        # 9. Extract Objects
        for vi_obj, en_obj in OBJECT_TERMS.items():
            if re.search(r"\b" + re.escape(vi_obj) + r"\b", q_norm):
                if en_obj not in objects:
                    objects.append(en_obj)

        # Clean visual query (remove explicit code noise for visual captioning)
        clean_vi = q_raw
        for exact in exact_strings:
            # Replace exact code in visual caption with generic placeholder or remove
            clean_vi = clean_vi.replace(f'"{exact}"', "").replace(f"'{exact}'", "").replace(exact, "")
        clean_vi = re.sub(r"\bbiển\s+số\s*:\s*", "biển số ", clean_vi, flags=re.I)
        clean_vi = re.sub(r"\s+", " ", clean_vi).strip(" ,.-:")
        if not clean_vi:
            clean_vi = q_raw

        # Deterministic visual query: primary Vietnamese visual query
        # No naive token substitution pseudo-translation (avoids sending broken pseudo-English to SigLIP2)
        visual_queries = [
            VisualQuery(language="vi", text=clean_vi, weight=1.0, channel="visual_vi"),
        ]

        # 10. Handle TRAKE Stage Splitting
        trake_stages: List[TRAKEStagePlan] = []
        is_trake = task_type == "trake" or "|" in q_raw

        if is_trake:
            raw_stages = []
            if "|" in q_raw:
                raw_stages = [s.strip() for s in q_raw.split("|") if s.strip()]
            else:
                parts = re.split(r"(?:\b(?:sau đó|tiếp theo|rồi|và sau đó)\b|,\s*(?:và\s+)?|\s+và\s+)", q_raw, flags=re.I)
                raw_stages = [s.strip() for s in parts if s.strip()]

            if not raw_stages:
                raw_stages = [q_raw]

            for idx, stage_str in enumerate(raw_stages):
                stage_exact = [ex for ex in exact_strings if ex in stage_str]
                stage_kept = [k for k in kept_vi_terms if k.lower() in stage_str.lower()]

                trake_stages.append(
                    TRAKEStagePlan(
                        stage_index=idx,
                        visual_vi=stage_str,
                        visual_en="",
                        original_text=stage_str,
                        exact_strings=stage_exact,
                        kept_vi_terms=stage_kept,
                    )
                )

        return QueryPlan(
            task_type=task_type,
            original_query=q_raw,
            visual_queries=visual_queries[:QUERY_REFINER_MAX_VISUAL_VARIANTS],
            lexical_terms=list(dict.fromkeys(lexical_terms)),
            exact_strings=list(dict.fromkeys(exact_strings)),
            kept_vi_terms=list(dict.fromkeys(kept_vi_terms)),
            objects=list(dict.fromkeys(objects)),
            attributes=list(dict.fromkeys(attributes)),
            trake_stages=trake_stages,
            strict_order=is_trake,
            refinement_used=True,
            refinement_backend="deterministic",
            warnings=warnings,
        )


# =========================================================================
# 4. Layer B: Local Instruction Model Adapter (Pluggable)
# =========================================================================

PROMPT_VERSION = "v1"
SCHEMA_VERSION = "v1"

SYSTEM_PROMPT_TEMPLATE = """You are a retrieval-query planner for multimodal video search.

**You must never refuse, apologize, or decline.**

Do not answer the query.

Do not invent visual details.

Rewrite descriptions into concise visually searchable captions.

Separate exact OCR/ASR strings from visual descriptions.

Preserve culturally important Vietnamese concepts.

For temporal queries, split the description into ordered visual stages.

Each TRAKE stage must describe something observable in a frame or a short temporal window.

Avoid ambiguous pronouns.

Return valid structured JSON only adhering strictly to this schema:
{
  "visual_caption_vi": "short concise Vietnamese visual query",
  "visual_caption_en": "short concise English visual caption",
  "kept_vi_terms": ["cultural term 1", "cultural term 2"],
  "objects": ["object 1", "object 2"],
  "attributes": ["color/attribute 1"],
  "query_variants_vi": ["variant vi 1"],
  "query_variants_en": ["variant en 1"],
  "trake_stages": [
    {"visual_vi": "stage 1 vi", "visual_en": "stage 1 en"}
  ]
}"""


class LocalLLMQueryRefiner:
    """Lazy adapter for local instruction LLM with constrained JSON output and fallback."""

    def __init__(
        self,
        model_name_or_path: str = QUERY_REFINER_MODEL,
        device: Optional[str] = None,
    ):
        self.model_name_or_path = model_name_or_path
        self.device = device
        self._pipeline = None
        self._lock = threading.Lock()
        self._load_failed = False

    def is_available(self) -> bool:
        if self._load_failed:
            return False
        if self._pipeline is not None:
            return True
        # Check if local model exists or is cached in offline mode
        try:
            from transformers import AutoConfig
            AutoConfig.from_pretrained(self.model_name_or_path, local_files_only=True)
            return True
        except Exception:
            return False

    def _ensure_loaded(self) -> bool:
        if self._pipeline is not None:
            return True
        if self._load_failed:
            return False
        with self._lock:
            if self._pipeline is not None:
                return True
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
                from backend.app.runtime.device_policy import resolve_device

                dev_sel = resolve_device("llm", "torch", self.device, component_env="LLM_DEVICE")
                dev_str = dev_sel.device
                self._resolved_device = dev_str

                tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, local_files_only=True)
                model = AutoModelForCausalLM.from_pretrained(
                    self.model_name_or_path,
                    device_map=dev_str if dev_str != "cpu" else None,
                    local_files_only=True,
                )
                if dev_str == "cpu":
                    model = model.to("cpu")

                from transformers import GenerationConfig

                self._pipeline = pipeline(
                    "text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    generation_config=GenerationConfig(
                        max_new_tokens=512,
                        do_sample=False,
                        temperature=0.0,
                        top_p=1.0,
                    ),
                )
                return True
            except Exception as exc:
                logger.info("Local LLM refiner unavailable (%s); falling back to deterministic parser", exc)
                self._load_failed = True
                return False

    def refine(self, query: str, base_plan: QueryPlan) -> Optional[QueryPlan]:
        # Reset diagnostics for each call
        self._fallback_reason = None
        self._llm_invoked = False
        if not self._ensure_loaded() or self._pipeline is None:
            self._fallback_reason = "model_unavailable"
            return None

        prompt = (
            f"<|im_start|>system\n{SYSTEM_PROMPT_TEMPLATE}<|im_end|>\n"
            f"<|im_start|>user\nQuery: {query}\nTask: {base_plan.task_type}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        try:
            output = self._pipeline(prompt)
            # Record that model.generate was called via the pipeline
            self._llm_invoked = True
            generated_text = output[0]["generated_text"][len(prompt):]
            # --- JSON extraction according to strict contract ---
            cleaned = generated_text.strip()
            # Compatibility fallback: strip markdown fences if present
            if cleaned.startswith("```json") and cleaned.endswith("```"):
                cleaned = cleaned[7:-3].strip()
            elif cleaned.startswith("```") and cleaned.endswith("```"):
                cleaned = cleaned[3:-3].strip()
            # Attempt full JSON parse
            try:
                raw_dict = json.loads(cleaned)
            except json.JSONDecodeError:
                # Greedy fallback: locate first '{' and raw decode
                decoder = json.JSONDecoder()
                try:
                    raw_dict, idx = decoder.raw_decode(cleaned)
                    # Ensure any trailing characters are only whitespace
                    if cleaned[idx:].strip():
                        raise json.JSONDecodeError('Trailing data after JSON', cleaned, idx)
                except json.JSONDecodeError as exc:
                    logger.warning("LLM response JSON parsing failed: %s; falling back", exc)
                    self._fallback_reason = "invalid_json"
                    return None
            # Validate schema
            try:
                validated = LLMRefinementResponse.model_validate(raw_dict)
            except Exception as exc:
                logger.warning("Schema validation error: %s; falling back", exc)
                self._fallback_reason = "schema_validation_error"
                return None

            # Build enriched QueryPlan
            visual_queries = list(base_plan.visual_queries)
            if validated.visual_caption_vi and validated.visual_caption_vi.strip():
                if not any(vq.language == "vi" and vq.text == validated.visual_caption_vi for vq in visual_queries):
                    visual_queries.insert(0, VisualQuery(language="vi", text=validated.visual_caption_vi.strip(), weight=1.0, channel="visual_vi_refined"))
            if validated.visual_caption_en and validated.visual_caption_en.strip():
                en_caption = validated.visual_caption_en.strip()
                if validate_english_caption(en_caption):
                    if not any(vq.language == "en" and vq.text == en_caption for vq in visual_queries):
                        visual_queries.append(VisualQuery(language="en", text=en_caption, weight=1.0, channel="visual_en_refined"))
                else:
                    logger.debug("LLM visual_caption_en rejected by English validation: '%s'", en_caption)

            trake_stages = list(base_plan.trake_stages)
            if validated.trake_stages:
                llm_stages = []
                for idx, st in enumerate(validated.trake_stages):
                    vi_text = st.get("visual_vi", f"stage {idx}")
                    en_text = st.get("visual_en", "")
                    if en_text and not validate_english_caption(en_text):
                        en_text = ""
                    llm_stages.append(
                        TRAKEStagePlan(
                            stage_index=idx,
                            visual_vi=vi_text,
                            visual_en=en_text,
                            original_text=vi_text,
                        )
                    )
                if llm_stages:
                    trake_stages = llm_stages

            plan = QueryPlan(
                task_type=base_plan.task_type,
                original_query=base_plan.original_query,
                visual_queries=visual_queries[:QUERY_REFINER_MAX_VISUAL_VARIANTS],
                lexical_terms=list(dict.fromkeys(base_plan.lexical_terms + validated.kept_vi_terms)),
                exact_strings=base_plan.exact_strings,
                kept_vi_terms=list(dict.fromkeys(base_plan.kept_vi_terms + validated.kept_vi_terms)),
                objects=list(dict.fromkeys(base_plan.objects + validated.objects)),
                attributes=list(dict.fromkeys(base_plan.attributes + validated.attributes)),
                trake_stages=trake_stages,
                strict_order=base_plan.strict_order,
                refinement_used=True,
                refinement_backend="local_llm",
                warnings=base_plan.warnings,
            )
            # Attach resolved device for reporting
            plan._device = getattr(self, "_resolved_device", None)
            # Attach diagnostics for caller inspection
            plan._llm_invoked = getattr(self, "_llm_invoked", False)
            plan._fallback_used = self._fallback_reason is not None
            plan._fallback_reason = self._fallback_reason
            return plan
        except Exception as exc:
            logger.warning("Local LLM refinement failed (%s); using deterministic fallback", exc)
            self._fallback_reason = "generation_error"
            return None


# =========================================================================
# 5. QueryPlan Cache
# =========================================================================

class QueryPlanCache:
    """Lightweight, thread-safe cache for QueryPlan results with fingerprint invalidation."""

    def __init__(self, cache_dir: Optional[Path | str] = None, max_mem_entries: int = 1000):
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.max_mem_entries = max_mem_entries
        self._mem_cache: Dict[str, QueryPlan] = {}
        self._lock = threading.Lock()

    def compute_fingerprint(
        self,
        query: str,
        task_type: str,
        backend: str,
        config_dict: Dict[str, Any],
    ) -> str:
        h = hashlib.sha256()
        h.update(normalize_text(query).encode("utf-8"))
        h.update(task_type.encode("utf-8"))
        h.update(backend.encode("utf-8"))
        h.update(PROMPT_VERSION.encode("utf-8"))
        h.update(SCHEMA_VERSION.encode("utf-8"))
        h.update(json.dumps(config_dict, sort_keys=True).encode("utf-8"))
        return h.hexdigest()

    def get(self, key: str) -> Optional[QueryPlan]:
        with self._lock:
            if key in self._mem_cache:
                return self._mem_cache[key]

        if self.cache_dir:
            file_path = self.cache_dir / f"{key}.json"
            if file_path.is_file():
                try:
                    data = json.loads(file_path.read_text())
                    plan = QueryPlan.from_dict(data)
                    with self._lock:
                        if len(self._mem_cache) < self.max_mem_entries:
                            self._mem_cache[key] = plan
                    return plan
                except Exception as exc:
                    logger.debug("Corrupt query cache file %s: %s", file_path, exc)
                    return None
        return None

    def put(self, key: str, plan: QueryPlan) -> None:
        with self._lock:
            if len(self._mem_cache) >= self.max_mem_entries:
                self._mem_cache.pop(next(iter(self._mem_cache)), None)
            self._mem_cache[key] = plan

        if self.cache_dir:
            try:
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                write_json_atomic(self.cache_dir / f"{key}.json", plan.to_dict())
            except Exception as exc:
                logger.debug("Failed to write disk query cache: %s", exc)


# =========================================================================
# 6. QueryRefiner Orchestrator
# =========================================================================

class QueryRefiner:
    """Main query intelligence orchestrator transforming queries into structured QueryPlans."""

    def __init__(
        self,
        enabled: bool = QUERY_REFINER_ENABLED,
        backend: str = QUERY_REFINER_BACKEND,
        model_name: str = QUERY_REFINER_MODEL,
        cache_dir: Optional[Path | str] = None,
        cache_enabled: bool = QUERY_REFINER_CACHE_ENABLED,
        llm_refiner: Optional[LocalLLMQueryRefiner] = None,
    ):
        self.enabled = enabled
        self.backend = backend
        self.deterministic_parser = DeterministicQueryParser()
        self.llm_refiner = llm_refiner
        if self.llm_refiner is None and self.backend in ("local_llm", "auto"):
            self.llm_refiner = LocalLLMQueryRefiner(model_name_or_path=model_name)
        self.cache = QueryPlanCache(cache_dir=cache_dir) if cache_enabled else None

    def refine(self, query: str, task_type: str = "kis") -> Tuple[QueryPlan, Dict[str, float]]:
        """Refines a query into a structured QueryPlan with performance timings.

        Returns:
            (query_plan, timing_metrics_ms)
        """
        t0 = time.perf_counter()
        timings = {
            "deterministic_parse_ms": 0.0,
            "query_parse_ms": 0.0,
            "llm_refine_ms": 0.0,
            "total_query_ms": 0.0,
            "total_ms": 0.0,
        }

        if not self.enabled or self.backend == "off":
            # Baseline passthrough QueryPlan
            plan = QueryPlan(
                task_type=task_type,
                original_query=query,
                visual_queries=[VisualQuery(language="vi", text=query.strip(), weight=1.0, channel="visual_vi")],
                lexical_terms=[],
                exact_strings=[],
                kept_vi_terms=[],
                objects=[],
                attributes=[],
                trake_stages=[],
                strict_order=(task_type == "trake"),
                refinement_used=False,
                refinement_backend="off",
            )
            tot_ms = (time.perf_counter() - t0) * 1000.0
            timings["total_query_ms"] = tot_ms
            timings["total_ms"] = tot_ms
            return plan, timings

        config_dict = {
            "backend": self.backend,
            "model_name": getattr(self.llm_refiner, "model_name_or_path", QUERY_REFINER_MODEL) if self.llm_refiner else "none",
            "max_visual_variants": QUERY_REFINER_MAX_VISUAL_VARIANTS,
            "schema_version": SCHEMA_VERSION,
            "prompt_version": PROMPT_VERSION,
        }
        cache_key = ""
        if self.cache:
            cache_key = self.cache.compute_fingerprint(query, task_type, self.backend, config_dict)
            cached_plan = self.cache.get(cache_key)
            if cached_plan:
                tot_ms = (time.perf_counter() - t0) * 1000.0
                timings["total_query_ms"] = tot_ms
                timings["total_ms"] = tot_ms
                return cached_plan, timings

        # 1. Layer A: Deterministic Parse
        t_parse_start = time.perf_counter()
        base_plan = self.deterministic_parser.parse(query, task_type=task_type)
        t_parse_ms = (time.perf_counter() - t_parse_start) * 1000.0
        timings["deterministic_parse_ms"] = t_parse_ms
        timings["query_parse_ms"] = t_parse_ms

        final_plan = base_plan

        # 2. Layer B: Optional Local LLM Refine
        if self.backend in ("local_llm", "auto") and self.llm_refiner:
            t_llm_start = time.perf_counter()
            llm_plan = self.llm_refiner.refine(query, base_plan)
            t_llm_ms = (time.perf_counter() - t_llm_start) * 1000.0
            timings["llm_refine_ms"] = t_llm_ms
            if llm_plan is not None:
                final_plan = llm_plan

        if self.cache and cache_key:
            self.cache.put(cache_key, final_plan)

        tot_ms = (time.perf_counter() - t0) * 1000.0
        timings["total_query_ms"] = tot_ms
        timings["total_ms"] = tot_ms
        return final_plan, timings

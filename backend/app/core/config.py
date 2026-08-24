# Configuration flags for the project
import os


def _environment_bool(name, default):
    raw_value = os.getenv(name, default)
    normalized = raw_value.strip().lower()
    if normalized in {"1", "true", "yes"}:
        return True
    if normalized in {"0", "false", "no"}:
        return False
    raise ValueError(f"{name} must be one of: 1, 0, true, false, yes, no")


SIGLIP_ENABLED = _environment_bool("SIGLIP_ENABLED", "true")
SIGLIP2_MODEL = os.getenv(
    "SIGLIP2_MODEL", "google/siglip2-base-patch16-224"
).strip()
if not SIGLIP2_MODEL:
    raise ValueError("SIGLIP2_MODEL must be a non-empty model ID or local path")
# Long text encoding is isolated behind a mode flag so deployments can restore
# the historical first-window behavior with SIGLIP_LONG_TEXT_MODE=truncate.
SIGLIP_LONG_TEXT_MODE = os.getenv("SIGLIP_LONG_TEXT_MODE", "chunk_mean").lower()
SIGLIP_TEXT_MAX_LENGTH = int(os.getenv("SIGLIP_TEXT_MAX_LENGTH", "64"))
SIGLIP_TEXT_CHUNK_STRIDE = int(os.getenv("SIGLIP_TEXT_CHUNK_STRIDE", "8"))
SIGLIP_TEXT_MAX_CHUNKS = int(os.getenv("SIGLIP_TEXT_MAX_CHUNKS", "8"))
FASTER_WHISPER_MODEL = "small"
DINO_ENABLED = False
E5_ENABLED = True
BM25_ENABLED = True

VECTOR_STORE = "FAISS"  # options: FAISS, QDRANT, MILVUS
# Adaptive OCR settings
OCR_BACKEND = os.getenv("OCR_BACKEND", "auto").lower()
OCR_CPU_BACKEND = os.getenv("OCR_CPU_BACKEND", "tesseract").lower()
OCR_GPU_BACKEND = os.getenv("OCR_GPU_BACKEND", "paddleocr").lower()
OCR_FALLBACK_BACKEND = os.getenv("OCR_FALLBACK_BACKEND", "tesseract").lower()
OCR_PADDLE_DEVICE = os.getenv("OCR_PADDLE_DEVICE", "auto").lower()
OCR_PADDLE_MIN_CONFIDENCE = float(os.getenv("OCR_PADDLE_MIN_CONFIDENCE", "0.50"))
OCR_FALLBACK_ON_EMPTY = os.getenv("OCR_FALLBACK_ON_EMPTY", "1").lower() in ("1", "true", "yes")
OCR_FALLBACK_ON_ERROR = os.getenv("OCR_FALLBACK_ON_ERROR", "1").lower() in ("1", "true", "yes")
OCR_FALLBACK_ON_LOW_CONFIDENCE = os.getenv("OCR_FALLBACK_ON_LOW_CONFIDENCE", "1").lower() in ("1", "true", "yes")
ENABLE_LLM_EXPANSION = False
ENABLE_LLM_ANSWERING = True
CAPTION_ENABLED = False

# QA context window (ms)
QA_CONTEXT_BEFORE_MS = 5000
QA_CONTEXT_AFTER_MS = 5000

# TRAKE settings
TRAKE_CANDIDATE_VIDEOS = 10
TRAKE_BEAM_WIDTH = 30
TRAKE_TEMPORAL_REFINE_ENABLED = os.getenv("TRAKE_TEMPORAL_REFINE_ENABLED", "true").lower() in ("1", "true", "yes")
TRAKE_TEMPORAL_REFINE_WINDOW_SECONDS = float(os.getenv("TRAKE_TEMPORAL_REFINE_WINDOW_SECONDS", "2.5"))
TRAKE_TEMPORAL_REFINE_SAMPLE_FPS = float(os.getenv("TRAKE_TEMPORAL_REFINE_SAMPLE_FPS", "5.0"))
TRAKE_TEMPORAL_REFINE_MAX_REGIONS_PER_VIDEO = int(os.getenv("TRAKE_TEMPORAL_REFINE_MAX_REGIONS_PER_VIDEO", "3"))
TRAKE_TEMPORAL_REFINE_MAX_TOTAL_REGIONS = int(os.getenv("TRAKE_TEMPORAL_REFINE_MAX_TOTAL_REGIONS", "6"))
TRAKE_TEMPORAL_REFINE_MAX_FRAMES_PER_REGION = int(os.getenv("TRAKE_TEMPORAL_REFINE_MAX_FRAMES_PER_REGION", "50"))
TRAKE_TEMPORAL_REFINE_CACHE_ENABLED = os.getenv("TRAKE_TEMPORAL_REFINE_CACHE_ENABLED", "true").lower() in ("1", "true", "yes")

# Query Refiner settings
QUERY_REFINER_ENABLED = os.getenv("QUERY_REFINER_ENABLED", "true").lower() in ("1", "true", "yes")
QUERY_REFINER_BACKEND = os.getenv("QUERY_REFINER_BACKEND", "auto").lower()
QUERY_REFINER_MODEL = os.getenv("QUERY_REFINER_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
QUERY_REFINER_MAX_VISUAL_VARIANTS = int(os.getenv("QUERY_REFINER_MAX_VISUAL_VARIANTS", "4"))
QUERY_REFINER_CACHE_ENABLED = os.getenv("QUERY_REFINER_CACHE_ENABLED", "true").lower() in ("1", "true", "yes")
QUERY_REFINER_RRF_K = int(os.getenv("QUERY_REFINER_RRF_K", "60"))
DEBUG_QUERY_PLAN = os.getenv("DEBUG_QUERY_PLAN", "false").lower() in ("1", "true", "yes")

# Evidence-aware Reranker settings
RERANKER_ENABLED = os.getenv("RERANKER_ENABLED", "true").lower() in ("1", "true", "yes")

# TRAKE Temporal Coherence settings
TRAKE_COHERENCE_MODE = os.getenv("TRAKE_COHERENCE_MODE", "diagnostic").lower()

# Visual Frame Sampling Optimization settings
VISUAL_SAMPLING_MODE = os.getenv("VISUAL_SAMPLING_MODE", "legacy").lower()
VISUAL_GLOBAL_SAMPLE_SECONDS = float(os.getenv("VISUAL_GLOBAL_SAMPLE_SECONDS", "5.0"))
VISUAL_DEDUP_ENABLED = os.getenv("VISUAL_DEDUP_ENABLED", "false").lower() in ("true", "1", "yes")
VISUAL_DEDUP_THRESHOLD = float(os.getenv("VISUAL_DEDUP_THRESHOLD", "0.97"))

# Local Dense Frame Refinement settings
LOCAL_REFINE_ENABLED = os.getenv("LOCAL_REFINE_ENABLED", "false").lower() in ("true", "1", "yes")
LOCAL_REFINE_WINDOW_SECONDS = float(os.getenv("LOCAL_REFINE_WINDOW_SECONDS", "10.0"))
LOCAL_REFINE_INTERVAL_SECONDS = float(os.getenv("LOCAL_REFINE_INTERVAL_SECONDS", "0.5"))
LOCAL_REFINE_MAX_REGIONS = int(os.getenv("LOCAL_REFINE_MAX_REGIONS", "5"))

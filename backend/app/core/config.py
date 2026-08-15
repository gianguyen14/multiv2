# Configuration flags for the project
import os
SIGLIP_ENABLED = True
SIGLIP2_MODEL = "google/siglip2-base-patch16-224"
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
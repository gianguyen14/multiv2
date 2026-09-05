import io
import logging
import os
from email.parser import BytesParser
from email.policy import default as email_policy
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field, field_validator

from backend.app.services.configured_search import ConfiguredSearch

logger = logging.getLogger(__name__)

MAX_IMAGE_UPLOAD_BYTES = 15 * 1024 * 1024
SUPPORTED_IMAGE_FORMATS = {"JPEG", "PNG", "WEBP"}


def _debug_api_errors_enabled() -> bool:
    return os.getenv("DEBUG_API_ERRORS", "false").strip().lower() in {"1", "true", "yes"}


def _unavailable_detail(kind: str, exc: Exception) -> str:
    """Return a client-safe 503 message; raw exception text is debug-only."""
    if _debug_api_errors_enabled():
        return f"{kind} is unavailable: {type(exc).__name__}: {exc}"
    return f"{kind} is unavailable"


async def _read_limited_body(request: Request, limit: int = MAX_IMAGE_UPLOAD_BYTES) -> bytes:
    body = bytearray()
    async for chunk in request.stream():
        if len(body) + len(chunk) > limit:
            raise HTTPException(413, "image file exceeds 15MB limit")
        body.extend(chunk)
    return bytes(body)


def _multipart_file(body: bytes, content_type: str) -> bytes:
    message = BytesParser(policy=email_policy).parsebytes(
        b"Content-Type: " + content_type.encode("latin-1")
        + b"\r\nMIME-Version: 1.0\r\n\r\n" + body
    )
    if not message.is_multipart():
        raise HTTPException(400, "invalid multipart image upload")
    for part in message.iter_parts():
        if part.get_param("name", header="content-disposition") == "file":
            payload = part.get_payload(decode=True)
            if payload:
                return payload
            raise HTTPException(400, "empty image upload")
    raise HTTPException(400, "multipart upload is missing file field")


def _decode_image(body: bytes) -> Image.Image:
    try:
        with Image.open(io.BytesIO(body)) as opened:
            if opened.format not in SUPPORTED_IMAGE_FORMATS:
                raise HTTPException(400, "unsupported image format; use JPEG, PNG, or WebP")
            opened.load()
            return opened.convert("RGB")
    except HTTPException:
        raise
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise HTTPException(400, "invalid or corrupt image") from exc


class SearchRequest(BaseModel):
    query: str = Field(default="", max_length=2000)
    query_type: Literal["kis", "qa", "trake"] = "kis"
    events: list[str] = Field(default_factory=list, max_length=20)
    top_k: int = Field(default=100, ge=1, le=1000)
    temporal_refine: bool = Field(default=True)
    query_refine: bool = Field(default=True)
    rerank: bool = Field(default=True)
    debug_query_plan: bool = Field(default=False)

    @field_validator("events")
    @classmethod
    def validate_events(cls, events):
        if any(not event.strip() or len(event) > 2000 for event in events):
            raise ValueError("events must be non-empty and at most 2000 characters")
        return events


def _build_configured_search(media_root):
    """Select the production search backend.

    ``SEARCH_BACKEND`` is the canonical selector; ``SEARCH_ENCODER`` is accepted
    as a legacy alias. The production default is ``qwen3_vl`` (Qwen3-VL-Embedding-2B
    over the packed 47,430 x 1024-d DB). ``siglip2`` remains available only as an
    explicit legacy mode. Unknown values fail loudly at startup so a deployment can
    never silently fall back to the wrong embedding space.
    """
    backend = (os.getenv("SEARCH_BACKEND") or os.getenv("SEARCH_ENCODER") or "qwen3_vl")
    backend = backend.strip().lower()
    if backend in ("qwen3_vl", "qwen3-vl", "qwen"):
        from backend.app.services.qwen_runtime_search import QwenRuntimeSearch

        return QwenRuntimeSearch(processed_root=media_root)
    if backend == "siglip2":
        return ConfiguredSearch(media_root)
    raise RuntimeError(
        f"Unknown SEARCH_BACKEND={backend!r}; supported values: qwen3_vl (default), siglip2 (legacy)"
    )


def create_app(search_handler=None, media_root=None, configured_search=None):
    app = FastAPI(title="AIC 2026 Retrieval")

    @app.middleware("http")
    async def add_security_headers(request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "SAMEORIGIN"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        return response

    frontend = Path(__file__).parents[2] / "frontend" / "src"
    frontend_root = frontend.resolve()
    configured_search = configured_search or _build_configured_search(media_root)
    uses_configured_search = search_handler is None
    search_handler = search_handler or (configured_search.handle if configured_search.configured else None)
    media_root = Path(media_root or configured_search.processed_root or "data/processed/videos").resolve()

    def _frontend_asset(subdir: str, filename: str):
        if not filename or "/" in filename or "\\" in filename or ".." in filename:
            raise HTTPException(404, "asset not found")
        asset_dir = (frontend_root / subdir).resolve()
        path = (asset_dir / filename).resolve()
        if (not asset_dir.is_relative_to(frontend_root)
                or not path.is_relative_to(asset_dir) or not path.is_file()):
            raise HTTPException(404, "asset not found")
        return FileResponse(path)

    @app.get("/")
    def index():
        return FileResponse(frontend / "index.html")

    @app.get("/styles/{filename}")
    def style(filename: str):
        return _frontend_asset("styles", filename)

    @app.get("/scripts/{filename}")
    def script(filename: str):
        return _frontend_asset("scripts", filename)

    def health_payload():
        from backend.app.runtime.device_policy import runtime_summary
        return {"status": "ok", "search_configured": search_handler is not None,
            "processed_root": str(configured_search.processed_root) if configured_search.configured else None,
            "search": configured_search.status(), "compute": runtime_summary()}

    @app.get("/health")
    @app.get("/health/live")
    def health():
        return health_payload()

    @app.get("/health/ready")
    def ready():
        payload = health_payload()
        if search_handler is None:
            raise HTTPException(503, detail={**payload, "status": "not_ready", "reason": "search is not configured"})
        if uses_configured_search:
            readiness = configured_search.readiness()
            if not readiness["ready"]:
                raise HTTPException(503, detail={**payload, "status": "not_ready", **readiness})
            payload["search_readiness"] = readiness
        return {**payload, "status": "ready"}

    @app.post("/api/search")
    def search(request: SearchRequest):
        if search_handler is None:
            raise HTTPException(503, "search index is not configured")
        if not request.query.strip() and not request.events:
            raise HTTPException(400, "query is required")
        try:
            results = search_handler(request.model_dump())
            resp = {"results": results}
            debug_on = request.debug_query_plan or os.getenv("DEBUG_QUERY_PLAN", "false").lower() in ("1", "true", "yes")
            if debug_on and uses_configured_search and configured_search.last_query_plan:
                resp["query_plan"] = configured_search.last_query_plan.to_dict()
                resp["query_metrics"] = configured_search.last_query_metrics
            return resp
        except (FileNotFoundError, RuntimeError) as exc:
            logger.exception("search unavailable")
            raise HTTPException(503, _unavailable_detail("search", exc)) from exc
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.post("/api/search/image")
    async def search_image_endpoint(
        request: Request,
        top_k: int = Query(default=100, ge=1, le=1000),
        raw: bool = Query(default=False)
    ):
        if not configured_search.configured:
            raise HTTPException(503, "search index is not configured")
        capabilities = configured_search.status().get("capabilities") or {}
        if capabilities and not capabilities.get("image", True):
            raise HTTPException(503, "image search is not supported by the active search backend")
        content_type = request.headers.get("content-type", "").lower()
        if not (content_type.startswith("multipart/form-data")
                or content_type.split(";", 1)[0].strip() in {"image/jpeg", "image/png", "image/webp"}):
            raise HTTPException(415, "content type must be JPEG, PNG, WebP, or multipart/form-data")
        body = await _read_limited_body(request)
        if not body:
            raise HTTPException(400, "empty image upload")
        if content_type.startswith("multipart/form-data"):
            body = _multipart_file(body, request.headers["content-type"])
        image = _decode_image(body)
        try:
            results = configured_search.search_image(image, top_k=top_k, deduplicate=not raw)
            return {"results": results}
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        except (FileNotFoundError, RuntimeError) as exc:
            logger.exception("image search failed")
            raise HTTPException(503, _unavailable_detail("image search", exc)) from exc
        finally:
            image.close()

    from fastapi.middleware.cors import CORSMiddleware
    allowed_origins = os.getenv("ALLOWED_ORIGINS")
    if allowed_origins:
        origins = [origin.strip() for origin in allowed_origins.split(",") if origin.strip()]
        if "*" in origins:
            raise RuntimeError("ALLOWED_ORIGINS must list explicit origins when credentials are enabled")
    else:
        origins = ["http://localhost:3000", "http://127.0.0.1:3000"]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    @app.get("/api/frames/{video_id}/{filename}")
    def frame(video_id: str, filename: str):
        if Path(filename).suffix.lower() not in {".jpg", ".webp", ".png"} or "/" in filename or "\\" in filename:
            raise HTTPException(404, "frame not found")
        frame_root = (media_root / video_id / "frames").resolve()
        path = (frame_root / filename).resolve()
        if (not frame_root.is_relative_to(media_root) or not path.is_relative_to(frame_root)
                or not path.is_file()):
            raise HTTPException(404, "frame not found")
        return FileResponse(path, headers={"Cache-Control": "public, max-age=86400"})

    return app


app = create_app()

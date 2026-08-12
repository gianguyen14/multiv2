import io
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from PIL import Image

from backend.app.services.search_service import SearchService

router = APIRouter()
_search_service: Optional[SearchService] = None


def configure_search_service(service: SearchService) -> None:
    global _search_service
    _search_service = service


def _service() -> SearchService:
    if _search_service is None:
        raise HTTPException(status_code=503, detail="Search service is not configured")
    return _search_service


@router.post("/search/image")
async def search_image(
    image: UploadFile = File(...),
    top_k: int = Form(10),
):
    contents = await image.read()
    try:
        query_image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid image file") from exc
    results = _service().search_images([query_image], top_k)
    return {"results": results[0] if results else []}


@router.post("/search/text")
async def search_text(text: str = Form(...), top_k: int = Form(10)):
    results = _service().search_text([text], top_k)
    return {"results": results[0] if results else []}

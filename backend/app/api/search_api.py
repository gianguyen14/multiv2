"""[DEPRECATED / INACTIVE STACK]
This module is a historical milestone prototype and is NOT part of the active runtime architecture.
The authoritative production stack is:
  - CLI: projectctl.py
  - Service: backend.app.services.configured_search.ConfiguredSearch
  - API: backend.app.main:create_app
  - Index: backend.app.video.frame_index (CURRENT generation)
"""

from typing import Any, Optional


from PIL import Image

from backend.app.services.search_service import SearchService

_search_service: Optional[SearchService] = None


def configure_search_service(service: SearchService) -> None:
    global _search_service
    _search_service = service


def _get_search_service() -> SearchService:
    if _search_service is None:
        raise RuntimeError("Search service is not configured")
    return _search_service


def search_text_endpoint(text: str, top_k: int) -> dict:
    return {"results": _get_search_service().search_by_text(text, top_k)}


def search_image_endpoint(image: Any, top_k: int) -> dict:
    return {"results": _get_search_service().search_by_image(image, top_k)}

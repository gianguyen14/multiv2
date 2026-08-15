"""[DEPRECATED / INACTIVE STACK]
This module is a historical milestone prototype and is NOT part of the active runtime architecture.
The authoritative production stack is:
  - CLI: projectctl.py
  - Service: backend.app.services.configured_search.ConfiguredSearch
  - API: backend.app.main:create_app
  - Index: backend.app.video.frame_index (CURRENT generation)
"""

from typing import Any, Optional


from backend.app.services.advanced_search_service import AdvancedSearchService

_service: Optional[AdvancedSearchService] = None


def configure_advanced_search_service(service: AdvancedSearchService) -> None:
    global _service
    _service = service


def _get_service() -> AdvancedSearchService:
    if _service is None:
        raise RuntimeError("Advanced search service is not configured")
    return _service


def _response(results: list, expand_factor: int) -> dict:
    return {"results": results, "meta": {"reranked": True, "expand_factor": expand_factor}}


def search_text_advanced(text: str, top_k: int, expand_factor: int = 3) -> dict:
    service = _get_service()
    service.retriever.expand_factor = expand_factor
    return _response(service.search_text(text, top_k), expand_factor)


def search_image_advanced(image: Any, top_k: int, expand_factor: int = 3) -> dict:
    service = _get_service()
    service.retriever.expand_factor = expand_factor
    return _response(service.search_image(image, top_k), expand_factor)


def search_hybrid(
    text: Optional[str], image: Any, top_k: int, expand_factor: int = 3
) -> dict:
    service = _get_service()
    service.retriever.expand_factor = expand_factor
    return _response(service.search_hybrid(text, image, top_k), expand_factor)

from __future__ import annotations

from fastapi import APIRouter
import httpx

from ..config import settings
from ..vectorstore import vs

router = APIRouter(prefix="/status", tags=["status"])


@router.get("")
async def get_status():
    # Check Ollama model availability
    model_available = False
    tags = []
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.get(f"{settings.llm_base_url.rstrip('/')}/api/tags")
            r.raise_for_status()
            tags = [m.get("name") for m in r.json().get("models", [])]
            model_available = settings.llm_model in tags
    except Exception:
        pass

    # Check Qdrant points count
    points = None
    try:
        cnt = vs.client.count(collection_name=vs.collection, exact=True)
        # qdrant-client returns CountResponse with 'count'
        points = int(getattr(cnt, "count", 0))
    except Exception:
        try:
            info = vs.client.get_collection(vs.collection)
            points = int(getattr(info, "points_count", 0) or 0)
        except Exception:
            points = None

    return {
        "llm_model": settings.llm_model,
        "model_available": model_available,
        "available_models": tags,
        "qdrant_collection": vs.collection,
        "points": points,
        "docs_dir": str(settings.docs_dir),
    }

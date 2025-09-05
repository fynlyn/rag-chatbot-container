from __future__ import annotations

from fastapi import APIRouter

from ..config import settings
from ..embeddings import embeddings
from ..loaders import load_all
from ..vectorstore import vs

router = APIRouter(prefix="/ingest", tags=["ingest"])


@router.post("/run")
async def run_ingest():
    docs = load_all(settings.docs_dir)
    if not docs:
        return {"status": "ok", "indexed": 0}
    texts = [d["text"] for d in docs]
    vecs = embeddings.embed(texts)
    ids = [d["id"] for d in docs]
    # Include text in payload for easier retrieval context
    payloads = [{**d["metadata"], "text": d["text"]} for d in docs]
    vs.upsert(ids, vecs, payloads)
    return {"status": "ok", "indexed": len(ids)}

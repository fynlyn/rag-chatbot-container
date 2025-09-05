from __future__ import annotations

from typing import List

from fastapi import APIRouter
from pydantic import BaseModel

from ..config import settings
from ..embeddings import embeddings
from ..llm import ollama
from ..vectorstore import vs

router = APIRouter(prefix="/chat", tags=["chat"])


class ChatRequest(BaseModel):
    query: str
    stream: bool = False
    top_k: int | None = None


def build_prompt(query: str, contexts: List[str]) -> str:
    context_block = "\n\n".join(f"- {c}" for c in contexts)
    return (
        f"System: {settings.system_prompt}\n\n"
        f"Context:\n{context_block}\n\n"
        f"User question: {query}\n\n"
        f"Answer:"
    )


@router.post("/ask")
async def ask(req: ChatRequest):
    q = req.query.strip()
    if not q:
        return {"answer": ""}

    # E5 recommends query prefix
    query_text = f"query: {q}"
    qvec = embeddings.embed_query(query_text)
    results = vs.search(qvec, top_k=req.top_k or settings.top_k)
    final_contexts: List[str] = []
    for r in results:
        payload = r.payload or {}
        if isinstance(payload, dict):
            txt = payload.get("text")
            if txt:
                final_contexts.append(str(txt))
            elif payload.get("source"):
                final_contexts.append(f"See: {payload['source']}")
    if not final_contexts:
        final_contexts = ["No specific context retrieved."]

    prompt = build_prompt(q, final_contexts)

    if req.stream:
        # Note: Streaming is served via SSE in main app route /chat/stream
        ans = await ollama.generate(
            settings.llm_model, 
            prompt, 
            settings.llm_temperature, 
            system=None,
            max_tokens=200,
            timeout=30.0
        )
        return {"answer": ans, "sources": final_contexts}

    ans = await ollama.generate(
        settings.llm_model, 
        prompt, 
        settings.llm_temperature, 
        system=None,
        max_tokens=200,
        timeout=30.0
    )
    return {"answer": ans, "sources": final_contexts}

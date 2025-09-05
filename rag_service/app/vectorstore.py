from __future__ import annotations

from typing import List, Optional

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from .config import settings


class VectorStore:
    def __init__(self):
        self.client = QdrantClient(url=settings.qdrant_url)
        self.collection = settings.qdrant_collection
        self._ensure_collection()

    def _ensure_collection(self):
        exists = False
        try:
            info = self.client.get_collection(self.collection)
            exists = info is not None
        except Exception:
            exists = False
        if exists and settings.recreate_collection:
            self.client.delete_collection(self.collection)
            exists = False
        if not exists:
            # Create collection with cosine distance; size from embedding model
            from .embeddings import embeddings

            dim = embeddings.dim
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=qmodels.VectorParams(size=dim, distance=qmodels.Distance.COSINE),
            )

    def upsert(self, ids: List[int], vectors: List[List[float]], payloads: List[dict]):
        self.client.upsert(
            collection_name=self.collection,
            points=qmodels.Batch(
                ids=ids,
                vectors=vectors,
                payloads=payloads,
            ),
        )

    def search(self, vector: List[float], top_k: int = 5, filter: Optional[qmodels.Filter] = None):
        return self.client.search(
            collection_name=self.collection,
            query_vector=vector,
            limit=top_k,
            query_filter=filter,
            with_payload=True,
        )


vs = VectorStore()

from __future__ import annotations

from typing import List

from sentence_transformers import SentenceTransformer

from .config import settings


class EmbeddingModel:
    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or settings.embedding_model
        # Lazy-load the model at first use
        self._model: SentenceTransformer | None = None

    def _ensure(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed(self, texts: List[str]) -> List[List[float]]:
        model = self._ensure()
        # E5 expects instruction prefixes; we'll apply a simple one for queries
        return model.encode(texts, normalize_embeddings=True).tolist()

    def embed_query(self, text: str) -> List[float]:
        model = self._ensure()
        return model.encode([text], normalize_embeddings=True)[0].tolist()

    @property
    def dim(self) -> int:
        model = self._ensure()
        return int(model.get_sentence_embedding_dimension())


embeddings = EmbeddingModel()

from __future__ import annotations

import random
from typing import List

from .config import settings


class EmbeddingModel:
    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or settings.embedding_model
        print(f"EmbeddingModel initialized with {self.model_name}")

    def embed(self, texts: List[str]) -> List[List[float]]:
        # Dummy embeddings for debugging
        return [[random.random() for _ in range(384)] for _ in texts]

    def embed_query(self, text: str) -> List[float]:
        # Dummy query embedding
        return [random.random() for _ in range(384)]

    @property
    def dim(self) -> int:
        return 384


embeddings = EmbeddingModel()

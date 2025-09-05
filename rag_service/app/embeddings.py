from __future__ import annotations

import random
from typing import List

from .config import settings


class EmbeddingModel:
    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or settings.embedding_model
        print(f"EmbeddingModel initialized with {self.model_name}")
        
        # Try to use a lighter embedding model if available
        try:
            from sentence_transformers import SentenceTransformer
            # Use a smaller, faster model
            self.model = SentenceTransformer('all-MiniLM-L6-v2')  # 384 dimensions, faster
            self._use_real_embeddings = True
            print("Using real sentence-transformers embeddings (all-MiniLM-L6-v2)")
        except Exception as e:
            print(f"Failed to load sentence-transformers: {e}")
            print("Using dummy embeddings for testing")
            self.model = None
            self._use_real_embeddings = False

    def embed(self, texts: List[str]) -> List[List[float]]:
        if self._use_real_embeddings and self.model:
            try:
                return self.model.encode(texts).tolist()
            except Exception as e:
                print(f"Error encoding texts: {e}, falling back to dummy")
        
        # Dummy embeddings fallback
        return [[random.random() for _ in range(384)] for _ in texts]

    def embed_query(self, text: str) -> List[float]:
        if self._use_real_embeddings and self.model:
            try:
                return self.model.encode([text])[0].tolist()
            except Exception as e:
                print(f"Error encoding query: {e}, falling back to dummy")
        
        # Dummy query embedding fallback
        return [random.random() for _ in range(384)]

    @property
    def dim(self) -> int:
        return 384


embeddings = EmbeddingModel()

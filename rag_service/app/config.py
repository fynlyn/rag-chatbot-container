from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from pydantic import Field, validator
import os
import yaml
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RAG_", extra="ignore")

    # Paths
    docs_dir: Path = Field(default=Path("/data/docs"))
    # Vector DB
    qdrant_url: str = Field(default_factory=lambda: "http://qdrant:6333")
    qdrant_collection: str = Field(default="company-files")
    # Embeddings
    embedding_model: str = Field(default="intfloat/multilingual-e5-base")
    embedding_dim: int | None = None  # auto-detected
    # LLM
    llm_model: str = Field(default="llama3.1:8b")  # Ollama model tag
    llm_temperature: float = 0.2
    llm_base_url: str = Field(default_factory=lambda: "http://ollama:11434")
    # Chunking
    chunk_size: int = 1000
    chunk_overlap: int = 100
    # Indexing
    recreate_collection: bool = False

    # UI
    system_prompt: str = (
        "You are a helpful assistant answering employee questions from internal documents.\n"
        "Use only the provided context. If unsure, say you cannot find the answer in the available resources."
    )
    top_k: int = 5

    @validator("docs_dir", pre=True)
    def _expand_docs(cls, v):
        return Path(v).expanduser()


def load_settings() -> Settings:
    # Default env-provided values
    cfg_path = os.getenv("CONFIG_PATH", "/config/config.yaml")
    data = {}
    if Path(cfg_path).exists():
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        except Exception:
            data = {}
    # Environment wins over file (handled by BaseSettings merge)
    return Settings(**data)


settings = load_settings()

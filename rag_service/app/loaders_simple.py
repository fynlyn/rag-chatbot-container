from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable, List

from pypdf import PdfReader

from .config import settings


SUPPORTED_EXTS = {".txt", ".md", ".pdf"}


def file_id(path: Path) -> str:
    return hashlib.sha1(str(path).encode("utf-8")).hexdigest()


def iter_files(root: Path) -> Iterable[Path]:
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS:
            yield p


def simple_text_split(text: str, chunk_size: int = 1000, chunk_overlap: int = 100) -> List[str]:
    """Simple text splitter without langchain dependency."""
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start = max(start + chunk_size - chunk_overlap, start + 1)
        if start >= len(text):
            break
    return chunks


def load_file(path: Path) -> List[dict]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        try:
            reader = PdfReader(str(path))
            text = "\n".join(page.extract_text() for page in reader.pages)
        except Exception:
            text = f"Failed to read PDF: {path}"
    else:
        # Handle .txt and .md as plain text
        try:
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
        except Exception:
            text = f"Failed to read file: {path}"
    
    chunks = simple_text_split(text, settings.chunk_size, settings.chunk_overlap)
    items = []
    for i, chunk_text in enumerate(chunks):
        # E5 document prefix improves retrieval quality
        prefixed = f"passage: {chunk_text}"
        items.append(
            {
                "id": f"{file_id(path)}-{i}",
                "text": prefixed,
                "metadata": {"source": str(path), "chunk": i},
            }
        )
    return items


def load_all(root: Path) -> List[dict]:
    all_chunks: List[dict] = []
    for p in iter_files(root):
        all_chunks.extend(load_file(p))
    return all_chunks

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable, List

from .config import settings


SUPPORTED_EXTS = {".txt", ".md", ".pdf"}


def file_id(path: Path) -> str:
    return hashlib.sha1(str(path).encode("utf-8")).hexdigest()


def iter_files(root: Path) -> Iterable[Path]:
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS:
            yield p


def simple_text_splitter(text: str, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
    """Simple text splitting without heavy dependencies"""
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        if end >= len(text):
            chunks.append(text[start:])
            break
        
        # Try to split at word boundary
        while end > start and text[end] not in ' \n\t':
            end -= 1
        if end == start:  # fallback if no word boundary found
            end = start + chunk_size
            
        chunks.append(text[start:end])
        start = end - overlap
        
    return chunks


def load_file(path: Path) -> List[dict]:
    suffix = path.suffix.lower()
    
    try:
        if suffix == ".pdf":
            import pypdf
            with open(path, 'rb') as f:
                reader = pypdf.PdfReader(f)
                text = "\n".join(page.extract_text() for page in reader.pages)
        else:
            # Handle .txt, .md as text
            with open(path, 'r', encoding='utf-8') as f:
                text = f.read()
                
        chunks = simple_text_splitter(text, settings.chunk_size, settings.chunk_overlap)
        
        items = []
        for i, chunk in enumerate(chunks):
            # E5 document prefix improves retrieval quality
            prefixed = f"passage: {chunk.strip()}"
            items.append({
                "id": f"{file_id(path)}-{i}",
                "text": prefixed,
                "metadata": {"source": str(path), "chunk": i},
            })
        return items
        
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return []


def load_all(root: Path) -> List[dict]:
    all_chunks: List[dict] = []
    for p in iter_files(root):
        all_chunks.extend(load_file(p))
    return all_chunks

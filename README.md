# RAG Chatbot Container (All-in-One)

An all-in-one, Dockerized RAG stack you can run on any machine. It uses:

- Ollama for local LLMs (LLama 3.1 8B, Qwen2.5 14B, DeepSeek R1, etc.)
- Qdrant as a local vector database
- Sentence-Transformers `intfloat/multilingual-e5-base` for multilingual embeddings
- FastAPI backend with a minimal web chat UI

You can customize via a single config file and environment variables. No external APIs required.

## Quick start

Prereqs: Docker & Docker Compose. For GPU acceleration, install NVIDIA Container Toolkit. CPU works too (slower).

1) Place your documents under `data/docs/` (txt, md, pdf supported)
2) Optionally edit `config/config.yaml`
3) Start the stack

```
docker compose up -d --build
```

4) Open the UI at http://localhost:8000 and click "Index Documents" once. Then chat.

The first run will download models:
- Ollama model (default `llama3.1:8b`). You can switch models in config.
- Sentence-Transformers `intfloat/multilingual-e5-base` for embeddings.

## Configuration

All defaults live in `config/config.yaml`. Environment variables with prefix `RAG_` override values, and `CONFIG_PATH` can point to a different YAML.

Key options:

- docs_dir: Path inside the container with your docs (`/data/docs`)
- qdrant_url: Vector DB URL (default internal `http://qdrant:6333`)
- qdrant_collection: Collection name (default `company-files`)
- embedding_model: Defaults to `intfloat/multilingual-e5-base`
- chunk_size / chunk_overlap: Text splitting
- llm_base_url: Ollama base URL (internal)
- llm_model: One of:
	- `llama3.1:8b` (default)
	- `qwen2.5:14b`
	- `deepseek-r1:latest` (reasoning; slower)
- llm_temperature: Defaults to 0.2
- top_k: Retrieval depth (default 5)
- recreate_collection: If true, recreates the collection at startup
- system_prompt: Instruction string

Environment overrides examples:

```
RAG_LLM_MODEL=qwen2.5:14b docker compose up -d
RAG_TOP_K=8 docker compose restart rag
```

## Endpoints

- POST `/ingest/run` — index files from `docs_dir`
- POST `/chat/ask` — JSON body `{ "query": "...", "top_k": 5, "stream": false }`
- GET `/chat/stream?q=...` — Server-sent events streaming answer
- GET `/health` — service health

## Notes on models

The container starts an Ollama service. The API will auto-pull the model you set the first time you query it. To pre-pull manually on the host:

```
docker compose exec ollama ollama pull llama3.1:8b
```

Alternate models:
- `qwen2.5:14b` for larger capability
- `deepseek-r1:latest` for reasoning tasks

## n8n parity

This stack mirrors your n8n workflow but uses local components:
- Vector: Qdrant instead of Pinecone
- Embeddings: `intfloat/multilingual-e5-base` instead of Gemini
- LLM: Ollama models instead of Gemini Chat

You could still import this into n8n by calling the HTTP endpoints above from an n8n workflow. If you need a ready-made n8n JSON for this local stack, say the word and I’ll add it.

## Development

Project layout:

- `docker-compose.yml` — services: qdrant, ollama, rag (FastAPI)
- `rag_service/` — FastAPI app
- `config/config.yaml` — default configuration
- `data/docs/` — mount your files here

To view logs:

```
docker compose logs -f rag
```

To rebuild after code changes:

```
docker compose up -d --build rag
```

## Troubleshooting

- Long first run: model downloads can take time.
- CPU only: Ollama will run on CPU if no GPU is available.
- Permissions: ensure `data/docs` is readable by Docker.
- PDFs: Unstructured parsing can vary; consider converting tricky PDFs to text/markdown.
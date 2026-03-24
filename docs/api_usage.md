# FastAPI RAG System Usage

## Starting the Server

```bash
uv sync --group dev
PYTHONPATH=. uvicorn src.main:app --reload
```

Server runs at: http://localhost:8000

## Interfaces

| URL | Description |
|-----|-------------|
| http://localhost:8000/chat | Interactive RAG chat UI |
| http://localhost:8000/docs | Swagger API documentation |
| http://localhost:8000/info | System status (JSON) |
| http://localhost:8000/ingest/status | Ingestion progress (JSON) |

## API Endpoints

### Health Check
```
GET /
```

### System Info
```
GET /info
```
Returns ingestion status and vector store metadata.

### Chat UI
```
GET /chat
```
Interactive web interface for asking questions and viewing answers with sources.

### Data Ingestion
```
POST /ingest
{
  "document_limit": 20,    // null = all 3200 passages
  "chunk_size": 1000,       // max characters per text chunk
  "batch_size": 5,          // embeddings per API call
  "overwrite": true          // replace existing index
}
```
Runs in the background. Monitor with `GET /ingest/status`.

### Ingestion Status
```
GET /ingest/status
```
Returns `"not_started"`, `"in_progress"`, `"completed"`, or `"failed"`.

### Similarity Search (Retrieval Only)
```
POST /query
{
  "query": "Uruguay capital city",
  "top_k": 5,
  "similarity_threshold": 0.0
}
```
Returns ranked document chunks with similarity scores. No LLM generation.

### RAG Generation (Full Answer)
```
POST /rag
{
  "query": "What is the capital of Uruguay?",
  "top_k": 3
}
```
Retrieves relevant chunks, sends them to GPT-4o, returns a synthesized answer plus sources.

## Typical Workflow

1. Start server
2. Ingest data via `/ingest` (or use Swagger UI)
3. Monitor progress via `/ingest/status`
4. Ask questions via `/chat` or `/rag`

## Testing the API

```bash
# Manual test script (requires running server)
python scripts/test_api.py

# Automated pytest tests
pytest tests/test_api.py -v
```
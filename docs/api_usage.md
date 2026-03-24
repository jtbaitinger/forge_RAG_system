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
| http://localhost:8000/chat | Interactive RAG chat UI with knowledge base overview |
| http://localhost:8000/docs | Swagger API documentation |
| http://localhost:8000/info | System status (JSON) |
| http://localhost:8000/ingest/status | Ingestion progress (JSON) |
| http://localhost:8000/knowledge | Knowledge base topic analysis (JSON) |

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

**Features:**
- **Knowledge Base Overview**: Expandable panel showing what topics the system knows about
- **Interactive chat**: Ask questions, get answers with source citations
- **Source transparency**: Each answer shows which documents were used with similarity scores
- **Configurable retrieval**: Adjust number of sources retrieved (1-10)
- **Real-time status**: Shows system status (ready, offline, no data loaded)
- **Smart polling**: Efficiently checks for system status changes without overloading the API

### Data Ingestion
```
POST /ingest
{
  "document_limit": 20,    // null = all 3200 passages
  "chunk_size": 1000,       // max characters per text chunk
  "batch_size": 5,          // embeddings per API call
  "overwrite": true,        // replace existing index
  "use_ai_titles": false    // NEW: use GPT-4o to generate meaningful titles
}
```
Runs in the background. Monitor with `GET /ingest/status`.

**AI Title Generation**: When `use_ai_titles: true`, each Wikipedia passage gets analyzed by GPT-4o to generate a concise, descriptive title (e.g., "Uruguay" instead of numeric ID "0"). This improves the knowledge base overview but adds ~30 seconds to ingestion time.

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

### Knowledge Base Overview (NEW)
```
GET /knowledge
```
Returns analysis of what topics are stored in the vector database:
```json
{
  "topics": {
    "wikipedia": {
      "count": 19,
      "sample_titles": ["Uruguay", "Montevideo", "South America", ...]
    }
  },
  "total_documents": 20,
  "unique_titles_sampled": 19,
  "sample_titles": ["Uruguay", "Argentina", "Brazil", ...]
}
```
This endpoint samples the vector database to show users what knowledge is available for questioning. Used by the chat UI to display the "Knowledge Base Overview" panel.

## Typical Workflow

1. **Start server**: `uvicorn src.main:app --reload`
2. **Visit chat UI**: Go to http://localhost:8000/chat
3. **Ingest data**: Use the API or web UI to run ingestion
   - For better experience: `{"use_ai_titles": true}` (takes longer but generates meaningful titles)
   - Monitor progress via `/ingest/status` or the chat UI status badge
4. **Explore knowledge**: Expand the "📚 Knowledge Base Overview" panel to see what topics are available
5. **Ask questions**: Use the chat interface or directly via `/rag` API
6. **View sources**: Each answer shows which documents were used with similarity scores

## Enhanced Features (Recent Additions)

- **AI-Generated Titles**: Enable `use_ai_titles: true` during ingestion for meaningful topic names
- **Knowledge Base Analysis**: The `/knowledge` endpoint and chat UI panel show what the system knows
- **Optimized Frontend**: Smart polling reduces API calls and improves performance
- **Better UX**: Real-time status updates, expandable source citations, configurable retrieval

## Testing the API

```bash
# Manual test script (requires running server)
python scripts/test_api.py

# Automated pytest tests
pytest tests/test_api.py -v
```
# How the RAG System Works

This document explains, in plain language, how every part of this system works — from raw Wikipedia text to a generated answer.

## What is RAG?

RAG stands for **Retrieval-Augmented Generation**. Instead of asking an LLM to answer from its training data alone, we:

1. **Retrieve** relevant documents from our own knowledge base
2. **Augment** the LLM's prompt with those documents
3. **Generate** an answer grounded in real source material

This gives the LLM factual context it wouldn't otherwise have, reducing hallucination.

## The Two Phases

### Phase 1: Ingestion (Offline — Run Once)

This builds the searchable knowledge base.

```
Wikipedia passages → Text chunks → Embedding vectors → Vector database (on disk)
```

**Step by step:**

1. **Load data** (`data_loader.py`): Downloads the `rag-mini-wikipedia` dataset from HuggingFace. This dataset has 3,200 short Wikipedia passages, each with an `id` and a `passage` field.

2. **Chunk text** (`embedding_generator.py`): Each passage is split into smaller text chunks (controlled by `chunk_size`, default 1000 characters with 200 overlap). Overlap ensures no information is lost at chunk boundaries. Short passages may stay as a single chunk.

3. **Generate embeddings** (`embedding_generator.py`): Each chunk is sent to Azure OpenAI's `text-embedding-3-large` model, which returns a vector of **3,072 floating-point numbers**. This vector is a mathematical representation of the chunk's meaning. Semantically similar text produces similar vectors.

4. **Store in vector database** (`vector_store.py`): All chunks and their embeddings are stored in a LlamaIndex `VectorStoreIndex`, which persists to disk as JSON files. This is the knowledge base.

**You control how much data is loaded** with the `document_limit` parameter during ingestion. Setting it to `20` loads 20 passages; setting it to `null` (or omitting it) loads all 3,200.

### Phase 2: Query (Online — Every Question)

```
User question → Embed question → Find similar chunks → Build prompt → GPT-4o → Answer
```

**Step by step:**

1. **Embed the question**: Your question ("What is Uruguay?") is converted to a 3,072-dimensional vector using the **same** embedding model used during ingestion.

2. **Similarity search** (`vector_store.py`): The question vector is compared against every stored chunk vector using **cosine similarity** — a measure of how much two vectors point in the same direction. The top-K most similar chunks are returned, ranked by score (0.0 = unrelated, 1.0 = identical meaning).

3. **Build augmented prompt**: LlamaIndex constructs a prompt like:
   ```
   Context information is below.
   ---------------------
   [retrieved chunk 1]
   [retrieved chunk 2]
   [retrieved chunk 3]
   ---------------------
   Given the context information and not prior knowledge, answer the query.
   Query: What is Uruguay?
   Answer:
   ```

4. **Generate answer**: This prompt is sent to Azure OpenAI's `gpt-4o`, which reads the retrieved chunks and synthesizes a natural language answer. The model is instructed to use **only** the provided context.

5. **Return answer + sources**: The API returns both the generated answer and the source chunks with their similarity scores, so you can verify where the answer came from.

## How It "Knows" What to Retrieve

The system does **not** understand topics or categories. It uses pure math:

- The embedding model maps text into a 3,072-dimensional space where **meaning** is encoded as direction
- "Uruguay" and "Montevideo" and "South American country" all produce vectors that point in similar directions
- "Uruguay" and "quantum physics" produce vectors that point in very different directions
- Cosine similarity measures how aligned two vectors are

This is **semantic search**, not keyword search. The query "What is the capital?" will find passages about Montevideo even if they don't contain the word "capital", because the embedding model learned that "capital" and "Montevideo" are semantically related.

## Key Parameters and Their Effects

| Parameter | Where | Default | Effect |
|-----------|-------|---------|--------|
| `document_limit` | `/ingest` | null (all) | How many Wikipedia passages to load. More = broader knowledge, longer ingestion |
| `chunk_size` | `/ingest` | 1000 | Max characters per text chunk. Smaller = more precise retrieval, more chunks |
| `batch_size` | `/ingest` | 5 | How many chunks to embed at once. Smaller = avoids rate limits, slower |
| `top_k` | `/query`, `/rag` | 3-5 | How many chunks to retrieve. More = broader context, but may dilute relevance |
| `similarity_threshold` | `/query` | 0.0 | Minimum similarity score. Higher = only highly relevant results |
| `overwrite` | `/ingest` | false | Whether to replace an existing index |

## Source Code Modules

| Module | Responsibility |
|--------|---------------|
| `src/main.py` | FastAPI application — all HTTP endpoints, background ingestion |
| `src/data_loader.py` | Loads Wikipedia passages from HuggingFace, converts to LlamaIndex Documents |
| `src/embedding_generator.py` | Converts documents to text chunks, generates embeddings via Azure OpenAI |
| `src/vector_store.py` | Creates/saves/loads vector index, performs similarity search, provides query engine |
| `src/llamaindex_models.py` | **Model isolation layer** — all Azure OpenAI access must go through here |
| `src/ailab/utils/azure.py` | Azure authentication — gets bearer tokens via DefaultAzureCredential |

## Data Flow Diagram

```
                         INGESTION (one-time)
                         ====================
HuggingFace Dataset
        |
        v
  data_loader.py -----> List[Document]
                              |
                              v
              embedding_generator.py -----> List[TextNode] (with embeddings)
                                                |
                                                v
                              vector_store.py -----> VectorStoreIndex (saved to disk)


                         QUERY (per-question)
                         ====================
User Question
        |
        v
  vector_store.py -----> embed question (same model)
        |                       |
        |                       v
        |               cosine similarity against all stored chunks
        |                       |
        |                       v
        |               top-K most similar chunks
        |                       |
        v                       v
  LlamaIndex Query Engine -----> augmented prompt
        |
        v
  GPT-4o (via llamaindex_models.py) -----> generated answer + sources
```

## What the Knowledge Base Contains

The system **only knows what you ingested**. There is no other data source.

- With `document_limit: 20` — the system knows about ~20 Wikipedia topics
- With `document_limit: null` — the system knows about ~3,200 Wikipedia passages
- The dataset covers a broad range of Wikipedia topics (countries, science, history, etc.)

To check what's currently loaded, visit `GET /info` which shows the ingestion status and index metadata.

## The Chat UI

Visit `http://localhost:8000/chat` for an interactive chat interface that:
- Sends your question to the `/rag` endpoint
- Displays the generated answer
- Shows expandable source documents with similarity scores
- Checks system status (whether data has been ingested)
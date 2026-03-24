# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Wikipedia RAG (Retrieval-Augmented Generation) system built with Python, FastAPI, and LlamaIndex. It implements a complete RAG pipeline using Azure OpenAI models with controlled access through model isolation layers.

## Development Commands

### Environment Setup
```bash
# Set up virtual environment and install dependencies
uv venv
uv sync

# For development dependencies
uv sync --group dev
```

### Authentication
```bash
# Required for accessing Azure OpenAI models
azd auth login --scope api://ailab/Model.Access

# Verify authentication
azd auth token --output json | jq -r '.expiresOn'
```

### Testing
```bash
# Run tests
pytest

# Run tests with specific path
pytest tests/
```

### API Server
```bash
# Start FastAPI server (when implemented)
# Command will be available after implementation
uvicorn main:app --reload
```

## Architecture & Core Principles

### Design Principles
1. **Simple (Rich Hickey)**: Independent, unentangled components in clear data pipelines
2. **Explainable**: Observable, inspectable steps from query to answer - no black boxes

### Data Flow
```
User Query → Embedding → Vector Search → Retrieve Top-K Docs → Augment Prompt → GPT-4o → Answer
```

### Key Components
- **Ingestion**: HuggingFace Wikipedia dataset → embeddings → local vector database
- **Retrieval**: Query embeddings → similarity search → relevant documents
- **Generation**: Augmented prompts → Azure GPT-4o → synthesized answers

## Model Access (CRITICAL)

**All model access MUST go through the model isolation layer in `src/llamaindex_models.py`.**

### Correct Usage
```python
# ✅ CORRECT: Use controlled model access
from llamaindex_models import get_gpt4o, get_text_embedding_3_large

llm = get_gpt4o(temperature=0.7)
embedding = get_text_embedding_3_large()
```

### Prohibited Usage
```python
# ❌ NEVER: Direct model instantiation bypasses security
from llama_index.llms.azure_openai import AzureOpenAI
llm = AzureOpenAI(model="gpt-4o", ...)  # This violates model isolation
```

### Available Models
- **Chat**: `gpt-4o` (GPT-4o via Azure OpenAI)
- **Embeddings**: `text-embedding-3-large` (Azure OpenAI)

## Project Structure

### Source Code
- `src/llamaindex_models.py` - **Model isolation layer (USE THIS FOR ALL MODEL ACCESS)**
- `src/ailab/utils/azure.py` - Azure authentication utilities
- `src/ailab/` - AI Lab utilities package

### Documentation & Examples
- `docs/architecture.md` - System architecture diagrams
- `docs/authentication.md` - Azure setup details
- `docs/model_isolation.md` - Model access patterns
- `docs/llamaindex_examples/` - Standalone usage examples
- `instructions.md` - Complete project brief
- `STATUS.md` - Implementation checklist

### Data Sources
- Wikipedia passages: `hf://datasets/rag-datasets/rag-mini-wikipedia/data/passages.parquet/part.0.parquet`
- Test questions: `hf://datasets/rag-datasets/rag-mini-wikipedia/data/test.parquet/part.0.parquet`

## Development Workflow

### Implementation Approach
1. **Test-Driven Development**: Create code → create tests → run tests → only proceed when tests pass
2. **FastAPI Endpoints**: Each major functionality needs corresponding API endpoints
3. **Jupyter Observability**: Create notebooks to inspect and demonstrate system behavior

### Key Implementation Tasks (from STATUS.md)
1. **Ingestion**: Load HuggingFace data → embeddings → vector database
2. **Query & Retrieval**: Query embeddings → top-K similarity search
3. **RAG Generation**: Augmented prompts → LLM → synthesized answers

### Authentication Issues
```bash
# If authentication fails:
azd auth logout
azd auth login --scope api://ailab/Model.Access
```

### Dependency Issues
```bash
# If imports fail:
uv sync --reinstall
```

## Model Configuration Details

- **API Version**: `2024-10-01-preview` (for both chat and embeddings)
- **Endpoint**: `https://ct-enterprisechat-api.azure-api.net/` (default, can be overridden with `AILAB_ENDPOINT`)
- **Authentication**: Azure DefaultAzureCredential with `api://ailab/Model.Access` scope

## Testing Strategy

- **Integration Tests**: Verify both internal behavior and API consistency
- **Test Organization**: Internal tests + Integration tests
- **Logs**: Available for debugging test failures
- See `docs/testing.md` for detailed testing guide

## Important Notes

- **Never bypass model isolation**: All LlamaIndex models must use `src/llamaindex_models.py`
- **Observability focus**: Every major operation should be observable via notebooks
- **Simple architecture**: Prefer clear data pipelines over complex interconnected objects
- **Authentication required**: Azure login is mandatory for model access
- **UV package manager**: Use `uv` commands, not `pip`

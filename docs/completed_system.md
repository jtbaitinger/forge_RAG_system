# RAG System - Complete

## Quick Start

```bash
# Authenticate
azd auth login --scope api://ailab/Model.Access

# Install dependencies
uv sync --group dev

# Start server
PYTHONPATH=. uvicorn src.main:app --reload
```

Then visit: http://localhost:8000/chat

## Project Structure

```
src/
  main.py                  # FastAPI application + chat UI
  data_loader.py           # HuggingFace Wikipedia data loading
  embedding_generator.py   # Embedding generation pipeline
  vector_store.py          # Vector database (create, save, load, search)
  llamaindex_models.py     # Model isolation layer (all Azure access)
  ailab/utils/azure.py     # Azure authentication utilities
  static/chat.html         # RAG chat web interface

tests/                     # Pytest test suite
scripts/                   # Manual test and demo scripts
notebooks/                 # Jupyter observability notebook
docs/                      # Full documentation
```

## Key Commands

```bash
# Start server
PYTHONPATH=. uvicorn src.main:app --reload

# Run tests
python scripts/run_tests.py -v

# Interactive demo
python scripts/demo_step_by_step.py

# Jupyter analysis
jupyter lab notebooks/rag_observability.ipynb
```

## Documentation

- [How It Works](how_it_works.md) — Full explanation of the RAG pipeline
- [Architecture](architecture.md) — System diagram
- [API Usage](api_usage.md) — All endpoints and the chat UI
- [Model Isolation](model_isolation.md) — Controlled Azure model access
- [Authentication](authentication.md) — Azure setup
- [Testing Guide](testing_guide.md) — Running and writing tests
- [Examples](examples.md) — Standalone example scripts
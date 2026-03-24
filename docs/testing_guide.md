# Testing Guide

## Test Structure

The test suite follows test-driven development principles with comprehensive coverage:

```
tests/
├── __init__.py
├── test_data_loader.py      # Unit tests for data loading
├── test_embedding_generator.py  # Unit tests for embeddings
├── test_vector_store.py     # Unit tests for vector storage
└── test_api.py             # Integration tests for FastAPI
```

## Running Tests

### Install Test Dependencies
```bash
uv sync --group dev
```

### Run All Tests
```bash
python scripts/run_tests.py
```

### Run Specific Test Categories
```bash
# Unit tests only (fast, minimal API calls)
python scripts/run_tests.py --category unit

# API tests only (slower, requires embeddings)
python scripts/run_tests.py --category api

# Integration tests (manual demo)
python scripts/run_tests.py --category integration
```

### Run Individual Test Files
```bash
# Data loading tests (no API calls)
pytest tests/test_data_loader.py -v

# Embedding tests (requires Azure auth)
pytest tests/test_embedding_generator.py -v

# Vector store tests (requires Azure auth)
pytest tests/test_vector_store.py -v

# API tests (full integration)
pytest tests/test_api.py -v
```

## Test Categories

### 🚀 Unit Tests (Fast)
- **Data Loading**: HuggingFace dataset loading, document conversion
- **Embedding Generation**: Model access, node creation, statistics
- **Vector Storage**: Index creation, similarity search, persistence

### 🌐 API Tests (Slower)
- **Health Endpoints**: Status checks, system info
- **Ingestion Endpoints**: Background processing, status tracking
- **Query Endpoints**: Similarity search, RAG generation
- **Validation**: Request/response validation

### 🔄 Integration Tests (Manual)
- **Complete Pipeline**: End-to-end data flow
- **Interactive Demo**: Step-by-step component testing

## Test Requirements

### Authentication
Tests requiring Azure OpenAI access need:
```bash
azd auth login --scope api://ailab/Model.Access
```

### Rate Limits
- Unit tests use `batch_size=1` to minimize API calls
- API tests include timeouts for embedding generation
- Tests automatically clean up storage after completion

## Test Data

Tests use minimal data sets:
- **Unit tests**: 2-3 sample documents
- **API tests**: 3 Wikipedia passages
- **All tests**: Automatic cleanup of test storage

## Troubleshooting

### Rate Limit Errors
If you hit rate limits:
```bash
# Run only data loading tests (no API calls)
pytest tests/test_data_loader.py

# Run with longer delays
pytest tests/ -v --tb=short
```

### Authentication Errors
Check Azure authentication:
```bash
azd auth token --output json | jq -r '.expiresOn'
```

### Storage Cleanup
Tests clean up automatically, but manual cleanup:
```bash
rm -rf ./vector_storage ./test_vector_storage
```

## Test Output Example

```
RAG System Test Runner
========================================

🧪 Running Unit Tests...
============================================================
RUNNING: Data Loader Tests
============================================================
✅ PASSED

============================================================
RUNNING: Embedding Generator Tests
============================================================
✅ PASSED

🎉 ALL TESTS PASSED!
```
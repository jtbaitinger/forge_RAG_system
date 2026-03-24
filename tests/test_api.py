"""Tests for FastAPI application."""

import pytest
import time
import shutil
from pathlib import Path
from fastapi.testclient import TestClient

from src.main import app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture(autouse=True)
def cleanup_test_storage():
    """Clean up test storage and reset app state before and after each test."""
    # Clean up before test
    for storage_path in ["./vector_storage", "./test_vector_storage"]:
        if Path(storage_path).exists():
            try:
                shutil.rmtree(storage_path)
            except Exception:
                pass

    # Reset app state before each test
    from src.main import app_state
    app_state["vector_store"] = None
    app_state["ingestion_status"] = "not_started"
    app_state["ingestion_progress"] = {}

    yield

    # Clean up after test
    for storage_path in ["./vector_storage", "./test_vector_storage"]:
        if Path(storage_path).exists():
            try:
                shutil.rmtree(storage_path)
            except Exception:
                pass

    # Reset app state after each test
    app_state["vector_store"] = None
    app_state["ingestion_status"] = "not_started"
    app_state["ingestion_progress"] = {}


class TestHealthEndpoints:
    """Test health and info endpoints."""

    def test_health_check(self, client):
        """Test health check endpoint."""
        response = client.get("/")

        assert response.status_code == 200
        data = response.json()

        assert "status" in data
        assert "message" in data
        assert "components" in data
        assert data["status"] == "healthy"

    def test_system_info(self, client):
        """Test system info endpoint."""
        response = client.get("/info")

        assert response.status_code == 200
        data = response.json()

        assert "ingestion_status" in data
        assert "vector_store_available" in data


class TestIngestionEndpoints:
    """Test data ingestion endpoints."""

    def test_ingestion_without_existing_store(self, client):
        """Test data ingestion when no store exists."""
        ingest_data = {
            "document_limit": 2,
            "chunk_size": 500,
            "batch_size": 1,
            "overwrite": True
        }

        response = client.post("/ingest", json=ingest_data)

        assert response.status_code == 200
        data = response.json()

        assert data["status"] == "started"
        assert "message" in data

    def test_ingestion_status(self, client):
        """Test ingestion status endpoint."""
        response = client.get("/ingest/status")

        assert response.status_code == 200
        data = response.json()

        assert "status" in data
        assert "progress" in data

    def test_ingestion_with_existing_store_no_overwrite(self, client):
        """Test ingestion fails when store exists and overwrite=False."""
        # First ingestion
        ingest_data = {
            "document_limit": 1,
            "batch_size": 1,
            "overwrite": True
        }
        client.post("/ingest", json=ingest_data)

        # Wait for completion
        for _ in range(30):
            status_response = client.get("/ingest/status")
            status = status_response.json()["status"]
            if status in ["completed", "failed"]:
                break
            time.sleep(2)

        # Try second ingestion without overwrite
        ingest_data["overwrite"] = False
        response = client.post("/ingest", json=ingest_data)

        assert response.status_code == 409  # Conflict

    def test_ingestion_in_progress_conflict(self, client):
        """Test that concurrent ingestion requests are rejected."""
        # Manually set up clean state for this test
        from src.main import app_state
        app_state["vector_store"] = None
        app_state["ingestion_status"] = "not_started"
        app_state["ingestion_progress"] = {}

        ingest_data = {
            "document_limit": 1,
            "batch_size": 1,
            "overwrite": True
        }

        # Start first ingestion
        response1 = client.post("/ingest", json=ingest_data)
        assert response1.status_code == 200

        # Verify the status was set
        assert app_state["ingestion_status"] == "in_progress"

        # Try second ingestion immediately - should conflict
        response2 = client.post("/ingest", json=ingest_data)
        assert response2.status_code == 409  # Should conflict


class TestQueryEndpoints:
    """Test query and RAG endpoints."""

    def setup_vector_store(self, client):
        """Helper to set up vector store for testing queries."""
        ingest_data = {
            "document_limit": 3,
            "chunk_size": 300,
            "batch_size": 1,
            "overwrite": True
        }

        # Start ingestion
        response = client.post("/ingest", json=ingest_data)
        assert response.status_code == 200

        # Wait for completion
        for _ in range(60):  # Increased timeout for embedding generation
            status_response = client.get("/ingest/status")
            status_data = status_response.json()

            if status_data["status"] == "completed":
                return True
            elif status_data["status"] == "failed":
                pytest.fail(f"Ingestion failed: {status_data.get('progress', {})}")

            time.sleep(3)

        pytest.fail("Ingestion timeout")

    def test_query_without_vector_store(self, client):
        """Test query endpoint without vector store."""
        # Explicitly ensure no vector store
        from src.main import app_state
        app_state["vector_store"] = None

        query_data = {
            "query": "test query",
            "top_k": 2
        }

        response = client.post("/query", json=query_data)
        assert response.status_code == 404

    def test_similarity_search(self, client):
        """Test similarity search endpoint."""
        # Set up vector store
        self.setup_vector_store(client)

        # Test query
        query_data = {
            "query": "Uruguay capital",
            "top_k": 2,
            "similarity_threshold": 0.0
        }

        response = client.post("/query", json=query_data)
        assert response.status_code == 200

        data = response.json()
        assert "query" in data
        assert "results" in data
        assert "total_results" in data
        assert data["query"] == "Uruguay capital"
        assert isinstance(data["results"], list)

        # Check result structure
        if data["results"]:
            result = data["results"][0]
            assert "rank" in result
            assert "text" in result
            assert "metadata" in result
            assert "similarity_score" in result
            assert "node_id" in result

    def test_similarity_search_with_threshold(self, client):
        """Test similarity search with threshold."""
        # Set up vector store
        self.setup_vector_store(client)

        # Test with high threshold
        query_data = {
            "query": "Uruguay",
            "top_k": 5,
            "similarity_threshold": 0.7
        }

        response = client.post("/query", json=query_data)
        assert response.status_code == 200

        data = response.json()
        assert data["total_results"] <= 5

        # All results should meet threshold
        for result in data["results"]:
            score = result["similarity_score"]
            if score is not None:
                assert score >= 0.7

    def test_rag_generation_without_vector_store(self, client):
        """Test RAG endpoint without vector store."""
        # Explicitly ensure no vector store
        from src.main import app_state
        app_state["vector_store"] = None

        rag_data = {
            "query": "What is Uruguay?",
            "top_k": 3
        }

        response = client.post("/rag", json=rag_data)
        assert response.status_code == 404

    def test_rag_generation(self, client):
        """Test RAG generation endpoint."""
        # Set up vector store
        self.setup_vector_store(client)

        # Test RAG query
        rag_data = {
            "query": "What is Uruguay?",
            "top_k": 2
        }

        response = client.post("/rag", json=rag_data)
        assert response.status_code == 200

        data = response.json()
        assert "query" in data
        assert "answer" in data
        assert "sources" in data
        assert data["query"] == "What is Uruguay?"
        assert len(data["answer"]) > 0
        assert isinstance(data["sources"], list)

        # Check source structure
        if data["sources"]:
            source = data["sources"][0]
            assert "rank" in source
            assert "text" in source
            assert "metadata" in source


class TestAPIValidation:
    """Test API request validation."""

    def test_invalid_ingest_request(self, client):
        """Test validation of ingest request."""
        # Reset app state to ensure clean test
        from src.main import app_state
        app_state["vector_store"] = None
        app_state["ingestion_status"] = "not_started"

        # Invalid batch_size (negative)
        ingest_data = {
            "document_limit": 5,
            "batch_size": -1
        }

        response = client.post("/ingest", json=ingest_data)
        assert response.status_code == 422  # Validation error

    def test_invalid_query_request(self, client):
        """Test validation of query request."""
        # Missing required query field
        query_data = {
            "top_k": 5
        }

        response = client.post("/query", json=query_data)
        assert response.status_code == 422  # Validation error

    def test_invalid_rag_request(self, client):
        """Test validation of RAG request."""
        # Missing required query field
        rag_data = {
            "top_k": 3
        }

        response = client.post("/rag", json=rag_data)
        assert response.status_code == 422  # Validation error
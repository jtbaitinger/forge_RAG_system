"""Tests for vector store functionality."""

import pytest
import shutil
from pathlib import Path
from llama_index.core import Document
from llama_index.core.schema import TextNode

from vector_store import RAGVectorStore, create_vector_store_from_nodes, load_vector_store
from embedding_generator import generate_document_embeddings


@pytest.fixture
def sample_documents():
    """Create sample documents for testing."""
    return [
        Document(
            text="Uruguay is a small country in South America.",
            metadata={"title": "Uruguay", "source": "test"}
        ),
        Document(
            text="Montevideo is the capital and largest city of Uruguay.",
            metadata={"title": "Montevideo", "source": "test"}
        )
    ]


@pytest.fixture
def embedded_nodes(sample_documents):
    """Create embedded nodes for testing."""
    nodes, _ = generate_document_embeddings(sample_documents, batch_size=1)
    return nodes


@pytest.fixture
def test_storage_dir():
    """Create a temporary storage directory for testing."""
    storage_dir = "./test_vector_storage"
    yield storage_dir

    # Clean up after test
    if Path(storage_dir).exists():
        shutil.rmtree(storage_dir)


class TestRAGVectorStore:
    """Test RAG vector store functionality."""

    def test_init(self, test_storage_dir):
        """Test RAGVectorStore initialization."""
        store = RAGVectorStore(
            storage_dir=test_storage_dir,
            index_name="test_index"
        )

        assert store.storage_dir == Path(test_storage_dir)
        assert store.index_name == "test_index"
        assert store._index is None
        assert store.storage_dir.exists()

    def test_create_index_from_nodes(self, embedded_nodes, test_storage_dir):
        """Test creating vector index from nodes."""
        store = RAGVectorStore(
            storage_dir=test_storage_dir,
            index_name="test_index"
        )

        index = store.create_index_from_nodes(embedded_nodes)

        assert index is not None
        assert store._index is not None
        assert store.stats is not None

    def test_create_index_from_documents(self, sample_documents, test_storage_dir):
        """Test creating vector index from documents."""
        store = RAGVectorStore(
            storage_dir=test_storage_dir,
            index_name="test_index"
        )

        index = store.create_index_from_documents(sample_documents)

        assert index is not None
        assert store._index is not None

    def test_save_and_load_index(self, embedded_nodes, test_storage_dir):
        """Test saving and loading vector index."""
        # Create and save index
        store = RAGVectorStore(
            storage_dir=test_storage_dir,
            index_name="test_index"
        )
        store.create_index_from_nodes(embedded_nodes)

        save_success = store.save_index(overwrite=True)
        assert save_success

        # Verify files were created
        index_path = Path(test_storage_dir) / "test_index"
        assert index_path.exists()

        # Load index in new store
        new_store = RAGVectorStore(
            storage_dir=test_storage_dir,
            index_name="test_index"
        )
        loaded_index = new_store.load_index()

        assert loaded_index is not None
        assert new_store._index is not None

    def test_query_similar_documents(self, embedded_nodes, test_storage_dir):
        """Test similarity search functionality."""
        store = RAGVectorStore(
            storage_dir=test_storage_dir,
            index_name="test_index"
        )
        store.create_index_from_nodes(embedded_nodes)

        # Test query
        results = store.query_similar_documents("Uruguay capital", top_k=2)

        assert isinstance(results, list)
        assert len(results) <= 2

        if results:
            result = results[0]
            assert 'rank' in result
            assert 'text' in result
            assert 'metadata' in result
            assert 'similarity_score' in result
            assert 'node_id' in result

    def test_query_with_threshold(self, embedded_nodes, test_storage_dir):
        """Test similarity search with threshold."""
        store = RAGVectorStore(
            storage_dir=test_storage_dir,
            index_name="test_index"
        )
        store.create_index_from_nodes(embedded_nodes)

        # Test with high threshold (should return fewer results)
        results_high = store.query_similar_documents(
            "Uruguay", top_k=5, similarity_threshold=0.8
        )
        results_low = store.query_similar_documents(
            "Uruguay", top_k=5, similarity_threshold=0.0
        )

        assert len(results_high) <= len(results_low)

    def test_get_query_engine(self, embedded_nodes, test_storage_dir):
        """Test query engine creation."""
        store = RAGVectorStore(
            storage_dir=test_storage_dir,
            index_name="test_index"
        )
        store.create_index_from_nodes(embedded_nodes)

        query_engine = store.get_query_engine(similarity_top_k=2)

        assert query_engine is not None

        # Test actual query
        response = query_engine.query("What is Uruguay?")
        assert response is not None
        assert len(str(response)) > 0

    def test_get_index_info(self, embedded_nodes, test_storage_dir):
        """Test index information retrieval."""
        store = RAGVectorStore(
            storage_dir=test_storage_dir,
            index_name="test_index"
        )

        # Before creating index
        info = store.get_index_info()
        assert info["status"] == "no_index_loaded"

        # After creating index
        store.create_index_from_nodes(embedded_nodes)
        info = store.get_index_info()
        assert info["status"] == "loaded"
        assert "storage_path" in info


class TestConvenienceFunctions:
    """Test convenience functions for vector store."""

    def test_create_vector_store_from_nodes(self, embedded_nodes, test_storage_dir):
        """Test convenience function for creating vector store."""
        store = create_vector_store_from_nodes(
            embedded_nodes,
            storage_dir=test_storage_dir,
            index_name="convenience_test",
            save=True
        )

        assert store is not None
        assert store._index is not None

        # Verify it was saved
        index_path = Path(test_storage_dir) / "convenience_test"
        assert index_path.exists()

    def test_load_vector_store(self, embedded_nodes, test_storage_dir):
        """Test convenience function for loading vector store."""
        # First create and save a store
        create_vector_store_from_nodes(
            embedded_nodes,
            storage_dir=test_storage_dir,
            index_name="load_test",
            save=True
        )

        # Then load it
        loaded_store = load_vector_store(
            storage_dir=test_storage_dir,
            index_name="load_test"
        )

        assert loaded_store is not None
        assert loaded_store._index is not None

        # Test that it works
        results = loaded_store.query_similar_documents("Uruguay", top_k=1)
        assert isinstance(results, list)
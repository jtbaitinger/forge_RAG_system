"""Tests for embedding generation functionality."""

import pytest
from llama_index.core import Document
from llama_index.core.schema import TextNode

from embedding_generator import EmbeddingGenerator, generate_document_embeddings, generate_query_embedding


@pytest.fixture
def sample_documents():
    """Create sample documents for testing."""
    return [
        Document(
            text="Uruguay is a country in South America.",
            metadata={"title": "Uruguay", "source": "test"}
        ),
        Document(
            text="Montevideo is the capital of Uruguay.",
            metadata={"title": "Montevideo", "source": "test"}
        )
    ]


class TestEmbeddingGenerator:
    """Test embedding generation functionality."""

    def test_init(self):
        """Test EmbeddingGenerator initialization."""
        generator = EmbeddingGenerator(chunk_size=500, chunk_overlap=100)

        assert generator.chunk_size == 500
        assert generator.chunk_overlap == 100
        assert generator._embedding_model is None
        assert generator._node_parser is None

    def test_documents_to_nodes(self, sample_documents):
        """Test document to node conversion."""
        generator = EmbeddingGenerator(chunk_size=100, chunk_overlap=20)

        nodes = generator.documents_to_nodes(sample_documents)

        assert isinstance(nodes, list)
        assert len(nodes) >= len(sample_documents)  # Could be more due to chunking
        assert all(isinstance(node, TextNode) for node in nodes)

        # Check first node has expected attributes
        first_node = nodes[0]
        assert hasattr(first_node, 'text')
        assert hasattr(first_node, 'metadata')

    def test_generate_embeddings_for_nodes(self, sample_documents):
        """Test embedding generation for nodes."""
        generator = EmbeddingGenerator()

        # Convert to nodes first
        nodes = generator.documents_to_nodes(sample_documents)

        # Generate embeddings (use small batch size)
        embedded_nodes = generator.generate_embeddings_for_nodes(nodes, batch_size=1)

        assert len(embedded_nodes) == len(nodes)

        # Check that at least some embeddings were generated
        successful_embeddings = sum(1 for node in embedded_nodes
                                  if hasattr(node, 'embedding') and node.embedding)
        assert successful_embeddings > 0

        # Check embedding properties
        for node in embedded_nodes:
            if hasattr(node, 'embedding') and node.embedding:
                assert isinstance(node.embedding, list)
                assert len(node.embedding) == 3072  # text-embedding-3-large dimension
                assert all(isinstance(val, float) for val in node.embedding)

    def test_generate_embeddings_for_documents(self, sample_documents):
        """Test end-to-end document embedding generation."""
        generator = EmbeddingGenerator()

        embedded_nodes, stats = generator.generate_embeddings_for_documents(
            sample_documents, batch_size=1
        )

        assert isinstance(embedded_nodes, list)
        assert stats is not None
        assert hasattr(stats, 'total_documents')
        assert hasattr(stats, 'total_nodes')
        assert stats.total_documents == len(sample_documents)

    def test_generate_query_embedding(self):
        """Test query embedding generation."""
        generator = EmbeddingGenerator()

        query = "What is the capital of Uruguay?"
        embedding = generator.generate_query_embedding(query)

        assert isinstance(embedding, list)
        assert len(embedding) == 3072
        assert all(isinstance(val, float) for val in embedding)

    def test_get_embedding_stats(self, sample_documents):
        """Test embedding statistics retrieval."""
        generator = EmbeddingGenerator()

        # Initially no stats
        assert generator.get_embedding_stats() is None

        # Generate embeddings
        generator.generate_embeddings_for_documents(sample_documents, batch_size=1)

        # Now should have stats
        stats = generator.get_embedding_stats()
        assert stats is not None
        assert stats.total_documents == len(sample_documents)


class TestConvenienceFunctions:
    """Test convenience functions for embedding generation."""

    def test_generate_document_embeddings(self, sample_documents):
        """Test convenience function for document embeddings."""
        embedded_nodes, stats = generate_document_embeddings(
            sample_documents,
            chunk_size=200,
            batch_size=1
        )

        assert isinstance(embedded_nodes, list)
        assert stats is not None
        assert len(embedded_nodes) >= len(sample_documents)

    def test_generate_query_embedding(self):
        """Test convenience function for query embedding."""
        query = "Test query about Uruguay"
        embedding = generate_query_embedding(query)

        assert isinstance(embedding, list)
        assert len(embedding) == 3072
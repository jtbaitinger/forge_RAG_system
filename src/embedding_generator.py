"""Embedding generation module for RAG system.

This module provides functions to generate embeddings from Wikipedia passages using
the controlled Azure OpenAI text-embedding-3-large model. It handles batch processing,
error handling, and efficient processing of large document sets.
"""

import logging
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import pandas as pd
from llama_index.core import Document
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.schema import TextNode

# Import our controlled model access
from llamaindex_models import get_text_embedding_3_large

# Set up logging
logger = logging.getLogger(__name__)


@dataclass
class EmbeddingStats:
    """Statistics about embedding generation process."""
    total_documents: int
    total_nodes: int
    total_tokens_processed: int
    processing_time_seconds: float
    embeddings_per_second: float
    average_embedding_dimension: int
    failed_embeddings: int


class EmbeddingGenerator:
    """Handles embedding generation for documents using controlled Azure OpenAI models."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """Initialize the embedding generator.

        Args:
            chunk_size: Maximum size of text chunks for embedding.
            chunk_overlap: Overlap between consecutive chunks.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._embedding_model = None
        self._node_parser = None
        self.stats: Optional[EmbeddingStats] = None

    def _get_embedding_model(self):
        """Get the controlled embedding model, creating it if needed."""
        if self._embedding_model is None:
            logger.info("Creating controlled text-embedding-3-large model")
            self._embedding_model = get_text_embedding_3_large()
            logger.info(f"Embedding model created: {type(self._embedding_model).__name__}")
        return self._embedding_model

    def _get_node_parser(self):
        """Get the node parser, creating it if needed."""
        if self._node_parser is None:
            self._node_parser = SimpleNodeParser.from_defaults(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )
            logger.info(f"Node parser created with chunk_size={self.chunk_size}, overlap={self.chunk_overlap}")
        return self._node_parser

    def documents_to_nodes(self, documents: List[Document]) -> List[TextNode]:
        """Convert documents to text nodes for embedding.

        Args:
            documents: List of LlamaIndex Document objects.

        Returns:
            List of TextNode objects ready for embedding.
        """
        logger.info(f"Converting {len(documents)} documents to nodes")

        parser = self._get_node_parser()
        nodes = parser.get_nodes_from_documents(documents)

        logger.info(f"Created {len(nodes)} text nodes from {len(documents)} documents")
        return nodes

    def generate_embeddings_for_nodes(self,
                                    nodes: List[TextNode],
                                    batch_size: int = 10) -> List[TextNode]:
        """Generate embeddings for text nodes.

        Args:
            nodes: List of TextNode objects to embed.
            batch_size: Number of nodes to process in each batch.

        Returns:
            List of TextNode objects with embeddings added.
        """
        logger.info(f"Generating embeddings for {len(nodes)} nodes (batch_size={batch_size})")

        start_time = time.time()
        embedding_model = self._get_embedding_model()

        embedded_nodes = []
        failed_count = 0
        total_tokens = 0

        for i in range(0, len(nodes), batch_size):
            batch = nodes[i:i + batch_size]
            batch_texts = [node.text for node in batch]

            logger.debug(f"Processing batch {i//batch_size + 1}/{(len(nodes) + batch_size - 1)//batch_size}")

            try:
                # Generate embeddings for this batch
                batch_embeddings = []
                for text in batch_texts:
                    try:
                        embedding = embedding_model.get_text_embedding(text)
                        batch_embeddings.append(embedding)
                        total_tokens += len(text.split())  # Rough token count
                    except Exception as e:
                        logger.warning(f"Failed to embed text (length={len(text)}): {e}")
                        batch_embeddings.append(None)
                        failed_count += 1

                # Add embeddings to nodes
                for node, embedding in zip(batch, batch_embeddings):
                    if embedding is not None:
                        node.embedding = embedding
                    embedded_nodes.append(node)

            except Exception as e:
                logger.error(f"Batch embedding failed: {e}")
                # Add nodes without embeddings
                embedded_nodes.extend(batch)
                failed_count += len(batch)

        end_time = time.time()
        processing_time = end_time - start_time

        # Calculate statistics
        successful_embeddings = len(nodes) - failed_count
        embeddings_per_second = successful_embeddings / processing_time if processing_time > 0 else 0

        # Get embedding dimension from first successful embedding
        avg_dimension = 0
        for node in embedded_nodes:
            if hasattr(node, 'embedding') and node.embedding:
                avg_dimension = len(node.embedding)
                break

        self.stats = EmbeddingStats(
            total_documents=0,  # Will be set by caller
            total_nodes=len(nodes),
            total_tokens_processed=total_tokens,
            processing_time_seconds=processing_time,
            embeddings_per_second=embeddings_per_second,
            average_embedding_dimension=avg_dimension,
            failed_embeddings=failed_count
        )

        logger.info(f"Embedding generation complete: {successful_embeddings}/{len(nodes)} successful")
        logger.info(f"Processing time: {processing_time:.2f}s ({embeddings_per_second:.1f} embeddings/sec)")

        return embedded_nodes

    def generate_embeddings_for_documents(self,
                                        documents: List[Document],
                                        batch_size: int = 10) -> Tuple[List[TextNode], EmbeddingStats]:
        """Generate embeddings for a list of documents.

        Args:
            documents: List of LlamaIndex Document objects.
            batch_size: Number of nodes to process in each batch.

        Returns:
            Tuple of (embedded_nodes, statistics).
        """
        logger.info(f"Starting embedding generation for {len(documents)} documents")

        # Convert documents to nodes
        nodes = self.documents_to_nodes(documents)

        # Generate embeddings
        embedded_nodes = self.generate_embeddings_for_nodes(nodes, batch_size=batch_size)

        # Update stats with document count
        if self.stats:
            self.stats.total_documents = len(documents)

        logger.info("Document embedding generation complete")
        return embedded_nodes, self.stats

    def generate_query_embedding(self, query: str) -> List[float]:
        """Generate embedding for a single query string.

        Args:
            query: Query string to embed.

        Returns:
            Embedding vector as list of floats.
        """
        logger.debug(f"Generating embedding for query: '{query[:50]}...'")

        embedding_model = self._get_embedding_model()

        try:
            embedding = embedding_model.get_text_embedding(query)
            logger.debug(f"Query embedding generated (dimension: {len(embedding)})")
            return embedding
        except Exception as e:
            logger.error(f"Failed to generate query embedding: {e}")
            raise

    def get_embedding_stats(self) -> Optional[EmbeddingStats]:
        """Get the latest embedding generation statistics.

        Returns:
            EmbeddingStats object or None if no embeddings generated yet.
        """
        return self.stats


# Convenience functions for simple usage
def generate_document_embeddings(documents: List[Document],
                               chunk_size: int = 1000,
                               chunk_overlap: int = 200,
                               batch_size: int = 10) -> Tuple[List[TextNode], EmbeddingStats]:
    """Generate embeddings for documents with default settings.

    Args:
        documents: List of LlamaIndex Document objects.
        chunk_size: Maximum size of text chunks.
        chunk_overlap: Overlap between chunks.
        batch_size: Batch size for processing.

    Returns:
        Tuple of (embedded_nodes, statistics).
    """
    generator = EmbeddingGenerator(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return generator.generate_embeddings_for_documents(documents, batch_size=batch_size)


def generate_query_embedding(query: str) -> List[float]:
    """Generate embedding for a query with default settings.

    Args:
        query: Query string to embed.

    Returns:
        Embedding vector as list of floats.
    """
    generator = EmbeddingGenerator()
    return generator.generate_query_embedding(query)
"""Vector store module for RAG system.

This module provides functionality to create, manage, and query a local vector database
using LlamaIndex. It handles storage of embeddings, metadata management, and similarity
search operations for the RAG pipeline.
"""

import logging
import json
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict

from llama_index.core import VectorStoreIndex, Document, StorageContext, load_index_from_storage
from llama_index.core.schema import TextNode
from llama_index.core.vector_stores import SimpleVectorStore

# Import our controlled model access
from llamaindex_models import get_text_embedding_3_large, get_gpt4o

# Set up logging
logger = logging.getLogger(__name__)


@dataclass
class VectorStoreStats:
    """Statistics about the vector store."""
    total_documents: int
    total_nodes: int
    index_size_mb: float
    embedding_dimension: int
    storage_path: str
    created_timestamp: str
    last_updated_timestamp: str


class RAGVectorStore:
    """Manages vector storage and retrieval for the RAG system."""

    def __init__(self,
                 storage_dir: str = "./vector_storage",
                 index_name: str = "wikipedia_rag_index"):
        """Initialize the vector store.

        Args:
            storage_dir: Directory to store the vector index and metadata.
            index_name: Name of the vector index.
        """
        self.storage_dir = Path(storage_dir)
        self.index_name = index_name
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        self._index: Optional[VectorStoreIndex] = None
        self._embedding_model = None
        self._llm = None
        self.stats: Optional[VectorStoreStats] = None

    def _get_embedding_model(self):
        """Get the controlled embedding model."""
        if self._embedding_model is None:
            logger.info("Creating controlled text-embedding-3-large model for vector store")
            self._embedding_model = get_text_embedding_3_large()
        return self._embedding_model

    def _get_llm(self):
        """Get the controlled LLM."""
        if self._llm is None:
            logger.info("Creating controlled GPT-4o model for vector store")
            self._llm = get_gpt4o(temperature=0.1)
        return self._llm

    def create_index_from_nodes(self, nodes: List[TextNode]) -> VectorStoreIndex:
        """Create a new vector index from text nodes.

        Args:
            nodes: List of TextNode objects with embeddings.

        Returns:
            Created VectorStoreIndex.
        """
        logger.info(f"Creating vector index from {len(nodes)} nodes")

        # Get controlled models
        embedding_model = self._get_embedding_model()
        llm = self._get_llm()

        # Create the index with controlled models
        self._index = VectorStoreIndex(
            nodes=nodes,
            embed_model=embedding_model,
            llm=llm
        )

        logger.info(f"Vector index created successfully with {len(nodes)} nodes")

        # Calculate and store statistics
        self._update_stats(nodes)

        return self._index

    def create_index_from_documents(self, documents: List[Document]) -> VectorStoreIndex:
        """Create a new vector index from documents (will generate embeddings).

        Args:
            documents: List of Document objects.

        Returns:
            Created VectorStoreIndex.
        """
        logger.info(f"Creating vector index from {len(documents)} documents")

        # Get controlled models
        embedding_model = self._get_embedding_model()
        llm = self._get_llm()

        # Create the index (this will automatically generate embeddings)
        self._index = VectorStoreIndex.from_documents(
            documents=documents,
            embed_model=embedding_model,
            llm=llm
        )

        logger.info(f"Vector index created successfully from {len(documents)} documents")

        # Calculate and store statistics
        self._update_stats_from_index()

        return self._index

    def save_index(self, overwrite: bool = False) -> bool:
        """Save the vector index to persistent storage.

        Args:
            overwrite: Whether to overwrite existing index.

        Returns:
            True if saved successfully, False otherwise.
        """
        if self._index is None:
            logger.error("No index to save. Create an index first.")
            return False

        index_path = self.storage_dir / self.index_name

        if index_path.exists() and not overwrite:
            logger.warning(f"Index already exists at {index_path}. Use overwrite=True to replace.")
            return False

        try:
            logger.info(f"Saving vector index to {index_path}")

            # Save the index
            self._index.storage_context.persist(persist_dir=str(index_path))

            # Save metadata and statistics
            self._save_metadata()

            logger.info("Vector index saved successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to save vector index: {e}")
            return False

    def load_index(self) -> Optional[VectorStoreIndex]:
        """Load a vector index from persistent storage.

        Returns:
            Loaded VectorStoreIndex or None if loading fails.
        """
        index_path = self.storage_dir / self.index_name

        if not index_path.exists():
            logger.warning(f"No saved index found at {index_path}")
            return None

        try:
            logger.info(f"Loading vector index from {index_path}")

            # Get controlled models for the loaded index
            embedding_model = self._get_embedding_model()
            llm = self._get_llm()

            # Load the storage context
            storage_context = StorageContext.from_defaults(persist_dir=str(index_path))

            # Load the index with controlled models
            self._index = load_index_from_storage(
                storage_context,
                embed_model=embedding_model,
                llm=llm
            )

            # Load metadata and statistics
            self._load_metadata()

            logger.info("Vector index loaded successfully")
            return self._index

        except Exception as e:
            logger.error(f"Failed to load vector index: {e}")
            return None

    def query_similar_documents(self,
                              query: str,
                              top_k: int = 5,
                              similarity_threshold: float = 0.0) -> List[Dict[str, Any]]:
        """Query the vector store for similar documents.

        Args:
            query: Query string.
            top_k: Number of most similar documents to return.
            similarity_threshold: Minimum similarity score threshold.

        Returns:
            List of dictionaries containing document information and similarity scores.
        """
        if self._index is None:
            logger.error("No index available. Create or load an index first.")
            return []

        logger.info(f"Querying vector store: '{query[:50]}...' (top_k={top_k})")

        try:
            # Create a retriever from the index
            retriever = self._index.as_retriever(similarity_top_k=top_k)

            # Perform the query
            retrieved_nodes = retriever.retrieve(query)

            # Format results
            results = []
            for i, node in enumerate(retrieved_nodes):
                score = getattr(node, 'score', None)

                # Apply similarity threshold
                if score is not None and score < similarity_threshold:
                    continue

                result = {
                    'rank': i + 1,
                    'text': node.text,
                    'metadata': node.metadata,
                    'similarity_score': score,
                    'node_id': node.node_id
                }
                results.append(result)

            logger.info(f"Retrieved {len(results)} documents above threshold {similarity_threshold}")
            return results

        except Exception as e:
            logger.error(f"Failed to query vector store: {e}")
            return []

    def get_query_engine(self, **kwargs):
        """Get a query engine for RAG generation.

        Args:
            **kwargs: Additional arguments for query engine configuration.

        Returns:
            Configured query engine.
        """
        if self._index is None:
            logger.error("No index available. Create or load an index first.")
            return None

        logger.info("Creating query engine for RAG generation")

        # Set default parameters
        default_params = {
            'similarity_top_k': 5,
            'response_mode': 'compact'
        }
        default_params.update(kwargs)

        return self._index.as_query_engine(**default_params)

    def get_index_info(self) -> Dict[str, Any]:
        """Get information about the current index.

        Returns:
            Dictionary with index information.
        """
        if self._index is None:
            return {"status": "no_index_loaded"}

        try:
            # Get basic index information
            info = {
                "status": "loaded",
                "storage_path": str(self.storage_dir / self.index_name),
                "has_embeddings": True,
            }

            # Add statistics if available
            if self.stats:
                info.update(asdict(self.stats))

            return info

        except Exception as e:
            logger.error(f"Failed to get index info: {e}")
            return {"status": "error", "error": str(e)}

    def _update_stats(self, nodes: List[TextNode]):
        """Update statistics based on nodes."""
        import datetime

        # Calculate basic stats
        total_nodes = len(nodes)
        embedding_dim = 0

        # Get embedding dimension from first node with embedding
        for node in nodes:
            if hasattr(node, 'embedding') and node.embedding:
                embedding_dim = len(node.embedding)
                break

        # Calculate storage size (approximate)
        storage_size_mb = 0.0
        index_path = self.storage_dir / self.index_name
        if index_path.exists():
            storage_size_mb = sum(f.stat().st_size for f in index_path.rglob('*') if f.is_file()) / (1024 * 1024)

        timestamp = datetime.datetime.now().isoformat()

        self.stats = VectorStoreStats(
            total_documents=0,  # Will be updated when available
            total_nodes=total_nodes,
            index_size_mb=storage_size_mb,
            embedding_dimension=embedding_dim,
            storage_path=str(self.storage_dir / self.index_name),
            created_timestamp=timestamp,
            last_updated_timestamp=timestamp
        )

    def _update_stats_from_index(self):
        """Update statistics from the loaded index."""
        if self._index is None:
            return

        # Create basic stats
        import datetime
        timestamp = datetime.datetime.now().isoformat()

        # Calculate storage size
        storage_size_mb = 0.0
        index_path = self.storage_dir / self.index_name
        if index_path.exists():
            storage_size_mb = sum(f.stat().st_size for f in index_path.rglob('*') if f.is_file()) / (1024 * 1024)

        self.stats = VectorStoreStats(
            total_documents=0,  # Not easily accessible from index
            total_nodes=0,      # Not easily accessible from index
            index_size_mb=storage_size_mb,
            embedding_dimension=3072,  # Known for text-embedding-3-large
            storage_path=str(self.storage_dir / self.index_name),
            created_timestamp=timestamp,
            last_updated_timestamp=timestamp
        )

    def _save_metadata(self):
        """Save metadata and statistics."""
        if self.stats is None:
            return

        metadata_file = self.storage_dir / f"{self.index_name}_metadata.json"

        try:
            with open(metadata_file, 'w') as f:
                json.dump(asdict(self.stats), f, indent=2)
            logger.info(f"Metadata saved to {metadata_file}")
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")

    def _load_metadata(self):
        """Load metadata and statistics."""
        metadata_file = self.storage_dir / f"{self.index_name}_metadata.json"

        if not metadata_file.exists():
            logger.info("No metadata file found")
            return

        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)

            # Update timestamp
            import datetime
            metadata['last_updated_timestamp'] = datetime.datetime.now().isoformat()

            self.stats = VectorStoreStats(**metadata)
            logger.info(f"Metadata loaded from {metadata_file}")
        except Exception as e:
            logger.error(f"Failed to load metadata: {e}")


# Convenience functions
def create_vector_store_from_nodes(nodes: List[TextNode],
                                 storage_dir: str = "./vector_storage",
                                 index_name: str = "wikipedia_rag_index",
                                 save: bool = True) -> RAGVectorStore:
    """Create and optionally save a vector store from nodes.

    Args:
        nodes: List of TextNode objects with embeddings.
        storage_dir: Directory to store the index.
        index_name: Name for the index.
        save: Whether to save the index to disk.

    Returns:
        RAGVectorStore instance.
    """
    store = RAGVectorStore(storage_dir=storage_dir, index_name=index_name)
    store.create_index_from_nodes(nodes)

    if save:
        store.save_index(overwrite=True)

    return store


def load_vector_store(storage_dir: str = "./vector_storage",
                     index_name: str = "wikipedia_rag_index") -> Optional[RAGVectorStore]:
    """Load an existing vector store.

    Args:
        storage_dir: Directory containing the stored index.
        index_name: Name of the index to load.

    Returns:
        RAGVectorStore instance or None if loading fails.
    """
    store = RAGVectorStore(storage_dir=storage_dir, index_name=index_name)

    if store.load_index():
        return store

    return None
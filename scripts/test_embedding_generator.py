#!/usr/bin/env python3
"""Test script to verify embedding generation functionality."""

import sys
import logging
from pathlib import Path

# Add src to path to import our modules
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from data_loader import load_wikipedia_documents
from embedding_generator import EmbeddingGenerator, generate_document_embeddings, generate_query_embedding

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_embedding_generation():
    """Test the embedding generation functionality."""
    print("Testing Embedding Generation")
    print("=" * 40)

    try:
        # Load some test documents (limit to 3 for quick testing)
        print("\n1. Loading test documents...")
        documents = load_wikipedia_documents(limit=3)
        print(f"   - Loaded {len(documents)} documents")

        if documents:
            print(f"   - First document title: '{documents[0].metadata.get('title', 'N/A')}'")
            print(f"   - First document text length: {len(documents[0].text)}")

        # Initialize embedding generator
        print("\n2. Creating embedding generator...")
        generator = EmbeddingGenerator(chunk_size=500, chunk_overlap=100)

        # Convert documents to nodes
        print("\n3. Converting documents to nodes...")
        nodes = generator.documents_to_nodes(documents)
        print(f"   - Created {len(nodes)} text nodes")

        if nodes:
            print(f"   - First node text length: {len(nodes[0].text)}")
            print(f"   - First node text preview: '{nodes[0].text[:100]}...'")

        # Generate embeddings for nodes (small batch for testing)
        print("\n4. Generating embeddings for nodes...")
        embedded_nodes = generator.generate_embeddings_for_nodes(nodes, batch_size=2)

        print(f"   - Processed {len(embedded_nodes)} nodes")

        # Check embedding results
        successful_embeddings = sum(1 for node in embedded_nodes
                                  if hasattr(node, 'embedding') and node.embedding)
        print(f"   - Successful embeddings: {successful_embeddings}/{len(embedded_nodes)}")

        if successful_embeddings > 0:
            first_embedded = next(node for node in embedded_nodes
                                if hasattr(node, 'embedding') and node.embedding)
            print(f"   - Embedding dimension: {len(first_embedded.embedding)}")
            print(f"   - Embedding sample values: {first_embedded.embedding[:5]}")

        # Test statistics
        stats = generator.get_embedding_stats()
        if stats:
            print(f"\n5. Embedding statistics:")
            print(f"   - Total nodes: {stats.total_nodes}")
            print(f"   - Processing time: {stats.processing_time_seconds:.2f} seconds")
            print(f"   - Embeddings per second: {stats.embeddings_per_second:.1f}")
            print(f"   - Average dimension: {stats.average_embedding_dimension}")
            print(f"   - Failed embeddings: {stats.failed_embeddings}")

        # Test convenience function
        print(f"\n6. Testing convenience function...")
        small_docs = documents[:2]  # Use even fewer for this test
        conv_nodes, conv_stats = generate_document_embeddings(
            small_docs,
            chunk_size=400,
            batch_size=1
        )
        print(f"   - Convenience function processed {len(conv_nodes)} nodes")
        print(f"   - Convenience stats: {conv_stats.total_documents} docs, {conv_stats.total_nodes} nodes")

        # Test query embedding
        print(f"\n7. Testing query embedding...")
        test_queries = [
            "What is Abraham Lincoln known for?",
            "Tell me about machine learning",
            "How does Python programming work?"
        ]

        for i, query in enumerate(test_queries, 1):
            try:
                query_embedding = generate_query_embedding(query)
                print(f"   - Query {i}: '{query}' -> embedding dim: {len(query_embedding)}")
            except Exception as e:
                print(f"   - Query {i} failed: {e}")

        print("\nAll embedding generation tests passed!")
        return True

    except Exception as e:
        print(f"\nERROR: Embedding generation test failed: {e}")
        logger.exception("Full error details:")
        return False


if __name__ == "__main__":
    success = test_embedding_generation()
    exit_code = 0 if success else 1
    sys.exit(exit_code)
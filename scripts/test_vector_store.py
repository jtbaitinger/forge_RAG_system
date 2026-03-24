#!/usr/bin/env python3
"""Test script to verify vector store functionality."""

import sys
import shutil
from pathlib import Path

# Add src to path to import our modules
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from data_loader import load_wikipedia_documents
from embedding_generator import generate_document_embeddings
from vector_store import RAGVectorStore, create_vector_store_from_nodes, load_vector_store


def test_vector_store():
    """Test the vector store functionality."""
    print("Testing Vector Store")
    print("=" * 30)

    # Test storage directory
    test_storage_dir = "./test_vector_storage"
    test_index_name = "test_wikipedia_index"

    try:
        # Clean up any existing test data
        test_path = Path(test_storage_dir)
        if test_path.exists():
            shutil.rmtree(test_path)
            print("   - Cleaned up existing test data")

        # Load test documents (small set for testing)
        print("\n1. Loading test documents...")
        documents = load_wikipedia_documents(limit=5)
        print(f"   - Loaded {len(documents)} documents")

        # Generate embeddings
        print("\n2. Generating embeddings...")
        embedded_nodes, embed_stats = generate_document_embeddings(
            documents,
            chunk_size=300,
            batch_size=2
        )
        print(f"   - Generated embeddings for {len(embedded_nodes)} nodes")
        print(f"   - Embedding dimension: {embed_stats.average_embedding_dimension}")

        # Create vector store
        print("\n3. Creating vector store...")
        vector_store = RAGVectorStore(
            storage_dir=test_storage_dir,
            index_name=test_index_name
        )

        # Create index from nodes
        print("\n4. Building vector index...")
        index = vector_store.create_index_from_nodes(embedded_nodes)
        print(f"   - Vector index created successfully")

        # Get index info
        info = vector_store.get_index_info()
        print(f"   - Index status: {info.get('status')}")
        print(f"   - Storage path: {info.get('storage_path')}")

        # Test similarity search
        print("\n5. Testing similarity search...")
        test_queries = [
            "Tell me about Uruguay",
            "What is the capital city?",
            "Information about South America"
        ]

        for i, query in enumerate(test_queries, 1):
            print(f"\n   Query {i}: '{query}'")
            results = vector_store.query_similar_documents(query, top_k=2)

            if results:
                for j, result in enumerate(results):
                    score = result.get('similarity_score', 'N/A')
                    text_preview = result['text'][:80] + "..."
                    print(f"     Result {j+1}: Score={score:.3f}" if isinstance(score, float) else f"     Result {j+1}: Score={score}")
                    print(f"                Text: {text_preview}")
            else:
                print(f"     No results found")

        # Test saving index
        print("\n6. Saving vector index...")
        save_success = vector_store.save_index(overwrite=True)
        print(f"   - Save successful: {save_success}")

        if save_success:
            # Verify saved files exist
            index_path = Path(test_storage_dir) / test_index_name
            print(f"   - Index directory exists: {index_path.exists()}")

            if index_path.exists():
                files = list(index_path.rglob('*'))
                print(f"   - Files saved: {len(files)}")

        # Test loading index
        print("\n7. Testing index loading...")
        new_vector_store = RAGVectorStore(
            storage_dir=test_storage_dir,
            index_name=test_index_name
        )

        loaded_index = new_vector_store.load_index()
        print(f"   - Index loaded successfully: {loaded_index is not None}")

        if loaded_index:
            # Test query on loaded index
            print("\n8. Testing query on loaded index...")
            query = "Uruguay capital"
            results = new_vector_store.query_similar_documents(query, top_k=1)

            if results:
                result = results[0]
                score = result.get('similarity_score', 'N/A')
                print(f"   - Query successful: Score={score}")
                print(f"   - Text preview: {result['text'][:100]}...")
            else:
                print("   - No results from loaded index")

        # Test convenience functions
        print("\n9. Testing convenience functions...")

        # Clean up for convenience function test
        if Path(test_storage_dir).exists():
            shutil.rmtree(test_storage_dir)

        # Test create_vector_store_from_nodes
        conv_store = create_vector_store_from_nodes(
            embedded_nodes[:3],  # Use fewer nodes
            storage_dir=test_storage_dir,
            index_name=test_index_name,
            save=True
        )
        print(f"   - create_vector_store_from_nodes: OK")

        # Test load_vector_store
        loaded_store = load_vector_store(
            storage_dir=test_storage_dir,
            index_name=test_index_name
        )
        print(f"   - load_vector_store: {loaded_store is not None}")

        # Test query engine
        print("\n10. Testing query engine...")
        if loaded_store:
            query_engine = loaded_store.get_query_engine(similarity_top_k=2)
            print(f"   - Query engine created: {query_engine is not None}")

            if query_engine:
                try:
                    # Test a simple query
                    response = query_engine.query("What is Uruguay?")
                    print(f"   - Query response: {str(response)[:100]}...")
                except Exception as e:
                    print(f"   - Query failed: {e}")

        print("\nAll vector store tests passed!")
        return True

    except Exception as e:
        print(f"\nERROR: Vector store test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        # Clean up test data
        try:
            test_path = Path(test_storage_dir)
            if test_path.exists():
                shutil.rmtree(test_path)
                print("\n   - Cleaned up test data")
        except Exception as e:
            print(f"\n   - Cleanup failed: {e}")


if __name__ == "__main__":
    success = test_vector_store()
    exit_code = 0 if success else 1
    sys.exit(exit_code)
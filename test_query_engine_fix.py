#!/usr/bin/env python3
"""Quick test to verify the query engine fix."""

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

def test_query_engine_fix():
    """Test that query engine works with controlled models."""
    print("Testing Query Engine Fix")
    print("=" * 30)

    try:
        from data_loader import load_wikipedia_documents
        from embedding_generator import generate_document_embeddings
        from vector_store import RAGVectorStore
        import shutil

        # Clean up
        test_dir = "./test_query_fix"
        if Path(test_dir).exists():
            shutil.rmtree(test_dir)

        print("1. Loading 1 document (minimal test)...")
        documents = load_wikipedia_documents(limit=1)

        print("2. Generating embeddings...")
        nodes, stats = generate_document_embeddings(documents, batch_size=1)

        print("3. Creating vector store...")
        store = RAGVectorStore(storage_dir=test_dir, index_name="fix_test")
        store.create_index_from_nodes(nodes)

        print("4. Testing query engine (the fix)...")
        query_engine = store.get_query_engine(similarity_top_k=1)

        if query_engine:
            print("   - Query engine created successfully!")

            print("5. Testing actual query...")
            response = query_engine.query("What is this about?")
            print(f"   - Query response: {str(response)[:100]}...")
            print("   - Query engine fix working!")
        else:
            print("   - Query engine creation failed")
            return False

        # Cleanup
        if Path(test_dir).exists():
            shutil.rmtree(test_dir)

        print("\nQuery engine fix successful!")
        return True

    except Exception as e:
        print(f"Query engine fix failed: {e}")
        return False

if __name__ == "__main__":
    test_query_engine_fix()
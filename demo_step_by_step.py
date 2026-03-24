#!/usr/bin/env python3
"""Step-by-step demo script to test and observe each RAG component.

Run this script to see each component working individually, with detailed
output so you can understand what's happening at each step.
"""

import sys
from pathlib import Path
import time

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

def demo_data_loading():
    """Demo 1: Wikipedia Data Loading"""
    print("=" * 60)
    print("DEMO 1: WIKIPEDIA DATA LOADING")
    print("=" * 60)

    try:
        from data_loader import WikipediaDataLoader

        print("\n🔄 Loading Wikipedia data...")
        loader = WikipediaDataLoader()

        # Load a small sample to see the data structure
        print("Loading 3 passages for inspection...")
        passages_df = loader.load_passages(limit=3)

        print(f"\n📊 DATA SUMMARY:")
        print(f"   Loaded: {len(passages_df)} passages")
        print(f"   Columns: {list(passages_df.columns)}")

        print(f"\n📄 SAMPLE PASSAGE:")
        if len(passages_df) > 0:
            first_passage = passages_df.iloc[0]
            print(f"   ID: {first_passage.get('id', 'N/A')}")
            print(f"   Text: {first_passage.get('passage', 'N/A')[:200]}...")

        # Convert to LlamaIndex Documents
        print(f"\n🔄 Converting to LlamaIndex Documents...")
        documents = loader.passages_to_documents(passages_df)

        print(f"📋 DOCUMENT SUMMARY:")
        print(f"   Created: {len(documents)} documents")
        if documents:
            first_doc = documents[0]
            print(f"   First doc metadata: {first_doc.metadata}")
            print(f"   First doc text length: {len(first_doc.text)}")

        print(f"\n✅ Data loading demo complete!")
        return documents

    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        return None


def demo_embedding_generation(documents):
    """Demo 2: Embedding Generation"""
    print("\n" + "=" * 60)
    print("DEMO 2: EMBEDDING GENERATION")
    print("=" * 60)

    if not documents:
        print("❌ No documents available. Run data loading first.")
        return None

    try:
        from embedding_generator import EmbeddingGenerator

        print(f"\n🔄 Generating embeddings for {len(documents)} documents...")
        generator = EmbeddingGenerator(chunk_size=300, chunk_overlap=50)

        # Convert to nodes first
        print("Converting documents to text nodes...")
        nodes = generator.documents_to_nodes(documents)
        print(f"   Created: {len(nodes)} text nodes")

        # Show a sample node
        if nodes:
            print(f"\n📄 SAMPLE NODE:")
            print(f"   Text length: {len(nodes[0].text)}")
            print(f"   Text preview: {nodes[0].text[:150]}...")

        # Generate embeddings (small batch to avoid rate limits)
        print(f"\n🔄 Generating embeddings (batch size: 1 to avoid rate limits)...")
        start_time = time.time()

        embedded_nodes = generator.generate_embeddings_for_nodes(nodes, batch_size=1)

        elapsed = time.time() - start_time

        # Check results
        successful = sum(1 for node in embedded_nodes if hasattr(node, 'embedding') and node.embedding)

        print(f"\n📊 EMBEDDING RESULTS:")
        print(f"   Processed nodes: {len(embedded_nodes)}")
        print(f"   Successful embeddings: {successful}/{len(embedded_nodes)}")
        print(f"   Processing time: {elapsed:.1f} seconds")

        if successful > 0:
            # Find first embedded node
            first_embedded = next(node for node in embedded_nodes if hasattr(node, 'embedding') and node.embedding)
            print(f"   Embedding dimension: {len(first_embedded.embedding)}")
            print(f"   Sample embedding values: {first_embedded.embedding[:3]}")

        # Get statistics
        stats = generator.get_embedding_stats()
        if stats:
            print(f"\n📈 DETAILED STATISTICS:")
            print(f"   Embeddings per second: {stats.embeddings_per_second:.2f}")
            print(f"   Failed embeddings: {stats.failed_embeddings}")

        print(f"\n✅ Embedding generation demo complete!")
        return embedded_nodes

    except Exception as e:
        print(f"❌ Embedding generation failed: {e}")
        return None


def demo_vector_storage(embedded_nodes):
    """Demo 3: Vector Storage and Similarity Search"""
    print("\n" + "=" * 60)
    print("DEMO 3: VECTOR STORAGE & SIMILARITY SEARCH")
    print("=" * 60)

    if not embedded_nodes:
        print("❌ No embedded nodes available. Run embedding generation first.")
        return None

    try:
        from vector_store import RAGVectorStore
        import shutil

        # Clean up any previous test data
        test_dir = "./demo_vector_storage"
        if Path(test_dir).exists():
            shutil.rmtree(test_dir)

        print(f"\n🔄 Creating vector store...")
        vector_store = RAGVectorStore(storage_dir=test_dir, index_name="demo_index")

        # Create index
        print("Building vector index...")
        index = vector_store.create_index_from_nodes(embedded_nodes)
        print(f"   Index created successfully!")

        # Get index info
        info = vector_store.get_index_info()
        print(f"\n📊 INDEX INFORMATION:")
        print(f"   Status: {info.get('status')}")
        print(f"   Storage path: {info.get('storage_path')}")

        # Test similarity search
        print(f"\n🔍 TESTING SIMILARITY SEARCH:")
        test_queries = [
            "Uruguay",
            "South America",
            "capital city"
        ]

        for query in test_queries:
            print(f"\n   Query: '{query}'")
            results = vector_store.query_similar_documents(query, top_k=2)

            if results:
                for i, result in enumerate(results, 1):
                    score = result.get('similarity_score', 0)
                    text_preview = result['text'][:100] + "..."
                    print(f"     {i}. Similarity: {score:.3f}")
                    print(f"        Text: {text_preview}")
            else:
                print("     No results found")

        # Test save/load
        print(f"\n💾 TESTING PERSISTENCE:")
        print("Saving index to disk...")
        save_success = vector_store.save_index(overwrite=True)
        print(f"   Save successful: {save_success}")

        if save_success:
            print("Loading index from disk...")
            new_store = RAGVectorStore(storage_dir=test_dir, index_name="demo_index")
            loaded = new_store.load_index()
            print(f"   Load successful: {loaded is not None}")

            if loaded:
                # Test query on loaded index
                results = new_store.query_similar_documents("Uruguay", top_k=1)
                if results:
                    score = results[0].get('similarity_score', 0)
                    print(f"   Test query on loaded index: {score:.3f} similarity")

        print(f"\n✅ Vector storage demo complete!")
        return vector_store

    except Exception as e:
        print(f"❌ Vector storage failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def demo_complete_pipeline():
    """Demo 4: Complete Pipeline Test"""
    print("\n" + "=" * 60)
    print("DEMO 4: COMPLETE PIPELINE")
    print("=" * 60)

    print("\n🔄 Running complete pipeline...")

    # Step 1: Load data
    documents = demo_data_loading()
    if not documents:
        return False

    # Step 2: Generate embeddings
    embedded_nodes = demo_embedding_generation(documents)
    if not embedded_nodes:
        return False

    # Step 3: Create vector store
    vector_store = demo_vector_storage(embedded_nodes)
    if not vector_store:
        return False

    print(f"\n🎉 COMPLETE PIPELINE SUCCESS!")
    print("All components working together:")
    print("   ✅ Data loading from HuggingFace")
    print("   ✅ Embedding generation with Azure OpenAI")
    print("   ✅ Vector storage and similarity search")
    print("   ✅ Persistence (save/load)")

    return True


if __name__ == "__main__":
    print("RAG SYSTEM - STEP BY STEP DEMO")
    print("This will test each component individually so you can see how it works")
    print("\nChoose what to test:")
    print("1. Data Loading only")
    print("2. Embedding Generation only (needs data loading first)")
    print("3. Vector Storage only (needs embeddings first)")
    print("4. Complete Pipeline (all steps)")
    print("5. Quick test (3 documents, minimal output)")

    choice = input("\nEnter your choice (1-5): ").strip()

    if choice == "1":
        demo_data_loading()
    elif choice == "2":
        docs = demo_data_loading()
        if docs:
            demo_embedding_generation(docs)
    elif choice == "3":
        docs = demo_data_loading()
        if docs:
            nodes = demo_embedding_generation(docs)
            if nodes:
                demo_vector_storage(nodes)
    elif choice == "4":
        demo_complete_pipeline()
    elif choice == "5":
        # Quick test with minimal rate limit impact
        print("\n🚀 QUICK TEST - Minimal API calls")
        docs = demo_data_loading()
        if docs and len(docs) > 0:
            # Use only 1 document to minimize API calls
            from embedding_generator import generate_document_embeddings
            nodes, stats = generate_document_embeddings(docs[:1], batch_size=1)
            if nodes:
                print(f"\n✅ Quick test successful!")
                print(f"   Processed: 1 document -> {len(nodes)} nodes")
                print(f"   Embeddings: {stats.average_embedding_dimension} dimensions")
            else:
                print("❌ Quick test failed")
    else:
        print("Invalid choice. Running complete pipeline...")
        demo_complete_pipeline()
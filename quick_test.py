#!/usr/bin/env python3
"""Quick test to verify imports and basic functionality."""

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

print("Quick Import Test")
print("=" * 30)

try:
    print("1. Testing basic imports...")
    import pandas as pd
    print("   - pandas imported")

    from datasets import load_dataset
    print("   - datasets imported")

    from llama_index.core import Document
    print("   - llama_index.core imported")

    print("\n2. Testing our custom modules...")
    from llamaindex_models import get_text_embedding_3_large
    print("   - llamaindex_models imported")

    from data_loader import WikipediaDataLoader
    print("   - data_loader imported")

    print("\n3. Testing model access...")
    try:
        embed_model = get_text_embedding_3_large()
        print("   - Embedding model created successfully")
    except Exception as e:
        print(f"   - Model access failed (expected if not authenticated): {e}")

    print("\nAll basic imports successful!")

except Exception as e:
    print(f"Import failed: {e}")
    sys.exit(1)
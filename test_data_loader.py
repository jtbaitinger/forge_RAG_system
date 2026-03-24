#!/usr/bin/env python3
"""Test script to verify Wikipedia data loading functionality."""

import sys
import logging
from pathlib import Path

# Add src to path to import our module
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from data_loader import WikipediaDataLoader, load_wikipedia_documents, load_test_questions

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_data_loading():
    """Test the Wikipedia data loading functionality."""
    print(" Testing Wikipedia Data Loading")
    print("=" * 50)

    try:
        # Initialize data loader
        print("\n1. Creating WikipediaDataLoader...")
        loader = WikipediaDataLoader()

        # Test loading passages (limit to 5 for quick testing)
        print("\n2. Loading Wikipedia passages (limited to 5)...")
        passages_df = loader.load_passages(limit=5)

        print(f"   OK Loaded {len(passages_df)} passages")
        print(f"   Data Columns: {list(passages_df.columns)}")

        if len(passages_df) > 0:
            print(f"   Sample Sample title: '{passages_df.iloc[0].get('title', 'N/A')}'")
            print(f"   Text Sample text length: {len(str(passages_df.iloc[0].get('text', '')))}")

        # Test converting to LlamaIndex Documents
        print("\n3. Converting to LlamaIndex Documents...")
        documents = loader.passages_to_documents(passages_df)

        print(f"   OK Created {len(documents)} LlamaIndex Documents")
        if documents:
            print(f"   Sample First document title: '{documents[0].metadata.get('title', 'N/A')}'")
            print(f"   Text First document text preview: '{documents[0].text[:100]}...'")

        # Test loading test questions (limit to 3)
        print("\n4. Loading test questions (limited to 3)...")
        test_df = loader.load_test_questions(limit=3)

        print(f"   OK Loaded {len(test_df)} test questions")
        print(f"   Data Columns: {list(test_df.columns)}")

        if len(test_df) > 0:
            first_question_cols = [col for col in test_df.columns if 'question' in col.lower()]
            if first_question_cols:
                print(f"   Question Sample question: '{test_df.iloc[0][first_question_cols[0]]}'")

        # Test data summary
        print("\n5. Getting data summary...")
        summary = loader.get_data_summary()

        print(f"   Summary Summary: {summary}")

        # Test convenience functions
        print("\n6. Testing convenience functions...")
        docs_direct = load_wikipedia_documents(limit=2)
        print(f"   OK load_wikipedia_documents() returned {len(docs_direct)} documents")

        test_direct = load_test_questions(limit=2)
        print(f"   OK load_test_questions() returned {len(test_direct)} questions")

        print("\nOK All data loading tests passed!")
        return True

    except Exception as e:
        print(f"\nERROR Data loading test failed: {e}")
        logger.exception("Full error details:")
        return False


if __name__ == "__main__":
    success = test_data_loading()
    exit_code = 0 if success else 1
    sys.exit(exit_code)
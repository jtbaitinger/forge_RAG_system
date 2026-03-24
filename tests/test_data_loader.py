"""Tests for data loading functionality."""

import pytest
import pandas as pd
from llama_index.core import Document

from data_loader import WikipediaDataLoader, load_wikipedia_documents, load_test_questions


class TestWikipediaDataLoader:
    """Test Wikipedia data loading functionality."""

    def test_load_passages_basic(self):
        """Test basic passage loading."""
        loader = WikipediaDataLoader()

        # Load a small number for testing
        passages_df = loader.load_passages(limit=5)

        assert isinstance(passages_df, pd.DataFrame)
        assert len(passages_df) <= 5
        assert len(passages_df) > 0
        assert 'passage' in passages_df.columns
        assert 'id' in passages_df.columns

    def test_load_passages_with_limit(self):
        """Test passage loading with different limits."""
        loader = WikipediaDataLoader()

        # Test different limits
        for limit in [1, 3, 5]:
            passages_df = loader.load_passages(limit=limit)
            assert len(passages_df) == limit

    def test_load_test_questions(self):
        """Test loading test questions."""
        loader = WikipediaDataLoader()

        test_df = loader.load_test_questions(limit=3)

        assert isinstance(test_df, pd.DataFrame)
        assert len(test_df) <= 3
        assert len(test_df) > 0
        assert 'question' in test_df.columns
        assert 'answer' in test_df.columns
        assert 'id' in test_df.columns

    def test_passages_to_documents(self):
        """Test conversion from passages to LlamaIndex Documents."""
        loader = WikipediaDataLoader()

        # Load passages
        passages_df = loader.load_passages(limit=3)

        # Convert to documents
        documents = loader.passages_to_documents(passages_df)

        assert isinstance(documents, list)
        assert len(documents) == len(passages_df)
        assert all(isinstance(doc, Document) for doc in documents)

        # Check first document
        first_doc = documents[0]
        assert hasattr(first_doc, 'text')
        assert hasattr(first_doc, 'metadata')
        assert 'doc_id' in first_doc.metadata
        assert 'source' in first_doc.metadata
        assert first_doc.metadata['source'] == 'wikipedia'

    def test_get_data_summary(self):
        """Test data summary functionality."""
        loader = WikipediaDataLoader()

        # Load some data
        passages_df = loader.load_passages(limit=2)
        test_df = loader.load_test_questions(limit=2)

        summary = loader.get_data_summary()

        assert isinstance(summary, dict)
        assert 'passages_loaded' in summary
        assert 'test_questions_loaded' in summary
        assert 'dataset_name' in summary
        assert summary['passages_loaded'] == 2
        assert summary['test_questions_loaded'] == 2


class TestConvenienceFunctions:
    """Test convenience functions for data loading."""

    def test_load_wikipedia_documents(self):
        """Test the convenience function for loading documents."""
        documents = load_wikipedia_documents(limit=2)

        assert isinstance(documents, list)
        assert len(documents) == 2
        assert all(isinstance(doc, Document) for doc in documents)

    def test_load_test_questions(self):
        """Test the convenience function for loading test questions."""
        questions_df = load_test_questions(limit=2)

        assert isinstance(questions_df, pd.DataFrame)
        assert len(questions_df) == 2
        assert 'question' in questions_df.columns
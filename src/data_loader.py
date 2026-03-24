"""Wikipedia data loading module for RAG system.

This module provides functions to load Wikipedia passages from the HuggingFace
rag-mini-wikipedia dataset. It handles the data loading, preprocessing, and
conversion to LlamaIndex Document format for embedding generation.
"""

import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
import pandas as pd
from datasets import load_dataset, Dataset
from llama_index.core import Document

# Set up logging
logger = logging.getLogger(__name__)


class WikipediaDataLoader:
    """Handles loading and preprocessing Wikipedia data from HuggingFace datasets."""

    def __init__(self, cache_dir: Optional[str] = None):
        """Initialize the data loader.

        Args:
            cache_dir: Directory to cache downloaded datasets. If None, uses default.
        """
        self.cache_dir = cache_dir
        self.dataset_name = "rag-datasets/rag-mini-wikipedia"
        self.passages_file = "data/passages.parquet/part.0.parquet"
        self.test_file = "data/test.parquet/part.0.parquet"
        self._passages_data: Optional[pd.DataFrame] = None
        self._test_data: Optional[pd.DataFrame] = None

    def load_passages(self, limit: Optional[int] = None) -> pd.DataFrame:
        """Load Wikipedia passages from HuggingFace dataset.

        Args:
            limit: Maximum number of passages to load. If None, loads all.

        Returns:
            DataFrame with passages data containing columns like 'text', 'title', etc.
        """
        logger.info(f"Loading Wikipedia passages from {self.dataset_name}")

        try:
            # Load the dataset from HuggingFace
            dataset = load_dataset(
                self.dataset_name,
                data_files=self.passages_file,
                cache_dir=self.cache_dir
            )

            # Convert to pandas DataFrame for easier manipulation
            df = dataset['train'].to_pandas()

            if limit:
                df = df.head(limit)
                logger.info(f"Limited passages to {limit} entries")

            logger.info(f"Loaded {len(df)} Wikipedia passages")
            logger.info(f"Columns available: {list(df.columns)}")

            # Cache the loaded data
            self._passages_data = df

            return df

        except Exception as e:
            logger.error(f"Failed to load Wikipedia passages: {e}")
            raise

    def load_test_questions(self, limit: Optional[int] = None) -> pd.DataFrame:
        """Load test questions from HuggingFace dataset.

        Args:
            limit: Maximum number of test questions to load. If None, loads all.

        Returns:
            DataFrame with test questions and answers.
        """
        logger.info(f"Loading test questions from {self.dataset_name}")

        try:
            # Load the test dataset
            dataset = load_dataset(
                self.dataset_name,
                data_files=self.test_file,
                cache_dir=self.cache_dir
            )

            # Convert to pandas DataFrame
            df = dataset['train'].to_pandas()

            if limit:
                df = df.head(limit)
                logger.info(f"Limited test questions to {limit} entries")

            logger.info(f"Loaded {len(df)} test questions")
            logger.info(f"Columns available: {list(df.columns)}")

            # Cache the loaded data
            self._test_data = df

            return df

        except Exception as e:
            logger.error(f"Failed to load test questions: {e}")
            raise

    def passages_to_documents(self,
                            passages_df: Optional[pd.DataFrame] = None,
                            text_column: str = "passage",
                            title_column: str = "id") -> List[Document]:
        """Convert passages DataFrame to LlamaIndex Document objects.

        Args:
            passages_df: DataFrame with passages. If None, uses cached data.
            text_column: Name of the column containing passage text.
            title_column: Name of the column containing passage titles.

        Returns:
            List of LlamaIndex Document objects ready for embedding.
        """
        if passages_df is None:
            if self._passages_data is None:
                raise ValueError("No passages data available. Call load_passages() first.")
            passages_df = self._passages_data

        logger.info(f"Converting {len(passages_df)} passages to LlamaIndex Documents")

        documents = []
        for idx, row in passages_df.iterrows():
            # Extract text and metadata
            text = str(row.get(text_column, ""))
            title = str(row.get(title_column, f"Document_{idx}"))

            # Create metadata dictionary with all available columns
            metadata = {
                "doc_id": str(idx),
                "title": title,
                "source": "wikipedia",
                "dataset": self.dataset_name
            }

            # Add any additional columns as metadata
            for col in passages_df.columns:
                if col not in [text_column, title_column]:
                    metadata[col] = str(row[col])

            # Create LlamaIndex Document
            doc = Document(
                text=text,
                metadata=metadata
            )
            documents.append(doc)

        logger.info(f"Created {len(documents)} LlamaIndex Documents")
        return documents

    def get_data_summary(self) -> Dict[str, Any]:
        """Get a summary of loaded data.

        Returns:
            Dictionary with data statistics and sample information.
        """
        summary = {
            "passages_loaded": len(self._passages_data) if self._passages_data is not None else 0,
            "test_questions_loaded": len(self._test_data) if self._test_data is not None else 0,
            "dataset_name": self.dataset_name
        }

        if self._passages_data is not None:
            summary["passages_columns"] = list(self._passages_data.columns)
            summary["average_text_length"] = self._passages_data.get('text', pd.Series([])).str.len().mean()
            summary["sample_titles"] = self._passages_data.get('title', pd.Series([])).head(3).tolist()

        if self._test_data is not None:
            summary["test_columns"] = list(self._test_data.columns)

        return summary


# Convenience functions for simple usage
def load_wikipedia_passages(limit: Optional[int] = None,
                          cache_dir: Optional[str] = None) -> pd.DataFrame:
    """Load Wikipedia passages with default settings.

    Args:
        limit: Maximum number of passages to load.
        cache_dir: Directory for caching datasets.

    Returns:
        DataFrame with Wikipedia passages.
    """
    loader = WikipediaDataLoader(cache_dir=cache_dir)
    return loader.load_passages(limit=limit)


def load_wikipedia_documents(limit: Optional[int] = None,
                           cache_dir: Optional[str] = None) -> List[Document]:
    """Load Wikipedia passages as LlamaIndex Documents.

    Args:
        limit: Maximum number of passages to load.
        cache_dir: Directory for caching datasets.

    Returns:
        List of LlamaIndex Document objects.
    """
    loader = WikipediaDataLoader(cache_dir=cache_dir)
    passages_df = loader.load_passages(limit=limit)
    return loader.passages_to_documents(passages_df)


def load_test_questions(limit: Optional[int] = None,
                       cache_dir: Optional[str] = None) -> pd.DataFrame:
    """Load test questions with default settings.

    Args:
        limit: Maximum number of test questions to load.
        cache_dir: Directory for caching datasets.

    Returns:
        DataFrame with test questions and answers.
    """
    loader = WikipediaDataLoader(cache_dir=cache_dir)
    return loader.load_test_questions(limit=limit)
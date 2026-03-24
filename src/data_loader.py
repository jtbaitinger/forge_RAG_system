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

# Import controlled model access for title generation
from llamaindex_models import get_gpt4o

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
                            title_column: str = "id",
                            use_ai_titles: bool = False) -> List[Document]:
        """Convert passages DataFrame to LlamaIndex Document objects.

        Args:
            passages_df: DataFrame with passages. If None, uses cached data.
            text_column: Name of the column containing passage text.
            title_column: Name of the column containing passage titles.
            use_ai_titles: Whether to use AI (GPT-4o) to generate titles from passage content.

        Returns:
            List of LlamaIndex Document objects ready for embedding.
        """
        if passages_df is None:
            if self._passages_data is None:
                raise ValueError("No passages data available. Call load_passages() first.")
            passages_df = self._passages_data

        logger.info(f"Converting {len(passages_df)} passages to LlamaIndex Documents")

        # Generate AI titles in batch if requested
        ai_titles = []
        if use_ai_titles:
            logger.info("Generating AI-powered titles for passages...")
            texts = [str(row.get(text_column, "")) for _, row in passages_df.iterrows()]
            ai_titles = self._generate_ai_titles_batch(texts)

        documents = []
        for idx, row in passages_df.iterrows():
            # Extract text and metadata
            text = str(row.get(text_column, ""))

            # Use AI-generated title if available, otherwise extract from text
            if use_ai_titles and idx < len(ai_titles):
                title = ai_titles[idx]
            else:
                title = self._extract_title_from_text(text, fallback=f"Document_{idx}")

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

    def _extract_title_from_text(self, text: str, fallback: str = "Unknown") -> str:
        """Extract a meaningful title from passage text.

        Args:
            text: The passage text to extract title from.
            fallback: Fallback title if extraction fails.

        Returns:
            Extracted title or fallback.
        """
        if not text or len(text.strip()) < 10:
            return fallback

        # Clean the text
        text = text.strip()
        import re

        # Strategy 1: Look for proper nouns or named entities at start
        # Pattern: "Uruguay is...", "The Beatles were...", "Albert Einstein was..."
        entity_patterns = [
            r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:is|was|are|were)\s+',  # "Uruguay is", "Albert Einstein was"
            r'^(The\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:is|was|are|were)\s+',  # "The Beatles were"
        ]

        for pattern in entity_patterns:
            match = re.search(pattern, text)
            if match:
                title = match.group(1).strip()
                if 3 <= len(title) <= 60:  # Reasonable title length
                    return title

        # Strategy 2: Look for locations, countries, organizations
        location_patterns = [
            r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*),?\s+(?:officially|also\s+known|located|situated)',
            r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:country|nation|city|state|province)',
        ]

        for pattern in location_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                title = match.group(1).strip()
                if 3 <= len(title) <= 60:
                    return title

        # Strategy 3: Find capitalized phrases that look like titles
        # Look for 2-4 capitalized words at the start
        title_match = re.match(r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})', text)
        if title_match:
            title = title_match.group(1).strip()
            if 3 <= len(title) <= 60:
                return title

        # Strategy 4: Look for quoted titles or titles in parentheses
        quoted_patterns = [
            r'"([^"]{3,50})"',
            r"'([^']{3,50})'",
            r'\(([^)]{3,50})\)',
        ]

        for pattern in quoted_patterns:
            match = re.search(pattern, text)
            if match:
                title = match.group(1).strip()
                # Check if it looks like a title (has capital letters)
                if re.search(r'[A-Z]', title) and 3 <= len(title) <= 60:
                    return title

        # Strategy 5: Extract key context from the text - first sentence without common stopwords
        sentences = re.split(r'[.!?]+', text)
        if sentences and len(sentences[0]) > 10:
            first_sentence = sentences[0].strip()

            # Remove common Wikipedia prefixes
            prefixes_to_remove = [
                r'^According to ',
                r'^As of ',
                r'^In \d{4},?\s*',
                r'^During ',
                r'^After ',
                r'^Before ',
                r'^Since ',
                r'^For most of ',
            ]

            for prefix in prefixes_to_remove:
                first_sentence = re.sub(prefix, '', first_sentence, flags=re.IGNORECASE)

            # Take meaningful part of the sentence
            words = first_sentence.split()
            if len(words) >= 3:
                # Look for the main subject - often first few words contain it
                potential_title = ' '.join(words[:min(8, len(words))])
                if len(potential_title) <= 70:
                    return potential_title

        # Final fallback: use first few meaningful words
        words = [w for w in text.split() if len(w) > 2]  # Skip short words
        if len(words) >= 2:
            return ' '.join(words[:5])

        return fallback[:50]

    def _generate_ai_titles_batch(self, texts: List[str]) -> List[str]:
        """Generate concise titles for passages using AI.

        Args:
            texts: List of passage texts to generate titles for.

        Returns:
            List of AI-generated titles corresponding to each passage.
        """
        logger.info(f"Generating AI titles for {len(texts)} passages")

        try:
            # Get the controlled GPT-4o model
            llm = get_gpt4o(temperature=0.1)  # Low temperature for consistent titles

            titles = []
            batch_size = 5  # Process in small batches to avoid token limits

            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                batch_titles = self._generate_ai_titles_single_batch(llm, batch_texts)
                titles.extend(batch_titles)

                if i % 10 == 0:  # Progress logging
                    logger.info(f"Generated titles for {len(titles)}/{len(texts)} passages")

            logger.info(f"AI title generation complete: {len(titles)} titles generated")
            return titles

        except Exception as e:
            logger.error(f"AI title generation failed: {e}")
            # Fallback to regex extraction
            return [self._extract_title_from_text(text, f"Document_{i}") for i, text in enumerate(texts)]

    def _generate_ai_titles_single_batch(self, llm, texts: List[str]) -> List[str]:
        """Generate titles for a single batch of texts.

        Args:
            llm: The LLM instance to use.
            texts: Batch of texts to process.

        Returns:
            List of generated titles for this batch.
        """
        # Create a prompt that asks for concise titles
        passages_text = ""
        for i, text in enumerate(texts):
            # Truncate very long passages to fit in prompt
            truncated = text[:500] + "..." if len(text) > 500 else text
            passages_text += f"Passage {i+1}: {truncated}\n\n"

        prompt = f"""Generate a concise, descriptive title (2-6 words) for each Wikipedia passage below. The title should capture the main subject or topic.

{passages_text}

Respond with exactly {len(texts)} titles, one per line, in the format:
1. Title for passage 1
2. Title for passage 2
etc.

Keep titles short, specific, and focused on the main subject (person, place, thing, concept)."""

        try:
            response = llm.complete(prompt)
            response_text = str(response)

            # Parse the numbered response
            titles = []
            for line in response_text.split('\n'):
                line = line.strip()
                # Look for numbered format: "1. Title" or "1) Title" or just "Title"
                import re
                match = re.match(r'^\d+[\.\)]\s*(.+)', line)
                if match:
                    title = match.group(1).strip()
                    # Clean up the title
                    title = re.sub(r'^["\']|["\']$', '', title)  # Remove quotes
                    if title and len(title) <= 100:  # Reasonable title length
                        titles.append(title)

            # Ensure we have the right number of titles
            while len(titles) < len(texts):
                titles.append(f"Document_{len(titles)}")

            return titles[:len(texts)]  # Truncate if we got too many

        except Exception as e:
            logger.error(f"Failed to generate titles for batch: {e}")
            # Fallback titles
            return [f"Document_{i}" for i in range(len(texts))]


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
                           cache_dir: Optional[str] = None,
                           use_ai_titles: bool = False) -> List[Document]:
    """Load Wikipedia passages as LlamaIndex Documents.

    Args:
        limit: Maximum number of passages to load.
        cache_dir: Directory for caching datasets.
        use_ai_titles: Whether to use AI to generate titles from passage content.

    Returns:
        List of LlamaIndex Document objects.
    """
    loader = WikipediaDataLoader(cache_dir=cache_dir)
    passages_df = loader.load_passages(limit=limit)
    return loader.passages_to_documents(passages_df, use_ai_titles=use_ai_titles)


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
"""
BM25 Retriever for keyword-based document search.

This module provides BM25 (Best Matching 25) retrieval functionality,
which uses term frequency and inverse document frequency for keyword matching.
This is essential for queries where exact keyword matches matter.
"""

from __future__ import annotations

import re
from typing import Any

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from src.utils.logging import LoggerMixin


class BM25Retriever(LoggerMixin):
    """
    BM25 retriever for keyword-based document search.
    
    BM25 excels at finding documents containing specific keywords,
    making it complementary to semantic search for hybrid retrieval.
    
    Args:
        documents: List of documents to index.
        k: Number of documents to retrieve (default: 4).
        
    Example:
        >>> from langchain_core.documents import Document
        >>> docs = [Document(page_content="Python is great")]
        >>> retriever = BM25Retriever(docs, k=2)
        >>> results = retriever.retrieve("Python")
    """
    
    def __init__(
        self,
        documents: list[Document] | None = None,
        k: int = 4,
    ) -> None:
        """
        Initialize the BM25 retriever.
        
        Args:
            documents: List of documents to index (can be added later).
            k: Number of documents to retrieve.
        """
        super().__init__()
        
        self.k = k
        self._documents: list[Document] = []
        self._bm25: BM25Okapi | None = None
        self._tokenized_corpus: list[list[str]] = []
        
        if documents:
            self.add_documents(documents)
        
        self.logger.info("BM25Retriever initialized", k=k)
    
    def _tokenize(self, text: str) -> list[str]:
        """
        Tokenize text for BM25.
        
        Performs simple whitespace tokenization with lowercasing
        and basic punctuation removal. Works for both English and Indonesian.
        
        Args:
            text: Text to tokenize.
            
        Returns:
            List of tokens.
        """
        # Lowercase and remove punctuation except for alphanumeric and spaces
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Split on whitespace and filter empty strings
        tokens = [token for token in text.split() if token]
        
        return tokens
    
    def add_documents(self, documents: list[Document]) -> None:
        """
        Add documents to the BM25 index.
        
        Args:
            documents: Documents to add.
        """
        self.logger.info("Adding documents to BM25 index", count=len(documents))
        
        self._documents.extend(documents)
        
        # Tokenize all documents
        self._tokenized_corpus = [
            self._tokenize(doc.page_content) for doc in self._documents
        ]
        
        # Rebuild BM25 index
        if self._tokenized_corpus:
            self._bm25 = BM25Okapi(self._tokenized_corpus)
        
        self.logger.info(
            "BM25 index updated",
            total_documents=len(self._documents),
        )
    
    def clear(self) -> None:
        """Clear all documents from the index."""
        self._documents = []
        self._tokenized_corpus = []
        self._bm25 = None
        self.logger.info("BM25 index cleared")
    
    def retrieve(self, query: str) -> list[tuple[Document, float]]:
        """
        Retrieve documents matching the query.
        
        Args:
            query: Search query string.
            
        Returns:
            List of (Document, score) tuples ordered by BM25 score.
        """
        if not self._bm25 or not self._documents:
            self.logger.warning("BM25 index is empty, returning no results")
            return []
        
        # Tokenize query
        query_tokens = self._tokenize(query)
        
        if not query_tokens:
            self.logger.warning("Query tokenized to empty, returning no results")
            return []
        
        # Get BM25 scores
        scores = self._bm25.get_scores(query_tokens)
        
        # Get top k documents with scores
        doc_scores = list(zip(self._documents, scores))
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Return top k
        results = doc_scores[:self.k]
        
        self.logger.debug(
            "BM25 retrieval completed",
            query=query[:100],
            num_results=len(results),
            top_score=results[0][1] if results else 0.0,
        )
        
        return results
    
    @property
    def document_count(self) -> int:
        """Get the number of indexed documents."""
        return len(self._documents)

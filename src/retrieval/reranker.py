"""
Document reranker for improving retrieval quality.

This module provides optional reranking functionality using cross-encoder models
to reorder retrieved documents based on fine-grained relevance scoring.
"""

from __future__ import annotations

from typing import List

from src.retrieval.retriever import RetrievedDocument
from src.utils.logging import LoggerMixin


class Reranker(LoggerMixin):
    """
    Reranker for improving document retrieval quality.
    
    Uses a cross-encoder model to rerank retrieved documents by computing
    fine-grained relevance scores between the query and each document.
    Cross-encoders are more accurate than bi-encoders (embeddings) but slower,
    making them ideal for reranking a small set of candidates.
    
    Args:
        model_name: HuggingFace cross-encoder model name.
        top_k: Number of documents to return after reranking.
        score_threshold: Minimum reranking score to include (optional).
        
    Example:
        >>> from src.retrieval import Reranker
        >>> 
        >>> reranker = Reranker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
        >>> reranked = reranker.rerank("What is Python?", retrieved_docs, top_k=3)
        >>> # Documents are now reordered by cross-encoder scores
    """
    
    DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        top_k: int = 4,
        score_threshold: float | None = None,
    ) -> None:
        """
        Initialize the reranker.
        
        Args:
            model_name: HuggingFace cross-encoder model name.
            top_k: Number of documents to return after reranking.
            score_threshold: Minimum score threshold (optional).
            
        Raises:
            ImportError: If sentence-transformers is not installed.
        """
        super().__init__()
        
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")
        
        self.model_name = model_name
        self.top_k = top_k
        self.score_threshold = score_threshold
        
        # Lazy load the model
        self._model = None
        
        self.logger.info(
            "Reranker initialized",
            model_name=model_name,
            top_k=top_k,
            score_threshold=score_threshold
        )
    
    @property
    def model(self):
        """Lazy-load the cross-encoder model."""
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError:
                self.logger.error("sentence-transformers not installed")
                raise ImportError(
                    "sentence-transformers is required for reranking. "
                    "Install it with: pip install sentence-transformers"
                )
            
            self.logger.info("Loading cross-encoder model", model=self.model_name)
            self._model = CrossEncoder(self.model_name)
            self.logger.info("Cross-encoder model loaded", model=self.model_name)
        
        return self._model
    
    def rerank(
        self,
        query: str,
        documents: List[RetrievedDocument],
        top_k: int | None = None
    ) -> List[RetrievedDocument]:
        """
        Rerank documents using cross-encoder model.
        
        Computes fine-grained relevance scores between the query and each document,
        then reorders documents by these scores.
        
        Args:
            query: Search query.
            documents: List of retrieved documents to rerank.
            top_k: Number of documents to return (uses self.top_k if None).
            
        Returns:
            Reranked list of documents, ordered by cross-encoder scores.
            
        Example:
            >>> reranker = Reranker()
            >>> docs = [...]  # Retrieved documents
            >>> reranked = reranker.rerank("machine learning basics", docs, top_k=5)
            >>> reranked[0].score > reranked[1].score
            True
        """
        if not query or not query.strip():
            self.logger.warning("Empty query provided to rerank")
            return documents
        
        if not documents:
            self.logger.warning("No documents provided to rerank")
            return documents
        
        k = top_k if top_k is not None else self.top_k
        
        self.logger.info(
            "Reranking documents",
            query=query,
            num_docs=len(documents),
            top_k=k
        )
        
        try:
            # Prepare query-document pairs for cross-encoder
            pairs = [(query, doc.content) for doc in documents]
            
            # Compute cross-encoder scores
            self.logger.debug("Computing cross-encoder scores")
            scores = self.model.predict(pairs)
            
            # Create new RetrievedDocument objects with updated scores
            reranked_docs = []
            for doc, score in zip(documents, scores):
                # Convert score to float and normalize if needed
                normalized_score = float(score)
                
                # Create new document with updated score
                reranked_doc = RetrievedDocument(
                    content=doc.content,
                    metadata=doc.metadata,
                    score=normalized_score,
                    source=doc.source,
                    chunk_index=doc.chunk_index
                )
                reranked_docs.append(reranked_doc)
            
            # Sort by score (descending)
            reranked_docs.sort(key=lambda d: d.score, reverse=True)
            
            # Apply score threshold if set
            if self.score_threshold is not None:
                original_count = len(reranked_docs)
                reranked_docs = [
                    doc for doc in reranked_docs
                    if doc.score >= self.score_threshold
                ]
                
                if len(reranked_docs) < original_count:
                    self.logger.debug(
                        "Filtered reranked results by score threshold",
                        threshold=self.score_threshold,
                        original_count=original_count,
                        filtered_count=len(reranked_docs)
                    )
            
            # Return top k
            result = reranked_docs[:k]
            
            self.logger.info(
                "Reranking complete",
                query=query,
                original_count=len(documents),
                reranked_count=len(result),
                top_score=result[0].score if result else 0.0
            )
            
            return result
            
        except Exception as e:
            self.logger.error(
                "Failed to rerank documents",
                query=query,
                error=str(e),
                exc_info=True
            )
            # Fallback: return original documents
            return documents[:k]
    
    def compute_scores(
        self,
        query: str,
        documents: List[RetrievedDocument]
    ) -> List[float]:
        """
        Compute cross-encoder scores without reranking.
        
        Useful for getting scores without modifying the document order.
        
        Args:
            query: Search query.
            documents: List of documents to score.
            
        Returns:
            List of scores in the same order as input documents.
            
        Example:
            >>> reranker = Reranker()
            >>> scores = reranker.compute_scores("What is AI?", docs)
            >>> max(scores)
            0.95
        """
        if not query or not documents:
            return []
        
        try:
            pairs = [(query, doc.content) for doc in documents]
            scores = self.model.predict(pairs)
            return [float(score) for score in scores]
            
        except Exception as e:
            self.logger.error(
                "Failed to compute scores",
                query=query,
                error=str(e),
                exc_info=True
            )
            return [0.0] * len(documents)


class RerankerConfig:
    """
    Configuration presets for different reranking scenarios.
    
    Provides predefined model configurations optimized for different use cases.
    """
    
    # Fast reranking with reasonable quality
    FAST = {
        "model_name": "cross-encoder/ms-marco-TinyBERT-L-2-v2",
        "top_k": 4,
    }
    
    # Balanced speed and quality (default)
    BALANCED = {
        "model_name": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "top_k": 4,
    }
    
    # High quality, slower
    HIGH_QUALITY = {
        "model_name": "cross-encoder/ms-marco-MiniLM-L-12-v2",
        "top_k": 4,
    }
    
    @classmethod
    def get_fast_reranker(cls) -> Reranker:
        """Get a fast reranker instance."""
        return Reranker(**cls.FAST)
    
    @classmethod
    def get_balanced_reranker(cls) -> Reranker:
        """Get a balanced reranker instance."""
        return Reranker(**cls.BALANCED)
    
    @classmethod
    def get_high_quality_reranker(cls) -> Reranker:
        """Get a high-quality reranker instance."""
        return Reranker(**cls.HIGH_QUALITY)

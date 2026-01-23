"""
Hybrid Retriever combining BM25 keyword search with semantic search.

This module provides a hybrid retrieval approach that combines:
- BM25: Excellent for exact keyword matches (e.g., "rektor" -> "Rektor")
- Semantic: Understanding meaning and context

The hybrid approach uses Reciprocal Rank Fusion (RRF) to merge results.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langchain_core.documents import Document

from src.core.vectorstore import VectorStoreManager
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.retriever import RAGRetriever, RetrievedDocument
from src.utils.logging import LoggerMixin


@dataclass
class HybridSearchConfig:
    """
    Configuration for hybrid search.
    
    Attributes:
        semantic_weight: Weight for semantic search (0-1, default 0.5).
        bm25_weight: Weight for BM25 search (0-1, default 0.5).
        k: Total number of documents to return.
        semantic_k: Number of candidates from semantic search.
        bm25_k: Number of candidates from BM25 search.
        rrf_k: RRF constant (default 60, higher = smoother ranking).
    """
    semantic_weight: float = 0.5
    bm25_weight: float = 0.5
    k: int = 5
    semantic_k: int = 15
    bm25_k: int = 15
    rrf_k: int = 60


class HybridRetriever(LoggerMixin):
    """
    Hybrid retriever combining BM25 and semantic search.
    
    Uses Reciprocal Rank Fusion (RRF) to combine results from both
    retrieval methods, providing the best of both worlds:
    - BM25 catches exact keyword matches
    - Semantic search catches meaning/context
    
    Args:
        vectorstore: Vector store manager for semantic search.
        config: Hybrid search configuration.
        
    Example:
        >>> hybrid = HybridRetriever(vectorstore)
        >>> results = await hybrid.retrieve("siapa rektor unnes?")
    """
    
    def __init__(
        self,
        vectorstore: VectorStoreManager,
        config: HybridSearchConfig | None = None,
    ) -> None:
        """
        Initialize the hybrid retriever.
        
        Args:
            vectorstore: Vector store manager for semantic search.
            config: Optional configuration (uses defaults if not provided).
        """
        super().__init__()
        
        self.vectorstore = vectorstore
        self.config = config or HybridSearchConfig()
        
        # Initialize BM25 retriever (will be populated lazily)
        self._bm25_retriever: BM25Retriever | None = None
        self._documents_loaded = False
        
        self.logger.info(
            "HybridRetriever initialized",
            semantic_weight=self.config.semantic_weight,
            bm25_weight=self.config.bm25_weight,
            k=self.config.k,
        )
    
    async def _ensure_bm25_loaded(self) -> None:
        """
        Ensure BM25 index is loaded with documents from vectorstore.
        
        This loads all documents from ChromaDB into the BM25 index.
        Since retriever instances are created per-request via dependency injection,
        we load fresh data each time to ensure consistency.
        """
        self.logger.info("Loading documents into BM25 index...")
        
        # Get all documents from ChromaDB collection
        collection = self.vectorstore.collection
        results = collection.get(include=["documents", "metadatas"])
        
        if not results["documents"]:
            self.logger.warning("No documents found in ChromaDB collection")
            self._bm25_retriever = BM25Retriever([], k=self.config.bm25_k)
            self._documents_loaded = True
            return
        
        # Convert to LangChain Documents
        documents = []
        for i, content in enumerate(results["documents"]):
            metadata = results["metadatas"][i] if results["metadatas"] else {}
            documents.append(Document(page_content=content, metadata=metadata))
        
        # Initialize BM25 with documents
        self._bm25_retriever = BM25Retriever(documents, k=self.config.bm25_k)
        self._documents_loaded = True
        
        self.logger.info(
            "BM25 index loaded",
            document_count=len(documents),
        )
    
    def invalidate_bm25_cache(self) -> None:
        """
        Invalidate BM25 cache to force reload on next query.
        
        Call this after ingesting new documents.
        """
        self._documents_loaded = False
        self._bm25_retriever = None
        self.logger.info("BM25 cache invalidated")
    
    async def retrieve(self, query: str) -> list[RetrievedDocument]:
        """
        Retrieve documents using hybrid search.
        
        Combines BM25 keyword search and semantic search using
        Reciprocal Rank Fusion (RRF).
        
        Args:
            query: Search query string.
            
        Returns:
            List of RetrievedDocument objects ordered by combined score.
        """
        if not query or not query.strip():
            self.logger.warning("Empty query provided")
            return []
        
        # Determine the retrieval mode based on weights
        is_semantic_only = self.config.bm25_weight == 0.0
        is_bm25_only = self.config.semantic_weight == 0.0
        
        self.logger.info(
            "Hybrid retrieval starting",
            query=query[:100],
            config=self.config,
            is_semantic_only=is_semantic_only,
            is_bm25_only=is_bm25_only,
        )
        
        semantic_results: list[tuple[Document, float]] = []
        bm25_results: list[tuple[Document, float]] = []
        
        # Only run searches that have non-zero weight
        if not is_bm25_only:
            semantic_results = await self._semantic_search(query)
            self.logger.info(
                "Semantic search results",
                query=query,
                num_results=len(semantic_results),
            )
            for i, (doc, score) in enumerate(semantic_results[:5]):
                self.logger.info(
                    f"Semantic result #{i+1}",
                    semantic_rank=i + 1,
                    semantic_score=f"{score:.4f}",
                    source=doc.metadata.get("source", "unknown"),
                    chunk_index=doc.metadata.get("chunk_index", 0),
                    content_preview=doc.page_content[:200].replace("\n", " "),
                )
        
        if not is_semantic_only:
            await self._ensure_bm25_loaded()
            bm25_results = self._bm25_search(query)
            self.logger.info(
                "BM25 search results",
                query=query,
                num_results=len(bm25_results),
            )
            for i, (doc, score) in enumerate(bm25_results[:5]):
                self.logger.info(
                    f"BM25 result #{i+1}",
                    bm25_rank=i + 1,
                    bm25_score=f"{score:.4f}",
                    source=doc.metadata.get("source", "unknown"),
                    chunk_index=doc.metadata.get("chunk_index", 0),
                    content_preview=doc.page_content[:200].replace("\n", " "),
                )
        
        # Log intermediate results for debugging
        self.logger.info(
            "Search results before fusion",
            semantic_count=len(semantic_results),
            bm25_count=len(bm25_results),
            semantic_top=(
                semantic_results[0][0].page_content[:100] 
                if semantic_results else "N/A"
            ),
            bm25_top=(
                bm25_results[0][0].page_content[:100] 
                if bm25_results else "N/A"
            ),
        )
        
        # Combine using RRF
        combined = self._reciprocal_rank_fusion(semantic_results, bm25_results)
        
        # Determine the retrieval method label for results
        if is_semantic_only:
            method_label = "semantic"
        elif is_bm25_only:
            method_label = "bm25"
        else:
            method_label = None  # Use per-document method from RRF
        
        # Convert to RetrievedDocument
        results = [
            RetrievedDocument(
                content=doc.page_content,
                metadata=doc.metadata,
                score=score,
                source=doc.metadata.get("source", "unknown"),
                chunk_index=doc.metadata.get("chunk_index", 0),
                retrieval_method=method_label if method_label else retrieval_method,
            )
            for doc, score, retrieval_method in combined[:self.config.k]
        ]
        
        self.logger.info(
            "Hybrid retrieval completed",
            query=query[:50],
            num_results=len(results),
            top_score=results[0].score if results else 0.0,
        )
        
        return results
    
    async def _semantic_search(
        self, query: str
    ) -> list[tuple[Document, float]]:
        """
        Perform semantic search using vectorstore.
        
        Args:
            query: Search query.
            
        Returns:
            List of (Document, score) tuples.
        """
        try:
            results = await self.vectorstore.search_with_score(
                query=query,
                k=self.config.semantic_k,
            )
            
            # Convert L2 distance to similarity score (lower L2 = higher similarity)
            # Using exponential decay: score = exp(-distance)
            import math
            converted = []
            for doc, distance in results:
                similarity = math.exp(-distance)
                converted.append((doc, similarity))
            
            return converted
            
        except Exception as e:
            self.logger.error(f"Semantic search failed: {e}")
            return []
    
    def _bm25_search(self, query: str) -> list[tuple[Document, float]]:
        """
        Perform BM25 keyword search.
        
        Args:
            query: Search query.
            
        Returns:
            List of (Document, score) tuples.
        """
        if not self._bm25_retriever:
            return []
        
        return self._bm25_retriever.retrieve(query)
    
    def _reciprocal_rank_fusion(
        self,
        semantic_results: list[tuple[Document, float]],
        bm25_results: list[tuple[Document, float]],
    ) -> list[tuple[Document, float, str]]:
        """
        Combine results using Reciprocal Rank Fusion (RRF).
        
        RRF formula: score = sum(1 / (k + rank)) for each result list
        
        This method is robust and doesn't require score normalization.
        
        Args:
            semantic_results: Results from semantic search.
            bm25_results: Results from BM25 search.
            
        Returns:
            Combined and ranked list of (Document, score, retrieval_method) tuples.
            retrieval_method is "semantic", "bm25", or "hybrid" (found by both).
        """
        k = self.config.rrf_k
        semantic_weight = self.config.semantic_weight
        bm25_weight = self.config.bm25_weight
        
        # Dictionary to accumulate RRF scores by document content
        # Using content as key since same doc may come from both retrievers
        # Now also tracking which retriever(s) found each document
        doc_scores: dict[str, tuple[Document, float, set[str]]] = {}
        
        # Process semantic results
        for rank, (doc, _) in enumerate(semantic_results, start=1):
            content_key = doc.page_content
            rrf_score = semantic_weight * (1.0 / (k + rank))
            
            if content_key in doc_scores:
                existing_doc, existing_score, methods = doc_scores[content_key]
                methods.add("semantic")
                doc_scores[content_key] = (existing_doc, existing_score + rrf_score, methods)
            else:
                doc_scores[content_key] = (doc, rrf_score, {"semantic"})
        
        # Process BM25 results
        for rank, (doc, _) in enumerate(bm25_results, start=1):
            content_key = doc.page_content
            rrf_score = bm25_weight * (1.0 / (k + rank))
            
            if content_key in doc_scores:
                existing_doc, existing_score, methods = doc_scores[content_key]
                methods.add("bm25")
                doc_scores[content_key] = (existing_doc, existing_score + rrf_score, methods)
            else:
                doc_scores[content_key] = (doc, rrf_score, {"bm25"})
        
        # Sort by combined score and determine retrieval method
        sorted_results = sorted(
            doc_scores.values(),
            key=lambda x: x[1],
            reverse=True,
        )
        
        # Convert methods set to retrieval_method string
        results_with_method = []
        for doc, score, methods in sorted_results:
            if len(methods) == 2:
                retrieval_method = "hybrid"
            else:
                retrieval_method = list(methods)[0]
            results_with_method.append((doc, score, retrieval_method))
        
        self.logger.debug(
            "RRF fusion completed",
            total_unique_docs=len(results_with_method),
            top_scores=[f"{score:.4f}" for _, score, _ in results_with_method[:5]],
        )
        
        return results_with_method


# Create a wrapper that matches RAGRetriever interface for compatibility
class HybridRAGRetriever(HybridRetriever):
    """
    Hybrid retriever with RAGRetriever-compatible interface.
    
    This class wraps HybridRetriever to provide the same interface
    as RAGRetriever, making it a drop-in replacement.
    """
    
    def __init__(
        self,
        vectorstore: VectorStoreManager,
        k: int = 5,
        semantic_weight: float = 0.5,
        bm25_weight: float = 0.5,
        **kwargs: Any,
    ) -> None:
        """
        Initialize hybrid RAG retriever.
        
        Args:
            vectorstore: Vector store manager.
            k: Number of documents to return.
            semantic_weight: Weight for semantic search.
            bm25_weight: Weight for BM25 search.
            **kwargs: Additional arguments (ignored for compatibility).
        """
        config = HybridSearchConfig(
            k=k,
            semantic_weight=semantic_weight,
            bm25_weight=bm25_weight,
            semantic_k=k * 3,
            bm25_k=k * 3,
        )
        super().__init__(vectorstore, config)
    
    async def retrieve_with_scores(
        self, query: str
    ) -> list[tuple[Document, float]]:
        """
        Retrieve documents with scores (for compatibility).
        
        Args:
            query: Search query.
            
        Returns:
            List of (Document, score) tuples.
        """
        results = await self.retrieve(query)
        return [
            (result.to_langchain_document(), result.score)
            for result in results
        ]

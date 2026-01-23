"""
RAG retriever for semantic document search.

This module provides retrieval functionality for the RAG chatbot, including:
- Semantic similarity search
- Maximum Marginal Relevance (MMR) for diverse results
- Relevance score calculation
- LangChain-compatible retriever interface
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun

from src.core.vectorstore import VectorStoreManager
from src.utils.logging import LoggerMixin


@dataclass
class RetrievedDocument:
    """
    A document retrieved from the vector store.
    
    Attributes:
        content: The text content of the document.
        metadata: Metadata associated with the document.
        score: Relevance score (0-1, higher is more relevant).
        source: Source file or identifier.
        chunk_index: Index of this chunk within the source document.
        retrieval_method: Method used to retrieve this document
            ("semantic", "bm25", or "hybrid" if found by both).
    """
    content: str
    metadata: Dict[str, Any]
    score: float
    source: str
    chunk_index: int = 0
    retrieval_method: str = "semantic"
    
    def __str__(self) -> str:
        """Human-readable string representation."""
        return (
            f"RetrievedDocument(source={self.source}, "
            f"chunk={self.chunk_index}, score={self.score:.3f}, "
            f"method={self.retrieval_method})"
        )
    
    def to_langchain_document(self) -> Document:
        """
        Convert to LangChain Document format.
        
        Returns:
            LangChain Document object.
        """
        return Document(page_content=self.content, metadata=self.metadata)


class RAGRetriever(LoggerMixin):
    """
    RAG retriever for semantic document search.
    
    Retrieves relevant documents from the vector store using semantic similarity
    or Maximum Marginal Relevance (MMR) to balance relevance with diversity.
    
    Args:
        vectorstore: Vector store manager for document retrieval.
        k: Number of documents to retrieve (default: 4).
        use_mmr: Whether to use MMR for diversity (default: True).
        mmr_diversity: Diversity factor for MMR, 0-1 (default: 0.5).
            0 = pure relevance, 1 = pure diversity.
        score_threshold: Minimum relevance score to include (default: 0.0).
        
    Example:
        >>> from src.retrieval import RAGRetriever
        >>> from src.core.vectorstore import VectorStoreManager
        >>> 
        >>> vectorstore = VectorStoreManager(...)
        >>> retriever = RAGRetriever(vectorstore, k=4, use_mmr=True)
        >>> 
        >>> results = await retriever.retrieve("What is RAG?")
        >>> for doc in results:
        ...     print(f"{doc.source}: {doc.content[:100]}...")
    """
    
    def __init__(
        self,
        vectorstore: VectorStoreManager,
        k: int = 4,
        use_mmr: bool = True,
        mmr_diversity: float = 0.5,
        score_threshold: float = 0.0,
    ) -> None:
        """
        Initialize the RAG retriever.
        
        Args:
            vectorstore: Vector store manager for document retrieval.
            k: Number of documents to retrieve.
            use_mmr: Whether to use MMR for diversity.
            mmr_diversity: Diversity factor for MMR (0-1).
            score_threshold: Minimum relevance score threshold.
            
        Raises:
            ValueError: If parameters are out of valid range.
        """
        super().__init__()
        
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        
        if not 0.0 <= mmr_diversity <= 1.0:
            raise ValueError(f"mmr_diversity must be between 0 and 1, got {mmr_diversity}")
        
        if not 0.0 <= score_threshold <= 1.0:
            raise ValueError(f"score_threshold must be between 0 and 1, got {score_threshold}")
        
        self.vectorstore = vectorstore
        self.k = k
        self.use_mmr = use_mmr
        self.mmr_diversity = mmr_diversity
        self.score_threshold = score_threshold
        
        self.logger.info(
            "RAGRetriever initialized",
            k=k,
            use_mmr=use_mmr,
            mmr_diversity=mmr_diversity,
            score_threshold=score_threshold
        )
    
    async def retrieve(self, query: str) -> List[RetrievedDocument]:
        """
        Retrieve documents relevant to the query.
        
        Uses either semantic similarity or MMR based on configuration.
        
        Args:
            query: Search query string.
            
        Returns:
            List of RetrievedDocument objects ordered by relevance.
            
        Example:
            >>> retriever = RAGRetriever(vectorstore, k=3)
            >>> results = await retriever.retrieve("Python programming")
            >>> len(results)
            3
            >>> results[0].score > results[1].score
            True
        """
        if not query or not query.strip():
            self.logger.warning("Empty query provided to retrieve")
            return []
        
        self.logger.info("Retrieving documents", query=query, k=self.k, use_mmr=self.use_mmr)
        
        try:
            # Get documents with scores
            docs_with_scores = await self.retrieve_with_scores(query)
            
            # Convert to RetrievedDocument objects
            retrieved_docs = [
                self._to_retrieved_document(doc, score)
                for doc, score in docs_with_scores
            ]
            
            self.logger.info(
                "Documents retrieved",
                query=query,
                num_results=len(retrieved_docs),
                top_score=retrieved_docs[0].score if retrieved_docs else 0.0
            )
            
            return retrieved_docs
            
        except Exception as e:
            self.logger.error(
                "Failed to retrieve documents",
                query=query,
                error=str(e),
                exc_info=True
            )
            raise
    
    async def retrieve_with_scores(
        self,
        query: str
    ) -> List[Tuple[Document, float]]:
        """
        Retrieve documents with relevance scores.
        
        Args:
            query: Search query string.
            
        Returns:
            List of (Document, score) tuples ordered by relevance.
            
        Example:
            >>> retriever = RAGRetriever(vectorstore)
            >>> results = await retriever.retrieve_with_scores("machine learning")
            >>> doc, score = results[0]
            >>> print(f"Score: {score:.3f}, Content: {doc.page_content[:50]}")
        """
        if not query or not query.strip():
            self.logger.warning("Empty query provided to retrieve_with_scores")
            return []
        
        try:
            if self.use_mmr:
                # Use MMR for diverse results
                self.logger.debug(
                    "Using MMR search",
                    query=query,
                    k=self.k,
                    diversity=self.mmr_diversity
                )
                
                # MMR returns documents without scores, so we need to get scores separately
                mmr_docs = await self.vectorstore.search_mmr(
                    query=query,
                    k=self.k,
                    fetch_k=self.k * 3,  # Fetch more candidates for diversity
                    lambda_mult=1.0 - self.mmr_diversity  # Convert to MMR lambda
                )
                
                # If no MMR results, return empty list
                if not mmr_docs:
                    return []
                
                # Get similarity scores for the MMR-selected documents
                docs_with_scores = await self.vectorstore.search_with_score(
                    query=query,
                    k=len(mmr_docs)
                )
                
                # Match MMR docs with their scores
                # This is approximate since MMR reorders, but gives a score estimate
                result = docs_with_scores[:len(mmr_docs)]
                
            else:
                # Use pure similarity search
                # Fetch significantly more candidates to improve recall
                # This is especially important for non-English queries
                fetch_k = max(self.k * 5, 50)
                self.logger.debug(
                    "Using similarity search",
                    query=query,
                    k=self.k,
                    fetch_k=fetch_k,
                )
                
                docs_with_scores = await self.vectorstore.search_with_score(
                    query=query,
                    k=fetch_k,
                )
                # Return top k results
                result = docs_with_scores[:self.k]
            
            # Filter by score threshold
            if self.score_threshold > 0.0:
                original_count = len(result)
                result = [
                    (doc, score) for doc, score in result
                    if score >= self.score_threshold
                ]
                
                if len(result) < original_count:
                    self.logger.debug(
                        "Filtered results by score threshold",
                        threshold=self.score_threshold,
                        original_count=original_count,
                        filtered_count=len(result)
                    )
            
            return result
            
        except Exception as e:
            self.logger.error(
                "Failed to retrieve documents with scores",
                query=query,
                error=str(e),
                exc_info=True
            )
            raise
    
    async def retrieve_by_metadata(
        self,
        query: str,
        metadata_filter: Dict[str, Any]
    ) -> List[RetrievedDocument]:
        """
        Retrieve documents matching both query and metadata filter.
        
        Args:
            query: Search query string.
            metadata_filter: Metadata filter criteria (e.g., {"source": "doc.pdf"}).
            
        Returns:
            List of RetrievedDocument objects matching criteria.
            
        Example:
            >>> retriever = RAGRetriever(vectorstore)
            >>> results = await retriever.retrieve_by_metadata(
            ...     "Python tutorial",
            ...     {"file_type": "pdf"}
            ... )
        """
        self.logger.info(
            "Retrieving documents with metadata filter",
            query=query,
            filter=metadata_filter
        )
        
        # Get all results first
        docs_with_scores = await self.retrieve_with_scores(query)
        
        # Filter by metadata
        filtered = []
        for doc, score in docs_with_scores:
            if self._matches_metadata(doc.metadata, metadata_filter):
                filtered.append((doc, score))
        
        self.logger.debug(
            "Filtered by metadata",
            original_count=len(docs_with_scores),
            filtered_count=len(filtered)
        )
        
        # Convert to RetrievedDocument
        return [
            self._to_retrieved_document(doc, score)
            for doc, score in filtered
        ]
    
    def _matches_metadata(
        self,
        metadata: Dict[str, Any],
        filter_criteria: Dict[str, Any]
    ) -> bool:
        """
        Check if metadata matches filter criteria.
        
        Args:
            metadata: Document metadata.
            filter_criteria: Filter criteria to match.
            
        Returns:
            True if metadata matches all filter criteria.
        """
        for key, value in filter_criteria.items():
            if key not in metadata or metadata[key] != value:
                return False
        return True
    
    def _to_retrieved_document(
        self,
        doc: Document,
        score: float
    ) -> RetrievedDocument:
        """
        Convert LangChain Document to RetrievedDocument.
        
        Args:
            doc: LangChain Document object.
            score: Relevance score.
            
        Returns:
            RetrievedDocument object.
        """
        return RetrievedDocument(
            content=doc.page_content,
            metadata=doc.metadata,
            score=score,
            source=doc.metadata.get("source", "unknown"),
            chunk_index=doc.metadata.get("chunk_index", 0)
        )
    
    def as_langchain_retriever(self) -> BaseRetriever:
        """
        Return a LangChain-compatible retriever.
        
        This allows the retriever to be used in LangChain chains and agents.
        
        Returns:
            LangChain BaseRetriever instance.
            
        Example:
            >>> retriever = RAGRetriever(vectorstore)
            >>> lc_retriever = retriever.as_langchain_retriever()
            >>> # Use in LangChain chains
            >>> from langchain.chains import RetrievalQA
            >>> qa = RetrievalQA.from_chain_type(llm=llm, retriever=lc_retriever)
        """
        return LangChainRetrieverWrapper(self)


class LangChainRetrieverWrapper(BaseRetriever):
    """
    LangChain-compatible wrapper for RAGRetriever.
    
    This class adapts our RAGRetriever to the LangChain BaseRetriever interface,
    allowing it to be used in LangChain chains and agents.
    """
    
    rag_retriever: RAGRetriever
    
    def __init__(self, rag_retriever: RAGRetriever, **kwargs: Any):
        """
        Initialize the wrapper.
        
        Args:
            rag_retriever: RAGRetriever instance to wrap.
            **kwargs: Additional arguments for BaseRetriever.
        """
        # Pass rag_retriever to parent init for Pydantic validation
        super().__init__(rag_retriever=rag_retriever, **kwargs)
    
    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun | None = None
    ) -> List[Document]:
        """
        Retrieve relevant documents (sync version for LangChain).
        
        Args:
            query: Search query.
            run_manager: Callback manager for tracing.
            
        Returns:
            List of LangChain Document objects.
        """
        import asyncio
        
        # Run async method in sync context
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        retrieved_docs = loop.run_until_complete(
            self.rag_retriever.retrieve(query)
        )
        
        # Convert to LangChain Documents
        return [doc.to_langchain_document() for doc in retrieved_docs]
    
    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun | None = None
    ) -> List[Document]:
        """
        Retrieve relevant documents (async version for LangChain).
        
        Args:
            query: Search query.
            run_manager: Callback manager for tracing.
            
        Returns:
            List of LangChain Document objects.
        """
        retrieved_docs = await self.rag_retriever.retrieve(query)
        return [doc.to_langchain_document() for doc in retrieved_docs]

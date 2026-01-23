"""
Retrieval module for RAG chatbot.

This module provides document retrieval functionality including:
- Semantic similarity search
- Maximum Marginal Relevance (MMR) for diversity
- Hybrid search (BM25 + semantic)
- Optional reranking with cross-encoder models
"""

from src.retrieval.retriever import (
    RAGRetriever,
    RetrievedDocument,
    LangChainRetrieverWrapper,
)
from src.retrieval.reranker import Reranker, RerankerConfig
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.hybrid import (
    HybridRetriever,
    HybridRAGRetriever,
    HybridSearchConfig,
)

__all__ = [
    "RAGRetriever",
    "RetrievedDocument",
    "LangChainRetrieverWrapper",
    "Reranker",
    "RerankerConfig",
    "BM25Retriever",
    "HybridRetriever",
    "HybridRAGRetriever",
    "HybridSearchConfig",
]
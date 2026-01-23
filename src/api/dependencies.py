"""
FastAPI dependency injection for RAG chatbot components.

This module provides dependency functions that create and inject
RAG system components into API route handlers.
"""

from functools import lru_cache
from typing import Annotated, Union

from fastapi import Depends

from src.api.models import RetrievalMethod
from src.chat.chain import RAGChatChain
from src.chat.memory import ConversationMemory
from src.config.settings import Settings
from src.core.embeddings import get_embeddings
from src.core.llm import get_llm
from src.core.vectorstore import VectorStoreManager
from src.ingestion.chunker import TextChunker
from src.ingestion.loader import DocumentLoaderFactory
from src.ingestion.pipeline import IngestionPipeline
from src.retrieval.retriever import RAGRetriever
from src.retrieval.hybrid import HybridRAGRetriever
from src.utils.logging import get_logger

logger = get_logger(__name__)


@lru_cache()
def get_settings() -> Settings:
    """
    Get application settings (cached).
    
    Returns:
        Application settings instance
    """
    return Settings()


def get_vectorstore(
    settings: Annotated[Settings, Depends(get_settings)]
) -> VectorStoreManager:
    """
    Get vector store manager instance.
    
    Args:
        settings: Application settings
        
    Returns:
        Initialized VectorStoreManager
    """
    embeddings = get_embeddings(settings)
    vectorstore = VectorStoreManager(
        settings=settings.chroma,
        embeddings=embeddings,
    )
    
    logger.info(
        "vectorstore_dependency_created",
        collection=settings.chroma.collection,
    )
    
    return vectorstore


@lru_cache()
def get_conversation_memory() -> ConversationMemory:
    """
    Get conversation memory instance (cached singleton).
    
    The memory must be cached to persist sessions across requests.
    
    Returns:
        Initialized ConversationMemory (singleton)
    """
    settings = get_settings()
    memory = ConversationMemory(
        window_size=settings.retrieval.retrieval_k,
        session_timeout=60,
    )
    
    logger.info(
        "conversation_memory_dependency_created",
        window_size=settings.retrieval.retrieval_k,
    )
    
    return memory


def get_retriever(
    settings: Annotated[Settings, Depends(get_settings)],
    vectorstore: Annotated[VectorStoreManager, Depends(get_vectorstore)],
) -> HybridRAGRetriever:
    """
    Get hybrid RAG retriever instance (default).
    
    Uses hybrid search combining BM25 keyword matching with semantic search
    for better retrieval of documents with specific terms.
    
    Args:
        settings: Application settings
        vectorstore: Vector store manager
        
    Returns:
        Initialized HybridRAGRetriever
    """
    retriever = HybridRAGRetriever(
        vectorstore=vectorstore,
        k=settings.retrieval.retrieval_k,
        semantic_weight=0.5,  # Balance between semantic and keyword
        bm25_weight=0.5,
    )
    
    logger.info(
        "hybrid_retriever_dependency_created",
        k=settings.retrieval.retrieval_k,
        semantic_weight=0.5,
        bm25_weight=0.5,
    )
    
    return retriever


def get_semantic_retriever(
    settings: Annotated[Settings, Depends(get_settings)],
    vectorstore: Annotated[VectorStoreManager, Depends(get_vectorstore)],
) -> RAGRetriever:
    """
    Get pure semantic RAG retriever instance.
    
    Uses only vector similarity search - faster but may miss keyword matches.
    
    Args:
        settings: Application settings
        vectorstore: Vector store manager
        
    Returns:
        Initialized RAGRetriever (semantic only)
    """
    retriever = RAGRetriever(
        vectorstore=vectorstore,
        k=settings.retrieval.retrieval_k,
        use_mmr=settings.retrieval.use_mmr,
    )
    
    logger.info(
        "semantic_retriever_dependency_created",
        k=settings.retrieval.retrieval_k,
        use_mmr=settings.retrieval.use_mmr,
    )
    
    return retriever


def create_retriever_for_method(
    method: RetrievalMethod,
    settings: Settings,
    vectorstore: VectorStoreManager,
    k: int | None = None,
) -> Union[RAGRetriever, HybridRAGRetriever]:
    """
    Factory function to create retriever based on method.
    
    Args:
        method: Retrieval method to use
        settings: Application settings
        vectorstore: Vector store manager
        k: Number of documents to retrieve (overrides settings if provided)
        
    Returns:
        Appropriate retriever instance
    """
    # Use provided k or fall back to settings
    num_docs = k if k is not None else settings.retrieval.retrieval_k
    
    if method == RetrievalMethod.SEMANTIC:
        # Pure semantic search - disable MMR for best relevance matching
        retriever = RAGRetriever(
            vectorstore=vectorstore,
            k=num_docs,
            use_mmr=False,  # Disable MMR for pure semantic similarity
        )
        logger.info(
            "retriever_created",
            method="semantic",
            k=num_docs,
            use_mmr=False,
        )
    elif method == RetrievalMethod.BM25:
        # BM25-only: use hybrid with 0 semantic weight
        retriever = HybridRAGRetriever(
            vectorstore=vectorstore,
            k=num_docs,
            semantic_weight=0.0,
            bm25_weight=1.0,
        )
        logger.info(
            "retriever_created",
            method="bm25",
            k=num_docs,
        )
    else:  # HYBRID (default)
        retriever = HybridRAGRetriever(
            vectorstore=vectorstore,
            k=num_docs,
            semantic_weight=0.5,
            bm25_weight=0.5,
        )
        logger.info(
            "retriever_created",
            method="hybrid",
            k=num_docs,
        )
    
    return retriever


def get_chat_chain(
    settings: Annotated[Settings, Depends(get_settings)],
    retriever: Annotated[RAGRetriever, Depends(get_retriever)],
    memory: Annotated[ConversationMemory, Depends(get_conversation_memory)],
) -> RAGChatChain:
    """
    Get RAG chat chain instance.
    
    Args:
        settings: Application settings
        retriever: RAG retriever
        memory: Conversation memory
        
    Returns:
        Initialized RAGChatChain
    """
    llm = get_llm(settings)
    
    chat_chain = RAGChatChain(
        llm=llm,
        retriever=retriever,
        memory=memory,
        use_mmr=settings.retrieval.use_mmr,
    )
    
    logger.info(
        "chat_chain_dependency_created",
        llm_provider=settings.llm.llm_provider,
        use_mmr=settings.retrieval.use_mmr,
    )
    
    return chat_chain


def get_ingestion_pipeline(
    settings: Annotated[Settings, Depends(get_settings)],
    vectorstore: Annotated[VectorStoreManager, Depends(get_vectorstore)],
) -> IngestionPipeline:
    """
    Get document ingestion pipeline instance.
    
    Args:
        settings: Application settings
        vectorstore: Vector store manager
        
    Returns:
        Initialized IngestionPipeline
    """
    loader_factory = DocumentLoaderFactory()
    chunker = TextChunker(
        chunk_size=settings.chunking.chunk_size,
        chunk_overlap=settings.chunking.chunk_overlap,
    )
    
    pipeline = IngestionPipeline(
        loader_factory=loader_factory,
        chunker=chunker,
        vectorstore=vectorstore,
    )
    
    logger.info(
        "ingestion_pipeline_dependency_created",
        chunk_size=settings.chunking.chunk_size,
        chunk_overlap=settings.chunking.chunk_overlap,
    )
    
    return pipeline


# Type aliases for cleaner route signatures
ChatChainDep = Annotated[RAGChatChain, Depends(get_chat_chain)]
RetrieverDep = Annotated[HybridRAGRetriever, Depends(get_retriever)]
IngestionPipelineDep = Annotated[IngestionPipeline, Depends(get_ingestion_pipeline)]
MemoryDep = Annotated[ConversationMemory, Depends(get_conversation_memory)]
VectorStoreDep = Annotated[VectorStoreManager, Depends(get_vectorstore)]
SettingsDep = Annotated[Settings, Depends(get_settings)]


__all__ = [
    "get_settings",
    "get_vectorstore",
    "get_conversation_memory",
    "get_retriever",
    "get_semantic_retriever",
    "get_chat_chain",
    "get_ingestion_pipeline",
    "create_retriever_for_method",
    "ChatChainDep",
    "RetrieverDep",
    "IngestionPipelineDep",
    "MemoryDep",
    "VectorStoreDep",
    "SettingsDep",
]

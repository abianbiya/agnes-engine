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
    Get vector store manager instance (cached singleton).

    Args:
        settings: Application settings

    Returns:
        Initialized VectorStoreManager
    """
    return _build_vectorstore()


@lru_cache()
def _build_vectorstore() -> VectorStoreManager:
    # ponytail: cached because each instance opens a fresh Chroma HTTP client
    # (heartbeat + get_or_create_collection + count) on first use.
    settings = get_settings()
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


@lru_cache()
def get_llm_cached() -> object:
    """Get the chat model (cached singleton, keeps its HTTP connection pool)."""
    return get_llm(get_settings())


def get_retriever(
    settings: Annotated[Settings, Depends(get_settings)],
    vectorstore: Annotated[VectorStoreManager, Depends(get_vectorstore)],
) -> HybridRAGRetriever:
    """
    Get hybrid RAG retriever instance (cached singleton, default).

    Uses hybrid search combining BM25 keyword matching with semantic search
    for better retrieval of documents with specific terms.

    Args:
        settings: Application settings
        vectorstore: Vector store manager

    Returns:
        Initialized HybridRAGRetriever
    """
    return create_retriever_for_method(
        RetrievalMethod.HYBRID, settings, vectorstore
    )


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
    Factory function to create retriever based on method (cached per method+k).

    Caching matters: a fresh HybridRAGRetriever re-reads the entire Chroma
    collection and rebuilds the BM25 index on its first query.

    Args:
        method: Retrieval method to use
        settings: Application settings
        vectorstore: Vector store manager
        k: Number of documents to retrieve (overrides settings if provided)

    Returns:
        Appropriate retriever instance
    """
    num_docs = k if k is not None else settings.retrieval.retrieval_k
    return _build_retriever(method, num_docs, settings.retrieval.use_mmr, vectorstore)


@lru_cache(maxsize=16)
def _build_retriever(
    method: RetrievalMethod,
    num_docs: int,
    use_mmr: bool,
    vectorstore: VectorStoreManager,
) -> Union[RAGRetriever, HybridRAGRetriever]:
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


def invalidate_retriever_caches() -> None:
    """Drop cached retrievers so BM25 reindexes after ingestion/deletion."""
    _build_retriever.cache_clear()
    _build_chat_chain.cache_clear()
    logger.info("retriever_caches_invalidated")


def get_retriever_for_warmup() -> HybridRAGRetriever:
    """Default retriever, for startup warmup."""
    return create_retriever_for_method(
        RetrievalMethod.HYBRID, get_settings(), _build_vectorstore()
    )


def get_chat_chain(
    settings: Annotated[Settings, Depends(get_settings)],
    retriever: Annotated[RAGRetriever, Depends(get_retriever)],
    memory: Annotated[ConversationMemory, Depends(get_conversation_memory)],
) -> RAGChatChain:
    """
    Get RAG chat chain instance (cached singleton).

    Args:
        settings: Application settings
        retriever: RAG retriever
        memory: Conversation memory

    Returns:
        Initialized RAGChatChain
    """
    return _build_chat_chain(retriever, memory, settings.retrieval.use_mmr)


def get_chat_chain_for_method(
    method: RetrievalMethod,
    settings: Settings,
    vectorstore: VectorStoreManager | None = None,
) -> RAGChatChain:
    """Get a cached chat chain for the given retrieval method."""
    retriever = create_retriever_for_method(
        method, settings, vectorstore or _build_vectorstore()
    )
    return _build_chat_chain(
        retriever, get_conversation_memory(), settings.retrieval.use_mmr
    )


@lru_cache(maxsize=16)
def _build_chat_chain(
    retriever: Union[RAGRetriever, HybridRAGRetriever],
    memory: ConversationMemory,
    use_mmr: bool,
) -> RAGChatChain:
    chat_chain = RAGChatChain(
        llm=get_llm_cached(),
        retriever=retriever,
        memory=memory,
        use_mmr=use_mmr,
    )

    logger.info(
        "chat_chain_dependency_created",
        llm_provider=get_settings().llm.llm_provider,
        use_mmr=use_mmr,
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
    "get_llm_cached",
    "get_retriever",
    "get_semantic_retriever",
    "get_chat_chain",
    "get_chat_chain_for_method",
    "get_ingestion_pipeline",
    "create_retriever_for_method",
    "invalidate_retriever_caches",
    "get_retriever_for_warmup",
    "ChatChainDep",
    "RetrieverDep",
    "IngestionPipelineDep",
    "MemoryDep",
    "VectorStoreDep",
    "SettingsDep",
]

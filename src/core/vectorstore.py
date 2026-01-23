"""
ChromaDB Vector Store Manager for RAG Chatbot.

This module provides a manager class for ChromaDB operations including:
- Connection management and health checks
- Document storage and retrieval
- Search with MMR support
- Collection management
"""

from pathlib import Path
from typing import Any

import chromadb
from chromadb.api.models.Collection import Collection
from chromadb.config import Settings as ChromaSettings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from src.config.settings import ChromaSettings as AppChromaSettings
from src.utils.logging import LoggerMixin, get_logger

logger = get_logger(__name__)


class VectorStoreManager(LoggerMixin):
    """
    Manager class for ChromaDB vector store operations.

    This class handles:
    - Connection to ChromaDB (HTTP or persistent)
    - Document addition with metadata
    - Semantic search with configurable k
    - MMR (Maximum Marginal Relevance) search
    - Collection management
    - Health checks
    """

    def __init__(
        self,
        settings: AppChromaSettings,
        embeddings: Embeddings,
    ) -> None:
        """
        Initialize the Vector Store Manager.

        Args:
            settings: ChromaDB configuration settings.
            embeddings: Embeddings instance for vectorization.

        Example:
            >>> from src.config.settings import get_settings
            >>> from src.core.embeddings import get_embeddings
            >>> settings = get_settings().chroma
            >>> embeddings = get_embeddings()
            >>> manager = VectorStoreManager(settings, embeddings)
        """
        self.settings = settings
        self.embeddings = embeddings
        self._client: chromadb.ClientAPI | None = None
        self._collection: Collection | None = None
        self._vectorstore: Chroma | None = None

        self.logger.info(
            "vector_store_manager_initialized",
            collection=settings.collection,
            host=settings.host,
            port=settings.port,
        )

    @property
    def collection_name(self) -> str:
        """
        Get the collection name.

        Returns:
            The name of the ChromaDB collection.
        """
        return self.settings.collection

    @property
    def client(self) -> chromadb.ClientAPI:
        """
        Get or create ChromaDB client.

        Returns:
            ChromaDB client instance.

        Raises:
            ConnectionError: If unable to connect to ChromaDB.
        """
        if self._client is None:
            self._client = self._create_client()
        return self._client

    @property
    def collection(self) -> Collection:
        """
        Get or create ChromaDB collection.

        Returns:
            ChromaDB collection instance.
        """
        if self._collection is None:
            self._collection = self.client.get_or_create_collection(
                name=self.settings.collection,
                metadata={"hnsw:space": "cosine"},
            )
            self.logger.info(
                "collection_created",
                collection=self.settings.collection,
                count=self._collection.count(),
            )
        return self._collection

    @property
    def vectorstore(self) -> Chroma:
        """
        Get or create LangChain Chroma vectorstore.

        Returns:
            LangChain Chroma instance.
        """
        if self._vectorstore is None:
            self._vectorstore = Chroma(
                client=self.client,
                collection_name=self.settings.collection,
                embedding_function=self.embeddings,
            )
            self.logger.info(
                "vectorstore_created",
                collection=self.settings.collection,
            )
        return self._vectorstore

    def _create_client(self) -> chromadb.ClientAPI:
        """
        Create ChromaDB client based on configuration.

        Returns:
            ChromaDB client instance.

        Raises:
            ConnectionError: If unable to connect to ChromaDB.
        """
        try:
            if self.settings.in_memory:
                self.logger.info("creating_in_memory_client")
                return chromadb.Client()

            # HTTP client for production
            self.logger.info(
                "creating_http_client",
                host=self.settings.host,
                port=self.settings.port,
            )
            client = chromadb.HttpClient(
                host=self.settings.host,
                port=self.settings.port,
            )

            # Test connection
            client.heartbeat()
            self.logger.info("chromadb_connection_successful")
            return client

        except Exception as e:
            self.logger.error(
                "chromadb_connection_failed",
                error=str(e),
                host=self.settings.host,
                port=self.settings.port,
                exc_info=True,
            )
            raise ConnectionError(
                f"Failed to connect to ChromaDB at {self.settings.host}:{self.settings.port}"
            ) from e

    async def connect(self) -> None:
        """
        Establish connection to ChromaDB.

        This method forces client initialization and tests the connection.

        Raises:
            ConnectionError: If connection fails.
        """
        _ = self.client
        _ = self.collection
        self.logger.info("vector_store_connected")

    async def health_check(self) -> bool:
        """
        Check ChromaDB connection health.

        Returns:
            True if healthy, False otherwise.
        """
        try:
            heartbeat = self.client.heartbeat()
            is_healthy = heartbeat is not None
            self.logger.info("health_check", healthy=is_healthy)
            return is_healthy
        except Exception as e:
            self.logger.error("health_check_failed", error=str(e))
            return False

    async def add_documents(
        self,
        documents: list[Document],
        ids: list[str] | None = None,
    ) -> list[str]:
        """
        Add documents to the vector store.

        Args:
            documents: List of LangChain Document objects.
            ids: Optional list of document IDs.

        Returns:
            List of document IDs.

        Example:
            >>> from langchain_core.documents import Document
            >>> docs = [Document(page_content="Hello", metadata={"source": "test"})]
            >>> ids = await manager.add_documents(docs)
        """
        self.logger.info(
            "adding_documents",
            count=len(documents),
            has_ids=ids is not None,
        )

        try:
            if ids:
                result_ids = self.vectorstore.add_documents(documents=documents, ids=ids)
            else:
                result_ids = self.vectorstore.add_documents(documents=documents)

            self.logger.info("documents_added", count=len(result_ids))
            return result_ids

        except Exception as e:
            self.logger.error(
                "add_documents_failed",
                error=str(e),
                count=len(documents),
                exc_info=True,
            )
            raise

    async def search(
        self,
        query: str,
        k: int = 4,
        filter: dict[str, Any] | None = None,
    ) -> list[Document]:
        """
        Perform semantic similarity search.

        Args:
            query: Search query string.
            k: Number of results to return.
            filter: Optional metadata filter.

        Returns:
            List of relevant documents.

        Example:
            >>> results = await manager.search("What is Python?", k=4)
        """
        self.logger.info("searching", query_length=len(query), k=k, has_filter=filter is not None)

        try:
            documents = self.vectorstore.similarity_search(
                query=query,
                k=k,
                filter=filter,
            )

            self.logger.info("search_completed", results_count=len(documents))
            return documents

        except Exception as e:
            self.logger.error("search_failed", error=str(e), exc_info=True)
            raise

    async def search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: dict[str, Any] | None = None,
    ) -> list[tuple[Document, float]]:
        """
        Perform semantic search with relevance scores.

        Args:
            query: Search query string.
            k: Number of results to return.
            filter: Optional metadata filter.

        Returns:
            List of tuples (document, relevance_score).

        Example:
            >>> results = await manager.search_with_score("Python tutorial", k=4)
            >>> for doc, score in results:
            ...     print(f"Score: {score}, Content: {doc.page_content[:50]}")
        """
        self.logger.info(
            "searching_with_score",
            query_length=len(query),
            k=k,
            has_filter=filter is not None,
        )

        try:
            results = self.vectorstore.similarity_search_with_score(
                query=query,
                k=k,
                filter=filter,
            )

            self.logger.info("search_with_score_completed", results_count=len(results))
            return results

        except Exception as e:
            self.logger.error("search_with_score_failed", error=str(e), exc_info=True)
            raise

    async def search_mmr(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: dict[str, Any] | None = None,
    ) -> list[Document]:
        """
        Perform Maximum Marginal Relevance (MMR) search.

        MMR balances relevance and diversity in search results.

        Args:
            query: Search query string.
            k: Number of results to return.
            fetch_k: Number of candidates to fetch before MMR reranking.
            lambda_mult: Diversity factor (0=max diversity, 1=max relevance).
            filter: Optional metadata filter.

        Returns:
            List of diverse, relevant documents.

        Example:
            >>> results = await manager.search_mmr(
            ...     "machine learning",
            ...     k=4,
            ...     lambda_mult=0.5
            ... )
        """
        self.logger.info(
            "searching_mmr",
            query_length=len(query),
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
        )

        try:
            documents = self.vectorstore.max_marginal_relevance_search(
                query=query,
                k=k,
                fetch_k=fetch_k,
                lambda_mult=lambda_mult,
                filter=filter,
            )

            self.logger.info("search_mmr_completed", results_count=len(documents))
            return documents

        except Exception as e:
            self.logger.error("search_mmr_failed", error=str(e), exc_info=True)
            raise

    async def delete_documents(self, ids: list[str]) -> None:
        """
        Delete documents by IDs.

        Args:
            ids: List of document IDs to delete.

        Example:
            >>> await manager.delete_documents(["doc1", "doc2"])
        """
        self.logger.info("deleting_documents", count=len(ids))

        try:
            self.vectorstore.delete(ids=ids)
            self.logger.info("documents_deleted", count=len(ids))

        except Exception as e:
            self.logger.error("delete_documents_failed", error=str(e), exc_info=True)
            raise

    async def delete_collection(self) -> None:
        """
        Delete the entire collection.

        Warning: This operation is irreversible!

        Example:
            >>> await manager.delete_collection()
        """
        self.logger.warning(
            "deleting_collection",
            collection=self.settings.collection,
        )

        try:
            self.client.delete_collection(name=self.settings.collection)
            self._collection = None
            self._vectorstore = None
            self.logger.info("collection_deleted")

        except Exception as e:
            self.logger.error("delete_collection_failed", error=str(e), exc_info=True)
            raise

    def get_collection_count(self) -> int:
        """
        Get the number of documents in the collection.

        Returns:
            Document count.
        """
        count = self.collection.count()
        self.logger.info("collection_count", count=count)
        return count

    def list_documents(self) -> list[dict[str, Any]]:
        """
        List all documents in the collection with metadata.

        Returns a list of unique documents (grouped by source file) with:
        - source: Original file path
        - file_name: File name
        - file_type: File extension
        - chunk_count: Number of chunks from this document
        - sample_content: Preview of the first chunk's content

        Returns:
            List of document info dictionaries.

        Example:
            >>> docs = manager.list_documents()
            >>> for doc in docs:
            ...     print(f"{doc['file_name']}: {doc['chunk_count']} chunks")
        """
        self.logger.info("listing_documents")

        try:
            # Get all documents from collection
            results = self.collection.get(include=["documents", "metadatas"])

            # Handle None or empty results
            metadatas = results.get("metadatas") or []
            doc_contents = results.get("documents") or []

            if not metadatas:
                self.logger.info("list_documents_empty")
                return []

            # Group by source file
            documents: dict[str, dict[str, Any]] = {}

            for i, metadata in enumerate(metadatas):
                if metadata is None:
                    continue
                    
                source = metadata.get("source", "unknown")

                if source not in documents:
                    # Safely extract file name and type
                    try:
                        file_name = metadata.get("file_name") or (Path(source).name if source != "unknown" else "unknown")
                    except Exception:
                        file_name = "unknown"
                    
                    try:
                        file_type = metadata.get("file_type") or (Path(source).suffix.lstrip(".") if source != "unknown" else "unknown")
                    except Exception:
                        file_type = "unknown"
                    
                    documents[source] = {
                        "source": source,
                        "file_name": file_name,
                        "file_type": file_type,
                        "chunk_count": 0,
                        "sample_content": None,
                        "pages": set(),
                    }

                documents[source]["chunk_count"] += 1

                # Track pages for PDFs
                page = metadata.get("page")
                if page is not None:
                    documents[source]["pages"].add(page)

                # Store first chunk's content as sample
                if documents[source]["sample_content"] is None and i < len(doc_contents):
                    content = doc_contents[i]
                    if content:
                        documents[source]["sample_content"] = content[:200] + "..." if len(content) > 200 else content

            # Convert to list and clean up
            doc_list = []
            for doc in documents.values():
                # Convert pages set to sorted list or None
                pages = sorted(doc["pages"]) if doc["pages"] else None
                doc_info = {
                    "source": doc["source"],
                    "file_name": doc["file_name"],
                    "file_type": doc["file_type"],
                    "chunk_count": doc["chunk_count"],
                    "sample_content": doc["sample_content"],
                }
                if pages:
                    doc_info["pages"] = pages
                    doc_info["page_count"] = len(pages)

                doc_list.append(doc_info)

            self.logger.info("list_documents_completed", document_count=len(doc_list))
            return doc_list

        except Exception as e:
            self.logger.error("list_documents_failed", error=str(e), exc_info=True)
            raise

    def as_retriever(self, search_kwargs: dict[str, Any] | None = None) -> VectorStore:
        """
        Get a LangChain retriever interface.

        Args:
            search_kwargs: Optional search parameters (k, filter, etc.).

        Returns:
            LangChain VectorStoreRetriever.

        Example:
            >>> retriever = manager.as_retriever(search_kwargs={"k": 4})
            >>> docs = retriever.get_relevant_documents("What is RAG?")
        """
        return self.vectorstore.as_retriever(search_kwargs=search_kwargs or {"k": 4})


def get_vectorstore(
    settings: AppChromaSettings | None = None,
    embeddings: Embeddings | None = None,
) -> VectorStoreManager:
    """
    Convenience function to get a vector store manager.

    Args:
        settings: Optional ChromaDB settings. If None, loads from environment.
        embeddings: Optional embeddings instance. If None, creates from settings.

    Returns:
        VectorStoreManager instance.

    Example:
        >>> manager = get_vectorstore()
        >>> await manager.connect()
    """
    if settings is None:
        from src.config.settings import get_settings

        settings = get_settings().chroma

    if embeddings is None:
        from src.core.embeddings import get_embeddings

        embeddings = get_embeddings()

    return VectorStoreManager(settings, embeddings)

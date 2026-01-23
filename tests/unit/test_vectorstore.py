"""
Tests for vector store manager module.

Uses pytest for unit tests with mocked ChromaDB client.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from langchain_core.documents import Document

from src.config.settings import ChromaSettings
from src.core.vectorstore import VectorStoreManager, get_vectorstore


@pytest.fixture
def chroma_settings() -> ChromaSettings:
    """Create test ChromaDB settings."""
    return ChromaSettings(
        host="localhost",
        port=8000,
        collection="test_collection",
        in_memory=False,
    )


@pytest.fixture
def mock_embeddings() -> MagicMock:
    """Create mock embeddings."""
    mock = MagicMock()
    mock.embed_documents.return_value = [[0.1] * 1536]
    mock.embed_query.return_value = [0.1] * 1536
    return mock


@pytest.fixture
def mock_chroma_client(mock_chromadb_client: MagicMock) -> MagicMock:
    """Use shared mock ChromaDB client from conftest."""
    return mock_chromadb_client


class TestVectorStoreManager:
    """Tests for VectorStoreManager class."""

    def test_initialization(
        self,
        chroma_settings: ChromaSettings,
        mock_embeddings: MagicMock,
    ) -> None:
        """Test VectorStoreManager initialization."""
        manager = VectorStoreManager(chroma_settings, mock_embeddings)

        assert manager.settings == chroma_settings
        assert manager.embeddings == mock_embeddings
        assert manager._client is None
        assert manager._collection is None

    def test_client_property_creates_client(
        self,
        chroma_settings: ChromaSettings,
        mock_embeddings: MagicMock,
    ) -> None:
        """Test that client property creates client on first access."""
        manager = VectorStoreManager(chroma_settings, mock_embeddings)

        with patch.object(manager, "_create_client") as mock_create:
            mock_create.return_value = MagicMock()
            client = manager.client

            mock_create.assert_called_once()
            assert client is not None

    def test_collection_property_creates_collection(
        self,
        chroma_settings: ChromaSettings,
        mock_embeddings: MagicMock,
        mock_chroma_client: MagicMock,
    ) -> None:
        """Test that collection property creates collection on first access."""
        manager = VectorStoreManager(chroma_settings, mock_embeddings)
        manager._client = mock_chroma_client

        collection = manager.collection

        mock_chroma_client.get_or_create_collection.assert_called_once_with(
            name="test_collection",
            metadata={"hnsw:space": "cosine"},
        )
        assert collection is not None

    def test_create_client_http(
        self,
        chroma_settings: ChromaSettings,
        mock_embeddings: MagicMock,
    ) -> None:
        """Test creating HTTP ChromaDB client."""
        manager = VectorStoreManager(chroma_settings, mock_embeddings)

        with patch("src.core.vectorstore.chromadb.HttpClient") as mock_http:
            mock_client = MagicMock()
            mock_client.heartbeat.return_value = True
            mock_http.return_value = mock_client

            client = manager._create_client()

            mock_http.assert_called_once_with(
                host="localhost",
                port=8000,
            )
            mock_client.heartbeat.assert_called_once()
            assert client == mock_client

    def test_create_client_in_memory(
        self,
        mock_embeddings: MagicMock,
    ) -> None:
        """Test creating in-memory ChromaDB client."""
        settings = ChromaSettings(in_memory=True)
        manager = VectorStoreManager(settings, mock_embeddings)

        with patch("src.core.vectorstore.chromadb.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client

            client = manager._create_client()

            mock_client_class.assert_called_once()
            assert client == mock_client

    def test_create_client_connection_error(
        self,
        chroma_settings: ChromaSettings,
        mock_embeddings: MagicMock,
    ) -> None:
        """Test connection error handling."""
        manager = VectorStoreManager(chroma_settings, mock_embeddings)

        with patch("src.core.vectorstore.chromadb.HttpClient") as mock_http:
            mock_http.side_effect = Exception("Connection failed")

            with pytest.raises(ConnectionError, match="Failed to connect to ChromaDB"):
                manager._create_client()

    @pytest.mark.asyncio
    async def test_connect(
        self,
        chroma_settings: ChromaSettings,
        mock_embeddings: MagicMock,
        mock_chroma_client: MagicMock,
    ) -> None:
        """Test connect method."""
        manager = VectorStoreManager(chroma_settings, mock_embeddings)
        manager._client = mock_chroma_client

        await manager.connect()

        # Should access client and collection
        assert manager._client is not None
        assert manager._collection is not None

    @pytest.mark.asyncio
    async def test_health_check_success(
        self,
        chroma_settings: ChromaSettings,
        mock_embeddings: MagicMock,
        mock_chroma_client: MagicMock,
    ) -> None:
        """Test successful health check."""
        manager = VectorStoreManager(chroma_settings, mock_embeddings)
        manager._client = mock_chroma_client

        is_healthy = await manager.health_check()

        assert is_healthy is True
        mock_chroma_client.heartbeat.assert_called_once()

    @pytest.mark.asyncio
    async def test_health_check_failure(
        self,
        chroma_settings: ChromaSettings,
        mock_embeddings: MagicMock,
    ) -> None:
        """Test health check failure."""
        manager = VectorStoreManager(chroma_settings, mock_embeddings)

        mock_client = MagicMock()
        mock_client.heartbeat.side_effect = Exception("Connection lost")
        manager._client = mock_client

        is_healthy = await manager.health_check()

        assert is_healthy is False

    @pytest.mark.asyncio
    async def test_add_documents(
        self,
        chroma_settings: ChromaSettings,
        mock_embeddings: MagicMock,
    ) -> None:
        """Test adding documents."""
        manager = VectorStoreManager(chroma_settings, mock_embeddings)

        mock_vectorstore = MagicMock()
        mock_vectorstore.add_documents.return_value = ["id1", "id2"]
        manager._vectorstore = mock_vectorstore

        documents = [
            Document(page_content="Test 1", metadata={"source": "test"}),
            Document(page_content="Test 2", metadata={"source": "test"}),
        ]

        ids = await manager.add_documents(documents)

        mock_vectorstore.add_documents.assert_called_once_with(documents=documents)
        assert ids == ["id1", "id2"]

    @pytest.mark.asyncio
    async def test_add_documents_with_ids(
        self,
        chroma_settings: ChromaSettings,
        mock_embeddings: MagicMock,
    ) -> None:
        """Test adding documents with custom IDs."""
        manager = VectorStoreManager(chroma_settings, mock_embeddings)

        mock_vectorstore = MagicMock()
        mock_vectorstore.add_documents.return_value = ["custom1", "custom2"]
        manager._vectorstore = mock_vectorstore

        documents = [
            Document(page_content="Test 1", metadata={"source": "test"}),
        ]
        custom_ids = ["custom1"]

        ids = await manager.add_documents(documents, ids=custom_ids)

        mock_vectorstore.add_documents.assert_called_once_with(
            documents=documents,
            ids=custom_ids,
        )
        assert ids == ["custom1", "custom2"]

    @pytest.mark.asyncio
    async def test_search(
        self,
        chroma_settings: ChromaSettings,
        mock_embeddings: MagicMock,
    ) -> None:
        """Test similarity search."""
        manager = VectorStoreManager(chroma_settings, mock_embeddings)

        mock_vectorstore = MagicMock()
        expected_docs = [
            Document(page_content="Result 1", metadata={"source": "test"}),
            Document(page_content="Result 2", metadata={"source": "test"}),
        ]
        mock_vectorstore.similarity_search.return_value = expected_docs
        manager._vectorstore = mock_vectorstore

        results = await manager.search("test query", k=4)

        mock_vectorstore.similarity_search.assert_called_once_with(
            query="test query",
            k=4,
            filter=None,
        )
        assert results == expected_docs

    @pytest.mark.asyncio
    async def test_search_with_score(
        self,
        chroma_settings: ChromaSettings,
        mock_embeddings: MagicMock,
    ) -> None:
        """Test similarity search with scores."""
        manager = VectorStoreManager(chroma_settings, mock_embeddings)

        mock_vectorstore = MagicMock()
        expected_results = [
            (Document(page_content="Result 1"), 0.9),
            (Document(page_content="Result 2"), 0.8),
        ]
        mock_vectorstore.similarity_search_with_score.return_value = expected_results
        manager._vectorstore = mock_vectorstore

        results = await manager.search_with_score("test query", k=4)

        mock_vectorstore.similarity_search_with_score.assert_called_once_with(
            query="test query",
            k=4,
            filter=None,
        )
        assert results == expected_results

    @pytest.mark.asyncio
    async def test_search_mmr(
        self,
        chroma_settings: ChromaSettings,
        mock_embeddings: MagicMock,
    ) -> None:
        """Test MMR search."""
        manager = VectorStoreManager(chroma_settings, mock_embeddings)

        mock_vectorstore = MagicMock()
        expected_docs = [
            Document(page_content="Result 1", metadata={"source": "test"}),
        ]
        mock_vectorstore.max_marginal_relevance_search.return_value = expected_docs
        manager._vectorstore = mock_vectorstore

        results = await manager.search_mmr(
            "test query",
            k=4,
            fetch_k=20,
            lambda_mult=0.5,
        )

        mock_vectorstore.max_marginal_relevance_search.assert_called_once_with(
            query="test query",
            k=4,
            fetch_k=20,
            lambda_mult=0.5,
            filter=None,
        )
        assert results == expected_docs

    @pytest.mark.asyncio
    async def test_delete_documents(
        self,
        chroma_settings: ChromaSettings,
        mock_embeddings: MagicMock,
    ) -> None:
        """Test deleting documents."""
        manager = VectorStoreManager(chroma_settings, mock_embeddings)

        mock_vectorstore = MagicMock()
        manager._vectorstore = mock_vectorstore

        await manager.delete_documents(["id1", "id2"])

        mock_vectorstore.delete.assert_called_once_with(ids=["id1", "id2"])

    @pytest.mark.asyncio
    async def test_delete_collection(
        self,
        chroma_settings: ChromaSettings,
        mock_embeddings: MagicMock,
        mock_chroma_client: MagicMock,
    ) -> None:
        """Test deleting entire collection."""
        manager = VectorStoreManager(chroma_settings, mock_embeddings)
        manager._client = mock_chroma_client

        await manager.delete_collection()

        mock_chroma_client.delete_collection.assert_called_once_with(
            name="test_collection"
        )
        assert manager._collection is None
        assert manager._vectorstore is None

    def test_get_collection_count(
        self,
        chroma_settings: ChromaSettings,
        mock_embeddings: MagicMock,
        mock_chroma_client: MagicMock,
    ) -> None:
        """Test getting collection count."""
        manager = VectorStoreManager(chroma_settings, mock_embeddings)
        manager._client = mock_chroma_client

        count = manager.get_collection_count()

        assert count == 0  # Mock returns 0

    def test_as_retriever(
        self,
        chroma_settings: ChromaSettings,
        mock_embeddings: MagicMock,
    ) -> None:
        """Test getting LangChain retriever."""
        manager = VectorStoreManager(chroma_settings, mock_embeddings)

        mock_vectorstore = MagicMock()
        manager._vectorstore = mock_vectorstore

        retriever = manager.as_retriever(search_kwargs={"k": 4})

        mock_vectorstore.as_retriever.assert_called_once_with(
            search_kwargs={"k": 4}
        )


class TestGetVectorstore:
    """Tests for get_vectorstore convenience function."""

    def test_get_vectorstore_with_settings(
        self,
        chroma_settings: ChromaSettings,
        mock_embeddings: MagicMock,
    ) -> None:
        """Test get_vectorstore with provided settings."""
        manager = get_vectorstore(settings=chroma_settings, embeddings=mock_embeddings)

        assert isinstance(manager, VectorStoreManager)
        assert manager.settings == chroma_settings
        assert manager.embeddings == mock_embeddings

    def test_get_vectorstore_without_settings(self) -> None:
        """Test get_vectorstore loads settings from environment."""
        with patch("src.config.settings.get_settings") as mock_get_settings:
            mock_settings = MagicMock()
            mock_settings.chroma = ChromaSettings()
            mock_get_settings.return_value = mock_settings

            with patch("src.core.embeddings.get_embeddings") as mock_get_embeddings:
                mock_embeddings = MagicMock()
                mock_get_embeddings.return_value = mock_embeddings

                manager = get_vectorstore()

                assert isinstance(manager, VectorStoreManager)
                mock_get_settings.assert_called_once()
                mock_get_embeddings.assert_called_once()


@pytest.mark.unit
class TestVectorStoreManagerUnit:
    """Comprehensive unit tests for VectorStoreManager."""

    @pytest.mark.asyncio
    async def test_search_with_filter(
        self,
        chroma_settings: ChromaSettings,
        mock_embeddings: MagicMock,
    ) -> None:
        """Test search with metadata filter."""
        manager = VectorStoreManager(chroma_settings, mock_embeddings)

        mock_vectorstore = MagicMock()
        mock_vectorstore.similarity_search.return_value = []
        manager._vectorstore = mock_vectorstore

        filter_dict = {"source": "test.pdf"}
        await manager.search("query", k=4, filter=filter_dict)

        _, kwargs = mock_vectorstore.similarity_search.call_args
        assert kwargs["filter"] == filter_dict

    @pytest.mark.asyncio
    async def test_error_handling_on_add_documents(
        self,
        chroma_settings: ChromaSettings,
        mock_embeddings: MagicMock,
    ) -> None:
        """Test error handling when adding documents fails."""
        manager = VectorStoreManager(chroma_settings, mock_embeddings)

        mock_vectorstore = MagicMock()
        mock_vectorstore.add_documents.side_effect = Exception("Storage error")
        manager._vectorstore = mock_vectorstore

        documents = [Document(page_content="Test")]

        with pytest.raises(Exception, match="Storage error"):
            await manager.add_documents(documents)

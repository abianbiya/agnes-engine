"""
Unit tests for API routes module.

Tests all FastAPI endpoints with mocked dependencies.
"""

import io
from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, List
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest
from fastapi import status
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api.dependencies import (
    get_chat_chain,
    get_conversation_memory,
    get_ingestion_pipeline,
    get_retriever,
    get_settings,
    get_vectorstore,
)


# Mock models matching internal implementations
@dataclass
class MockSourceDocument:
    """Mock SourceDocument from chat.chain module."""
    filename: str
    page: int | None = None
    section: str | None = None
    relevance_score: float = 0.9
    content_preview: str | None = None


@dataclass
class MockChatResponse:
    """Mock ChatResponse from chat.chain module."""
    answer: str
    sources: List[MockSourceDocument]
    session_id: str
    metadata: Dict[str, Any] | None = None


@dataclass
class MockRetrievedDocument:
    """Mock RetrievedDocument from retrieval.retriever module."""
    content: str
    metadata: Dict[str, Any]
    score: float
    source: str
    chunk_index: int = 0


@dataclass
class MockIngestionResult:
    """Mock IngestionResult from ingestion.pipeline module."""
    success: bool
    file_path: str
    num_documents_loaded: int = 0
    num_chunks_created: int = 0
    num_chunks_stored: int = 0
    error: str | None = None


# Fixtures
@pytest.fixture
def mock_chat_chain():
    """Create mock RAG chat chain."""
    chain = AsyncMock()
    chain.chat = AsyncMock(return_value=MockChatResponse(
        answer="Test answer",
        sources=[MockSourceDocument(filename="test.pdf")],
        session_id="test-session-123",
    ))
    
    # Create an async generator factory for streaming
    def mock_stream_factory(**kwargs):
        async def mock_stream():
            for chunk in ["chunk1", "chunk2", "chunk3"]:
                yield chunk
        return mock_stream()
    
    chain.stream_chat = Mock(side_effect=mock_stream_factory)
    return chain


@pytest.fixture
def mock_retriever():
    """Create mock RAG retriever."""
    retriever = Mock()
    retriever.search = Mock(return_value=[
        MockRetrievedDocument(
            content="Test content",
            metadata={"source": "test.pdf", "page": 1},
            score=0.95,
            source="test.pdf",
        )
    ])
    retriever.mmr_search = Mock(return_value=[
        MockRetrievedDocument(
            content="Test content MMR",
            metadata={"source": "test2.pdf", "page": 2},
            score=0.90,
            source="test2.pdf",
        )
    ])
    return retriever


@pytest.fixture
def mock_pipeline():
    """Create mock ingestion pipeline."""
    pipeline = AsyncMock()
    pipeline.ingest_file = AsyncMock(return_value=MockIngestionResult(
        success=True,
        file_path="/data/test.pdf",
        num_documents_loaded=1,
        num_chunks_created=10,
        num_chunks_stored=10,
    ))
    return pipeline


@pytest.fixture
def mock_memory():
    """Create mock conversation memory."""
    memory = Mock()
    memory.list_sessions = Mock(return_value=["session-1", "session-2"])
    memory.clear_session = Mock()
    return memory


@pytest.fixture
def mock_vectorstore():
    """Create mock vectorstore."""
    vectorstore = Mock()
    vectorstore.health_check = Mock(return_value=True)
    return vectorstore


@pytest.fixture
def mock_settings():
    """Create mock settings."""
    settings = Mock()
    settings.api_version = "1.0.0"
    return settings


@pytest.fixture
def client(
    mock_chat_chain,
    mock_retriever,
    mock_pipeline,
    mock_memory,
    mock_vectorstore,
    mock_settings,
):
    """Create test client with mocked dependencies."""
    app = create_app()
    
    # Override dependencies
    app.dependency_overrides[get_chat_chain] = lambda: mock_chat_chain
    app.dependency_overrides[get_retriever] = lambda: mock_retriever
    app.dependency_overrides[get_ingestion_pipeline] = lambda: mock_pipeline
    app.dependency_overrides[get_conversation_memory] = lambda: mock_memory
    app.dependency_overrides[get_vectorstore] = lambda: mock_vectorstore
    app.dependency_overrides[get_settings] = lambda: mock_settings
    
    return TestClient(app)


# Test Chat Endpoint
class TestChatEndpoint:
    """Test POST /api/v1/chat endpoint."""

    def test_chat_success(self, client):
        """Should successfully process chat request."""
        response = client.post(
            "/api/v1/chat",
            json={"question": "What is AI?"}
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "answer" in data
        assert data["answer"] == "Test answer"
        assert "sources" in data
        assert len(data["sources"]) == 1
        assert data["sources"][0]["filename"] == "test.pdf"
        assert "session_id" in data
        assert data["session_id"] == "test-session-123"

    def test_chat_with_session_id(self, client, mock_chat_chain):
        """Should process chat with existing session ID."""
        response = client.post(
            "/api/v1/chat",
            json={
                "question": "Follow-up question",
                "session_id": "existing-session"
            }
        )
        
        assert response.status_code == status.HTTP_200_OK
        
        # Verify chat_chain.chat was called with session_id
        mock_chat_chain.chat.assert_called_once()
        call_kwargs = mock_chat_chain.chat.call_args[1]
        assert call_kwargs["question"] == "Follow-up question"
        assert call_kwargs["session_id"] == "existing-session"

    def test_chat_empty_question(self, client):
        """Should reject empty question."""
        response = client.post(
            "/api/v1/chat",
            json={"question": ""}
        )
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_chat_whitespace_question(self, client):
        """Should reject whitespace-only question."""
        response = client.post(
            "/api/v1/chat",
            json={"question": "   "}
        )
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_chat_missing_question(self, client):
        """Should reject request without question."""
        response = client.post(
            "/api/v1/chat",
            json={}
        )
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_chat_error_handling(self, client, mock_chat_chain):
        """Should handle errors gracefully."""
        mock_chat_chain.chat.side_effect = Exception("Chat processing failed")
        
        response = client.post(
            "/api/v1/chat",
            json={"question": "Test question"}
        )
        
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        data = response.json()
        assert "detail" in data


# Test Chat Stream Endpoint
class TestChatStreamEndpoint:
    """Test POST /api/v1/chat/stream endpoint."""

    def test_chat_stream_success(self, client):
        """Should successfully stream chat response."""
        response = client.post(
            "/api/v1/chat/stream",
            json={"question": "What is AI?"}
        )
        
        assert response.status_code == status.HTTP_200_OK
        assert response.headers["content-type"] == "text/event-stream; charset=utf-8"
        assert "no-cache" in response.headers["cache-control"]
        
        # Check SSE format
        content = response.text
        assert "data:" in content
        assert "chunk1" in content
        assert "chunk2" in content
        assert "chunk3" in content

    def test_chat_stream_with_session(self, client, mock_chat_chain):
        """Should stream with session ID."""
        response = client.post(
            "/api/v1/chat/stream",
            json={
                "question": "Follow-up question",
                "session_id": "test-session"
            }
        )
        
        assert response.status_code == status.HTTP_200_OK
        
        # Verify stream_chat was called with session_id
        mock_chat_chain.stream_chat.assert_called_once()
        call_kwargs = mock_chat_chain.stream_chat.call_args[1]
        assert call_kwargs["session_id"] == "test-session"

    def test_chat_stream_error_in_stream(self, client, mock_chat_chain):
        """Should handle errors during streaming."""
        async def error_gen(**kwargs):
            async def gen():
                yield "chunk1"
                raise Exception("Stream error")
            return gen()
        
        mock_chat_chain.stream_chat = Mock(side_effect=error_gen)
        
        response = client.post(
            "/api/v1/chat/stream",
            json={"question": "Test"}
        )
        
        # Should still return 200 (streaming started)
        assert response.status_code == status.HTTP_200_OK
        # Error should be in stream content
        content = response.text
        assert "error" in content.lower()


# Test Search Endpoint
class TestSearchEndpoint:
    """Test POST /api/v1/search endpoint."""

    def test_search_success(self, client, mock_retriever):
        """Should successfully search documents."""
        response = client.post(
            "/api/v1/search",
            json={"query": "test query"}
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "results" in data
        assert "count" in data
        assert "query" in data
        assert data["query"] == "test query"
        assert data["count"] == 1
        assert len(data["results"]) == 1
        
        result = data["results"][0]
        assert result["content"] == "Test content"
        assert result["score"] == 0.95
        assert result["source"] == "test.pdf"

    def test_search_with_limit(self, client, mock_retriever):
        """Should respect limit parameter."""
        response = client.post(
            "/api/v1/search",
            json={
                "query": "test query",
                "limit": 10
            }
        )
        
        assert response.status_code == status.HTTP_200_OK
        
        # Verify search was called with correct limit
        mock_retriever.search.assert_called_once()
        call_kwargs = mock_retriever.search.call_args[1]
        assert call_kwargs["k"] == 10

    def test_search_with_mmr(self, client, mock_retriever):
        """Should use MMR when requested."""
        response = client.post(
            "/api/v1/search",
            json={
                "query": "test query",
                "use_mmr": True
            }
        )
        
        assert response.status_code == status.HTTP_200_OK
        
        # Verify mmr_search was called instead of search
        mock_retriever.mmr_search.assert_called_once()
        mock_retriever.search.assert_not_called()
        
        data = response.json()
        assert data["results"][0]["content"] == "Test content MMR"

    def test_search_empty_query(self, client):
        """Should reject empty query."""
        response = client.post(
            "/api/v1/search",
            json={"query": ""}
        )
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_search_invalid_limit(self, client):
        """Should reject invalid limit values."""
        # Limit too high
        response = client.post(
            "/api/v1/search",
            json={"query": "test", "limit": 21}
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        
        # Limit too low
        response = client.post(
            "/api/v1/search",
            json={"query": "test", "limit": 0}
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_search_error_handling(self, client, mock_retriever):
        """Should handle search errors gracefully."""
        mock_retriever.search.side_effect = Exception("Search failed")
        
        response = client.post(
            "/api/v1/search",
            json={"query": "test"}
        )
        
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR


# Test Ingest Endpoint
class TestIngestEndpoint:
    """Test POST /api/v1/ingest endpoint."""

    def test_ingest_pdf_success(self, client, mock_pipeline):
        """Should successfully ingest PDF file."""
        # Create mock PDF file
        file_content = b"Mock PDF content"
        files = {
            "file": ("test.pdf", io.BytesIO(file_content), "application/pdf")
        }
        
        response = client.post("/api/v1/ingest", files=files)
        
        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()
        assert data["success"] is True
        assert data["filename"] == "test.pdf"
        assert data["file_type"] == "pdf"
        assert data["chunks_created"] == 10
        assert data["chunks_stored"] == 10
        assert data["documents_loaded"] == 1

    def test_ingest_txt_success(self, client, mock_pipeline):
        """Should successfully ingest TXT file."""
        file_content = b"Mock text content"
        files = {
            "file": ("test.txt", io.BytesIO(file_content), "text/plain")
        }
        
        response = client.post("/api/v1/ingest", files=files)
        
        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()
        assert data["success"] is True
        assert data["file_type"] == "txt"

    def test_ingest_md_success(self, client, mock_pipeline):
        """Should successfully ingest Markdown file."""
        file_content = b"# Mock Markdown"
        files = {
            "file": ("test.md", io.BytesIO(file_content), "text/markdown")
        }
        
        response = client.post("/api/v1/ingest", files=files)
        
        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()
        assert data["success"] is True
        assert data["file_type"] == "md"

    def test_ingest_invalid_file_type(self, client):
        """Should reject unsupported file types."""
        file_content = b"Mock content"
        files = {
            "file": ("test.docx", io.BytesIO(file_content), "application/vnd.openxmlformats-officedocument.wordprocessingml.document")
        }
        
        response = client.post("/api/v1/ingest", files=files)
        
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        data = response.json()
        assert "detail" in data
        # Detail is now a string representation of the details dict
        assert ".docx" in data["detail"]
        assert "filename" in data["detail"]

    def test_ingest_no_file(self, client):
        """Should reject request without file."""
        response = client.post("/api/v1/ingest")
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_ingest_error_handling(self, client, mock_pipeline):
        """Should handle ingestion errors gracefully."""
        mock_pipeline.ingest_file.side_effect = Exception("Ingestion failed")
        
        file_content = b"Mock content"
        files = {
            "file": ("test.pdf", io.BytesIO(file_content), "application/pdf")
        }
        
        response = client.post("/api/v1/ingest", files=files)
        
        # Now returns 422 because DocumentIngestionError maps to 422
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_ingest_failed_result(self, client, mock_pipeline):
        """Should return failed response when ingestion fails."""
        mock_pipeline.ingest_file.return_value = MockIngestionResult(
            success=False,
            file_path="/data/test.pdf",
            error="File parsing failed",
        )
        
        file_content = b"Mock content"
        files = {
            "file": ("test.pdf", io.BytesIO(file_content), "application/pdf")
        }
        
        response = client.post("/api/v1/ingest", files=files)
        
        # Should still return 201 but with success=False
        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()
        assert data["success"] is False
        assert data["error_message"] == "File parsing failed"


# Test Health Endpoint
class TestHealthEndpoint:
    """Test GET /api/v1/health endpoint."""

    def test_health_check_healthy(self, client, mock_vectorstore):
        """Should return healthy status when all services are up."""
        mock_vectorstore.health_check.return_value = True
        
        response = client.get("/api/v1/health")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["status"] == "healthy"
        assert data["services"]["vectorstore"] is True
        assert data["services"]["llm"] is True
        assert data["services"]["embeddings"] is True
        assert "version" in data

    def test_health_check_degraded(self, client, mock_vectorstore):
        """Should return degraded status when vectorstore is down."""
        mock_vectorstore.health_check.return_value = False
        
        response = client.get("/api/v1/health")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["status"] == "degraded"
        assert data["services"]["vectorstore"] is False

    def test_health_check_exception(self, client, mock_vectorstore):
        """Should handle health check exceptions."""
        mock_vectorstore.health_check.side_effect = Exception("Health check failed")
        
        response = client.get("/api/v1/health")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["status"] == "degraded"
        assert data["services"]["vectorstore"] is False


# Test Session Delete Endpoint
class TestSessionDeleteEndpoint:
    """Test DELETE /api/v1/session/{session_id} endpoint."""

    def test_delete_session_success(self, client, mock_memory):
        """Should successfully delete existing session."""
        mock_memory.list_sessions.return_value = ["test-session-123"]
        
        response = client.delete("/api/v1/session/test-session-123")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["message"] == "Session cleared successfully"
        assert data["session_id"] == "test-session-123"
        
        # Verify clear_session was called
        mock_memory.clear_session.assert_called_once_with("test-session-123")

    def test_delete_session_not_found(self, client, mock_memory):
        """Should return 404 for non-existent session."""
        mock_memory.list_sessions.return_value = ["other-session"]
        
        response = client.delete("/api/v1/session/non-existent")
        
        assert response.status_code == status.HTTP_404_NOT_FOUND
        data = response.json()
        assert "detail" in data
        # Detail is now a string representation of the details dict
        assert "session_id" in data["detail"]
        assert "non-existent" in data["detail"]
        
        # Verify clear_session was NOT called
        mock_memory.clear_session.assert_not_called()

    def test_delete_session_error_handling(self, client, mock_memory):
        """Should handle deletion errors gracefully."""
        mock_memory.list_sessions.return_value = ["test-session"]
        mock_memory.clear_session.side_effect = Exception("Delete failed")
        
        response = client.delete("/api/v1/session/test-session")
        
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR


# Test Root Endpoint
class TestRootEndpoint:
    """Test GET / endpoint."""

    def test_root_endpoint(self, client):
        """Should return API info at root endpoint."""
        response = client.get("/")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "message" in data or "name" in data or "version" in data

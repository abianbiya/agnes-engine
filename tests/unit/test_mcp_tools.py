"""
Unit tests for MCP tools module.

Tests MCP tool definitions, schemas, and MCPToolHandler with mocked components.
"""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from mcp.types import Tool

from src.chat.chain import ChatResponse, SourceDocument
from src.ingestion.pipeline import IngestionResult
from src.mcp.tools import (
    DOCUMENT_SEARCH_TOOL,
    INGEST_DOCUMENT_TOOL,
    LIST_DOCUMENTS_TOOL,
    MCPToolHandler,
    RAG_CHAT_TOOL,
    TOOLS,
)
from src.retrieval.retriever import RetrievedDocument


class TestToolDefinitions:
    """Test MCP tool schema definitions."""
    
    def test_rag_chat_tool_is_tool_type(self):
        """RAG chat tool should be a Tool instance."""
        assert isinstance(RAG_CHAT_TOOL, Tool)
    
    def test_rag_chat_tool_has_correct_name(self):
        """RAG chat tool should have correct name."""
        assert RAG_CHAT_TOOL.name == "rag_chat"
    
    def test_rag_chat_tool_has_description(self):
        """RAG chat tool should have a description."""
        assert isinstance(RAG_CHAT_TOOL.description, str)
        assert len(RAG_CHAT_TOOL.description) > 0
        assert "RAG" in RAG_CHAT_TOOL.description
    
    def test_rag_chat_tool_has_input_schema(self):
        """RAG chat tool should have input schema."""
        schema = RAG_CHAT_TOOL.inputSchema
        assert schema["type"] == "object"
        assert "question" in schema["properties"]
        assert "session_id" in schema["properties"]
        assert "question" in schema["required"]
    
    def test_document_search_tool_is_tool_type(self):
        """Document search tool should be a Tool instance."""
        assert isinstance(DOCUMENT_SEARCH_TOOL, Tool)
    
    def test_document_search_tool_has_correct_name(self):
        """Document search tool should have correct name."""
        assert DOCUMENT_SEARCH_TOOL.name == "document_search"
    
    def test_document_search_tool_has_input_schema(self):
        """Document search tool should have input schema."""
        schema = DOCUMENT_SEARCH_TOOL.inputSchema
        assert schema["type"] == "object"
        assert "query" in schema["properties"]
        assert "limit" in schema["properties"]
        assert "use_mmr" in schema["properties"]
        assert "query" in schema["required"]
    
    def test_document_search_tool_has_valid_defaults(self):
        """Document search tool should have valid default values."""
        schema = DOCUMENT_SEARCH_TOOL.inputSchema
        assert schema["properties"]["limit"]["default"] == 4
        assert schema["properties"]["use_mmr"]["default"] is False
        assert schema["properties"]["limit"]["minimum"] == 1
        assert schema["properties"]["limit"]["maximum"] == 20
    
    def test_ingest_document_tool_is_tool_type(self):
        """Ingest document tool should be a Tool instance."""
        assert isinstance(INGEST_DOCUMENT_TOOL, Tool)
    
    def test_ingest_document_tool_has_correct_name(self):
        """Ingest document tool should have correct name."""
        assert INGEST_DOCUMENT_TOOL.name == "ingest_document"
    
    def test_ingest_document_tool_has_input_schema(self):
        """Ingest document tool should have input schema."""
        schema = INGEST_DOCUMENT_TOOL.inputSchema
        assert schema["type"] == "object"
        assert "file_path" in schema["properties"]
        assert "file_path" in schema["required"]
    
    def test_list_documents_tool_is_tool_type(self):
        """List documents tool should be a Tool instance."""
        assert isinstance(LIST_DOCUMENTS_TOOL, Tool)
    
    def test_list_documents_tool_has_correct_name(self):
        """List documents tool should have correct name."""
        assert LIST_DOCUMENTS_TOOL.name == "list_documents"
    
    def test_list_documents_tool_has_input_schema(self):
        """List documents tool should have input schema with no required fields."""
        schema = LIST_DOCUMENTS_TOOL.inputSchema
        assert schema["type"] == "object"
        assert len(schema["required"]) == 0
    
    def test_tools_list_contains_all_tools(self):
        """TOOLS list should contain all 4 tool definitions."""
        assert len(TOOLS) == 4
        assert RAG_CHAT_TOOL in TOOLS
        assert DOCUMENT_SEARCH_TOOL in TOOLS
        assert INGEST_DOCUMENT_TOOL in TOOLS
        assert LIST_DOCUMENTS_TOOL in TOOLS


@pytest.fixture
def mock_chat_chain():
    """Create a mock RAG chat chain for testing."""
    chain = AsyncMock()
    
    # Mock memory
    chain.memory = MagicMock()
    chain.memory.create_session.return_value = "test-session-123"
    chain.memory.get_session_history.return_value = []
    
    # Mock chat response
    mock_response = ChatResponse(
        answer="This is a test answer based on the documents.",
        sources=[
            SourceDocument(
                filename="test.pdf",
                page=1,
                relevance_score=0.95,
                content_preview="Test content preview...",
            )
        ],
        session_id="test-session-123",
    )
    chain.chat = AsyncMock(return_value=mock_response)
    
    return chain


@pytest.fixture
def mock_retriever():
    """Create a mock retriever for testing."""
    retriever = MagicMock()
    
    # Mock search results
    mock_docs = [
        RetrievedDocument(
            content="This is the first relevant document.",
            metadata={"source": "doc1.txt", "page": 1},
            score=0.95,
            source="doc1.txt",
        ),
        RetrievedDocument(
            content="This is the second relevant document.",
            metadata={"source": "doc2.txt", "page": 2},
            score=0.88,
            source="doc2.txt",
        ),
    ]
    
    retriever.search.return_value = mock_docs
    retriever.mmr_search.return_value = mock_docs
    
    return retriever


@pytest.fixture
def mock_ingestion_pipeline():
    """Create a mock ingestion pipeline for testing."""
    pipeline = AsyncMock()
    
    # Mock ingestion result matching actual IngestionResult structure
    mock_result = IngestionResult(
        success=True,
        file_path="/path/to/test_document.pdf",
        num_documents_loaded=1,
        num_chunks_created=5,
        num_chunks_stored=5,
    )
    pipeline.ingest_file = AsyncMock(return_value=mock_result)
    
    return pipeline


@pytest.fixture
def tool_handler(mock_chat_chain, mock_retriever, mock_ingestion_pipeline):
    """Create MCPToolHandler with mocked dependencies."""
    return MCPToolHandler(
        chat_chain=mock_chat_chain,
        retriever=mock_retriever,
        ingestion_pipeline=mock_ingestion_pipeline,
    )


class TestMCPToolHandlerInitialization:
    """Test MCPToolHandler initialization."""
    
    def test_initialization_stores_components(
        self,
        mock_chat_chain,
        mock_retriever,
        mock_ingestion_pipeline,
    ):
        """Should store all provided components."""
        handler = MCPToolHandler(
            chat_chain=mock_chat_chain,
            retriever=mock_retriever,
            ingestion_pipeline=mock_ingestion_pipeline,
        )
        
        assert handler.chat_chain is mock_chat_chain
        assert handler.retriever is mock_retriever
        assert handler.ingestion_pipeline is mock_ingestion_pipeline
    
    def test_get_available_tools_returns_all_tools(self, tool_handler):
        """Should return all available tools."""
        tools = tool_handler.get_available_tools()
        
        assert len(tools) == 4
        assert all(isinstance(tool, Tool) for tool in tools)


class TestHandleRagChat:
    """Test handle_rag_chat method."""
    
    @pytest.mark.asyncio
    async def test_handle_rag_chat_with_session_id(self, tool_handler, mock_chat_chain):
        """Should handle RAG chat with provided session ID."""
        result = await tool_handler.handle_rag_chat(
            question="What is machine learning?",
            session_id="existing-session",
        )
        
        # Verify chat was called correctly
        mock_chat_chain.chat.assert_called_once_with(
            question="What is machine learning?",
            session_id="existing-session",
        )
        
        # Verify session history was checked
        mock_chat_chain.memory.get_session_history.assert_called_once_with(
            "existing-session"
        )
        
        # Verify result format
        assert "answer" in result
        assert "sources" in result
        assert "session_id" in result
        assert result["answer"] == "This is a test answer based on the documents."
    
    @pytest.mark.asyncio
    async def test_handle_rag_chat_without_session_id(
        self,
        tool_handler,
        mock_chat_chain,
    ):
        """Should create new session ID when not provided."""
        result = await tool_handler.handle_rag_chat(
            question="What is AI?",
        )
        
        # Verify new session was created
        mock_chat_chain.memory.create_session.assert_called_once()
        
        # Verify chat was called with generated session ID
        assert mock_chat_chain.chat.called
        call_kwargs = mock_chat_chain.chat.call_args.kwargs
        assert call_kwargs["session_id"] == "test-session-123"
        
        assert "session_id" in result
    
    @pytest.mark.asyncio
    async def test_handle_rag_chat_returns_sources(self, tool_handler):
        """Should include sources in response."""
        result = await tool_handler.handle_rag_chat(
            question="Test question?",
            session_id="test-session",
        )
        
        assert "sources" in result
        assert isinstance(result["sources"], list)
        assert len(result["sources"]) > 0
    
    @pytest.mark.asyncio
    async def test_handle_rag_chat_error_handling(self, tool_handler, mock_chat_chain):
        """Should raise exception when chat fails."""
        mock_chat_chain.chat.side_effect = Exception("Chat processing failed")
        
        with pytest.raises(Exception, match="Chat processing failed"):
            await tool_handler.handle_rag_chat(
                question="Test question?",
                session_id="test-session",
            )


class TestHandleDocumentSearch:
    """Test handle_document_search method."""
    
    @pytest.mark.asyncio
    async def test_handle_document_search_with_defaults(
        self,
        tool_handler,
        mock_retriever,
    ):
        """Should perform search with default parameters."""
        result = await tool_handler.handle_document_search(
            query="machine learning",
        )
        
        # Verify search was called correctly
        mock_retriever.search.assert_called_once_with("machine learning", k=4)
        
        # Verify result format
        assert "results" in result
        assert "count" in result
        assert "query" in result
        assert result["query"] == "machine learning"
        assert result["count"] == 2
        assert len(result["results"]) == 2
    
    @pytest.mark.asyncio
    async def test_handle_document_search_with_custom_limit(
        self,
        tool_handler,
        mock_retriever,
    ):
        """Should respect custom limit parameter."""
        result = await tool_handler.handle_document_search(
            query="neural networks",
            limit=10,
        )
        
        mock_retriever.search.assert_called_once_with("neural networks", k=10)
        assert result["count"] == 2
    
    @pytest.mark.asyncio
    async def test_handle_document_search_with_mmr(
        self,
        tool_handler,
        mock_retriever,
    ):
        """Should use MMR search when use_mmr is True."""
        result = await tool_handler.handle_document_search(
            query="deep learning",
            limit=5,
            use_mmr=True,
        )
        
        # Verify MMR search was used
        mock_retriever.mmr_search.assert_called_once_with("deep learning", k=5)
        mock_retriever.search.assert_not_called()
        
        assert result["count"] == 2
    
    @pytest.mark.asyncio
    async def test_handle_document_search_result_format(self, tool_handler):
        """Should format results correctly."""
        result = await tool_handler.handle_document_search(query="test")
        
        # Check result structure
        assert isinstance(result["results"], list)
        
        # Check individual result format
        for doc_result in result["results"]:
            assert "content" in doc_result
            assert "metadata" in doc_result
            assert "score" in doc_result
            assert "source" in doc_result
            assert isinstance(doc_result["score"], float)
    
    @pytest.mark.asyncio
    async def test_handle_document_search_rounds_scores(self, tool_handler):
        """Should round scores to 4 decimal places."""
        result = await tool_handler.handle_document_search(query="test")
        
        for doc_result in result["results"]:
            score = doc_result["score"]
            # Check that score has at most 4 decimal places
            assert len(str(score).split(".")[-1]) <= 4
    
    @pytest.mark.asyncio
    async def test_handle_document_search_error_handling(
        self,
        tool_handler,
        mock_retriever,
    ):
        """Should raise exception when search fails."""
        mock_retriever.search.side_effect = Exception("Search failed")
        
        with pytest.raises(Exception, match="Search failed"):
            await tool_handler.handle_document_search(query="test")


class TestHandleIngestDocument:
    """Test handle_ingest_document method."""
    
    @pytest.mark.asyncio
    async def test_handle_ingest_document_success(
        self,
        tool_handler,
        mock_ingestion_pipeline,
        tmp_path,
    ):
        """Should successfully ingest a document."""
        # Create a temporary test file
        test_file = tmp_path / "test.pdf"
        test_file.write_text("Test content")
        
        result = await tool_handler.handle_ingest_document(
            file_path=str(test_file),
        )
        
        # Verify ingestion was called
        mock_ingestion_pipeline.ingest_file.assert_called_once_with(str(test_file))
        
        # Verify result format matches updated response structure
        assert result["success"] is True
        assert result["filename"] == "test_document.pdf"
        assert result["chunks_created"] == 5
        assert result["chunks_stored"] == 5
        assert result["documents_loaded"] == 1
        assert result["file_type"] == "pdf"
        assert "file_path" in result
    
    @pytest.mark.asyncio
    async def test_handle_ingest_document_file_not_found(self, tool_handler):
        """Should raise FileNotFoundError for non-existent file."""
        with pytest.raises(FileNotFoundError):
            await tool_handler.handle_ingest_document(
                file_path="/non/existent/file.pdf",
            )
    
    @pytest.mark.asyncio
    async def test_handle_ingest_document_not_a_file(self, tool_handler, tmp_path):
        """Should raise ValueError when path is a directory."""
        # Create a directory instead of a file
        test_dir = tmp_path / "test_dir"
        test_dir.mkdir()
        
        with pytest.raises(ValueError, match="not a file"):
            await tool_handler.handle_ingest_document(
                file_path=str(test_dir),
            )
    
    @pytest.mark.asyncio
    async def test_handle_ingest_document_with_error(
        self,
        tool_handler,
        mock_ingestion_pipeline,
        tmp_path,
    ):
        """Should include error message in result when ingestion fails partially."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Test")
        
        # Mock partial failure with correct IngestionResult structure
        mock_result = IngestionResult(
            success=False,
            file_path=str(test_file),
            num_documents_loaded=0,
            num_chunks_created=0,
            num_chunks_stored=0,
            error="Failed to process document",
        )
        mock_ingestion_pipeline.ingest_file.return_value = mock_result
        
        result = await tool_handler.handle_ingest_document(str(test_file))
        
        assert result["success"] is False
        assert "error_message" in result
        assert result["error_message"] == "Failed to process document"
    
    @pytest.mark.asyncio
    async def test_handle_ingest_document_exception_handling(
        self,
        tool_handler,
        mock_ingestion_pipeline,
        tmp_path,
    ):
        """Should raise exception when ingestion fails completely."""
        test_file = tmp_path / "test.pdf"
        test_file.write_text("Test")
        
        mock_ingestion_pipeline.ingest_file.side_effect = Exception(
            "Ingestion failed"
        )
        
        with pytest.raises(Exception, match="Ingestion failed"):
            await tool_handler.handle_ingest_document(str(test_file))


class TestHandleListDocuments:
    """Test handle_list_documents method."""
    
    @pytest.mark.asyncio
    async def test_handle_list_documents_returns_dict(self, tool_handler):
        """Should return dictionary with documents and count."""
        result = await tool_handler.handle_list_documents()
        
        assert isinstance(result, dict)
        assert "documents" in result
        assert "count" in result
    
    @pytest.mark.asyncio
    async def test_handle_list_documents_structure(self, tool_handler):
        """Should return correct structure."""
        result = await tool_handler.handle_list_documents()
        
        assert isinstance(result["documents"], list)
        assert isinstance(result["count"], int)
        assert result["count"] == len(result["documents"])
    
    @pytest.mark.asyncio
    async def test_handle_list_documents_empty_list(self, tool_handler):
        """Should handle empty document list."""
        result = await tool_handler.handle_list_documents()
        
        # Current implementation returns empty list
        assert result["count"] == 0
        assert len(result["documents"]) == 0

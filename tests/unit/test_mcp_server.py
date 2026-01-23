"""
Unit tests for MCP server module.

Tests RAGMCPServer initialization, handler registration, and MCP protocol handlers.
"""

import json
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest
from mcp import types
from mcp.server import Server

from src.chat.chain import ChatResponse, SourceDocument
from src.ingestion.pipeline import IngestionResult
from src.mcp.server import RAGMCPServer, create_mcp_server
from src.retrieval.retriever import RetrievedDocument


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
        answer="This is a test answer.",
        sources=[
            SourceDocument(
                filename="test.pdf",
                page=1,
                relevance_score=0.95,
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
            content="Test content.",
            metadata={"source": "doc1.txt"},
            score=0.95,
            source="doc1.txt",
        ),
    ]
    
    retriever.search.return_value = mock_docs
    retriever.mmr_search.return_value = mock_docs
    
    return retriever


@pytest.fixture
def mock_ingestion_pipeline():
    """Create a mock ingestion pipeline for testing."""
    pipeline = AsyncMock()
    
    # Mock ingestion result
    mock_result = IngestionResult(
        success=True,
        file_path="/path/to/test.pdf",
        num_documents_loaded=1,
        num_chunks_created=5,
        num_chunks_stored=5,
    )
    pipeline.ingest_file = AsyncMock(return_value=mock_result)
    
    return pipeline


@pytest.fixture
def mock_vectorstore():
    """Create a mock vectorstore for testing."""
    vectorstore = MagicMock()
    return vectorstore


@pytest.fixture
def mcp_server(mock_chat_chain, mock_retriever, mock_ingestion_pipeline, mock_vectorstore):
    """Create RAGMCPServer with mocked dependencies."""
    return RAGMCPServer(
        chat_chain=mock_chat_chain,
        retriever=mock_retriever,
        ingestion_pipeline=mock_ingestion_pipeline,
        vectorstore=mock_vectorstore,
    )


class TestRAGMCPServerInitialization:
    """Test RAGMCPServer initialization."""
    
    def test_initialization_stores_components(
        self,
        mock_chat_chain,
        mock_retriever,
        mock_ingestion_pipeline,
        mock_vectorstore,
    ):
        """Should store all provided components."""
        server = RAGMCPServer(
            chat_chain=mock_chat_chain,
            retriever=mock_retriever,
            ingestion_pipeline=mock_ingestion_pipeline,
            vectorstore=mock_vectorstore,
        )
        
        assert server.chat_chain is mock_chat_chain
        assert server.retriever is mock_retriever
        assert server.ingestion_pipeline is mock_ingestion_pipeline
        assert server.vectorstore is mock_vectorstore
    
    def test_initialization_creates_handlers(self, mcp_server):
        """Should create tool and resource handlers."""
        assert hasattr(mcp_server, "tool_handler")
        assert hasattr(mcp_server, "resource_handler")
        assert mcp_server.tool_handler is not None
        assert mcp_server.resource_handler is not None
    
    def test_initialization_creates_mcp_server(self, mcp_server):
        """Should create MCP Server instance."""
        assert hasattr(mcp_server, "server")
        assert isinstance(mcp_server.server, Server)
    
    def test_initialization_with_custom_name_and_version(
        self,
        mock_chat_chain,
        mock_retriever,
        mock_ingestion_pipeline,
        mock_vectorstore,
    ):
        """Should accept custom server name and version."""
        server = RAGMCPServer(
            chat_chain=mock_chat_chain,
            retriever=mock_retriever,
            ingestion_pipeline=mock_ingestion_pipeline,
            vectorstore=mock_vectorstore,
            server_name="custom-server",
            server_version="2.0.0",
        )
        
        assert server.server.name == "custom-server"
    
    def test_default_server_name(self, mcp_server):
        """Should use default server name."""
        assert mcp_server.server.name == "rag-chatbot"


class TestGetServerInfo:
    """Test get_server_info method."""
    
    def test_get_server_info_returns_dict(self, mcp_server):
        """Should return dictionary with server information."""
        info = mcp_server.get_server_info()
        
        assert isinstance(info, dict)
    
    def test_get_server_info_has_required_fields(self, mcp_server):
        """Should include all required information fields."""
        info = mcp_server.get_server_info()
        
        assert "name" in info
        assert "version" in info
        assert "protocol_version" in info
        assert "tools" in info
        assert "capabilities" in info
    
    def test_get_server_info_tools_list(self, mcp_server):
        """Should list all available tools."""
        info = mcp_server.get_server_info()
        
        tools = info["tools"]
        assert isinstance(tools, list)
        assert len(tools) == 4
        assert "rag_chat" in tools
        assert "document_search" in tools
        assert "ingest_document" in tools
        assert "list_documents" in tools
    
    def test_get_server_info_capabilities(self, mcp_server):
        """Should declare server capabilities."""
        info = mcp_server.get_server_info()
        
        capabilities = info["capabilities"]
        assert isinstance(capabilities, dict)
        assert capabilities["tools"] is True
        assert capabilities["resources"] is True
        assert capabilities["logging"] is True


class TestRunStdio:
    """Test run_stdio method."""
    
    @pytest.mark.asyncio
    async def test_run_stdio_not_implemented_error(self, mcp_server):
        """Should raise NotImplementedError for SSE transport."""
        with pytest.raises(NotImplementedError):
            await mcp_server.run_sse()
    
    @pytest.mark.asyncio
    async def test_run_sse_with_custom_host_port(self, mcp_server):
        """Should accept custom host and port parameters."""
        with pytest.raises(NotImplementedError):
            await mcp_server.run_sse(host="127.0.0.1", port=9000)


class TestHandlerIntegration:
    """Test handler integration with MCP protocol."""
    
    @pytest.mark.asyncio
    async def test_tool_handler_available_through_server(self, mcp_server):
        """Tool handler should be accessible via server."""
        tools = mcp_server.tool_handler.get_available_tools()
        
        assert len(tools) == 4
        assert all(isinstance(tool, types.Tool) for tool in tools)
    
    @pytest.mark.asyncio
    async def test_resource_handler_available_through_server(self, mcp_server):
        """Resource handler should be accessible via server."""
        resources = await mcp_server.resource_handler.list_resources()
        
        assert isinstance(resources, list)
        assert len(resources) > 0


class TestToolCallRouting:
    """Test tool call routing logic."""
    
    @pytest.mark.asyncio
    async def test_handles_rag_chat_tool_arguments(
        self,
        mcp_server,
        mock_chat_chain,
    ):
        """Should correctly route rag_chat tool call."""
        # Directly test the tool handler (simulating what the MCP server would do)
        result = await mcp_server.tool_handler.handle_rag_chat(
            question="What is AI?",
            session_id="test-session",
        )
        
        mock_chat_chain.chat.assert_called_once()
        assert "answer" in result
    
    @pytest.mark.asyncio
    async def test_handles_document_search_tool_arguments(
        self,
        mcp_server,
        mock_retriever,
    ):
        """Should correctly route document_search tool call."""
        result = await mcp_server.tool_handler.handle_document_search(
            query="test query",
            limit=5,
            use_mmr=True,
        )
        
        mock_retriever.mmr_search.assert_called_once()
        assert "results" in result
    
    @pytest.mark.asyncio
    async def test_handles_ingest_document_tool_arguments(
        self,
        mcp_server,
        mock_ingestion_pipeline,
        tmp_path,
    ):
        """Should correctly route ingest_document tool call."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Test content")
        
        result = await mcp_server.tool_handler.handle_ingest_document(
            file_path=str(test_file),
        )
        
        mock_ingestion_pipeline.ingest_file.assert_called_once()
        assert "success" in result
    
    @pytest.mark.asyncio
    async def test_handles_list_documents_tool(self, mcp_server):
        """Should correctly route list_documents tool call."""
        result = await mcp_server.tool_handler.handle_list_documents()
        
        assert "documents" in result
        assert "count" in result


class TestResourceRouting:
    """Test resource routing logic."""
    
    @pytest.mark.asyncio
    async def test_resource_handler_lists_resources(self, mcp_server):
        """Should list resources through handler."""
        resources = await mcp_server.resource_handler.list_resources()
        
        assert isinstance(resources, list)
        assert len(resources) > 0
    
    @pytest.mark.asyncio
    async def test_resource_handler_gets_collection_resource(self, mcp_server):
        """Should get collection resource through handler."""
        uri = "collection://all"
        
        contents = await mcp_server.resource_handler.get_resource(uri)
        
        assert contents is not None
        assert str(contents.uri) == uri
    
    @pytest.mark.asyncio
    async def test_resource_handler_lists_templates(self, mcp_server):
        """Should list resource templates through handler."""
        templates = await mcp_server.resource_handler.list_resource_templates()
        
        assert isinstance(templates, list)
        assert len(templates) > 0


class TestCreateMCPServerFactory:
    """Test create_mcp_server factory function."""
    
    @pytest.mark.asyncio
    async def test_create_mcp_server_returns_server_instance(
        self,
        mock_chat_chain,
        mock_retriever,
        mock_ingestion_pipeline,
        mock_vectorstore,
    ):
        """Should create and return RAGMCPServer instance."""
        server = await create_mcp_server(
            chat_chain=mock_chat_chain,
            retriever=mock_retriever,
            ingestion_pipeline=mock_ingestion_pipeline,
            vectorstore=mock_vectorstore,
        )
        
        assert isinstance(server, RAGMCPServer)
        assert server.chat_chain is mock_chat_chain
        assert server.retriever is mock_retriever
        assert server.ingestion_pipeline is mock_ingestion_pipeline
        assert server.vectorstore is mock_vectorstore
    
    @pytest.mark.asyncio
    async def test_create_mcp_server_initializes_handlers(
        self,
        mock_chat_chain,
        mock_retriever,
        mock_ingestion_pipeline,
        mock_vectorstore,
    ):
        """Should initialize handlers correctly."""
        server = await create_mcp_server(
            chat_chain=mock_chat_chain,
            retriever=mock_retriever,
            ingestion_pipeline=mock_ingestion_pipeline,
            vectorstore=mock_vectorstore,
        )
        
        assert server.tool_handler is not None
        assert server.resource_handler is not None


class TestErrorHandling:
    """Test error handling in server operations."""
    
    @pytest.mark.asyncio
    async def test_tool_handler_error_propagates(
        self,
        mcp_server,
        mock_chat_chain,
    ):
        """Should propagate errors from tool handlers."""
        mock_chat_chain.chat.side_effect = Exception("Test error")
        
        with pytest.raises(Exception, match="Test error"):
            await mcp_server.tool_handler.handle_rag_chat(
                question="test",
                session_id="test",
            )
    
    @pytest.mark.asyncio
    async def test_resource_handler_error_propagates(self, mcp_server):
        """Should propagate errors from resource handlers."""
        with pytest.raises(ValueError):
            await mcp_server.resource_handler.get_resource("invalid://uri")

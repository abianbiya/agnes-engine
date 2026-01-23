"""
MCP (Model Context Protocol) module for the RAG chatbot.

This module provides MCP server implementation, tools, and resources
that expose RAG chatbot functionality to AI assistants like Claude Desktop.
"""

from src.mcp.resources import (
    COLLECTION_URI_SCHEME,
    DOCUMENTS_URI_SCHEME,
    MCPResourceHandler,
)
from src.mcp.server import RAGMCPServer, create_mcp_server
from src.mcp.tools import (
    DOCUMENT_SEARCH_TOOL,
    INGEST_DOCUMENT_TOOL,
    LIST_DOCUMENTS_TOOL,
    MCPToolHandler,
    RAG_CHAT_TOOL,
    TOOLS,
)

__all__ = [
    # Server
    "RAGMCPServer",
    "create_mcp_server",
    # Tool handler and tools
    "MCPToolHandler",
    "TOOLS",
    "RAG_CHAT_TOOL",
    "DOCUMENT_SEARCH_TOOL",
    "INGEST_DOCUMENT_TOOL",
    "LIST_DOCUMENTS_TOOL",
    # Resource handler and constants
    "MCPResourceHandler",
    "DOCUMENTS_URI_SCHEME",
    "COLLECTION_URI_SCHEME",
]

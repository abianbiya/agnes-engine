"""
Custom exception hierarchy for the RAG chatbot system.

This module defines a comprehensive exception hierarchy for error handling
throughout the application, with specific exceptions for different error types.
"""

from typing import Any, Dict, Optional


class RAGException(Exception):
    """
    Base exception for all RAG system errors.
    
    All custom exceptions in the RAG system should inherit from this base class.
    This allows for easy catching of all RAG-related errors.
    
    Attributes:
        message: Human-readable error message
        details: Optional dictionary with additional error context
        cause: Optional underlying exception that caused this error
    """
    
    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
    ):
        """
        Initialize RAG exception.
        
        Args:
            message: Human-readable error message
            details: Optional dictionary with additional error context
            cause: Optional underlying exception that caused this error
        """
        super().__init__(message)
        self.message = message
        self.details = details or {}
        self.cause = cause
    
    def __str__(self) -> str:
        """Return string representation of the exception."""
        if self.details:
            return f"{self.message} (details: {self.details})"
        return self.message
    
    def __repr__(self) -> str:
        """Return detailed string representation for debugging."""
        return (
            f"{self.__class__.__name__}("
            f"message={self.message!r}, "
            f"details={self.details!r}, "
            f"cause={self.cause!r})"
        )


# =============================================================================
# Document Ingestion Errors
# =============================================================================

class DocumentIngestionError(RAGException):
    """
    Error during document ingestion process.
    
    Raised when there are issues loading, parsing, or processing documents
    during the ingestion pipeline.
    """
    pass


class DocumentLoadError(DocumentIngestionError):
    """
    Error loading a document from file.
    
    Raised when a document cannot be loaded, e.g., file not found,
    unsupported format, or corrupted file.
    """
    pass


class DocumentParseError(DocumentIngestionError):
    """
    Error parsing document content.
    
    Raised when document content cannot be extracted or parsed correctly,
    e.g., malformed PDF, encoding issues.
    """
    pass


class ChunkingError(DocumentIngestionError):
    """
    Error during text chunking process.
    
    Raised when document text cannot be properly split into chunks.
    """
    pass


# =============================================================================
# Vector Store Errors
# =============================================================================

class VectorStoreError(RAGException):
    """
    Error with vector store operations.
    
    Raised for issues with vector database operations like connection,
    storage, or retrieval.
    """
    pass


class VectorStoreConnectionError(VectorStoreError):
    """
    Error connecting to vector store.
    
    Raised when unable to establish connection to ChromaDB or other
    vector database.
    """
    pass


class VectorStoreNotFoundError(VectorStoreError):
    """
    Requested collection or document not found in vector store.
    
    Raised when trying to access a collection or document that doesn't exist.
    """
    pass


class VectorStoreStorageError(VectorStoreError):
    """
    Error storing documents in vector store.
    
    Raised when documents cannot be stored, e.g., storage full,
    permission issues.
    """
    pass


class VectorStoreRetrievalError(VectorStoreError):
    """
    Error retrieving documents from vector store.
    
    Raised when search or retrieval operations fail.
    """
    pass


# =============================================================================
# LLM Errors
# =============================================================================

class LLMError(RAGException):
    """
    Error with Language Model operations.
    
    Raised for issues with LLM API calls, generation, or configuration.
    """
    pass


class LLMConnectionError(LLMError):
    """
    Error connecting to LLM API.
    
    Raised when unable to connect to OpenAI, Ollama, or other LLM provider.
    """
    pass


class LLMGenerationError(LLMError):
    """
    Error during text generation.
    
    Raised when LLM fails to generate a response, e.g., context too long,
    rate limit exceeded.
    """
    pass


class LLMAuthenticationError(LLMError):
    """
    Authentication error with LLM provider.
    
    Raised when API key is invalid or authentication fails.
    """
    pass


class LLMRateLimitError(LLMError):
    """
    Rate limit exceeded for LLM API.
    
    Raised when too many requests are made to the LLM API.
    """
    pass


# =============================================================================
# Embedding Errors
# =============================================================================

class EmbeddingError(RAGException):
    """
    Error with embedding generation.
    
    Raised for issues with creating embeddings from text.
    """
    pass


class EmbeddingConnectionError(EmbeddingError):
    """
    Error connecting to embedding service.
    
    Raised when unable to connect to embedding provider.
    """
    pass


class EmbeddingGenerationError(EmbeddingError):
    """
    Error during embedding generation.
    
    Raised when embeddings cannot be generated for text.
    """
    pass


# =============================================================================
# Configuration Errors
# =============================================================================

class ConfigurationError(RAGException):
    """
    Error in system configuration.
    
    Raised when configuration is invalid, missing, or inconsistent.
    """
    pass


class MissingConfigurationError(ConfigurationError):
    """
    Required configuration value is missing.
    
    Raised when a required configuration parameter is not provided.
    """
    pass


class InvalidConfigurationError(ConfigurationError):
    """
    Configuration value is invalid.
    
    Raised when a configuration parameter has an invalid value.
    """
    pass


# =============================================================================
# Retrieval Errors
# =============================================================================

class RetrievalError(RAGException):
    """
    Error during document retrieval.
    
    Raised for issues with retrieving relevant documents for a query.
    """
    pass


class EmptyRetrievalError(RetrievalError):
    """
    No documents retrieved for query.
    
    Raised when retrieval returns no results (may be expected in some cases).
    """
    pass


class RerankerError(RetrievalError):
    """
    Error during document reranking.
    
    Raised when reranking fails to process results.
    """
    pass


# =============================================================================
# Chat Errors
# =============================================================================

class ChatError(RAGException):
    """
    Error during chat operations.
    
    Raised for issues with chat chain, memory, or conversation handling.
    """
    pass


class ChatMemoryError(ChatError):
    """
    Error with conversation memory.
    
    Raised when unable to store or retrieve conversation history.
    """
    pass


class ChatGenerationError(ChatError):
    """
    Error generating chat response.
    
    Raised when chat chain fails to generate a response.
    """
    pass


class SessionNotFoundError(ChatError):
    """
    Chat session not found.
    
    Raised when trying to access a non-existent conversation session.
    """
    pass


# =============================================================================
# MCP Errors
# =============================================================================

class MCPError(RAGException):
    """
    Error with MCP server operations.
    
    Raised for issues with Model Context Protocol server.
    """
    pass


class MCPToolError(MCPError):
    """
    Error executing MCP tool.
    
    Raised when an MCP tool call fails.
    """
    pass


class MCPResourceError(MCPError):
    """
    Error accessing MCP resource.
    
    Raised when unable to retrieve an MCP resource.
    """
    pass


class MCPTransportError(MCPError):
    """
    Error with MCP transport layer.
    
    Raised for issues with stdio or SSE transport.
    """
    pass


# =============================================================================
# API Errors
# =============================================================================

class APIError(RAGException):
    """
    Error with REST API operations.
    
    Raised for issues with API endpoints, request handling, or responses.
    """
    pass


class ValidationError(APIError):
    """
    Request validation error.
    
    Raised when API request fails validation (note: Pydantic also has
    ValidationError, use this for custom validations).
    """
    pass


class NotFoundError(APIError):
    """
    Resource not found.
    
    Raised when a requested resource doesn't exist.
    """
    pass


class UnauthorizedError(APIError):
    """
    Authentication required.
    
    Raised when authentication is required but not provided.
    """
    pass


class ForbiddenError(APIError):
    """
    Access forbidden.
    
    Raised when user doesn't have permission to access resource.
    """
    pass


class RateLimitError(APIError):
    """
    API rate limit exceeded.
    
    Raised when too many requests are made to the API.
    """
    pass


# =============================================================================
# Utility Functions
# =============================================================================

def get_http_status_code(exception: Exception) -> int:
    """
    Map exception to appropriate HTTP status code.
    
    Args:
        exception: Exception instance
        
    Returns:
        HTTP status code (400-599)
    """
    # Map exception types to HTTP status codes
    status_map = {
        # Client errors (400-499)
        ValidationError: 400,
        NotFoundError: 404,
        SessionNotFoundError: 404,
        VectorStoreNotFoundError: 404,
        UnauthorizedError: 401,
        ForbiddenError: 403,
        LLMAuthenticationError: 401,
        DocumentLoadError: 400,
        DocumentParseError: 422,
        DocumentIngestionError: 422,
        ChunkingError: 422,
        RateLimitError: 429,
        LLMRateLimitError: 429,
        # Server errors (500-599)
        VectorStoreError: 503,
        VectorStoreConnectionError: 503,
        VectorStoreStorageError: 503,
        VectorStoreRetrievalError: 503,
        LLMError: 502,
        LLMConnectionError: 502,
        LLMGenerationError: 502,
        EmbeddingError: 502,
        EmbeddingConnectionError: 502,
        EmbeddingGenerationError: 502,
        RetrievalError: 500,
        ChatError: 500,
        ChatGenerationError: 500,
        MCPError: 500,
        ConfigurationError: 500,
        MissingConfigurationError: 500,
        InvalidConfigurationError: 500,
        RAGException: 500,
    }
    
    # Get status code for exception type or its parent classes
    for exc_type, status_code in status_map.items():
        if isinstance(exception, exc_type):
            return status_code
    
    # Default to 500 for unknown exceptions
    return 500


__all__ = [
    # Base exception
    "RAGException",
    # Document ingestion
    "DocumentIngestionError",
    "DocumentLoadError",
    "DocumentParseError",
    "ChunkingError",
    # Vector store
    "VectorStoreError",
    "VectorStoreConnectionError",
    "VectorStoreNotFoundError",
    "VectorStoreStorageError",
    "VectorStoreRetrievalError",
    # LLM
    "LLMError",
    "LLMConnectionError",
    "LLMGenerationError",
    "LLMAuthenticationError",
    "LLMRateLimitError",
    # Embeddings
    "EmbeddingError",
    "EmbeddingConnectionError",
    "EmbeddingGenerationError",
    # Configuration
    "ConfigurationError",
    "MissingConfigurationError",
    "InvalidConfigurationError",
    # Retrieval
    "RetrievalError",
    "EmptyRetrievalError",
    "RerankerError",
    # Chat
    "ChatError",
    "ChatMemoryError",
    "ChatGenerationError",
    "SessionNotFoundError",
    # MCP
    "MCPError",
    "MCPToolError",
    "MCPResourceError",
    "MCPTransportError",
    # API
    "APIError",
    "ValidationError",
    "NotFoundError",
    "UnauthorizedError",
    "ForbiddenError",
    "RateLimitError",
    # Utilities
    "get_http_status_code",
]

"""
Tests for custom exception classes and exception handling utilities.

This module tests the exception hierarchy, exception initialization,
string representations, and HTTP status code mappings.
"""

import pytest
from fastapi import status

from src.utils.exceptions import (
    # Base exception
    RAGException,
    # Document ingestion exceptions
    DocumentIngestionError,
    DocumentLoadError,
    DocumentParseError,
    ChunkingError,
    # Vector store exceptions
    VectorStoreError,
    VectorStoreConnectionError,
    VectorStoreNotFoundError,
    VectorStoreStorageError,
    VectorStoreRetrievalError,
    # LLM exceptions
    LLMError,
    LLMConnectionError,
    LLMGenerationError,
    LLMAuthenticationError,
    LLMRateLimitError,
    # Embedding exceptions
    EmbeddingError,
    EmbeddingConnectionError,
    EmbeddingGenerationError,
    # Configuration exceptions
    ConfigurationError,
    MissingConfigurationError,
    InvalidConfigurationError,
    # Retrieval exceptions
    RetrievalError,
    EmptyRetrievalError,
    RerankerError,
    # Chat exceptions
    ChatError,
    ChatMemoryError,
    ChatGenerationError,
    SessionNotFoundError,
    # MCP exceptions
    MCPError,
    MCPToolError,
    MCPResourceError,
    MCPTransportError,
    # API exceptions
    APIError,
    ValidationError,
    NotFoundError,
    UnauthorizedError,
    ForbiddenError,
    RateLimitError,
    # Utility functions
    get_http_status_code,
)


class TestExceptionHierarchy:
    """Test exception inheritance and hierarchy."""
    
    def test_base_exception_is_exception(self):
        """Test that RAGException inherits from Exception."""
        assert issubclass(RAGException, Exception)
    
    def test_document_exceptions_inherit_from_base(self):
        """Test document-related exceptions inherit from RAGException."""
        assert issubclass(DocumentIngestionError, RAGException)
        assert issubclass(DocumentLoadError, DocumentIngestionError)
        assert issubclass(DocumentParseError, DocumentIngestionError)
        assert issubclass(ChunkingError, DocumentIngestionError)
    
    def test_vectorstore_exceptions_inherit_from_base(self):
        """Test vector store exceptions inherit from RAGException."""
        assert issubclass(VectorStoreError, RAGException)
        assert issubclass(VectorStoreConnectionError, VectorStoreError)
        assert issubclass(VectorStoreNotFoundError, VectorStoreError)
        assert issubclass(VectorStoreStorageError, VectorStoreError)
        assert issubclass(VectorStoreRetrievalError, VectorStoreError)
    
    def test_llm_exceptions_inherit_from_base(self):
        """Test LLM exceptions inherit from RAGException."""
        assert issubclass(LLMError, RAGException)
        assert issubclass(LLMConnectionError, LLMError)
        assert issubclass(LLMGenerationError, LLMError)
        assert issubclass(LLMAuthenticationError, LLMError)
        assert issubclass(LLMRateLimitError, LLMError)
    
    def test_embedding_exceptions_inherit_from_base(self):
        """Test embedding exceptions inherit from RAGException."""
        assert issubclass(EmbeddingError, RAGException)
        assert issubclass(EmbeddingConnectionError, EmbeddingError)
        assert issubclass(EmbeddingGenerationError, EmbeddingError)
    
    def test_config_exceptions_inherit_from_base(self):
        """Test configuration exceptions inherit from RAGException."""
        assert issubclass(ConfigurationError, RAGException)
        assert issubclass(MissingConfigurationError, ConfigurationError)
        assert issubclass(InvalidConfigurationError, ConfigurationError)
    
    def test_retrieval_exceptions_inherit_from_base(self):
        """Test retrieval exceptions inherit from RAGException."""
        assert issubclass(RetrievalError, RAGException)
        assert issubclass(EmptyRetrievalError, RetrievalError)
        assert issubclass(RerankerError, RetrievalError)
    
    def test_chat_exceptions_inherit_from_base(self):
        """Test chat exceptions inherit from RAGException."""
        assert issubclass(ChatError, RAGException)
        assert issubclass(ChatMemoryError, ChatError)
        assert issubclass(ChatGenerationError, ChatError)
        assert issubclass(SessionNotFoundError, ChatError)
    
    def test_mcp_exceptions_inherit_from_base(self):
        """Test MCP exceptions inherit from RAGException."""
        assert issubclass(MCPError, RAGException)
        assert issubclass(MCPToolError, MCPError)
        assert issubclass(MCPResourceError, MCPError)
        assert issubclass(MCPTransportError, MCPError)
    
    def test_api_exceptions_inherit_from_base(self):
        """Test API exceptions inherit from RAGException."""
        assert issubclass(APIError, RAGException)
        assert issubclass(ValidationError, APIError)
        assert issubclass(NotFoundError, APIError)
        assert issubclass(UnauthorizedError, APIError)
        assert issubclass(ForbiddenError, APIError)
        assert issubclass(RateLimitError, APIError)


class TestExceptionInitialization:
    """Test exception initialization and attributes."""
    
    def test_base_exception_with_message_only(self):
        """Test initializing base exception with just a message."""
        exc = RAGException("Test error message")
        assert exc.message == "Test error message"
        assert exc.details == {}  # Default is empty dict, not None
        assert exc.cause is None
    
    def test_base_exception_with_all_params(self):
        """Test initializing base exception with all parameters."""
        cause = ValueError("Original error")
        details = {"key": "value", "count": 42}
        
        exc = RAGException(
            message="Test error",
            details=details,
            cause=cause,
        )
        
        assert exc.message == "Test error"
        assert exc.details == details
        assert exc.cause == cause
    
    def test_document_load_error_with_details(self):
        """Test DocumentLoadError with detailed context."""
        exc = DocumentLoadError(
            message="Failed to load document",
            details={"filename": "test.pdf", "path": "/tmp/test.pdf"},
            cause=FileNotFoundError("File not found"),
        )
        
        assert exc.message == "Failed to load document"
        assert exc.details["filename"] == "test.pdf"
        assert exc.details["path"] == "/tmp/test.pdf"
        assert isinstance(exc.cause, FileNotFoundError)
    
    def test_llm_rate_limit_error(self):
        """Test LLMRateLimitError initialization."""
        exc = LLMRateLimitError(
            message="Rate limit exceeded",
            details={"retry_after": 60, "limit": 100},
        )
        
        assert exc.message == "Rate limit exceeded"
        assert exc.details["retry_after"] == 60
        assert exc.details["limit"] == 100
    
    def test_session_not_found_error(self):
        """Test SessionNotFoundError initialization."""
        exc = SessionNotFoundError(
            message="Session abc123 not found",
            details={"session_id": "abc123"},
        )
        
        assert exc.message == "Session abc123 not found"
        assert exc.details["session_id"] == "abc123"


class TestExceptionStringRepresentation:
    """Test exception string representations."""
    
    def test_str_with_message_only(self):
        """Test __str__ with just a message."""
        exc = RAGException("Simple error")
        assert str(exc) == "Simple error"
    
    def test_str_with_message_and_details(self):
        """Test __str__ with message and details."""
        exc = RAGException(
            message="Error occurred",
            details={"code": 123},
        )
        result = str(exc)
        assert "Error occurred" in result
        assert "code" in result
        assert "123" in result
    
    def test_str_with_all_fields(self):
        """Test __str__ with all fields populated."""
        cause = ValueError("Original error")
        exc = RAGException(
            message="Wrapped error",
            details={"info": "test"},
            cause=cause,
        )
        result = str(exc)
        assert "Wrapped error" in result
        assert "info" in result
        # Note: __str__ doesn't include cause, only message and details
    
    def test_repr_includes_class_name(self):
        """Test __repr__ includes the class name."""
        exc = DocumentLoadError("Test error")
        result = repr(exc)
        assert "DocumentLoadError" in result
        assert "Test error" in result
    
    def test_repr_includes_details(self):
        """Test __repr__ includes details."""
        exc = ChatMemoryError(
            message="Memory error",
            details={"session_id": "test123"},
        )
        result = repr(exc)
        assert "ChatMemoryError" in result
        assert "session_id" in result


class TestHTTPStatusCodeMapping:
    """Test HTTP status code mapping for exceptions."""
    
    def test_validation_error_maps_to_400(self):
        """Test ValidationError maps to 400 Bad Request."""
        exc = ValidationError("Invalid input")
        assert get_http_status_code(exc) == status.HTTP_400_BAD_REQUEST
    
    def test_unauthorized_error_maps_to_401(self):
        """Test UnauthorizedError maps to 401 Unauthorized."""
        exc = UnauthorizedError("Authentication required")
        assert get_http_status_code(exc) == status.HTTP_401_UNAUTHORIZED
    
    def test_forbidden_error_maps_to_403(self):
        """Test ForbiddenError maps to 403 Forbidden."""
        exc = ForbiddenError("Access denied")
        assert get_http_status_code(exc) == status.HTTP_403_FORBIDDEN
    
    def test_not_found_error_maps_to_404(self):
        """Test NotFoundError maps to 404 Not Found."""
        exc = NotFoundError("Resource not found")
        assert get_http_status_code(exc) == status.HTTP_404_NOT_FOUND
    
    def test_session_not_found_maps_to_404(self):
        """Test SessionNotFoundError maps to 404."""
        exc = SessionNotFoundError("Session not found")
        assert get_http_status_code(exc) == status.HTTP_404_NOT_FOUND
    
    def test_document_ingestion_maps_to_422(self):
        """Test DocumentIngestionError maps to 422 Unprocessable Entity."""
        exc = DocumentIngestionError("Failed to process document")
        assert get_http_status_code(exc) == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_document_load_error_maps_to_422(self):
        """Test DocumentLoadError maps to 400 Bad Request."""
        exc = DocumentLoadError("Cannot load document")
        assert get_http_status_code(exc) == status.HTTP_400_BAD_REQUEST
    
    def test_rate_limit_error_maps_to_429(self):
        """Test RateLimitError maps to 429 Too Many Requests."""
        exc = RateLimitError("Too many requests")
        assert get_http_status_code(exc) == status.HTTP_429_TOO_MANY_REQUESTS
    
    def test_llm_rate_limit_maps_to_429(self):
        """Test LLMRateLimitError maps to 429."""
        exc = LLMRateLimitError("LLM rate limit exceeded")
        assert get_http_status_code(exc) == status.HTTP_429_TOO_MANY_REQUESTS
    
    def test_config_error_maps_to_500(self):
        """Test ConfigurationError maps to 500 Internal Server Error."""
        exc = ConfigurationError("Invalid configuration")
        assert get_http_status_code(exc) == status.HTTP_500_INTERNAL_SERVER_ERROR
    
    def test_missing_config_maps_to_500(self):
        """Test MissingConfigurationError maps to 500."""
        exc = MissingConfigurationError("Required config missing")
        assert get_http_status_code(exc) == status.HTTP_500_INTERNAL_SERVER_ERROR
    
    def test_llm_error_maps_to_502(self):
        """Test LLMError maps to 502 Bad Gateway."""
        exc = LLMError("LLM service failed")
        assert get_http_status_code(exc) == status.HTTP_502_BAD_GATEWAY
    
    def test_llm_connection_error_maps_to_502(self):
        """Test LLMConnectionError maps to 502."""
        exc = LLMConnectionError("Cannot connect to LLM")
        assert get_http_status_code(exc) == status.HTTP_502_BAD_GATEWAY
    
    def test_embedding_error_maps_to_502(self):
        """Test EmbeddingError maps to 502."""
        exc = EmbeddingError("Embedding service failed")
        assert get_http_status_code(exc) == status.HTTP_502_BAD_GATEWAY
    
    def test_vectorstore_error_maps_to_503(self):
        """Test VectorStoreError maps to 503 Service Unavailable."""
        exc = VectorStoreError("Vector store unavailable")
        assert get_http_status_code(exc) == status.HTTP_503_SERVICE_UNAVAILABLE
    
    def test_vectorstore_connection_maps_to_503(self):
        """Test VectorStoreConnectionError maps to 503."""
        exc = VectorStoreConnectionError("Cannot connect to vector store")
        assert get_http_status_code(exc) == status.HTTP_503_SERVICE_UNAVAILABLE
    
    def test_generic_rag_exception_maps_to_500(self):
        """Test generic RAGException maps to 500."""
        exc = RAGException("Generic error")
        assert get_http_status_code(exc) == status.HTTP_500_INTERNAL_SERVER_ERROR
    
    def test_retrieval_error_maps_to_500(self):
        """Test RetrievalError maps to 500."""
        exc = RetrievalError("Retrieval failed")
        assert get_http_status_code(exc) == status.HTTP_500_INTERNAL_SERVER_ERROR
    
    def test_chat_error_maps_to_500(self):
        """Test ChatError maps to 500."""
        exc = ChatError("Chat failed")
        assert get_http_status_code(exc) == status.HTTP_500_INTERNAL_SERVER_ERROR


class TestExceptionChaining:
    """Test exception chaining with cause parameter."""
    
    def test_chaining_preserves_original_exception(self):
        """Test that exception chaining preserves the original exception."""
        original = ValueError("Original error")
        wrapped = DocumentLoadError(
            message="Failed to load",
            cause=original,
        )
        
        assert wrapped.cause == original
        assert isinstance(wrapped.cause, ValueError)
    
    def test_multiple_level_chaining(self):
        """Test chaining multiple levels of exceptions."""
        level1 = IOError("IO error")
        level2 = DocumentLoadError("Load failed", cause=level1)
        level3 = DocumentIngestionError("Ingestion failed", cause=level2)
        
        assert level3.cause == level2
        assert level2.cause == level1
        assert level3.cause.cause == level1
    
    def test_chaining_with_details_and_cause(self):
        """Test exception with both details and cause."""
        original = FileNotFoundError("/path/to/file")
        exc = DocumentLoadError(
            message="File not found",
            details={"path": "/path/to/file", "retries": 3},
            cause=original,
        )
        
        assert exc.details["path"] == "/path/to/file"
        assert exc.details["retries"] == 3
        assert exc.cause == original


class TestExceptionDetails:
    """Test exception details dictionary handling."""
    
    def test_details_can_be_empty_dict(self):
        """Test details can be an empty dictionary."""
        exc = RAGException("Error", details={})
        assert exc.details == {}
    
    def test_details_preserves_data_types(self):
        """Test details dictionary preserves various data types."""
        details = {
            "string": "value",
            "int": 42,
            "float": 3.14,
            "bool": True,
            "list": [1, 2, 3],
            "dict": {"nested": "value"},
        }
        exc = RAGException("Error", details=details)
        
        assert exc.details["string"] == "value"
        assert exc.details["int"] == 42
        assert exc.details["float"] == 3.14
        assert exc.details["bool"] is True
        assert exc.details["list"] == [1, 2, 3]
        assert exc.details["dict"]["nested"] == "value"
    
    def test_details_none_by_default(self):
        """Test details is empty dict when not provided."""
        exc = RAGException("Error")
        assert exc.details == {}  # Default is empty dict, not None
    
    def test_details_can_contain_exception_info(self):
        """Test details can contain information about the error."""
        exc = LLMRateLimitError(
            message="Rate limit exceeded",
            details={
                "limit": 100,
                "current": 150,
                "reset_time": "2024-01-01T00:00:00Z",
                "retry_after_seconds": 3600,
            },
        )
        
        assert exc.details["limit"] == 100
        assert exc.details["current"] == 150
        assert "reset_time" in exc.details
        assert exc.details["retry_after_seconds"] == 3600

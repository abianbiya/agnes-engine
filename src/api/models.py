"""
API request and response models for the RAG chatbot.

This module defines Pydantic models for API requests and responses,
providing validation, serialization, and documentation.
"""

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class RetrievalMethod(str, Enum):
    """
    Retrieval method for document search.
    
    Attributes:
        SEMANTIC: Pure semantic/vector similarity search (fastest).
        BM25: Pure BM25 keyword-based search (good for exact matches).
        HYBRID: Combines semantic and BM25 using Reciprocal Rank Fusion (most accurate).
    """
    SEMANTIC = "semantic"
    BM25 = "bm25"
    HYBRID = "hybrid"


class SourceDocument(BaseModel):
    """
    Source document reference in chat responses.
    
    Attributes:
        filename: Name of the source document
        page: Page number (for PDFs)
        section: Section or heading name
        relevance_score: Relevance score from retrieval (0-1)
        content_preview: Preview of the content (optional)
        retrieval_method: Method used to retrieve this document
            ("semantic", "bm25", or "hybrid" if found by both)
    """
    filename: str = Field(..., description="Source document filename")
    page: Optional[int] = Field(None, description="Page number in document")
    section: Optional[str] = Field(None, description="Section or heading name")
    relevance_score: float = Field(..., description="Relevance score (0-1)", ge=0.0)
    content_preview: Optional[str] = Field(None, description="Preview of matched content")
    retrieval_method: str = Field(
        "semantic", 
        description="Retrieval method: 'semantic', 'bm25', or 'hybrid'"
    )
    
    @field_validator("relevance_score")
    @classmethod
    def clamp_relevance_score(cls, v: float) -> float:
        """Clamp score to valid range (ChromaDB can return slightly > 1.0)."""
        return min(max(v, 0.0), 1.0)

    class Config:
        json_schema_extra = {
            "example": {
                "filename": "machine_learning_guide.pdf",
                "page": 5,
                "section": "Neural Networks",
                "relevance_score": 0.95,
                "content_preview": "Neural networks are computational models inspired by...",
                "retrieval_method": "hybrid",
            }
        }


class ChatRequest(BaseModel):
    """
    Request model for chat endpoint.
    
    Attributes:
        question: User's question
        session_id: Optional session ID for conversation continuity
        retrieval_method: Retrieval method to use (semantic, bm25, hybrid)
    """
    question: str = Field(..., description="User's question", min_length=1, max_length=2000)
    session_id: Optional[str] = Field(None, description="Session ID for conversation continuity")
    retrieval_method: RetrievalMethod = Field(
        default=RetrievalMethod.HYBRID,
        description="Retrieval method: 'semantic' (vector search), 'bm25' (keyword), or 'hybrid' (both)"
    )

    @field_validator("question")
    @classmethod
    def validate_question(cls, v: str) -> str:
        """Validate question is not empty or whitespace only."""
        if not v.strip():
            raise ValueError("Question cannot be empty or whitespace only")
        return v.strip()

    class Config:
        json_schema_extra = {
            "example": {
                "question": "What is machine learning?",
                "session_id": "550e8400-e29b-41d4-a716-446655440000",
                "retrieval_method": "hybrid",
            }
        }


class ChatResponse(BaseModel):
    """
    Response model for chat endpoint.
    
    Attributes:
        answer: Generated answer from RAG system
        sources: List of source documents used
        session_id: Session ID for this conversation
        metadata: Optional metadata (e.g., token usage, processing time)
    """
    answer: str = Field(..., description="Generated answer")
    sources: List[SourceDocument] = Field(..., description="Source documents")
    session_id: str = Field(..., description="Session ID")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")

    class Config:
        json_schema_extra = {
            "example": {
                "answer": "Machine learning is a subset of artificial intelligence...",
                "sources": [
                    {
                        "filename": "ml_basics.pdf",
                        "page": 1,
                        "relevance_score": 0.95,
                    }
                ],
                "session_id": "550e8400-e29b-41d4-a716-446655440000",
                "metadata": {"processing_time": 1.23},
            }
        }


class SearchRequest(BaseModel):
    """
    Request model for document search endpoint.
    
    Attributes:
        query: Search query
        limit: Maximum number of results
        use_mmr: Whether to use MMR for diverse results
        retrieval_method: Retrieval method to use (semantic, bm25, hybrid)
    """
    query: str = Field(..., description="Search query", min_length=1, max_length=1000)
    limit: int = Field(4, description="Maximum number of results", ge=1, le=20)
    use_mmr: bool = Field(False, description="Use Maximum Marginal Relevance for diversity")
    retrieval_method: RetrievalMethod = Field(
        default=RetrievalMethod.HYBRID,
        description="Retrieval method: 'semantic' (vector search), 'bm25' (keyword), or 'hybrid' (both)"
    )

    @field_validator("query")
    @classmethod
    def validate_query(cls, v: str) -> str:
        """Validate query is not empty or whitespace only."""
        if not v.strip():
            raise ValueError("Query cannot be empty or whitespace only")
        return v.strip()

    class Config:
        json_schema_extra = {
            "example": {
                "query": "neural networks",
                "limit": 5,
                "use_mmr": True,
                "retrieval_method": "hybrid",
            }
        }


class SearchResult(BaseModel):
    """
    Single search result.
    
    Attributes:
        content: Document content snippet
        metadata: Document metadata
        score: Relevance score
        source: Source document identifier
    """
    content: str = Field(..., description="Document content snippet")
    metadata: Dict[str, Any] = Field(..., description="Document metadata")
    score: float = Field(..., description="Relevance score", ge=0.0)
    source: str = Field(..., description="Source document identifier")
    
    @field_validator("score")
    @classmethod
    def clamp_score(cls, v: float) -> float:
        """Clamp score to valid range (ChromaDB can return slightly > 1.0)."""
        return min(max(v, 0.0), 1.0)

    class Config:
        json_schema_extra = {
            "example": {
                "content": "Neural networks consist of layers of interconnected nodes...",
                "metadata": {"source": "deep_learning.pdf", "page": 12},
                "score": 0.92,
                "source": "deep_learning.pdf",
            }
        }


class SearchResponse(BaseModel):
    """
    Response model for search endpoint.
    
    Attributes:
        results: List of search results
        count: Number of results returned
        query: Original search query
    """
    results: List[SearchResult] = Field(..., description="Search results")
    count: int = Field(..., description="Number of results", ge=0)
    query: str = Field(..., description="Original search query")

    class Config:
        json_schema_extra = {
            "example": {
                "results": [
                    {
                        "content": "Neural networks...",
                        "metadata": {"source": "ml.pdf", "page": 5},
                        "score": 0.95,
                        "source": "ml.pdf",
                    }
                ],
                "count": 1,
                "query": "neural networks",
            }
        }


class IngestResponse(BaseModel):
    """
    Response model for document ingestion endpoint.
    
    Attributes:
        success: Whether ingestion was successful
        filename: Name of ingested file
        file_path: Path to the ingested file
        chunks_created: Number of chunks created
        chunks_stored: Number of chunks stored in vectorstore
        documents_loaded: Number of documents loaded from file
        file_type: Type of file (pdf, txt, md)
        error_message: Error message if ingestion failed
    """
    success: bool = Field(..., description="Ingestion success status")
    filename: str = Field(..., description="Ingested filename")
    file_path: Optional[str] = Field(None, description="Path to ingested file")
    chunks_created: int = Field(..., description="Number of chunks created", ge=0)
    chunks_stored: int = Field(..., description="Number of chunks stored", ge=0)
    documents_loaded: int = Field(..., description="Number of documents loaded", ge=0)
    file_type: str = Field(..., description="File type")
    error_message: Optional[str] = Field(None, description="Error message if failed")

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "filename": "research_paper.pdf",
                "file_path": "/data/research_paper.pdf",
                "chunks_created": 45,
                "chunks_stored": 45,
                "documents_loaded": 1,
                "file_type": "pdf",
            }
        }


class HealthResponse(BaseModel):
    """
    Response model for health check endpoint.
    
    Attributes:
        status: Overall system status
        services: Status of individual services
        version: API version
    """
    status: str = Field(..., description="Overall system status (healthy/degraded/unhealthy)")
    services: Dict[str, bool] = Field(..., description="Individual service status")
    version: str = Field(..., description="API version")

    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "services": {
                    "vectorstore": True,
                    "llm": True,
                    "embeddings": True,
                },
                "version": "1.0.0",
            }
        }


class ErrorResponse(BaseModel):
    """
    Standard error response model.
    
    Attributes:
        error: Error type/category
        message: Human-readable error message
        detail: Optional detailed error information
        correlation_id: Request correlation ID for tracing
    """
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    correlation_id: Optional[str] = Field(None, description="Request correlation ID")

    class Config:
        json_schema_extra = {
            "example": {
                "error": "ValidationError",
                "message": "Question cannot be empty",
                "detail": "Field 'question' is required and must not be empty",
                "correlation_id": "550e8400-e29b-41d4-a716-446655440000",
            }
        }


class SessionDeleteResponse(BaseModel):
    """
    Response model for session deletion endpoint.
    
    Attributes:
        message: Success message
        session_id: Deleted session ID
    """
    message: str = Field(..., description="Success message")
    session_id: str = Field(..., description="Deleted session ID")

    class Config:
        json_schema_extra = {
            "example": {
                "message": "Session cleared successfully",
                "session_id": "550e8400-e29b-41d4-a716-446655440000",
            }
        }


class CollectionClearResponse(BaseModel):
    """
    Response model for collection clear endpoint.
    
    Attributes:
        message: Success message
        documents_deleted: Number of documents deleted
    """
    message: str = Field(..., description="Success message")
    documents_deleted: int = Field(..., description="Number of documents deleted", ge=0)

    class Config:
        json_schema_extra = {
            "example": {
                "message": "Collection cleared successfully",
                "documents_deleted": 369,
            }
        }


class DocumentInfo(BaseModel):
    """
    Information about an ingested document.
    
    Attributes:
        source: Original file path
        file_name: Name of the document file
        file_type: File extension/type
        chunk_count: Number of chunks created from this document
        sample_content: Preview of the document content
        pages: List of page numbers (for PDFs)
        page_count: Total number of pages (for PDFs)
    """
    source: str = Field(..., description="Original file path")
    file_name: str = Field(..., description="Document file name")
    file_type: str = Field(..., description="File type (pdf, txt, md)")
    chunk_count: int = Field(..., description="Number of chunks", ge=0)
    sample_content: Optional[str] = Field(None, description="Preview of document content")
    pages: Optional[List[int]] = Field(None, description="Page numbers (for PDFs)")
    page_count: Optional[int] = Field(None, description="Total pages (for PDFs)")

    class Config:
        json_schema_extra = {
            "example": {
                "source": "/data/documents/research_paper.pdf",
                "file_name": "research_paper.pdf",
                "file_type": "pdf",
                "chunk_count": 45,
                "sample_content": "This paper presents a novel approach to...",
                "pages": [1, 2, 3, 4, 5],
                "page_count": 5,
            }
        }


class DocumentListResponse(BaseModel):
    """
    Response model for document listing endpoint.
    
    Attributes:
        documents: List of document information
        count: Total number of unique documents
        total_chunks: Total number of chunks across all documents
    """
    documents: List[DocumentInfo] = Field(..., description="List of documents")
    count: int = Field(..., description="Number of unique documents", ge=0)
    total_chunks: int = Field(..., description="Total chunks across all documents", ge=0)

    class Config:
        json_schema_extra = {
            "example": {
                "documents": [
                    {
                        "source": "/data/docs/guide.pdf",
                        "file_name": "guide.pdf",
                        "file_type": "pdf",
                        "chunk_count": 25,
                        "sample_content": "Welcome to the user guide...",
                        "pages": [1, 2, 3],
                        "page_count": 3,
                    }
                ],
                "count": 1,
                "total_chunks": 25,
            }
        }


__all__ = [
    "RetrievalMethod",
    "SourceDocument",
    "ChatRequest",
    "ChatResponse",
    "SearchRequest",
    "SearchResult",
    "SearchResponse",
    "IngestResponse",
    "HealthResponse",
    "ErrorResponse",
    "SessionDeleteResponse",
    "CollectionClearResponse",
    "DocumentInfo",
    "DocumentListResponse",
]

"""
API module for the RAG chatbot.

This module provides the FastAPI application with REST endpoints,
request/response models, dependencies, and middleware.
"""

from src.api.app import app, create_app
from src.api.models import (
    ChatRequest,
    ChatResponse,
    ErrorResponse,
    HealthResponse,
    IngestResponse,
    SearchRequest,
    SearchResponse,
    SearchResult,
    SessionDeleteResponse,
    SourceDocument,
)
from src.api.routes import router

__all__ = [
    # Application
    "app",
    "create_app",
    "router",
    # Request models
    "ChatRequest",
    "SearchRequest",
    # Response models
    "ChatResponse",
    "SearchResponse",
    "SearchResult",
    "IngestResponse",
    "HealthResponse",
    "ErrorResponse",
    "SessionDeleteResponse",
    # Nested models
    "SourceDocument",
]

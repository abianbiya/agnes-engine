"""
Main FastAPI application for the RAG chatbot.

This module creates and configures the FastAPI application with
routes, middleware, and event handlers.
"""

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.middleware import CorrelationIdMiddleware, RequestLoggingMiddleware
from src.api.models import ErrorResponse
from src.api.routes import router as api_router
from src.config.settings import Settings
from src.utils.exceptions import RAGException, get_http_status_code
from src.utils.logging import get_correlation_id, get_logger, setup_logging

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """
    Application lifespan context manager.
    
    Handles startup and shutdown events.
    
    Args:
        app: FastAPI application instance
        
    Yields:
        None
    """
    # Startup
    logger.info("application_starting", version=app.version)
    
    # Setup logging
    settings = Settings()
    setup_logging(
        log_level=settings.logging.log_level,
        log_format=settings.logging.log_format,
    )
    
    logger.info(
        "application_started",
        version=app.version,
        environment=settings.environment,
    )
    
    yield
    
    # Shutdown
    logger.info("application_shutting_down")
    
    # Cleanup resources here if needed
    # e.g., close database connections, etc.
    
    logger.info("application_stopped")


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.
    
    Returns:
        Configured FastAPI application instance
    """
    settings = Settings()
    
    # Create FastAPI app
    app = FastAPI(
        title="RAG Chatbot API",
        description="Production-grade Retrieval-Augmented Generation chatbot API",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )
    
    # Add CORS middleware
    if settings.api.cors_enabled:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.api.cors_origins_list,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        logger.info(
            "cors_middleware_configured",
            origins=settings.api.cors_origins_list,
        )
    
    # Add custom middleware
    app.add_middleware(RequestLoggingMiddleware)
    app.add_middleware(CorrelationIdMiddleware)
    
    logger.info("middleware_configured")
    
    # Include API router
    app.include_router(
        api_router,
        prefix="/api/v1",
        tags=["RAG Chatbot"],
    )
    
    logger.info("routes_registered")
    
    # Add exception handlers
    @app.exception_handler(RAGException)
    async def rag_exception_handler(
        request: Request,
        exc: RAGException,
    ) -> JSONResponse:
        """Handle all RAG custom exceptions."""
        correlation_id = get_correlation_id()
        status_code = get_http_status_code(exc)
        
        logger.error(
            "rag_exception",
            exception_type=exc.__class__.__name__,
            message=exc.message,
            details=exc.details,
            path=request.url.path,
            correlation_id=correlation_id,
            exc_info=True,
        )
        
        return JSONResponse(
            status_code=status_code,
            content=ErrorResponse(
                error=exc.__class__.__name__,
                message=exc.message,
                detail=str(exc.details) if exc.details else None,
                correlation_id=correlation_id,
            ).model_dump(),
        )
    
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        """Handle request validation errors."""
        correlation_id = get_correlation_id()
        
        logger.warning(
            "validation_error",
            errors=exc.errors(),
            body=exc.body,
            correlation_id=correlation_id,
        )
        
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=ErrorResponse(
                error="ValidationError",
                message="Request validation failed",
                detail=str(exc.errors()),
                correlation_id=correlation_id,
            ).model_dump(),
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(
        request: Request,
        exc: Exception,
    ) -> JSONResponse:
        """Handle general exceptions."""
        correlation_id = get_correlation_id()
        
        logger.error(
            "unhandled_exception",
            error=str(exc),
            path=request.url.path,
            correlation_id=correlation_id,
            exc_info=True,
        )
        
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=ErrorResponse(
                error="InternalServerError",
                message="An unexpected error occurred",
                detail=str(exc) if settings.is_debug else None,
                correlation_id=correlation_id,
            ).model_dump(),
        )
    
    # Add root endpoint
    @app.get(
        "/",
        tags=["Root"],
        summary="Root endpoint",
        description="Returns API information and status.",
    )
    async def root() -> dict:
        """Root endpoint with API information."""
        return {
            "name": "RAG Chatbot API",
            "version": "1.0.0",
            "status": "running",
            "docs": "/docs",
            "health": "/api/v1/health",
        }
    
    logger.info("application_created")
    
    return app


# Create application instance
app = create_app()


__all__ = ["app", "create_app"]

"""
FastAPI middleware for correlation ID handling and request logging.

This module provides middleware components for:
- Adding correlation IDs to requests
- Logging request/response details
- Error handling and logging
"""

import time
from typing import Callable
from uuid import uuid4

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from src.utils.logging import (
    clear_correlation_id,
    get_correlation_id,
    get_logger,
    set_correlation_id,
)

logger = get_logger(__name__)

# Header name for correlation ID
CORRELATION_ID_HEADER = "X-Correlation-ID"


class CorrelationIdMiddleware(BaseHTTPMiddleware):
    """
    Middleware to handle correlation IDs for request tracing.

    This middleware:
    - Extracts correlation ID from incoming request headers
    - Generates a new correlation ID if not present
    - Adds the correlation ID to response headers
    - Sets the correlation ID in the context for logging
    """

    def __init__(self, app: ASGIApp, header_name: str = CORRELATION_ID_HEADER) -> None:
        """
        Initialize the middleware.

        Args:
            app: The ASGI application.
            header_name: The header name for correlation ID.
        """
        super().__init__(app)
        self.header_name = header_name

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process the request and add correlation ID.

        Args:
            request: The incoming request.
            call_next: The next middleware/handler in the chain.

        Returns:
            The response with correlation ID header.
        """
        # Get correlation ID from header or generate new one
        correlation_id = request.headers.get(self.header_name) or str(uuid4())

        # Set correlation ID in context
        set_correlation_id(correlation_id)

        try:
            # Process request
            response = await call_next(request)

            # Add correlation ID to response headers
            response.headers[self.header_name] = correlation_id

            return response
        finally:
            # Clear correlation ID from context
            clear_correlation_id()


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware to log HTTP requests and responses.

    Logs:
    - Request method, path, and query parameters
    - Response status code and timing
    - Error details for failed requests
    """

    def __init__(
        self,
        app: ASGIApp,
        log_request_body: bool = False,
        log_response_body: bool = False,
        exclude_paths: list[str] | None = None,
    ) -> None:
        """
        Initialize the middleware.

        Args:
            app: The ASGI application.
            log_request_body: Whether to log request bodies.
            log_response_body: Whether to log response bodies.
            exclude_paths: Paths to exclude from logging (e.g., health checks).
        """
        super().__init__(app)
        self.log_request_body = log_request_body
        self.log_response_body = log_response_body
        self.exclude_paths = exclude_paths or ["/health", "/metrics", "/favicon.ico"]

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process the request and log details.

        Args:
            request: The incoming request.
            call_next: The next middleware/handler in the chain.

        Returns:
            The response.
        """
        # Skip logging for excluded paths
        if request.url.path in self.exclude_paths:
            return await call_next(request)

        # Record start time
        start_time = time.perf_counter()

        # Log request
        log_data = {
            "method": request.method,
            "path": request.url.path,
            "query_params": dict(request.query_params),
            "client_host": request.client.host if request.client else None,
            "user_agent": request.headers.get("user-agent"),
        }

        logger.info("request_started", **log_data)

        try:
            # Process request
            response = await call_next(request)

            # Calculate duration
            duration_ms = (time.perf_counter() - start_time) * 1000

            # Log response
            logger.info(
                "request_completed",
                method=request.method,
                path=request.url.path,
                status_code=response.status_code,
                duration_ms=round(duration_ms, 2),
            )

            return response

        except Exception as e:
            # Calculate duration
            duration_ms = (time.perf_counter() - start_time) * 1000

            # Log error
            logger.error(
                "request_failed",
                method=request.method,
                path=request.url.path,
                error_type=type(e).__name__,
                error_message=str(e),
                duration_ms=round(duration_ms, 2),
                exc_info=True,
            )
            raise


def get_correlation_id_from_request(request: Request) -> str:
    """
    Get correlation ID from request or context.

    Args:
        request: The FastAPI request object.

    Returns:
        The correlation ID.
    """
    # Try to get from header first
    correlation_id = request.headers.get(CORRELATION_ID_HEADER)
    if correlation_id:
        return correlation_id

    # Try to get from context
    ctx_correlation_id = get_correlation_id()
    if ctx_correlation_id:
        return ctx_correlation_id

    # Generate new one
    return str(uuid4())

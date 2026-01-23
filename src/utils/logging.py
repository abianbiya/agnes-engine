"""
Structured logging utilities for the RAG Chatbot application.

This module provides:
- Structured JSON logging for production
- Console logging for development
- Correlation ID support for request tracing
- Context-aware logging with bound variables
"""

import logging
import sys
from contextvars import ContextVar
from typing import Any
from uuid import uuid4

import structlog
from structlog.types import EventDict, Processor

# Context variable for correlation ID
correlation_id_var: ContextVar[str | None] = ContextVar("correlation_id", default=None)


def get_correlation_id() -> str | None:
    """
    Get the current correlation ID from context.

    Returns:
        The correlation ID if set, None otherwise.
    """
    return correlation_id_var.get()


def set_correlation_id(correlation_id: str | None = None) -> str:
    """
    Set or generate a correlation ID in the current context.

    Args:
        correlation_id: Optional correlation ID to set. If None, generates a new UUID.

    Returns:
        The correlation ID that was set.
    """
    cid = correlation_id or str(uuid4())
    correlation_id_var.set(cid)
    return cid


def clear_correlation_id() -> None:
    """Clear the correlation ID from the current context."""
    correlation_id_var.set(None)


def add_correlation_id(
    logger: logging.Logger, method_name: str, event_dict: EventDict
) -> EventDict:
    """
    Structlog processor to add correlation ID to log entries.

    Args:
        logger: The logger instance.
        method_name: The logging method name.
        event_dict: The event dictionary.

    Returns:
        Updated event dictionary with correlation ID.
    """
    correlation_id = get_correlation_id()
    if correlation_id:
        event_dict["correlation_id"] = correlation_id
    return event_dict


def add_app_context(
    logger: logging.Logger, method_name: str, event_dict: EventDict
) -> EventDict:
    """
    Structlog processor to add application context to log entries.

    Args:
        logger: The logger instance.
        method_name: The logging method name.
        event_dict: The event dictionary.

    Returns:
        Updated event dictionary with app context.
    """
    event_dict["app"] = "rag-chatbot"
    return event_dict


def setup_logging(
    log_level: str = "INFO",
    log_format: str = "json",
    app_name: str = "rag-chatbot",
) -> None:
    """
    Configure structured logging for the application.

    Args:
        log_level: The logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_format: The log format ('json' for production, 'console' for development).
        app_name: The application name to include in logs.
    """
    # Shared processors for all formats
    shared_processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
        add_correlation_id,
        add_app_context,
    ]

    if log_format == "json":
        # JSON format for production
        processors: list[Processor] = [
            *shared_processors,
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ]
        # Configure standard logging to output JSON as well
        logging.basicConfig(
            format="%(message)s",
            stream=sys.stdout,
            level=getattr(logging, log_level),
        )
    else:
        # Console format for development
        processors = [
            *shared_processors,
            structlog.dev.ConsoleRenderer(
                colors=True,
                exception_formatter=structlog.dev.plain_traceback,
            ),
        ]
        # Configure standard logging with readable format
        logging.basicConfig(
            format="%(levelname)s %(name)s %(message)s",
            stream=sys.stdout,
            level=getattr(logging, log_level),
        )

    # Configure structlog
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Suppress noisy third-party loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """
    Get a configured logger instance.

    Args:
        name: Optional logger name. If None, uses the calling module's name.

    Returns:
        A bound structlog logger instance.
    """
    return structlog.get_logger(name)


class LoggerMixin:
    """
    Mixin class to add logging capabilities to any class.

    Usage:
        class MyService(LoggerMixin):
            def do_something(self):
                self.logger.info("doing_something", key="value")
    """

    @property
    def logger(self) -> structlog.stdlib.BoundLogger:
        """Get a logger bound to this class."""
        if not hasattr(self, "_logger"):
            self._logger = get_logger(self.__class__.__name__)
        return self._logger


def log_function_call(
    logger: structlog.stdlib.BoundLogger | None = None,
    level: str = "debug",
    include_args: bool = True,
    include_result: bool = False,
) -> Any:
    """
    Decorator to log function calls.

    Args:
        logger: Optional logger instance. If None, creates one from function name.
        level: Log level for the entry/exit messages.
        include_args: Whether to include function arguments in logs.
        include_result: Whether to include function result in logs.

    Returns:
        Decorated function.

    Usage:
        @log_function_call()
        def my_function(arg1, arg2):
            return result
    """
    import functools

    def decorator(func: Any) -> Any:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            func_logger = logger or get_logger(func.__module__)
            log_method = getattr(func_logger, level)

            # Log entry
            log_data: dict[str, Any] = {"function": func.__name__}
            if include_args:
                log_data["args_count"] = len(args)
                log_data["kwargs_keys"] = list(kwargs.keys())

            log_method("function_called", **log_data)

            try:
                result = func(*args, **kwargs)

                # Log success
                result_data: dict[str, Any] = {"function": func.__name__, "status": "success"}
                if include_result:
                    result_data["result_type"] = type(result).__name__

                log_method("function_completed", **result_data)
                return result

            except Exception as e:
                # Log error
                func_logger.error(
                    "function_failed",
                    function=func.__name__,
                    error_type=type(e).__name__,
                    error_message=str(e),
                    exc_info=True,
                )
                raise

        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            func_logger = logger or get_logger(func.__module__)
            log_method = getattr(func_logger, level)

            # Log entry
            log_data: dict[str, Any] = {"function": func.__name__}
            if include_args:
                log_data["args_count"] = len(args)
                log_data["kwargs_keys"] = list(kwargs.keys())

            log_method("function_called", **log_data)

            try:
                result = await func(*args, **kwargs)

                # Log success
                result_data: dict[str, Any] = {"function": func.__name__, "status": "success"}
                if include_result:
                    result_data["result_type"] = type(result).__name__

                log_method("function_completed", **result_data)
                return result

            except Exception as e:
                # Log error
                func_logger.error(
                    "function_failed",
                    function=func.__name__,
                    error_type=type(e).__name__,
                    error_message=str(e),
                    exc_info=True,
                )
                raise

        import asyncio

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return wrapper

    return decorator

"""
Tests for logging utilities module.

Uses pytest for unit tests and Hypothesis for property-based testing.
"""

import logging
from unittest.mock import MagicMock, patch

import pytest
from hypothesis import given, settings as hypothesis_settings
from hypothesis import strategies as st

from src.utils.logging import (
    LoggerMixin,
    add_app_context,
    add_correlation_id,
    clear_correlation_id,
    get_correlation_id,
    get_logger,
    log_function_call,
    set_correlation_id,
    setup_logging,
)


class TestCorrelationId:
    """Tests for correlation ID management."""

    def test_set_and_get_correlation_id(self) -> None:
        """Test setting and getting correlation ID."""
        clear_correlation_id()

        # Initially None
        assert get_correlation_id() is None

        # Set specific ID
        set_correlation_id("test-123")
        assert get_correlation_id() == "test-123"

        # Clear ID
        clear_correlation_id()
        assert get_correlation_id() is None

    def test_set_correlation_id_generates_uuid(self) -> None:
        """Test that set_correlation_id generates UUID when not provided."""
        clear_correlation_id()

        cid = set_correlation_id()
        assert cid is not None
        assert len(cid) == 36  # UUID format
        assert get_correlation_id() == cid

        clear_correlation_id()

    def test_set_correlation_id_returns_value(self) -> None:
        """Test that set_correlation_id returns the set value."""
        clear_correlation_id()

        cid = set_correlation_id("my-custom-id")
        assert cid == "my-custom-id"

        clear_correlation_id()

    @given(st.text(min_size=1, max_size=100))
    @hypothesis_settings(max_examples=20)
    def test_correlation_id_roundtrip(self, cid: str) -> None:
        """Property: Any string set as correlation ID should be retrievable."""
        clear_correlation_id()

        set_correlation_id(cid)
        assert get_correlation_id() == cid

        clear_correlation_id()


class TestAddCorrelationId:
    """Tests for add_correlation_id processor."""

    def test_adds_correlation_id_when_set(self) -> None:
        """Test that processor adds correlation ID to event dict."""
        clear_correlation_id()
        set_correlation_id("test-cid")

        mock_logger = MagicMock(spec=logging.Logger)
        event_dict = {"event": "test"}
        result = add_correlation_id(mock_logger, "info", event_dict)

        assert result["correlation_id"] == "test-cid"
        clear_correlation_id()

    def test_no_correlation_id_when_not_set(self) -> None:
        """Test that processor doesn't add correlation ID when not set."""
        clear_correlation_id()

        mock_logger = MagicMock(spec=logging.Logger)
        event_dict = {"event": "test"}
        result = add_correlation_id(mock_logger, "info", event_dict)

        assert "correlation_id" not in result


class TestAddAppContext:
    """Tests for add_app_context processor."""

    def test_adds_app_name(self) -> None:
        """Test that processor adds app name to event dict."""
        mock_logger = MagicMock(spec=logging.Logger)
        event_dict = {"event": "test"}
        result = add_app_context(mock_logger, "info", event_dict)

        assert result["app"] == "rag-chatbot"


class TestSetupLogging:
    """Tests for setup_logging function."""

    def test_setup_json_logging(self) -> None:
        """Test setup with JSON format."""
        # This should not raise
        setup_logging(log_level="INFO", log_format="json")

    def test_setup_console_logging(self) -> None:
        """Test setup with console format."""
        # This should not raise
        setup_logging(log_level="DEBUG", log_format="console")

    def test_setup_with_different_levels(self) -> None:
        """Test setup with various log levels."""
        for level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            setup_logging(log_level=level, log_format="console")


class TestGetLogger:
    """Tests for get_logger function."""

    def test_get_logger_with_name(self) -> None:
        """Test getting logger with specific name."""
        setup_logging(log_format="console")
        logger = get_logger("test.module")
        assert logger is not None

    def test_get_logger_without_name(self) -> None:
        """Test getting logger without name."""
        setup_logging(log_format="console")
        logger = get_logger()
        assert logger is not None

    def test_logger_can_log(self) -> None:
        """Test that logger can actually log messages."""
        setup_logging(log_format="console", log_level="DEBUG")
        logger = get_logger("test")

        # These should not raise
        logger.debug("debug message")
        logger.info("info message")
        logger.warning("warning message")


class TestLoggerMixin:
    """Tests for LoggerMixin class."""

    def test_mixin_provides_logger(self) -> None:
        """Test that mixin provides logger property."""
        setup_logging(log_format="console")

        class TestClass(LoggerMixin):
            pass

        obj = TestClass()
        assert obj.logger is not None

    def test_mixin_logger_caching(self) -> None:
        """Test that logger is cached on instance."""
        setup_logging(log_format="console")

        class TestClass(LoggerMixin):
            pass

        obj = TestClass()
        logger1 = obj.logger
        logger2 = obj.logger
        assert logger1 is logger2


class TestLogFunctionCall:
    """Tests for log_function_call decorator."""

    def test_decorator_logs_function_call(self) -> None:
        """Test that decorator logs function entry and exit."""
        setup_logging(log_format="console", log_level="DEBUG")

        @log_function_call(level="info")
        def test_func(x: int, y: int) -> int:
            return x + y

        result = test_func(1, 2)
        assert result == 3

    def test_decorator_logs_function_error(self) -> None:
        """Test that decorator logs function errors."""
        setup_logging(log_format="console", log_level="DEBUG")

        @log_function_call(level="info")
        def failing_func() -> None:
            raise ValueError("test error")

        with pytest.raises(ValueError, match="test error"):
            failing_func()

    def test_decorator_with_async_function(self) -> None:
        """Test decorator with async function."""
        setup_logging(log_format="console", log_level="DEBUG")

        @log_function_call(level="info")
        async def async_func(x: int) -> int:
            return x * 2

        import asyncio

        result = asyncio.run(async_func(5))
        assert result == 10

    def test_decorator_preserves_function_metadata(self) -> None:
        """Test that decorator preserves function name and docstring."""

        @log_function_call()
        def documented_func() -> None:
            """This is a docstring."""
            pass

        assert documented_func.__name__ == "documented_func"
        assert documented_func.__doc__ == "This is a docstring."


class TestLoggingPropertyBased:
    """Property-based tests for logging utilities."""

    @given(st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=("L", "N", "P"))))
    @hypothesis_settings(max_examples=20)
    def test_logger_name_accepted(self, name: str) -> None:
        """Property: Any reasonable string should work as logger name."""
        setup_logging(log_format="console")
        logger = get_logger(name)
        assert logger is not None

    @given(
        st.sampled_from(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]),
        st.sampled_from(["json", "console"]),
    )
    @hypothesis_settings(max_examples=10)
    def test_setup_logging_combinations(self, level: str, format_: str) -> None:
        """Property: All valid level/format combinations should work."""
        # Should not raise
        setup_logging(log_level=level, log_format=format_)

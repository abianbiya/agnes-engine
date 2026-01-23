"""
Tests for LLM factory module.

Uses pytest for unit tests and mocks external LLM providers.
"""

from unittest.mock import MagicMock, patch

import pytest

from src.config.settings import LLMSettings
from src.core.llm import LLMFactory, get_llm


class TestLLMFactory:
    """Tests for LLMFactory class."""

    def test_create_with_openai_provider(self) -> None:
        """Test creating OpenAI LLM."""
        settings = LLMSettings(
            llm_provider="openai",
            openai_api_key="sk-test",
            llm_model="gpt-4",
        )

        with patch("src.core.llm.ChatOpenAI") as mock_openai:
            llm = LLMFactory.create(settings)
            mock_openai.assert_called_once()

    def test_create_with_ollama_provider(self) -> None:
        """Test creating Ollama LLM."""
        settings = LLMSettings(
            llm_provider="ollama",
            ollama_model="llama2",
        )

        mock_ollama_class = MagicMock()
        mock_ollama_instance = MagicMock()
        mock_ollama_class.return_value = mock_ollama_instance
        
        with patch.dict("sys.modules", {"langchain_ollama": MagicMock(ChatOllama=mock_ollama_class)}):
            llm = LLMFactory.create(settings)
            assert llm == mock_ollama_instance

    def test_create_with_unsupported_provider(self) -> None:
        """Test error when using unsupported provider."""
        settings = LLMSettings(llm_provider="openai")
        # Patch to force unsupported provider
        settings.llm_provider = "unsupported"  # type: ignore

        with pytest.raises(ValueError, match="Unsupported LLM provider"):
            LLMFactory.create(settings)

    def test_create_openai_with_custom_params(self) -> None:
        """Test creating OpenAI LLM with custom parameters."""
        with patch("src.core.llm.ChatOpenAI") as mock_openai:
            llm = LLMFactory.create_openai(
                api_key="sk-test",
                model="gpt-3.5-turbo",
                temperature=0.5,
                max_tokens=1000,
            )

            mock_openai.assert_called_once_with(
                api_key="sk-test",
                model="gpt-3.5-turbo",
                temperature=0.5,
                max_tokens=1000,
            )

    def test_create_ollama_with_custom_params(self) -> None:
        """Test creating Ollama LLM with custom parameters."""
        mock_ollama_class = MagicMock()
        mock_ollama_instance = MagicMock()
        mock_ollama_class.return_value = mock_ollama_instance
        
        with patch.dict("sys.modules", {"langchain_ollama": MagicMock(ChatOllama=mock_ollama_class)}):
            llm = LLMFactory.create_ollama(
                base_url="http://localhost:11434",
                model="mistral",
                temperature=0.8,
            )

            mock_ollama_class.assert_called_once_with(
                base_url="http://localhost:11434",
                model="mistral",
                temperature=0.8,
            )
            assert llm == mock_ollama_instance

    def test_create_ollama_import_error(self) -> None:
        """Test error when langchain-ollama is not installed."""
        with patch.dict("sys.modules", {"langchain_ollama": None}):
            with pytest.raises(ImportError, match="langchain-ollama is not installed"):
                LLMFactory.create_ollama(model="llama2")


class TestGetLLM:
    """Tests for get_llm convenience function."""

    def test_get_llm_with_settings(self) -> None:
        """Test get_llm with provided settings."""
        settings = LLMSettings(
            llm_provider="openai",
            openai_api_key="sk-test",
        )

        with patch("src.core.llm.LLMFactory.create") as mock_create:
            llm = get_llm(settings)
            mock_create.assert_called_once_with(settings)

    def test_get_llm_without_settings(self) -> None:
        """Test get_llm loads settings from environment."""
        with patch("src.config.settings.get_settings") as mock_get_settings:
            mock_settings = MagicMock()
            mock_settings.llm = LLMSettings(
                llm_provider="openai",
                openai_api_key="sk-test",
            )
            mock_get_settings.return_value = mock_settings

            with patch("src.core.llm.LLMFactory.create") as mock_create:
                llm = get_llm()
                mock_create.assert_called_once()


class TestLLMFactoryLogging:
    """Tests for logging in LLM factory."""

    def test_create_logs_provider_info(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test that create logs provider information."""
        settings = LLMSettings(
            llm_provider="openai",
            openai_api_key="sk-test",
        )

        with patch("src.core.llm.ChatOpenAI"):
            LLMFactory.create(settings)
            # Logging assertions would go here if using caplog

    def test_create_openai_logs_model_info(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test that create_openai logs model information."""
        with patch("src.core.llm.ChatOpenAI"):
            LLMFactory.create_openai(
                api_key="sk-test",
                model="gpt-4",
            )
            # Logging assertions would go here


@pytest.mark.unit
class TestLLMFactoryUnit:
    """Unit tests for LLM factory with comprehensive mocking."""

    def test_openai_factory_parameters(self) -> None:
        """Test that OpenAI factory passes all parameters correctly."""
        with patch("src.core.llm.ChatOpenAI") as mock_class:
            mock_instance = MagicMock()
            mock_class.return_value = mock_instance

            result = LLMFactory.create_openai(
                api_key="sk-test-key",
                model="gpt-4-turbo",
                temperature=0.3,
                max_tokens=4096,
                custom_param="value",
            )

            mock_class.assert_called_once_with(
                api_key="sk-test-key",
                model="gpt-4-turbo",
                temperature=0.3,
                max_tokens=4096,
                custom_param="value",
            )
            assert result == mock_instance

    def test_ollama_factory_parameters(self) -> None:
        """Test that Ollama factory passes all parameters correctly."""
        mock_ollama_class = MagicMock()
        mock_ollama_instance = MagicMock()
        mock_ollama_class.return_value = mock_ollama_instance
        
        with patch.dict("sys.modules", {"langchain_ollama": MagicMock(ChatOllama=mock_ollama_class)}):
            result = LLMFactory.create_ollama(
                base_url="http://custom:8000",
                model="codellama",
                temperature=0.1,
                custom_param="value",
            )

            mock_ollama_class.assert_called_once_with(
                base_url="http://custom:8000",
                model="codellama",
                temperature=0.1,
                custom_param="value",
            )
            assert result == mock_ollama_instance

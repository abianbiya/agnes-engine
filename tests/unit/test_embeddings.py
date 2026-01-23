"""
Tests for embeddings factory module.

Uses pytest for unit tests and mocks external embedding providers.
"""

from unittest.mock import MagicMock, patch

import pytest

from src.config.settings import EmbeddingSettings
from src.core.embeddings import EmbeddingsFactory, get_embeddings


class TestEmbeddingsFactory:
    """Tests for EmbeddingsFactory class."""

    def test_create_with_openai_provider(self) -> None:
        """Test creating OpenAI embeddings."""
        settings = EmbeddingSettings(
            embedding_provider="openai",
            embedding_model="text-embedding-ada-002",
        )

        with patch("src.core.embeddings.OpenAIEmbeddings") as mock_openai:
            embeddings = EmbeddingsFactory.create(settings, api_key="sk-test")
            mock_openai.assert_called_once()

    def test_create_openai_without_api_key_raises_error(self) -> None:
        """Test that OpenAI provider requires API key."""
        settings = EmbeddingSettings(embedding_provider="openai")

        with pytest.raises(ValueError, match="OpenAI API key is required"):
            EmbeddingsFactory.create(settings)

    def test_create_with_huggingface_provider(self) -> None:
        """Test creating HuggingFace embeddings."""
        settings = EmbeddingSettings(
            embedding_provider="huggingface",
            huggingface_embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        )

        # Mock the HuggingFaceEmbeddings class that will be imported
        mock_hf_class = MagicMock()
        mock_hf_instance = MagicMock()
        mock_hf_class.return_value = mock_hf_instance
        
        with patch.dict("sys.modules", {"langchain_huggingface": MagicMock(HuggingFaceEmbeddings=mock_hf_class)}):
            embeddings = EmbeddingsFactory.create(settings)
            assert embeddings == mock_hf_instance

    def test_create_with_ollama_provider(self) -> None:
        """Test creating Ollama embeddings."""
        settings = EmbeddingSettings(
            embedding_provider="ollama",
            embedding_model="llama2",
        )

        # Mock the OllamaEmbeddings class that will be imported
        mock_ollama_class = MagicMock()
        mock_ollama_instance = MagicMock()
        mock_ollama_class.return_value = mock_ollama_instance
        
        with patch.dict("sys.modules", {"langchain_ollama": MagicMock(OllamaEmbeddings=mock_ollama_class)}):
            embeddings = EmbeddingsFactory.create(settings)
            assert embeddings == mock_ollama_instance

    def test_create_with_unsupported_provider(self) -> None:
        """Test error when using unsupported provider."""
        settings = EmbeddingSettings(embedding_provider="openai")
        settings.embedding_provider = "unsupported"  # type: ignore

        with pytest.raises(ValueError, match="Unsupported embedding provider"):
            EmbeddingsFactory.create(settings, api_key="sk-test")

    def test_create_openai_with_custom_model(self) -> None:
        """Test creating OpenAI embeddings with custom model."""
        with patch("src.core.embeddings.OpenAIEmbeddings") as mock_openai:
            embeddings = EmbeddingsFactory.create_openai(
                api_key="sk-test",
                model="text-embedding-3-small",
            )

            mock_openai.assert_called_once_with(
                api_key="sk-test",
                model="text-embedding-3-small",
            )

    def test_create_huggingface_with_custom_model(self) -> None:
        """Test creating HuggingFace embeddings with custom model."""
        mock_hf_class = MagicMock()
        mock_hf_instance = MagicMock()
        mock_hf_class.return_value = mock_hf_instance
        
        with patch.dict("sys.modules", {"langchain_huggingface": MagicMock(HuggingFaceEmbeddings=mock_hf_class)}):
            embeddings = EmbeddingsFactory.create_huggingface(
                model_name="sentence-transformers/paraphrase-MiniLM-L6-v2",
            )

            mock_hf_class.assert_called_once_with(
                model_name="sentence-transformers/paraphrase-MiniLM-L6-v2",
            )
            assert embeddings == mock_hf_instance

    def test_create_ollama_with_custom_params(self) -> None:
        """Test creating Ollama embeddings with custom parameters."""
        mock_ollama_class = MagicMock()
        mock_ollama_instance = MagicMock()
        mock_ollama_class.return_value = mock_ollama_instance
        
        with patch.dict("sys.modules", {"langchain_ollama": MagicMock(OllamaEmbeddings=mock_ollama_class)}):
            embeddings = EmbeddingsFactory.create_ollama(
                model="mistral",
                base_url="http://custom:8000",
            )

            mock_ollama_class.assert_called_once_with(
                model="mistral",
                base_url="http://custom:8000",
            )
            assert embeddings == mock_ollama_instance

    def test_create_huggingface_import_error(self) -> None:
        """Test error when langchain-huggingface is not installed."""
        with patch.dict("sys.modules", {"langchain_huggingface": None}):
            with pytest.raises(ImportError, match="langchain-huggingface is not installed"):
                EmbeddingsFactory.create_huggingface()

    def test_create_ollama_import_error(self) -> None:
        """Test error when langchain-ollama is not installed."""
        with patch.dict("sys.modules", {"langchain_ollama": None}):
            with pytest.raises(ImportError, match="langchain-ollama is not installed"):
                EmbeddingsFactory.create_ollama()


class TestGetEmbeddings:
    """Tests for get_embeddings convenience function."""

    def test_get_embeddings_with_settings(self) -> None:
        """Test get_embeddings with provided settings."""
        settings = EmbeddingSettings(embedding_provider="openai")

        with patch("src.core.embeddings.EmbeddingsFactory.create") as mock_create:
            embeddings = get_embeddings(settings, api_key="sk-test")
            mock_create.assert_called_once_with(settings, api_key="sk-test")

    def test_get_embeddings_without_settings_loads_from_env(self) -> None:
        """Test get_embeddings loads settings from environment."""
        with patch("src.config.settings.get_settings") as mock_get_settings:
            mock_settings = MagicMock()
            mock_settings.embedding = EmbeddingSettings(embedding_provider="openai")
            mock_settings.llm.openai_api_key.get_secret_value.return_value = "sk-test"
            mock_get_settings.return_value = mock_settings

            with patch("src.core.embeddings.EmbeddingsFactory.create") as mock_create:
                embeddings = get_embeddings()
                mock_create.assert_called_once()

    def test_get_embeddings_extracts_api_key_for_openai(self) -> None:
        """Test that get_embeddings extracts OpenAI API key from settings."""
        with patch("src.config.settings.get_settings") as mock_get_settings:
            mock_settings = MagicMock()
            mock_settings.embedding = EmbeddingSettings(
                embedding_provider="openai",
            )
            mock_settings.llm.openai_api_key.get_secret_value.return_value = "sk-from-settings"
            mock_get_settings.return_value = mock_settings

            with patch("src.core.embeddings.EmbeddingsFactory.create") as mock_create:
                embeddings = get_embeddings()
                _, kwargs = mock_create.call_args
                assert kwargs["api_key"] == "sk-from-settings"


class TestEmbeddingsFactoryLogging:
    """Tests for logging in embeddings factory."""

    def test_create_logs_provider_info(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test that create logs provider information."""
        settings = EmbeddingSettings(embedding_provider="openai")

        with patch("src.core.embeddings.OpenAIEmbeddings"):
            EmbeddingsFactory.create(settings, api_key="sk-test")
            # Logging assertions would go here

    def test_create_openai_logs_model_info(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test that create_openai logs model information."""
        with patch("src.core.embeddings.OpenAIEmbeddings"):
            EmbeddingsFactory.create_openai(api_key="sk-test")
            # Logging assertions would go here


@pytest.mark.unit
class TestEmbeddingsFactoryUnit:
    """Unit tests for embeddings factory with comprehensive mocking."""

    def test_openai_factory_parameters(self) -> None:
        """Test that OpenAI factory passes all parameters correctly."""
        with patch("src.core.embeddings.OpenAIEmbeddings") as mock_class:
            mock_instance = MagicMock()
            mock_class.return_value = mock_instance

            result = EmbeddingsFactory.create_openai(
                api_key="sk-test-key",
                model="text-embedding-3-large",
                custom_param="value",
            )

            mock_class.assert_called_once_with(
                api_key="sk-test-key",
                model="text-embedding-3-large",
                custom_param="value",
            )
            assert result == mock_instance

    def test_huggingface_factory_parameters(self) -> None:
        """Test that HuggingFace factory passes all parameters correctly."""
        mock_hf_class = MagicMock()
        mock_hf_instance = MagicMock()
        mock_hf_class.return_value = mock_hf_instance
        
        with patch.dict("sys.modules", {"langchain_huggingface": MagicMock(HuggingFaceEmbeddings=mock_hf_class)}):
            result = EmbeddingsFactory.create_huggingface(
                model_name="custom/model",
                custom_param="value",
            )

            mock_hf_class.assert_called_once_with(
                model_name="custom/model",
                custom_param="value",
            )
            assert result == mock_hf_instance

    def test_ollama_factory_parameters(self) -> None:
        """Test that Ollama factory passes all parameters correctly."""
        mock_ollama_class = MagicMock()
        mock_ollama_instance = MagicMock()
        mock_ollama_class.return_value = mock_ollama_instance
        
        with patch.dict("sys.modules", {"langchain_ollama": MagicMock(OllamaEmbeddings=mock_ollama_class)}):
            result = EmbeddingsFactory.create_ollama(
                model="llama2",
                base_url="http://test:9000",
                custom_param="value",
            )

            mock_ollama_class.assert_called_once_with(
                model="llama2",
                base_url="http://test:9000",
                custom_param="value",
            )
            assert result == mock_ollama_instance

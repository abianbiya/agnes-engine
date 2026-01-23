"""
Tests for configuration settings module.

Uses pytest for unit tests and Hypothesis for property-based testing.
"""

import os
from unittest.mock import patch

import pytest
from hypothesis import given, settings as hypothesis_settings
from hypothesis import strategies as st

from src.config.settings import (
    ChunkingSettings,
    ChromaSettings,
    EmbeddingSettings,
    LLMSettings,
    LoggingSettings,
    RetrievalSettings,
    Settings,
    get_settings,
    reload_settings,
)
from tests.strategies import (
    api_key,
    chunk_size_and_overlap,
    environment_name,
    openai_model_name,
    temperature_value,
)


class TestLLMSettings:
    """Tests for LLM configuration settings."""

    def test_default_values(self) -> None:
        """Test default LLM settings."""
        settings = LLMSettings()
        assert settings.llm_model == "gpt-4"
        assert settings.llm_provider == "openai"
        assert settings.llm_temperature == 0.7
        assert settings.llm_max_tokens == 2048

    def test_temperature_validation_valid(self) -> None:
        """Test valid temperature values."""
        settings = LLMSettings(llm_temperature=0.0)
        assert settings.llm_temperature == 0.0

        settings = LLMSettings(llm_temperature=2.0)
        assert settings.llm_temperature == 2.0

    def test_temperature_validation_invalid(self) -> None:
        """Test invalid temperature values raise error."""
        with pytest.raises(ValueError):
            LLMSettings(llm_temperature=-0.1)

        with pytest.raises(ValueError):
            LLMSettings(llm_temperature=2.1)

    @given(st.floats(min_value=0.0, max_value=2.0, allow_nan=False))
    @hypothesis_settings(max_examples=50)
    def test_temperature_property_valid_range(self, temp: float) -> None:
        """Property: Any temperature in [0.0, 2.0] should be valid."""
        settings = LLMSettings(llm_temperature=temp)
        assert 0.0 <= settings.llm_temperature <= 2.0


class TestChunkingSettings:
    """Tests for chunking configuration settings."""

    def test_default_values(self) -> None:
        """Test default chunking settings."""
        settings = ChunkingSettings()
        assert settings.chunk_size == 1000
        assert settings.chunk_overlap == 200
        assert settings.max_file_size_mb == 50

    def test_overlap_must_be_less_than_size(self) -> None:
        """Test that overlap must be less than chunk size."""
        with pytest.raises(ValueError, match="chunk_overlap must be less than chunk_size"):
            ChunkingSettings(chunk_size=500, chunk_overlap=500)

        with pytest.raises(ValueError, match="chunk_overlap must be less than chunk_size"):
            ChunkingSettings(chunk_size=500, chunk_overlap=600)

    def test_valid_overlap_configuration(self) -> None:
        """Test valid overlap configuration."""
        settings = ChunkingSettings(chunk_size=1000, chunk_overlap=100)
        assert settings.chunk_size == 1000
        assert settings.chunk_overlap == 100

    @given(
        st.integers(min_value=100, max_value=10000),
        st.integers(min_value=0, max_value=1000),
    )
    @hypothesis_settings(max_examples=50)
    def test_chunk_settings_property(self, chunk_size: int, chunk_overlap: int) -> None:
        """Property: Valid chunk settings should have overlap < size."""
        if chunk_overlap >= chunk_size:
            with pytest.raises(ValueError):
                ChunkingSettings(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        else:
            settings = ChunkingSettings(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            assert settings.chunk_overlap < settings.chunk_size


class TestChromaSettings:
    """Tests for ChromaDB configuration settings."""

    def test_default_values(self) -> None:
        """Test default ChromaDB settings."""
        settings = ChromaSettings()
        assert settings.host == "localhost"
        assert settings.port == 8000
        assert settings.collection == "documents"

    def test_url_property(self) -> None:
        """Test URL property generation."""
        settings = ChromaSettings(host="chromadb", port=9000)
        assert settings.url == "http://chromadb:9000"

    def test_port_validation(self) -> None:
        """Test port validation."""
        with pytest.raises(ValueError):
            ChromaSettings(port=0)

        with pytest.raises(ValueError):
            ChromaSettings(port=70000)


class TestRetrievalSettings:
    """Tests for retrieval configuration settings."""

    def test_default_values(self) -> None:
        """Test default retrieval settings."""
        settings = RetrievalSettings()
        assert settings.retrieval_k == 4
        assert settings.use_mmr is True
        assert settings.mmr_diversity == 0.5

    def test_k_validation(self) -> None:
        """Test retrieval k validation."""
        settings = RetrievalSettings(retrieval_k=10)
        assert settings.retrieval_k == 10

        with pytest.raises(ValueError):
            RetrievalSettings(retrieval_k=0)

        with pytest.raises(ValueError):
            RetrievalSettings(retrieval_k=25)

    @given(st.floats(min_value=0.0, max_value=1.0, allow_nan=False))
    @hypothesis_settings(max_examples=50)
    def test_mmr_diversity_property(self, diversity: float) -> None:
        """Property: MMR diversity should be in [0.0, 1.0]."""
        settings = RetrievalSettings(mmr_diversity=diversity)
        assert 0.0 <= settings.mmr_diversity <= 1.0


class TestLoggingSettings:
    """Tests for logging configuration settings."""

    def test_default_values(self) -> None:
        """Test default logging settings."""
        settings = LoggingSettings()
        assert settings.log_level == "INFO"
        assert settings.log_format == "json"
        assert settings.debug is False

    def test_valid_log_levels(self) -> None:
        """Test valid log level values."""
        for level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            settings = LoggingSettings(log_level=level)
            assert settings.log_level == level

    def test_invalid_log_level(self) -> None:
        """Test invalid log level raises error."""
        with pytest.raises(ValueError):
            LoggingSettings(log_level="INVALID")


class TestSettings:
    """Tests for main Settings class."""

    def test_default_initialization(self) -> None:
        """Test Settings can be initialized with defaults."""
        # Use environment without OpenAI key requirement
        with patch.dict(os.environ, {"ENVIRONMENT": "development"}, clear=False):
            settings = Settings()
            assert settings.app_name == "rag-chatbot"
            assert settings.environment == "development"

    def test_nested_settings(self) -> None:
        """Test nested settings are properly initialized."""
        with patch.dict(os.environ, {"ENVIRONMENT": "development"}, clear=False):
            settings = Settings()
            assert isinstance(settings.llm, LLMSettings)
            assert isinstance(settings.chroma, ChromaSettings)
            assert isinstance(settings.chunking, ChunkingSettings)

    def test_is_production_property(self) -> None:
        """Test is_production property."""
        with patch.dict(os.environ, {"ENVIRONMENT": "production", "OPENAI_API_KEY": "sk-test"}):
            settings = Settings()
            assert settings.is_production is True

        with patch.dict(os.environ, {"ENVIRONMENT": "development"}):
            settings = Settings()
            assert settings.is_production is False

    def test_is_debug_property(self) -> None:
        """Test is_debug property."""
        with patch.dict(os.environ, {"ENVIRONMENT": "development", "DEBUG": "false"}):
            settings = Settings()
            # Development environment enables debug by default
            assert settings.is_debug is True

        with patch.dict(os.environ, {"ENVIRONMENT": "production", "DEBUG": "true", "OPENAI_API_KEY": "sk-test"}):
            settings = Settings()
            assert settings.is_debug is True

    def test_openai_key_required_in_production(self) -> None:
        """Test OpenAI API key is required in production with OpenAI provider."""
        with patch.dict(
            os.environ,
            {"ENVIRONMENT": "production", "LLM_PROVIDER": "openai", "OPENAI_API_KEY": ""},
            clear=True,
        ):
            with pytest.raises(ValueError, match="OPENAI_API_KEY is required"):
                Settings()

    def test_openai_key_not_required_for_ollama(self) -> None:
        """Test OpenAI API key is not required when using Ollama provider."""
        with patch.dict(
            os.environ,
            {"ENVIRONMENT": "production", "LLM_PROVIDER": "ollama", "OPENAI_API_KEY": ""},
        ):
            settings = Settings()
            assert settings.llm.llm_provider == "ollama"


class TestGetSettings:
    """Tests for get_settings function."""

    def test_get_settings_cached(self) -> None:
        """Test that get_settings returns cached instance."""
        reload_settings()  # Clear cache first

        with patch.dict(os.environ, {"ENVIRONMENT": "development"}):
            settings1 = get_settings()
            settings2 = get_settings()
            assert settings1 is settings2

    def test_reload_settings(self) -> None:
        """Test that reload_settings clears cache."""
        with patch.dict(os.environ, {"ENVIRONMENT": "development"}):
            settings1 = get_settings()
            settings2 = reload_settings()
            # After reload, should be different instances
            assert settings1 is not settings2


class TestSettingsPropertyBased:
    """Property-based tests for settings using Hypothesis."""

    @given(st.integers(min_value=1, max_value=20))
    @hypothesis_settings(max_examples=20)
    def test_retrieval_k_always_positive(self, k: int) -> None:
        """Property: retrieval_k should always be positive."""
        settings = RetrievalSettings(retrieval_k=k)
        assert settings.retrieval_k > 0

    @given(st.integers(min_value=1, max_value=65535))
    @hypothesis_settings(max_examples=20)
    def test_port_in_valid_range(self, port: int) -> None:
        """Property: Ports should be in valid range."""
        settings = ChromaSettings(port=port)
        assert 1 <= settings.port <= 65535

    @given(
        st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=("L", "N"))),
    )
    @hypothesis_settings(max_examples=20)
    def test_collection_name_accepted(self, name: str) -> None:
        """Property: Collection names should be accepted."""
        settings = ChromaSettings(collection=name)
        assert settings.collection == name


@pytest.mark.property
class TestPropertyBasedLLMSettings:
    """Enhanced property-based tests for LLM settings."""

    @given(temp=temperature_value())
    def test_temperature_validation_property(self, temp: float) -> None:
        """Property: Temperature should be validated correctly."""
        settings = LLMSettings(llm_temperature=temp)
        assert 0.0 <= settings.llm_temperature <= 2.0
        assert isinstance(settings.llm_temperature, float)

    @given(model=openai_model_name())
    def test_model_name_acceptance(self, model: str) -> None:
        """Property: Valid OpenAI model names should be accepted."""
        settings = LLMSettings(llm_model=model)
        assert settings.llm_model == model
        assert len(settings.llm_model) > 0

    @given(
        provider=st.sampled_from(["openai", "ollama"]),
        temp=temperature_value(),
        max_tokens=st.integers(min_value=100, max_value=4096),
    )
    def test_llm_settings_combination(
        self, provider: str, temp: float, max_tokens: int
    ) -> None:
        """Property: Valid LLM setting combinations should work."""
        settings = LLMSettings(
            llm_provider=provider,
            llm_temperature=temp,
            llm_max_tokens=max_tokens,
        )
        assert settings.llm_provider == provider
        assert 0.0 <= settings.llm_temperature <= 2.0
        assert settings.llm_max_tokens == max_tokens

    @given(
        temp=st.floats(min_value=-10.0, max_value=-0.001)
        | st.floats(min_value=2.001, max_value=10.0),
    )
    def test_temperature_out_of_bounds(self, temp: float) -> None:
        """Property: Invalid temperature values should be rejected."""
        with pytest.raises(ValueError):
            LLMSettings(llm_temperature=temp)


@pytest.mark.property
class TestPropertyBasedChunkingSettings:
    """Enhanced property-based tests for chunking settings."""

    @given(sizes=chunk_size_and_overlap())
    def test_valid_chunk_size_overlap_combinations(self, sizes: tuple) -> None:
        """Property: Valid chunk size/overlap pairs should work."""
        chunk_size, chunk_overlap = sizes
        # Ensure overlap doesn't exceed max constraint of 1000
        chunk_overlap = min(chunk_overlap, 1000)
        settings = ChunkingSettings(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        assert settings.chunk_size == chunk_size
        assert settings.chunk_overlap == chunk_overlap
        assert settings.chunk_overlap < settings.chunk_size

    @given(
        chunk_size=st.integers(min_value=100, max_value=1500),
    )
    def test_overlap_ratio_constraint(self, chunk_size: int) -> None:
        """Property: Overlap must be less than chunk size."""
        # Valid: overlap is less than both chunk size and max constraint (1000)
        valid_overlap = min(chunk_size // 2, 1000)
        settings = ChunkingSettings(chunk_size=chunk_size, chunk_overlap=valid_overlap)
        assert settings.chunk_overlap < settings.chunk_size
        
        # Invalid: overlap equals chunk size (when chunk_size <= 1000)
        if chunk_size <= 1000:
            with pytest.raises(ValueError):
                ChunkingSettings(chunk_size=chunk_size, chunk_overlap=chunk_size)

    @given(
        chunk_size=st.integers(min_value=100, max_value=2000),
        max_file_size=st.integers(min_value=1, max_value=200),
    )
    def test_file_size_limits(self, chunk_size: int, max_file_size: int) -> None:
        """Property: File size limits should be respected."""
        settings = ChunkingSettings(
            chunk_size=chunk_size,
            chunk_overlap=min(chunk_size // 2, 1000),  # Ensure under max constraint
            max_file_size_mb=max_file_size,
        )
        assert settings.max_file_size_mb == max_file_size
        assert settings.max_file_size_mb > 0

    @given(
        chunk_size=st.integers(min_value=50, max_value=99),
    )
    def test_minimum_chunk_size_validation(self, chunk_size: int) -> None:
        """Property: Chunk size must meet minimum requirements."""
        with pytest.raises(ValueError):
            ChunkingSettings(chunk_size=chunk_size, chunk_overlap=0)


@pytest.mark.property
class TestPropertyBasedChromaSettings:
    """Enhanced property-based tests for ChromaDB settings."""

    @given(
        host=st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=("L", "N", "Pd"))),
        port=st.integers(min_value=1, max_value=65535),
    )
    def test_url_generation_property(self, host: str, port: int) -> None:
        """Property: URL should be correctly generated from host and port."""
        settings = ChromaSettings(host=host, port=port)
        expected_url = f"http://{host}:{port}"
        assert settings.url == expected_url
        assert settings.host in settings.url
        assert str(settings.port) in settings.url

    @given(
        host=st.sampled_from(["localhost", "127.0.0.1", "chromadb", "db.example.com"]),
        collection=st.text(min_size=1, max_size=100, alphabet=st.characters(whitelist_categories=("L", "N", "Pd"))),
    )
    def test_chroma_configuration_variants(self, host: str, collection: str) -> None:
        """Property: Different ChromaDB configurations should work."""
        settings = ChromaSettings(host=host, collection=collection)
        assert settings.host == host
        assert settings.collection == collection
        assert settings.url.startswith("http://")

    @given(
        port=st.integers(min_value=-1000, max_value=0)
        | st.integers(min_value=65536, max_value=100000),
    )
    def test_invalid_port_rejection(self, port: int) -> None:
        """Property: Invalid ports should be rejected."""
        with pytest.raises(ValueError):
            ChromaSettings(port=port)


@pytest.mark.property
class TestPropertyBasedRetrievalSettings:
    """Enhanced property-based tests for retrieval settings."""

    @given(
        k=st.integers(min_value=1, max_value=20),
        diversity=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
        use_mmr=st.booleans(),
    )
    def test_retrieval_configuration_property(
        self, k: int, diversity: float, use_mmr: bool
    ) -> None:
        """Property: Valid retrieval configurations should work."""
        settings = RetrievalSettings(
            retrieval_k=k,
            mmr_diversity=diversity,
            use_mmr=use_mmr,
        )
        assert settings.retrieval_k == k
        assert 0.0 <= settings.mmr_diversity <= 1.0
        assert settings.use_mmr == use_mmr

    @given(
        diversity=st.floats(min_value=-10.0, max_value=-0.001)
        | st.floats(min_value=1.001, max_value=10.0),
    )
    def test_diversity_out_of_bounds(self, diversity: float) -> None:
        """Property: MMR diversity outside [0, 1] should be rejected."""
        with pytest.raises(ValueError):
            RetrievalSettings(mmr_diversity=diversity)

    @given(k=st.integers(min_value=1, max_value=20))
    def test_k_must_be_in_range(self, k: int) -> None:
        """Property: k must be in valid range."""
        settings = RetrievalSettings(retrieval_k=k)
        assert 1 <= settings.retrieval_k <= 20


@pytest.mark.property
class TestPropertyBasedLoggingSettings:
    """Enhanced property-based tests for logging settings."""

    @given(
        level=st.sampled_from(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]),
        log_format=st.sampled_from(["json", "console"]),
        debug=st.booleans(),
    )
    def test_logging_configuration_property(
        self, level: str, log_format: str, debug: bool
    ) -> None:
        """Property: Valid logging configurations should work."""
        settings = LoggingSettings(
            log_level=level,
            log_format=log_format,
            debug=debug,
        )
        assert settings.log_level == level
        assert settings.log_format == log_format
        assert settings.debug == debug

    @given(level=st.text(min_size=1, max_size=20).filter(
        lambda x: x not in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    ))
    def test_invalid_log_level_rejection(self, level: str) -> None:
        """Property: Invalid log levels should be rejected."""
        with pytest.raises(ValueError):
            LoggingSettings(log_level=level)


@pytest.mark.property
class TestPropertyBasedEmbeddingSettings:
    """Property-based tests for embedding settings."""

    @given(
        provider=st.sampled_from(["openai", "huggingface", "ollama"]),
        model=st.text(min_size=1, max_size=100, alphabet=st.characters(whitelist_categories=("L", "N", "Pd", "Pc"))),
    )
    def test_embedding_configuration_property(self, provider: str, model: str) -> None:
        """Property: Valid embedding configurations should work."""
        settings = EmbeddingSettings(
            embedding_provider=provider,
            embedding_model=model,
        )
        assert settings.embedding_provider == provider
        assert settings.embedding_model == model

    @given(provider=st.sampled_from(["openai", "huggingface", "ollama"]))
    def test_provider_defaults(self, provider: str) -> None:
        """Property: Each provider should have valid defaults."""
        settings = EmbeddingSettings(embedding_provider=provider)
        assert settings.embedding_provider == provider
        assert len(settings.embedding_model) > 0


@pytest.mark.property
class TestPropertyBasedSettingsIntegration:
    """Property-based integration tests for Settings class."""

    @given(
        env=environment_name(),
        debug=st.booleans(),
    )
    def test_environment_configuration_property(self, env: str, debug: bool) -> None:
        """Property: Valid environment configurations should work."""
        env_vars = {
            "ENVIRONMENT": env,
            "DEBUG": str(debug).lower(),
        }
        
        # Only require API key for production with OpenAI
        if env == "production":
            env_vars["OPENAI_API_KEY"] = "sk-test-key-for-testing"
        
        with patch.dict(os.environ, env_vars, clear=False):
            settings = Settings()
            assert settings.environment == env

    @given(
        chunk_size=st.integers(min_value=100, max_value=2000),
        retrieval_k=st.integers(min_value=1, max_value=20),
    )
    def test_nested_settings_consistency(self, chunk_size: int, retrieval_k: int) -> None:
        """Property: Nested settings should maintain consistency."""
        overlap = chunk_size // 2
        
        with patch.dict(os.environ, {"ENVIRONMENT": "development"}, clear=False):
            settings = Settings(
                chunking=ChunkingSettings(chunk_size=chunk_size, chunk_overlap=overlap),
                retrieval=RetrievalSettings(retrieval_k=retrieval_k),
            )
            assert settings.chunking.chunk_size == chunk_size
            assert settings.retrieval.retrieval_k == retrieval_k
            assert settings.chunking.chunk_overlap < settings.chunking.chunk_size


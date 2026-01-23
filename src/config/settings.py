"""
Configuration settings for the RAG Chatbot application.

This module uses Pydantic Settings to manage configuration from environment
variables and .env files with validation and type safety.
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMSettings(BaseSettings):
    """LLM provider configuration."""

    model_config = SettingsConfigDict(env_prefix="")

    openai_api_key: SecretStr = Field(
        default=SecretStr(""),
        description="OpenAI API key",
    )
    llm_model: str = Field(
        default="gpt-4",
        description="LLM model name",
    )
    llm_provider: Literal["openai", "ollama"] = Field(
        default="openai",
        description="LLM provider",
    )
    llm_temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="LLM temperature (creativity)",
    )
    llm_max_tokens: int = Field(
        default=2048,
        ge=1,
        le=128000,
        description="Maximum tokens in response",
    )

    # Ollama settings
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        description="Ollama base URL",
    )
    ollama_model: str = Field(
        default="llama2",
        description="Ollama model name",
    )

    @field_validator("llm_temperature")
    @classmethod
    def validate_temperature(cls, v: float) -> float:
        """Ensure temperature is within valid range."""
        if not 0.0 <= v <= 2.0:
            raise ValueError("Temperature must be between 0.0 and 2.0")
        return v


class EmbeddingSettings(BaseSettings):
    """Embedding model configuration."""

    model_config = SettingsConfigDict(
        env_prefix="",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    embedding_model: str = Field(
        default="bge-m3",
        description="Embedding model name (bge-m3 recommended for multilingual)",
    )
    embedding_provider: Literal["openai", "huggingface", "ollama"] = Field(
        default="ollama",
        description="Embedding provider",
    )
    huggingface_embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="HuggingFace embedding model",
    )
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        description="Ollama base URL for embeddings",
    )


class ChromaSettings(BaseSettings):
    """ChromaDB configuration."""

    model_config = SettingsConfigDict(env_prefix="CHROMA_")

    host: str = Field(
        default="localhost",
        description="ChromaDB host",
    )
    port: int = Field(
        default=8000,
        ge=1,
        le=65535,
        description="ChromaDB port",
    )
    collection: str = Field(
        default="documents",
        description="ChromaDB collection name",
    )
    persist_directory: str = Field(
        default="./data/chroma",
        description="ChromaDB persistence directory",
    )
    in_memory: bool = Field(
        default=False,
        description="Use in-memory ChromaDB",
    )

    @property
    def url(self) -> str:
        """Get ChromaDB URL."""
        return f"http://{self.host}:{self.port}"


class ChunkingSettings(BaseSettings):
    """Document chunking configuration."""

    model_config = SettingsConfigDict(env_prefix="")

    chunk_size: int = Field(
        default=1000,
        ge=100,
        le=10000,
        description="Chunk size in characters",
    )
    chunk_overlap: int = Field(
        default=200,
        ge=0,
        le=1000,
        description="Chunk overlap in characters",
    )
    max_file_size_mb: int = Field(
        default=50,
        ge=1,
        le=500,
        description="Maximum file size in MB",
    )

    @model_validator(mode="after")
    def validate_overlap_less_than_size(self) -> "ChunkingSettings":
        """Ensure overlap is less than chunk size."""
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        return self


class RetrievalSettings(BaseSettings):
    """Retrieval configuration."""

    model_config = SettingsConfigDict(env_prefix="")

    retrieval_k: int = Field(
        default=4,
        ge=1,
        le=20,
        description="Number of documents to retrieve",
    )
    use_mmr: bool = Field(
        default=True,
        description="Use Maximum Marginal Relevance",
    )
    mmr_diversity: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="MMR diversity factor",
    )
    use_reranker: bool = Field(
        default=False,
        description="Use reranking",
    )
    reranker_model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        description="Reranker model",
    )


class MCPSettings(BaseSettings):
    """MCP server configuration."""

    model_config = SettingsConfigDict(env_prefix="MCP_")

    server_port: int = Field(
        default=3000,
        ge=1,
        le=65535,
        description="MCP server port",
    )
    transport: Literal["stdio", "sse"] = Field(
        default="stdio",
        description="MCP transport type",
    )
    server_name: str = Field(
        default="rag-chatbot",
        description="MCP server name",
    )


class APISettings(BaseSettings):
    """API server configuration."""

    model_config = SettingsConfigDict(env_prefix="API_")

    port: int = Field(
        default=8080,
        ge=1,
        le=65535,
        description="API server port",
    )
    host: str = Field(
        default="0.0.0.0",
        description="API server host",
    )
    cors_enabled: bool = Field(
        default=True,
        alias="CORS_ENABLED",
        description="Enable CORS",
    )
    cors_origins: str = Field(
        default="http://localhost:3000,http://localhost:8080",
        alias="CORS_ORIGINS",
        description="Allowed CORS origins (comma-separated)",
    )
    rate_limit: int = Field(
        default=60,
        ge=1,
        le=1000,
        description="API rate limit per minute",
    )

    @property
    def cors_origins_list(self) -> list[str]:
        """Get CORS origins as a list."""
        return [origin.strip() for origin in self.cors_origins.split(",")]


class RedisSettings(BaseSettings):
    """Redis configuration."""

    model_config = SettingsConfigDict(env_prefix="REDIS_")

    host: str = Field(
        default="localhost",
        description="Redis host",
    )
    port: int = Field(
        default=6379,
        ge=1,
        le=65535,
        description="Redis port",
    )
    password: SecretStr = Field(
        default=SecretStr(""),
        description="Redis password",
    )
    db: int = Field(
        default=0,
        ge=0,
        le=15,
        description="Redis database number",
    )
    enabled: bool = Field(
        default=False,
        description="Enable Redis caching",
    )
    session_ttl: int = Field(
        default=3600,
        ge=60,
        le=86400,
        alias="SESSION_TTL",
        description="Session TTL in seconds",
    )

    @property
    def url(self) -> str:
        """Get Redis URL."""
        password = self.password.get_secret_value()
        if password:
            return f"redis://:{password}@{self.host}:{self.port}/{self.db}"
        return f"redis://{self.host}:{self.port}/{self.db}"


class LoggingSettings(BaseSettings):
    """Logging configuration."""

    model_config = SettingsConfigDict(env_prefix="")

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="Log level",
    )
    log_format: Literal["json", "console"] = Field(
        default="json",
        description="Log format",
    )
    debug: bool = Field(
        default=False,
        description="Enable debug mode",
    )


class Settings(BaseSettings):
    """Main application settings aggregating all configuration sections."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Application metadata
    app_name: str = Field(
        default="rag-chatbot",
        description="Application name",
    )
    environment: Literal["development", "staging", "production"] = Field(
        default="development",
        description="Application environment",
    )
    health_check_enabled: bool = Field(
        default=True,
        description="Enable health check endpoint",
    )

    # Nested settings
    llm: LLMSettings = Field(default_factory=LLMSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    chroma: ChromaSettings = Field(default_factory=ChromaSettings)
    chunking: ChunkingSettings = Field(default_factory=ChunkingSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    mcp: MCPSettings = Field(default_factory=MCPSettings)
    api: APISettings = Field(default_factory=APISettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)

    @model_validator(mode="after")
    def validate_api_key_for_openai(self) -> "Settings":
        """Ensure OpenAI API key is provided when using OpenAI provider."""
        if self.llm.llm_provider == "openai":
            api_key = self.llm.openai_api_key.get_secret_value()
            if not api_key or api_key == "":
                # Allow empty key in development for testing
                if self.environment == "production":
                    raise ValueError(
                        "OPENAI_API_KEY is required when using OpenAI provider in production"
                    )
        return self

    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment == "production"

    @property
    def is_debug(self) -> bool:
        """Check if debug mode is enabled."""
        return self.logging.debug or self.environment == "development"


@lru_cache
def get_settings() -> Settings:
    """
    Get cached application settings.

    Returns:
        Settings: Application settings instance.

    Note:
        Settings are cached after first load. Call `get_settings.cache_clear()`
        to reload settings from environment.
    """
    return Settings()


def reload_settings() -> Settings:
    """
    Reload settings from environment.

    Returns:
        Settings: Fresh application settings instance.
    """
    get_settings.cache_clear()
    return get_settings()

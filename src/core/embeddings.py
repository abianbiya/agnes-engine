"""
Embeddings factory for RAG Chatbot.

This module provides a factory pattern for creating embedding model instances
supporting multiple providers (OpenAI, HuggingFace, Ollama).
"""

from typing import Any

from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings

from src.config.settings import EmbeddingSettings
from src.utils.logging import LoggerMixin, get_logger

logger = get_logger(__name__)


class EmbeddingsFactory(LoggerMixin):
    """Factory class for creating embedding model instances."""

    @staticmethod
    def create(settings: EmbeddingSettings, api_key: str | None = None) -> Embeddings:
        """
        Create an embeddings instance based on provider settings.

        Args:
            settings: Embedding configuration settings.
            api_key: Optional API key (required for OpenAI).

        Returns:
            A LangChain Embeddings instance.

        Raises:
            ValueError: If provider is not supported or required params are missing.

        Example:
            >>> from src.config.settings import EmbeddingSettings
            >>> settings = EmbeddingSettings(embedding_provider="openai")
            >>> embeddings = EmbeddingsFactory.create(settings, api_key="sk-...")
        """
        logger.info(
            "creating_embeddings",
            provider=settings.embedding_provider,
            model=settings.embedding_model,
        )

        if settings.embedding_provider == "openai":
            if not api_key:
                raise ValueError("OpenAI API key is required for OpenAI embeddings")
            return EmbeddingsFactory.create_openai(
                api_key=api_key,
                model=settings.embedding_model,
            )
        elif settings.embedding_provider == "huggingface":
            return EmbeddingsFactory.create_huggingface(
                model_name=settings.huggingface_embedding_model,
            )
        elif settings.embedding_provider == "ollama":
            return EmbeddingsFactory.create_ollama(
                model=settings.embedding_model,
                base_url=settings.ollama_base_url,
            )
        else:
            raise ValueError(
                f"Unsupported embedding provider: {settings.embedding_provider}. "
                f"Supported providers: openai, huggingface, ollama"
            )

    @staticmethod
    def create_openai(
        api_key: str,
        model: str = "text-embedding-ada-002",
        **kwargs: Any,
    ) -> OpenAIEmbeddings:
        """
        Create an OpenAI embeddings instance.

        Args:
            api_key: OpenAI API key.
            model: Embedding model name.
            **kwargs: Additional arguments passed to OpenAIEmbeddings.

        Returns:
            OpenAIEmbeddings instance.

        Example:
            >>> embeddings = EmbeddingsFactory.create_openai(
            ...     api_key="sk-...",
            ...     model="text-embedding-ada-002"
            ... )
        """
        logger.info("creating_openai_embeddings", model=model)

        return OpenAIEmbeddings(
            api_key=api_key,
            model=model,
            **kwargs,
        )

    @staticmethod
    def create_huggingface(
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        **kwargs: Any,
    ) -> Embeddings:
        """
        Create a HuggingFace embeddings instance.

        Args:
            model_name: HuggingFace model name.
            **kwargs: Additional arguments passed to HuggingFaceEmbeddings.

        Returns:
            HuggingFaceEmbeddings instance.

        Example:
            >>> embeddings = EmbeddingsFactory.create_huggingface(
            ...     model_name="sentence-transformers/all-MiniLM-L6-v2"
            ... )
        """
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
        except ImportError:
            raise ImportError(
                "langchain-huggingface is not installed. "
                "Install it with: pip install langchain-huggingface"
            )

        logger.info("creating_huggingface_embeddings", model_name=model_name)

        return HuggingFaceEmbeddings(
            model_name=model_name,
            **kwargs,
        )

    @staticmethod
    def create_ollama(
        model: str = "llama2",
        base_url: str = "http://localhost:11434",
        **kwargs: Any,
    ) -> Embeddings:
        """
        Create an Ollama embeddings instance.

        Args:
            model: Ollama model name.
            base_url: Ollama server URL.
            **kwargs: Additional arguments passed to OllamaEmbeddings.

        Returns:
            OllamaEmbeddings instance.

        Example:
            >>> embeddings = EmbeddingsFactory.create_ollama(
            ...     model="llama2",
            ...     base_url="http://localhost:11434"
            ... )
        """
        try:
            from langchain_ollama import OllamaEmbeddings
        except ImportError:
            raise ImportError(
                "langchain-ollama is not installed. "
                "Install it with: pip install langchain-ollama"
            )

        logger.info(
            "creating_ollama_embeddings",
            model=model,
            base_url=base_url,
        )

        return OllamaEmbeddings(
            model=model,
            base_url=base_url,
            **kwargs,
        )


def get_embeddings(
    settings: EmbeddingSettings | None = None,
    api_key: str | None = None,
) -> Embeddings:
    """
    Convenience function to get an embeddings instance.

    Args:
        settings: Optional embedding settings. If None, loads from environment.
        api_key: Optional API key for providers that require it.

    Returns:
        A LangChain Embeddings instance.

    Example:
        >>> embeddings = get_embeddings()
        >>> vectors = embeddings.embed_documents(["Hello", "World"])
    """
    if settings is None:
        from src.config.settings import get_settings

        app_settings = get_settings()
        settings = app_settings.embedding
        # Get API key from LLM settings if using OpenAI
        if settings.embedding_provider == "openai" and not api_key:
            api_key = app_settings.llm.openai_api_key.get_secret_value()
    else:
        # Handle if full Settings object was passed instead of EmbeddingSettings
        from src.config.settings import Settings
        if isinstance(settings, Settings):
            full_settings = settings
            settings = full_settings.embedding
            # Get API key from LLM settings if using OpenAI
            if settings.embedding_provider == "openai" and not api_key:
                api_key = full_settings.llm.openai_api_key.get_secret_value()

    return EmbeddingsFactory.create(settings, api_key=api_key)

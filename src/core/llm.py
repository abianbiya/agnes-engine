"""
LLM provider factory for RAG Chatbot.

This module provides a factory pattern for creating LLM instances
supporting multiple providers (OpenAI, Ollama, etc.).
"""

from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI

from src.config.settings import LLMSettings
from src.utils.logging import LoggerMixin, get_logger

logger = get_logger(__name__)


class LLMFactory(LoggerMixin):
    """Factory class for creating LLM instances."""

    @staticmethod
    def create(settings: LLMSettings) -> BaseChatModel:
        """
        Create an LLM instance based on provider settings.

        Args:
            settings: LLM configuration settings.

        Returns:
            A LangChain BaseChatModel instance.

        Raises:
            ValueError: If provider is not supported.

        Example:
            >>> from src.config.settings import LLMSettings
            >>> settings = LLMSettings(llm_provider="openai")
            >>> llm = LLMFactory.create(settings)
        """
        logger.info(
            "creating_llm",
            provider=settings.llm_provider,
            model=settings.llm_model,
        )

        if settings.llm_provider == "openai":
            return LLMFactory.create_openai(
                api_key=settings.openai_api_key.get_secret_value(),
                model=settings.llm_model,
                temperature=settings.llm_temperature,
                max_tokens=settings.llm_max_tokens,
            )
        elif settings.llm_provider == "ollama":
            return LLMFactory.create_ollama(
                base_url=settings.ollama_base_url,
                model=settings.ollama_model,
                temperature=settings.llm_temperature,
            )
        else:
            raise ValueError(
                f"Unsupported LLM provider: {settings.llm_provider}. "
                f"Supported providers: openai, ollama"
            )

    @staticmethod
    def create_openai(
        api_key: str,
        model: str = "gpt-4",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs: Any,
    ) -> ChatOpenAI:
        """
        Create an OpenAI chat model instance.

        Args:
            api_key: OpenAI API key.
            model: Model name (e.g., 'gpt-4', 'gpt-3.5-turbo').
            temperature: Sampling temperature (0.0 to 2.0).
            max_tokens: Maximum tokens in response.
            **kwargs: Additional arguments passed to ChatOpenAI.

        Returns:
            ChatOpenAI instance.

        Example:
            >>> llm = LLMFactory.create_openai(
            ...     api_key="sk-...",
            ...     model="gpt-4",
            ...     temperature=0.7
            ... )
        """
        logger.info(
            "creating_openai_llm",
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        return ChatOpenAI(
            api_key=api_key,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

    @staticmethod
    def create_ollama(
        base_url: str = "http://localhost:11434",
        model: str = "llama2",
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> BaseChatModel:
        """
        Create an Ollama chat model instance.

        Args:
            base_url: Ollama server URL.
            model: Model name (e.g., 'llama2', 'mistral').
            temperature: Sampling temperature.
            **kwargs: Additional arguments passed to ChatOllama.

        Returns:
            ChatOllama instance.

        Example:
            >>> llm = LLMFactory.create_ollama(
            ...     base_url="http://localhost:11434",
            ...     model="llama2"
            ... )
        """
        try:
            from langchain_ollama import ChatOllama
        except ImportError:
            raise ImportError(
                "langchain-ollama is not installed. "
                "Install it with: pip install langchain-ollama"
            )

        logger.info(
            "creating_ollama_llm",
            base_url=base_url,
            model=model,
            temperature=temperature,
        )

        return ChatOllama(
            base_url=base_url,
            model=model,
            temperature=temperature,
            **kwargs,
        )


def get_llm(settings: LLMSettings | None = None) -> BaseChatModel:
    """
    Convenience function to get an LLM instance.

    Args:
        settings: Optional LLM settings. If None, loads from environment.

    Returns:
        A LangChain BaseChatModel instance.

    Example:
        >>> llm = get_llm()
        >>> response = llm.invoke("What is Python?")
    """
    if settings is None:
        from src.config.settings import get_settings

        settings = get_settings().llm
    else:
        # Handle if full Settings object was passed instead of LLMSettings
        from src.config.settings import Settings
        if isinstance(settings, Settings):
            settings = settings.llm

    return LLMFactory.create(settings)

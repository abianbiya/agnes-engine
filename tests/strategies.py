"""
Custom Hypothesis strategies for property-based testing.

This module provides custom strategies for generating test data
that conforms to the domain models of the RAG chatbot system.
"""

from typing import Any

from hypothesis import strategies as st
from langchain_core.documents import Document


# =============================================================================
# Text Strategies
# =============================================================================

@st.composite
def text_with_newlines(draw: Any) -> str:
    """
    Generate text that may contain newlines.
    
    Useful for testing text processing that should handle multi-line content.
    """
    lines = draw(st.lists(st.text(min_size=1, max_size=200), min_size=1, max_size=20))
    return "\n".join(lines)


@st.composite
def markdown_text(draw: Any) -> str:
    """
    Generate simple markdown-formatted text.
    
    Includes headers, paragraphs, lists, and code blocks.
    """
    elements = []
    
    # Add header
    header_level = draw(st.integers(min_value=1, max_value=3))
    header_text = draw(st.text(min_size=5, max_size=50, alphabet=st.characters(blacklist_characters="\n#")))
    elements.append("#" * header_level + " " + header_text)
    elements.append("")
    
    # Add paragraphs
    num_paragraphs = draw(st.integers(min_value=1, max_value=3))
    for _ in range(num_paragraphs):
        paragraph = draw(st.text(min_size=20, max_size=200, alphabet=st.characters(blacklist_characters="\n")))
        elements.append(paragraph)
        elements.append("")
    
    return "\n".join(elements)


@st.composite
def valid_filename(draw: Any) -> str:
    """
    Generate valid filename strings.
    
    Includes alphanumeric characters, underscores, hyphens, and dots.
    """
    name = draw(st.text(
        min_size=1,
        max_size=50,
        alphabet=st.characters(
            whitelist_categories=("Lu", "Ll", "Nd"),
            whitelist_characters="_-.",
        )
    ).filter(lambda x: len(x.strip("._-")) > 0))
    return name


@st.composite
def file_extension(draw: Any) -> str:
    """
    Generate common file extensions.
    """
    extensions = [".txt", ".md", ".pdf", ".doc", ".docx", ".json", ".csv"]
    return draw(st.sampled_from(extensions))


# =============================================================================
# Document Strategies
# =============================================================================

@st.composite
def document_metadata(draw: Any) -> dict[str, Any]:
    """
    Generate document metadata dictionary.
    
    Includes common fields like source, page, chunk_index, etc.
    """
    metadata = {}
    
    # Always include source
    filename = draw(valid_filename())
    extension = draw(file_extension())
    metadata["source"] = filename + extension
    
    # Optional fields
    if draw(st.booleans()):
        metadata["page"] = draw(st.integers(min_value=1, max_value=1000) | st.none())
    
    if draw(st.booleans()):
        metadata["chunk_index"] = draw(st.integers(min_value=0, max_value=100))
    
    if draw(st.booleans()):
        metadata["title"] = draw(st.text(min_size=5, max_size=100))
    
    if draw(st.booleans()):
        metadata["author"] = draw(st.text(min_size=3, max_size=50))
    
    return metadata


@st.composite
def langchain_document(
    draw: Any,
    min_content_size: int = 10,
    max_content_size: int = 1000,
) -> Document:
    """
    Generate LangChain Document objects.
    
    Args:
        min_content_size: Minimum size of page content
        max_content_size: Maximum size of page content
    
    Returns:
        Document object with content and metadata
    """
    content = draw(st.text(min_size=min_content_size, max_size=max_content_size))
    metadata = draw(document_metadata())
    
    return Document(page_content=content, metadata=metadata)


@st.composite
def document_list(
    draw: Any,
    min_docs: int = 1,
    max_docs: int = 10,
    min_content_size: int = 10,
    max_content_size: int = 500,
) -> list[Document]:
    """
    Generate a list of LangChain Document objects.
    
    Args:
        min_docs: Minimum number of documents
        max_docs: Maximum number of documents
        min_content_size: Minimum size of each document's content
        max_content_size: Maximum size of each document's content
    
    Returns:
        List of Document objects
    """
    num_docs = draw(st.integers(min_value=min_docs, max_value=max_docs))
    docs = [
        draw(langchain_document(
            min_content_size=min_content_size,
            max_content_size=max_content_size,
        ))
        for _ in range(num_docs)
    ]
    return docs


# =============================================================================
# Chunk Strategies
# =============================================================================

@st.composite
def chunk_size_and_overlap(draw: Any) -> tuple[int, int]:
    """
    Generate valid chunk_size and chunk_overlap pairs.
    
    Ensures:
    - chunk_size >= 100 (minimum required)
    - chunk_overlap < chunk_size
    - overlap is typically 10-50% of chunk_size
    """
    chunk_size = draw(st.integers(min_value=100, max_value=2000))
    # Overlap should be less than chunk_size and typically 10-50% of chunk_size
    max_overlap = min(chunk_size - 1, int(chunk_size * 0.5))
    chunk_overlap = draw(st.integers(min_value=0, max_value=max_overlap))
    
    return chunk_size, chunk_overlap


# =============================================================================
# Configuration Strategies
# =============================================================================

@st.composite
def api_key(draw: Any) -> str:
    """
    Generate API key-like strings.
    
    Format: prefix + random alphanumeric characters
    """
    prefix = draw(st.sampled_from(["sk-", "key-", "api-", ""]))
    key_length = draw(st.integers(min_value=20, max_value=64))
    key = draw(st.text(
        min_size=key_length,
        max_size=key_length,
        alphabet=st.characters(
            whitelist_categories=("Lu", "Ll", "Nd"),
        )
    ))
    return prefix + key


@st.composite
def openai_model_name(draw: Any) -> str:
    """
    Generate valid OpenAI model names.
    """
    models = [
        "gpt-4",
        "gpt-4-turbo",
        "gpt-4-turbo-preview",
        "gpt-3.5-turbo",
        "gpt-3.5-turbo-16k",
        "text-embedding-ada-002",
        "text-embedding-3-small",
        "text-embedding-3-large",
    ]
    return draw(st.sampled_from(models))


@st.composite
def temperature_value(draw: Any) -> float:
    """
    Generate valid temperature values for LLM (0.0 to 2.0).
    """
    return draw(st.floats(min_value=0.0, max_value=2.0))


@st.composite
def environment_name(draw: Any) -> str:
    """
    Generate valid environment names.
    
    Returns one of: development, staging, production
    """
    environments = ["development", "staging", "production"]
    return draw(st.sampled_from(environments))


# =============================================================================
# Session and ID Strategies
# =============================================================================

@st.composite
def session_id(draw: Any) -> str:
    """
    Generate session ID strings.
    
    Format: UUID-like or alphanumeric string
    """
    # Generate UUID-like format
    if draw(st.booleans()):
        parts = [
            draw(st.text(min_size=8, max_size=8, alphabet="0123456789abcdef")),
            draw(st.text(min_size=4, max_size=4, alphabet="0123456789abcdef")),
            draw(st.text(min_size=4, max_size=4, alphabet="0123456789abcdef")),
            draw(st.text(min_size=4, max_size=4, alphabet="0123456789abcdef")),
            draw(st.text(min_size=12, max_size=12, alphabet="0123456789abcdef")),
        ]
        return "-".join(parts)
    else:
        # Generate alphanumeric ID
        return draw(st.text(
            min_size=16,
            max_size=32,
            alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd"))
        ))


@st.composite
def correlation_id(draw: Any) -> str:
    """
    Generate correlation ID strings (UUID format).
    """
    parts = [
        draw(st.text(min_size=8, max_size=8, alphabet="0123456789abcdef")),
        draw(st.text(min_size=4, max_size=4, alphabet="0123456789abcdef")),
        draw(st.text(min_size=4, max_size=4, alphabet="0123456789abcdef")),
        draw(st.text(min_size=4, max_size=4, alphabet="0123456789abcdef")),
        draw(st.text(min_size=12, max_size=12, alphabet="0123456789abcdef")),
    ]
    return "-".join(parts)


# =============================================================================
# Chat and Query Strategies
# =============================================================================

@st.composite
def chat_question(draw: Any) -> str:
    """
    Generate realistic chat questions.
    """
    question_starters = [
        "What is",
        "How do I",
        "Can you explain",
        "Tell me about",
        "Why does",
        "When should",
        "Where can I find",
    ]
    starter = draw(st.sampled_from(question_starters))
    topic = draw(st.text(min_size=5, max_size=100, alphabet=st.characters(blacklist_characters="?\n")))
    return f"{starter} {topic}?"


@st.composite
def chat_history(draw: Any) -> list[tuple[str, str]]:
    """
    Generate chat history (list of question-answer pairs).
    """
    num_exchanges = draw(st.integers(min_value=0, max_value=5))
    history = []
    
    for _ in range(num_exchanges):
        question = draw(chat_question())
        answer = draw(st.text(min_size=20, max_size=500))
        history.append((question, answer))
    
    return history


# =============================================================================
# Vector and Embedding Strategies
# =============================================================================

@st.composite
def embedding_vector(
    draw: Any,
    dimension: int = 1536,
    value_range: tuple[float, float] = (-1.0, 1.0),
) -> list[float]:
    """
    Generate embedding vectors.
    
    Args:
        dimension: Vector dimension (default 1536 for OpenAI ada-002)
        value_range: Range of values for each dimension
    
    Returns:
        List of floats representing an embedding vector
    """
    return [
        draw(st.floats(min_value=value_range[0], max_value=value_range[1]))
        for _ in range(dimension)
    ]


@st.composite
def similarity_score(draw: Any) -> float:
    """
    Generate similarity scores (0.0 to 1.0).
    """
    return draw(st.floats(min_value=0.0, max_value=1.0))


# =============================================================================
# HTTP and API Strategies
# =============================================================================

@st.composite
def http_status_code(draw: Any) -> int:
    """
    Generate valid HTTP status codes.
    """
    codes = [200, 201, 204, 400, 401, 403, 404, 422, 429, 500, 502, 503]
    return draw(st.sampled_from(codes))


@st.composite
def content_type(draw: Any) -> str:
    """
    Generate Content-Type header values.
    """
    types = [
        "application/json",
        "application/pdf",
        "text/plain",
        "text/markdown",
        "text/html",
    ]
    return draw(st.sampled_from(types))


# Export all strategies
__all__ = [
    "text_with_newlines",
    "markdown_text",
    "valid_filename",
    "file_extension",
    "document_metadata",
    "langchain_document",
    "document_list",
    "chunk_size_and_overlap",
    "api_key",
    "openai_model_name",
    "temperature_value",
    "environment_name",
    "session_id",
    "correlation_id",
    "chat_question",
    "chat_history",
    "embedding_vector",
    "similarity_score",
    "http_status_code",
    "content_type",
]

"""
Pytest configuration and shared fixtures.

This module provides:
- Shared fixtures for all tests
- Hypothesis profile configuration
- Test environment setup
"""

import os
from collections.abc import Generator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from hypothesis import settings as hypothesis_settings, Verbosity

# Configure Hypothesis profiles
hypothesis_settings.register_profile(
    "ci",
    max_examples=100,
    deadline=1000,
    verbosity=Verbosity.normal,
)
hypothesis_settings.register_profile(
    "dev",
    max_examples=20,
    deadline=500,
    verbosity=Verbosity.verbose,
)
hypothesis_settings.register_profile(
    "debug",
    max_examples=10,
    deadline=None,
    verbosity=Verbosity.verbose,
)

# Load profile from environment or use dev
profile = os.environ.get("HYPOTHESIS_PROFILE", "dev")
hypothesis_settings.load_profile(profile)


@pytest.fixture(scope="session")
def test_data_dir() -> Path:
    """Get the test data directory path."""
    return Path(__file__).parent / "data"


@pytest.fixture(scope="session")
def sample_pdf_path(test_data_dir: Path) -> Path:
    """Get path to sample PDF file."""
    return test_data_dir / "sample.pdf"


@pytest.fixture(scope="session")
def sample_txt_path(test_data_dir: Path) -> Path:
    """Get path to sample TXT file."""
    return test_data_dir / "sample.txt"


@pytest.fixture(scope="session")
def sample_md_path(test_data_dir: Path) -> Path:
    """Get path to sample Markdown file."""
    return test_data_dir / "sample.md"


@pytest.fixture
def mock_env_vars() -> Generator[dict[str, str], None, None]:
    """Fixture to set environment variables for testing."""
    env_vars = {
        "ENVIRONMENT": "development",
        "OPENAI_API_KEY": "sk-test-key",
        "LLM_MODEL": "gpt-4",
        "LLM_PROVIDER": "openai",
        "CHROMA_HOST": "localhost",
        "CHROMA_PORT": "8000",
        "LOG_LEVEL": "DEBUG",
    }
    with patch.dict(os.environ, env_vars, clear=False):
        yield env_vars


@pytest.fixture
def mock_openai_client() -> MagicMock:
    """Create a mock OpenAI client."""
    mock = MagicMock()
    mock.chat.completions.create.return_value = MagicMock(
        choices=[
            MagicMock(
                message=MagicMock(content="This is a test response."),
                finish_reason="stop",
            )
        ],
        usage=MagicMock(prompt_tokens=10, completion_tokens=20, total_tokens=30),
    )
    return mock


@pytest.fixture
def mock_embeddings() -> MagicMock:
    """Create mock embeddings that return consistent vectors."""
    mock = MagicMock()
    # Return a 1536-dimensional vector (OpenAI ada-002 dimension)
    mock.embed_documents.return_value = [[0.1] * 1536]
    mock.embed_query.return_value = [0.1] * 1536
    return mock


@pytest.fixture
def mock_chromadb_client() -> MagicMock:
    """Create a mock ChromaDB client."""
    mock = MagicMock()
    mock_collection = MagicMock()
    mock_collection.count.return_value = 0
    mock_collection.add.return_value = None
    mock_collection.query.return_value = {
        "ids": [["doc1", "doc2"]],
        "documents": [["Document 1 content", "Document 2 content"]],
        "metadatas": [[{"source": "test.pdf"}, {"source": "test.txt"}]],
        "distances": [[0.1, 0.2]],
    }
    mock.get_or_create_collection.return_value = mock_collection
    mock.heartbeat.return_value = True
    return mock


@pytest.fixture
def sample_documents() -> list[dict[str, Any]]:
    """Create sample document data for testing."""
    return [
        {
            "page_content": "This is the first document about Python programming.",
            "metadata": {
                "source": "python_guide.pdf",
                "page": 1,
                "chunk_index": 0,
            },
        },
        {
            "page_content": "Machine learning is a subset of artificial intelligence.",
            "metadata": {
                "source": "ml_intro.txt",
                "page": None,
                "chunk_index": 0,
            },
        },
        {
            "page_content": "# Introduction\n\nThis is a markdown document about APIs.",
            "metadata": {
                "source": "api_docs.md",
                "page": None,
                "chunk_index": 0,
            },
        },
    ]


@pytest.fixture
def sample_chat_history() -> list[tuple[str, str]]:
    """Create sample chat history for testing."""
    return [
        ("What is Python?", "Python is a high-level programming language."),
        ("What are its main features?", "Python features include simplicity and readability."),
    ]


@pytest.fixture(autouse=True)
def reset_logging() -> Generator[None, None, None]:
    """Reset logging configuration between tests."""
    from src.utils.logging import clear_correlation_id

    clear_correlation_id()
    yield
    clear_correlation_id()


@pytest.fixture
def temp_document_dir(tmp_path: Path) -> Path:
    """Create a temporary directory with sample documents."""
    # Create sample files
    pdf_content = b"%PDF-1.4 fake pdf content"
    txt_content = "This is a sample text document.\n\nIt has multiple paragraphs."
    md_content = "# Sample Document\n\n## Introduction\n\nThis is a markdown file."

    # Write files
    (tmp_path / "sample.pdf").write_bytes(pdf_content)
    (tmp_path / "sample.txt").write_text(txt_content)
    (tmp_path / "sample.md").write_text(md_content)

    return tmp_path


# Markers for test categorization
def pytest_configure(config: pytest.Config) -> None:
    """Configure pytest markers."""
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "property: Property-based tests")
    config.addinivalue_line("markers", "slow: Slow tests")

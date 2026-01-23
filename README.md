# Production-Grade RAG Chatbot

A production-ready Retrieval-Augmented Generation (RAG) chatbot built with Python, **LangChain**, ChromaDB, and MCP support.

## Features

- **LangChain Integration**: Full LangChain orchestration for LLMs, embeddings, document loaders, and chains
- **Multiple LLM Providers**: OpenAI, Ollama support
- **Multiple Embedding Providers**: OpenAI, HuggingFace, Ollama
- **Vector Storage**: ChromaDB for efficient semantic search
- **Document Support**: PDF, TXT, Markdown ingestion
- **MCP Support**: Model Context Protocol for tool integration
- **Production Ready**: Docker, logging, health checks, structured configuration
- **Testing**: pytest + Hypothesis for unit and property-based tests

## Project Status

### ✅ Completed (Phase 1 & 2.1-2.2)

- [x] Project structure and configuration
- [x] Pydantic Settings with validation
- [x] Structured logging with correlation IDs
- [x] LLM Factory (OpenAI, Ollama)
- [x] Embeddings Factory (OpenAI, HuggingFace, Ollama)
- [x] Comprehensive test suite with pytest + Hypothesis

### 🚧 In Progress (Phase 2.3)

- [ ] Vector Store Manager (ChromaDB)
- [ ] Connection management and health checks
- [ ] Document operations (add/search/delete)

### 📋 Planned

- Phase 3: Document Ingestion Pipeline
- Phase 4: Retrieval System
- Phase 5: Chat System with Memory
- Phase 6: MCP Server
- Phase 7: REST API
- Phase 8: Docker Configuration
- Phase 9: Error Handling
- Phase 10: Integration Tests

## Technology Stack

| Category | Technology |
|----------|------------|
| **Framework** | LangChain, LangGraph |
| **LLM** | OpenAI GPT-4, Ollama |
| **Embeddings** | OpenAI ada-002, HuggingFace, Ollama |
| **Vector Store** | ChromaDB |
| **API** | FastAPI |
| **Testing** | pytest, Hypothesis |
| **Code Quality** | Black, Ruff, MyPy |
| **Logging** | structlog |
| **Containerization** | Docker, Docker Compose |

## Quick Start

### Prerequisites

```bash
# Python 3.11+
python --version

# Optional: Ollama for local LLMs
# brew install ollama
```

### Installation

```bash
# Clone repository
cd rag-chatbot

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e ".[dev]"

# Copy environment template
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### Run Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run property-based tests only
pytest -m property

# Run with Hypothesis debug profile
HYPOTHESIS_PROFILE=debug pytest
```

### Pre-commit Hooks

```bash
# Install pre-commit hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

## Project Structure

```
rag-chatbot/
├── src/
│   ├── config/settings.py      # Pydantic settings with validation
│   ├── core/
│   │   ├── llm.py               # ✅ LLM factory (OpenAI, Ollama)
│   │   ├── embeddings.py        # ✅ Embeddings factory
│   │   └── vectorstore.py       # 🚧 ChromaDB manager
│   ├── ingestion/               # 📋 Document loaders & chunking
│   ├── retrieval/               # 📋 RAG retriever & reranker
│   ├── chat/                    # 📋 Conversation chain & memory
│   ├── mcp/                     # 📋 MCP server & tools
│   ├── api/
│   │   ├── middleware.py        # ✅ Correlation ID middleware
│   │   ├── routes.py            # 📋 FastAPI endpoints
│   │   └── models.py            # 📋 Pydantic models
│   └── utils/
│       └── logging.py           # ✅ Structured logging
├── tests/
│   ├── conftest.py              # ✅ Shared fixtures
│   ├── unit/                    # ✅ Unit tests
│   ├── integration/             # 📋 Integration tests
│   └── property/                # 📋 Property-based tests
├── docker/                      # 📋 Dockerfiles & compose
├── documents/                   # Document ingestion directory
├── pyproject.toml               # ✅ Project configuration
└── .env.example                 # ✅ Environment template
```

## Configuration

All configuration is managed through environment variables. See `.env.example` for complete reference.

### Key Settings

```bash
# LLM Configuration
OPENAI_API_KEY=sk-your-key
LLM_MODEL=gpt-4
LLM_PROVIDER=openai

# Embedding Configuration
EMBEDDING_MODEL=text-embedding-ada-002
EMBEDDING_PROVIDER=openai

# ChromaDB Configuration
CHROMA_HOST=localhost
CHROMA_PORT=8000
CHROMA_COLLECTION=documents

# Application Configuration
LOG_LEVEL=INFO
DEBUG=false
```

## Testing Strategy

### Unit Tests (pytest)
- Fast, isolated tests with mocks
- Test business logic and edge cases
- Located in `tests/unit/`

### Property-Based Tests (Hypothesis)
- Generate random test inputs
- Test invariants and properties
- Catch edge cases automatically

### Integration Tests
- Test with real services (ChromaDB, etc.)
- Use testcontainers for isolation
- Located in `tests/integration/`

## Development

### Code Quality

```bash
# Format code
black src/ tests/

# Lint code
ruff check src/ tests/

# Type check
mypy src/

# Run all quality checks
pre-commit run --all-files
```

### Adding New Features

1. Update requirements in `pyproject.toml`
2. Implement feature with type hints and docstrings
3. Write unit tests (pytest)
4. Write property tests (Hypothesis) for data transformations
5. Update configuration in `settings.py` if needed
6. Run quality checks

## Documentation

- **Requirements**: `.specflow/specs/active/rag-chatbot/requirements.md`
- **Design**: `.specflow/specs/active/rag-chatbot/design.md`
- **Tasks**: `.specflow/specs/active/rag-chatbot/tasks.md`
- **Project Context**: `.specflow/project.md`

## License

MIT

## Contributing

1. Follow PEP 8 and project conventions
2. Write tests for all new code
3. Ensure all tests pass
4. Run pre-commit hooks
5. Update documentation

# =============================================================================
# Multi-stage Dockerfile for RAG Chatbot API Service
# =============================================================================
# Build stage: Create Python wheel files for dependencies
# Production stage: Copy wheels and application code, run as non-root user
# =============================================================================

# -----------------------------------------------------------------------------
# Stage 1: Builder - Build Python wheels for all dependencies
# -----------------------------------------------------------------------------
FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    make \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files
COPY pyproject.toml .

# Extract dependencies and create wheels
# This builds all dependencies as wheel files for faster installation in production
RUN pip install --no-cache-dir hatchling && \
    pip wheel --no-cache-dir --wheel-dir /app/wheels \
    langchain>=0.3.0 \
    langchain-openai>=0.2.0 \
    langchain-community>=0.3.0 \
    langchain-chroma>=0.1.0 \
    langchain-ollama>=0.2.0 \
    langgraph>=0.2.0 \
    chromadb>=0.5.0 \
    rank-bm25>=0.2.2 \
    pymupdf>=1.24.0 \
    markdown>=3.0 \
    unstructured>=0.15.0 \
    sentence-transformers>=3.0.0 \
    fastapi>=0.115.0 \
    uvicorn[standard]>=0.30.0 \
    python-multipart>=0.0.9 \
    mcp>=1.0.0 \
    pydantic>=2.9.0 \
    pydantic-settings>=2.5.0 \
    python-dotenv>=1.0.0 \
    structlog>=24.4.0 \
    httpx>=0.27.0 \
    aiofiles>=24.1.0 \
    redis>=5.0.0

# -----------------------------------------------------------------------------
# Stage 2: Production - Minimal image with only runtime dependencies
# -----------------------------------------------------------------------------
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user and group
RUN groupadd -r appuser && useradd -r -g appuser -u 1000 appuser

# Copy wheels from builder and install
COPY --from=builder /app/wheels /wheels
RUN pip install --no-cache /wheels/* && \
    rm -rf /wheels

# Copy application code
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser scripts/ ./scripts/

# Create directories for data and logs
RUN mkdir -p /app/data /app/logs /app/documents && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose API port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 --start-period=40s \
    CMD python scripts/healthcheck.py || exit 1

# Run the API server
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8080"]

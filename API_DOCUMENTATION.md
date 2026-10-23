# RAG Chatbot API Documentation

## Overview

The RAG (Retrieval-Augmented Generation) Chatbot API provides endpoints for conversational AI powered by document retrieval. It allows you to ingest documents, search through them semantically, and chat with an AI that uses retrieved context to generate accurate answers.

**Base URL:** `http://localhost:8080/api/v1`

**Version:** 1.0.0

---

## Table of Contents

- [Authentication](#authentication)
- [Endpoints](#endpoints)
  - [Health Check](#health-check)
  - [Chat](#chat)
  - [Chat Stream](#chat-stream)
  - [Search](#search)
  - [Ingest Document](#ingest-document)
  - [Ingest Batch](#ingest-batch)
  - [Ingest Text](#ingest-text)
  - [Clear Session](#clear-session)
- [Models](#models)
- [Error Handling](#error-handling)
- [Examples](#examples)
- [Administration](#administration)

---

## Authentication

Currently, the API does not require authentication. All endpoints are publicly accessible.

---

## Endpoints

### Health Check

Check the health status of the RAG system and its components.

**Endpoint:** `GET /health`

**Response:** `200 OK`

```json
{
  "status": "healthy",
  "services": {
    "vectorstore": true,
    "llm": true,
    "embeddings": true
  },
  "version": "1.0.0"
}
```

**Status Values:**
- `healthy` - All services operational
- `degraded` - Some services unavailable
- `unhealthy` - Critical services down

#### Example

```bash
curl http://localhost:8080/api/v1/health
```

---

### Chat

Send a question and receive an AI-generated answer based on ingested documents.

**Endpoint:** `POST /chat`

**Content-Type:** `application/json`

#### Request Body

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `question` | string | Yes | The user's question (1-2000 characters) |
| `session_id` | string | No | Session ID for conversation continuity. If not provided, a new session is created. |

```json
{
  "question": "What is machine learning?",
  "session_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

#### Response `200 OK`

| Field | Type | Description |
|-------|------|-------------|
| `answer` | string | Generated answer from RAG system |
| `sources` | array | List of source documents used |
| `session_id` | string | Session ID for this conversation |
| `metadata` | object | Optional metadata (processing time, tokens, etc.) |

```json
{
  "answer": "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed...",
  "sources": [
    {
      "filename": "ml_basics.pdf",
      "page": 5,
      "section": "Introduction",
      "relevance_score": 0.95,
      "content_preview": "Machine learning is a method of data analysis..."
    }
  ],
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "metadata": null
}
```

#### Example

```bash
curl -X POST http://localhost:8080/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are the different data types in Python?",
    "session_id": "my-session-123"
  }'
```

---

### Chat Stream

Send a question and receive a streaming response using Server-Sent Events (SSE).

**Endpoint:** `POST /chat/stream`

**Content-Type:** `application/json`

**Response Content-Type:** `text/event-stream`

#### Request Body

Same as [Chat](#chat) endpoint.

```json
{
  "question": "Explain neural networks in detail",
  "session_id": "my-session-123"
}
```

#### Response

Streaming response with SSE format:

```
data: Neural

data: networks

data: are

data: computational

data: models

data: ...
```

On error:
```
data: {"error": "Error message here"}
```

#### Example

```bash
curl -X POST http://localhost:8080/api/v1/chat/stream \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Explain neural networks",
    "session_id": "my-session-123"
  }'
```

---

### Search

Search for relevant documents using semantic similarity.

**Endpoint:** `POST /search`

**Content-Type:** `application/json`

#### Request Body

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `query` | string | Yes | - | Search query (1-1000 characters) |
| `limit` | integer | No | 4 | Maximum results (1-20) |
| `use_mmr` | boolean | No | false | Use Maximum Marginal Relevance for diverse results |

```json
{
  "query": "neural networks",
  "limit": 5,
  "use_mmr": true
}
```

#### Response `200 OK`

| Field | Type | Description |
|-------|------|-------------|
| `results` | array | List of search results |
| `count` | integer | Number of results returned |
| `query` | string | Original search query |

```json
{
  "results": [
    {
      "content": "Neural networks consist of layers of interconnected nodes...",
      "metadata": {
        "source": "/tmp/deep_learning.pdf",
        "page": 12,
        "chunk_index": 3
      },
      "score": 0.95,
      "source": "/tmp/deep_learning.pdf"
    }
  ],
  "count": 1,
  "query": "neural networks"
}
```

#### Example

```bash
curl -X POST http://localhost:8080/api/v1/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "exception handling in Python",
    "limit": 3
  }'
```

---

### Ingest Document

Upload and ingest a document file into the knowledge base.

**Endpoint:** `POST /ingest`

**Content-Type:** `multipart/form-data`

#### Request

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `file` | file | Yes | Document file to ingest |

**Supported File Types:**
- `.pdf` - PDF documents
- `.txt` - Plain text files
- `.md` - Markdown files

**Re-ingesting replaces.** Documents are identified by file name: uploading a file whose name was ingested before replaces its old chunks instead of adding duplicates. Two different documents with the same name will therefore overwrite each other.

**Markdown frontmatter.** A `.md` file that starts with YAML frontmatter (`document_type`, `document_number`, `year`, `title`, `status`, as in the JDIH exports) gets a header such as `[Peraturan Rektor Nomor 11 Tahun 2026 tentang ...]` on every chunk, so answers can cite the right regulation from any part of it.

#### Response `201 Created`

| Field | Type | Description |
|-------|------|-------------|
| `success` | boolean | Whether ingestion was successful |
| `filename` | string | Name of ingested file |
| `file_path` | string | Path to the ingested file |
| `chunks_created` | integer | Number of text chunks created |
| `chunks_stored` | integer | Number of chunks stored in vectorstore. Lower than `chunks_created` if some chunks failed to embed and were skipped (see `error_message`) |
| `documents_loaded` | integer | Number of documents loaded from file |
| `file_type` | string | Type of file (pdf, txt, md) |
| `error_message` | string | Error message if ingestion failed |

```json
{
  "success": true,
  "filename": "research_paper.pdf",
  "file_path": "/tmp/tmpabc123/research_paper.pdf",
  "chunks_created": 45,
  "chunks_stored": 45,
  "documents_loaded": 1,
  "file_type": "pdf",
  "error_message": null
}
```

#### Example

```bash
curl -X POST http://localhost:8080/api/v1/ingest \
  -F "file=@/path/to/document.pdf"
```

---

### Ingest Batch

Upload several documents and/or `.zip` archives in one request.

**Endpoint:** `POST /ingest/batch`

**Content-Type:** `multipart/form-data`

#### Request

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `files` | file (repeatable) | Yes | `.pdf`, `.txt`, `.md` files or `.zip` archives containing them. Repeat the field for each file. |

Zip handling:
- Supported files are taken from any folder in the archive; folder names are ignored (only the file name is kept).
- macOS `__MACOSX/` entries and hidden files are skipped silently.
- Unsupported files and files over 50 MB are reported as failed entries in the response; the rest of the batch still ingests.

As with [Ingest Document](#ingest-document), a file name that was ingested before replaces its old chunks.

#### Response `201 Created`

An array with one [Ingest Document response](#response-201-created) object per file (including skipped ones, with `success: false`).

```json
[
  {
    "success": true,
    "filename": "FAQ PPID UNNES.txt",
    "file_path": "/tmp/tmpx1y2z3/FAQ PPID UNNES.txt",
    "chunks_created": 12,
    "chunks_stored": 12,
    "documents_loaded": 1,
    "file_type": "txt",
    "error_message": null
  },
  {
    "success": false,
    "filename": "notes.docx",
    "file_path": null,
    "chunks_created": 0,
    "chunks_stored": 0,
    "documents_loaded": 0,
    "file_type": "docx",
    "error_message": "unsupported file type"
  }
]
```

#### Example

```bash
# a zip
curl -X POST http://localhost:8080/api/v1/ingest/batch \
  -F "files=@knowledge-base.zip"

# several files (can be mixed with zips)
curl -X POST http://localhost:8080/api/v1/ingest/batch \
  -F "files=@faq.txt" \
  -F "files=@peraturan-11-2026.md" \
  -F "files=@more-docs.zip"
```

> Large batches can take minutes (every chunk is embedded through the embedding model). Make sure any reverse proxy in front of the API allows a long enough timeout and a large enough request body.

---

### Ingest Text

Ingest raw text sent as JSON instead of a file.

**Endpoint:** `POST /ingest/text`

**Content-Type:** `application/json`

#### Request Body

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | Yes | Document name (1-200 characters). Identifies the document: sending the same `name` again replaces the previous version. End it in `.md` for markdown handling (frontmatter header, heading-aware chunking); anything else is treated as `.txt`. |
| `text` | string | Yes | Document content (non-empty) |

```json
{
  "name": "faq-wisuda.md",
  "text": "### Question:\nKapan wisuda periode Desember?\nAnswer:\nWisuda periode Desember dilaksanakan ..."
}
```

#### Response `201 Created`

Same as [Ingest Document](#response-201-created); `filename` is the normalized name (e.g. `faq-wisuda` becomes `faq-wisuda.txt`) and `file_path` is `null`.

```json
{
  "success": true,
  "filename": "faq-wisuda.md",
  "file_path": null,
  "chunks_created": 1,
  "chunks_stored": 1,
  "documents_loaded": 1,
  "file_type": "md",
  "error_message": null
}
```

#### Example

```bash
curl -X POST http://localhost:8080/api/v1/ingest/text \
  -H "Content-Type: application/json" \
  -d '{"name": "faq-wisuda.md", "text": "### Question:\nKapan wisuda?\nAnswer:\n..."}'
```

---

### Clear Session

Clear a conversation session and its history.

**Endpoint:** `DELETE /session/{session_id}`

#### Path Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `session_id` | string | Session ID to clear |

#### Response `200 OK`

```json
{
  "message": "Session cleared successfully",
  "session_id": "my-session-123"
}
```

#### Response `404 Not Found`

```json
{
  "error": "SessionNotFoundError",
  "message": "Session not found: my-session-123",
  "detail": null,
  "correlation_id": "abc-123-def"
}
```

#### Example

```bash
curl -X DELETE http://localhost:8080/api/v1/session/my-session-123
```

---

## Models

### SourceDocument

Source document reference in chat responses.

| Field | Type | Description |
|-------|------|-------------|
| `filename` | string | Source document filename |
| `page` | integer | Page number (for PDFs) |
| `section` | string | Section or heading name |
| `relevance_score` | float | Relevance score (0-1) |
| `content_preview` | string | Preview of matched content |

### SearchResult

Single search result.

| Field | Type | Description |
|-------|------|-------------|
| `content` | string | Document content snippet |
| `metadata` | object | Document metadata |
| `score` | float | Relevance score (0-1) |
| `source` | string | Source document identifier |

### ErrorResponse

Standard error response.

| Field | Type | Description |
|-------|------|-------------|
| `error` | string | Error type/category |
| `message` | string | Human-readable error message |
| `detail` | string | Detailed error information |
| `correlation_id` | string | Request correlation ID for tracing |

---

## Error Handling

The API uses standard HTTP status codes and returns structured error responses.

### HTTP Status Codes

| Code | Description |
|------|-------------|
| `200` | Success |
| `201` | Created (successful ingestion) |
| `400` | Bad Request (validation error) |
| `404` | Not Found (session not found) |
| `500` | Internal Server Error |

### Error Types

| Error Type | Description |
|------------|-------------|
| `ValidationError` | Invalid request parameters |
| `ChatGenerationError` | Failed to generate chat response |
| `RetrievalError` | Document retrieval failed |
| `DocumentLoadError` | Failed to load document |
| `DocumentParseError` | Failed to parse document |
| `DocumentIngestionError` | Failed to ingest document |
| `SessionNotFoundError` | Session does not exist |
| `ChatMemoryError` | Memory operation failed |

### Example Error Response

```json
{
  "error": "ValidationError",
  "message": "Unsupported file type: .docx",
  "detail": {
    "filename": "document.docx",
    "extension": ".docx",
    "allowed_extensions": [".pdf", ".txt", ".md"]
  },
  "correlation_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

---

## Examples

### Complete Workflow

#### 1. Check System Health

```bash
curl http://localhost:8080/api/v1/health
```

#### 2. Ingest a Document

```bash
curl -X POST http://localhost:8080/api/v1/ingest \
  -F "file=@python_guide.pdf"
```

Response:
```json
{
  "success": true,
  "filename": "python_guide.pdf",
  "chunks_created": 25,
  "chunks_stored": 25,
  "documents_loaded": 1,
  "file_type": "pdf"
}
```

#### 3. Search Documents

```bash
curl -X POST http://localhost:8080/api/v1/search \
  -H "Content-Type: application/json" \
  -d '{"query": "list comprehension", "limit": 3}'
```

#### 4. Chat with Context

```bash
curl -X POST http://localhost:8080/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "question": "How do I create a list comprehension in Python?",
    "session_id": "user-123"
  }'
```

#### 5. Follow-up Question (Same Session)

```bash
curl -X POST http://localhost:8080/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Can you show me a more complex example?",
    "session_id": "user-123"
  }'
```

#### 6. Clear Session When Done

```bash
curl -X DELETE http://localhost:8080/api/v1/session/user-123
```

---

## Administration

### Wipe Knowledge Base

Through the API (deletes every chunk in the collection; cannot be undone):

```bash
curl -X DELETE http://localhost:8080/api/v1/collection
```

Response:
```json
{
  "message": "Collection cleared successfully",
  "documents_deleted": 553
}
```

Or, to also remove the ChromaDB data volume:

```bash
# Stop containers
docker compose -f docker/docker-compose.yml down

# Remove ChromaDB data volume
docker volume rm docker_chroma-data

# Restart containers
docker compose -f docker/docker-compose.yml up -d
```

### Re-ingest Everything

Needed after changes to chunking or the embedding model, since existing chunks keep the old format:

```bash
curl -X DELETE http://localhost:8080/api/v1/collection
curl -X POST http://localhost:8080/api/v1/ingest/batch -F "files=@knowledge-base.zip"
```

To update a single document afterwards, ingest it again under the same file name; no wipe needed.

### View Logs

```bash
# API logs
docker logs rag-api

# ChromaDB logs
docker logs chromadb

# Follow logs in real-time
docker logs -f rag-api
```

### Service Ports

| Service | Port | Description |
|---------|------|-------------|
| RAG API | 8080 | Main API endpoint |
| ChromaDB | 8000 | Vector database |
| MCP Server | 3000 | MCP protocol server (SSE) |
| Redis | 6379 | Cache (optional) |

---

## Rate Limits

Currently, no rate limits are enforced. For production deployments, consider adding rate limiting via a reverse proxy or API gateway.

---

## Changelog

### Unreleased
- `POST /ingest/batch`: many files and/or `.zip` archives per request
- `POST /ingest/text`: ingest raw text as JSON
- Re-ingesting a file name replaces its old chunks instead of duplicating them
- Uploaded files keep their real name (sources no longer show `tmpXXXX.md`)
- Chunks split on markdown headings (Pasal, FAQ `### Question:`) and carry a header with the document (from frontmatter) and section
- Embeddings are sent in batches of 32 chunks; a chunk that fails to embed is skipped and reported in `error_message` instead of failing the whole file

### v1.0.0
- Initial release
- Chat endpoint with conversation memory
- Streaming chat support (SSE)
- Document ingestion (PDF, TXT, MD)
- Semantic search with MMR support
- Session management
- Health check endpoint

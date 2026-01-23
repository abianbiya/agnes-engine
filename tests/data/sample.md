# RAG Chatbot Documentation

## Introduction

This document provides an overview of the RAG (Retrieval-Augmented Generation) chatbot system. The system combines the power of large language models with a knowledge base to provide accurate, contextual responses.

## Architecture

The RAG chatbot consists of several key components:

### Document Ingestion
- Supports multiple document formats (PDF, TXT, MD)
- Chunks documents intelligently at natural boundaries
- Generates embeddings for semantic search
- Stores in ChromaDB vector database

### Retrieval System
- Semantic similarity search using embeddings
- Maximum Marginal Relevance (MMR) for diverse results
- Configurable number of retrieved documents
- Support for metadata filtering

### Chat Interface
- Conversational memory for context
- Streaming responses for better UX
- Tool integration via MCP
- RESTful API endpoints

## Getting Started

### Installation

```bash
pip install -e ".[dev]"
```

### Configuration

Create a `.env` file based on `.env.example`:

```
OPENAI_API_KEY=your-key-here
CHROMA_HOST=localhost
CHROMA_PORT=8000
```

### Usage

1. **Ingest Documents**
   ```python
   from src.ingestion import IngestionPipeline
   
   pipeline = IngestionPipeline(loader, chunker, vectorstore)
   result = await pipeline.ingest_file("document.pdf")
   ```

2. **Query the Chatbot**
   ```python
   from src.chat import ChatEngine
   
   engine = ChatEngine(llm, vectorstore)
   response = await engine.chat("What is RAG?")
   ```

## Features

- **Multi-format Support**: PDF, TXT, and Markdown files
- **Intelligent Chunking**: Preserves context with overlapping chunks
- **Semantic Search**: Find relevant information based on meaning, not just keywords
- **Conversational Memory**: Maintains context across multiple turns
- **Tool Integration**: Extend capabilities with MCP tools
- **Production Ready**: Comprehensive logging, error handling, and testing

## API Reference

### REST Endpoints

- `POST /ingest` - Upload and ingest a document
- `POST /chat` - Send a message to the chatbot
- `GET /health` - Check system health
- `GET /collections` - List available collections

## Best Practices

1. **Document Preparation**: Clean and format documents before ingestion
2. **Chunk Size**: Balance between context (larger) and precision (smaller)
3. **Retrieval Count**: More documents = more context but slower responses
4. **Error Handling**: Always check ingestion results before querying

## Troubleshooting

### Common Issues

**Issue**: Documents not loading
**Solution**: Check file format is supported (.pdf, .txt, .md)

**Issue**: Poor search results
**Solution**: Adjust chunk size and overlap parameters

**Issue**: Slow responses
**Solution**: Reduce number of retrieved documents

## Contributing

See CONTRIBUTING.md for development guidelines.

## License

MIT License - see LICENSE file for details.

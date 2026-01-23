"""
Ingestion module for RAG chatbot.

This module provides document ingestion functionality including:
- Document loading from various formats (PDF, TXT, MD)
- Text chunking with overlap
- Complete ingestion pipeline orchestration
"""

from src.ingestion.chunker import TextChunker
from src.ingestion.loader import DocumentLoaderFactory, load_document
from src.ingestion.pipeline import IngestionPipeline, IngestionResult

__all__ = [
    "DocumentLoaderFactory",
    "load_document",
    "TextChunker",
    "IngestionPipeline",
    "IngestionResult",
]
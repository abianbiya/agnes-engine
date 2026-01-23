"""
Ingestion pipeline for RAG chatbot.

This module orchestrates the complete document ingestion workflow:
1. Load documents from various formats
2. Chunk documents into optimal sizes
3. Generate embeddings
4. Store in vector database
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import List

from langchain_core.documents import Document

from src.core.vectorstore import VectorStoreManager
from src.ingestion.chunker import TextChunker
from src.ingestion.loader import DocumentLoaderFactory
from src.utils.logging import LoggerMixin


@dataclass
class IngestionResult:
    """
    Result of an ingestion operation.
    
    Attributes:
        success: Whether the ingestion was successful.
        file_path: Path to the ingested file.
        num_documents_loaded: Number of documents loaded from the file.
        num_chunks_created: Number of chunks created from the documents.
        num_chunks_stored: Number of chunks successfully stored in vector DB.
        error: Error message if ingestion failed (None if successful).
    """
    success: bool
    file_path: str
    num_documents_loaded: int = 0
    num_chunks_created: int = 0
    num_chunks_stored: int = 0
    error: str | None = None
    
    def __str__(self) -> str:
        """Human-readable string representation."""
        if self.success:
            return (
                f"✓ {self.file_path}: "
                f"{self.num_documents_loaded} docs → "
                f"{self.num_chunks_created} chunks → "
                f"{self.num_chunks_stored} stored"
            )
        else:
            return f"✗ {self.file_path}: {self.error}"


class IngestionPipeline(LoggerMixin):
    """
    Complete document ingestion pipeline.
    
    Orchestrates:
    1. Document loading (PDF, TXT, MD)
    2. Text chunking with overlap
    3. Embedding generation
    4. Vector storage in ChromaDB
    
    Args:
        loader_factory: Factory for loading documents.
        chunker: Text chunker for splitting documents.
        vectorstore: Vector store manager for storage.
        
    Example:
        >>> from src.ingestion.pipeline import IngestionPipeline
        >>> from src.ingestion.loader import DocumentLoaderFactory
        >>> from src.ingestion.chunker import TextChunker
        >>> from src.core.vectorstore import VectorStoreManager
        >>> 
        >>> loader = DocumentLoaderFactory()
        >>> chunker = TextChunker(chunk_size=1000, chunk_overlap=200)
        >>> vectorstore = VectorStoreManager(...)
        >>> 
        >>> pipeline = IngestionPipeline(loader, chunker, vectorstore)
        >>> result = await pipeline.ingest_file("path/to/document.pdf")
        >>> print(result)
        ✓ path/to/document.pdf: 10 docs → 45 chunks → 45 stored
    """
    
    def __init__(
        self,
        loader_factory: DocumentLoaderFactory,
        chunker: TextChunker,
        vectorstore: VectorStoreManager,
    ) -> None:
        """
        Initialize the ingestion pipeline.
        
        Args:
            loader_factory: Factory for loading documents from files.
            chunker: Text chunker for splitting documents.
            vectorstore: Vector store manager for storage.
        """
        super().__init__()
        
        self.loader_factory = loader_factory
        self.chunker = chunker
        self.vectorstore = vectorstore
        
        self.logger.info(
            "IngestionPipeline initialized",
            chunk_size=chunker.chunk_size,
            chunk_overlap=chunker.chunk_overlap,
            collection_name=vectorstore.collection_name
        )
    
    async def ingest_file(self, file_path: str | Path) -> IngestionResult:
        """
        Ingest a single file into the vector database.
        
        Complete workflow:
        1. Load document from file
        2. Chunk into smaller pieces
        3. Generate embeddings
        4. Store in vector database
        
        Args:
            file_path: Path to the file to ingest.
            
        Returns:
            IngestionResult with details of the operation.
            
        Example:
            >>> pipeline = IngestionPipeline(loader, chunker, vectorstore)
            >>> result = await pipeline.ingest_file("document.pdf")
            >>> result.success
            True
            >>> result.num_chunks_stored
            45
        """
        path = Path(file_path)
        path_str = str(path)
        
        self.logger.info("Starting file ingestion", file_path=path_str)
        
        try:
            # Step 1: Load documents
            self.logger.debug("Loading documents", file_path=path_str)
            documents = self.loader_factory.load(path)
            
            if not documents:
                error_msg = "No documents loaded from file"
                self.logger.warning(error_msg, file_path=path_str)
                return IngestionResult(
                    success=False,
                    file_path=path_str,
                    error=error_msg
                )
            
            self.logger.debug(
                "Documents loaded",
                file_path=path_str,
                num_documents=len(documents)
            )
            
            # Step 2: Chunk documents
            self.logger.debug("Chunking documents", file_path=path_str)
            chunks = self.chunker.chunk_documents(documents)
            
            if not chunks:
                error_msg = "No chunks created from documents"
                self.logger.warning(error_msg, file_path=path_str)
                return IngestionResult(
                    success=False,
                    file_path=path_str,
                    num_documents_loaded=len(documents),
                    error=error_msg
                )
            
            self.logger.debug(
                "Documents chunked",
                file_path=path_str,
                num_chunks=len(chunks)
            )
            
            # Step 3 & 4: Generate embeddings and store (handled by vectorstore)
            self.logger.debug("Storing chunks in vector database", file_path=path_str)
            ids = await self.vectorstore.add_documents(chunks)
            
            self.logger.info(
                "File ingestion complete",
                file_path=path_str,
                num_documents=len(documents),
                num_chunks=len(chunks),
                num_stored=len(ids)
            )
            
            return IngestionResult(
                success=True,
                file_path=path_str,
                num_documents_loaded=len(documents),
                num_chunks_created=len(chunks),
                num_chunks_stored=len(ids)
            )
            
        except Exception as e:
            error_msg = str(e)
            self.logger.error(
                "File ingestion failed",
                file_path=path_str,
                error=error_msg,
                exc_info=True
            )
            return IngestionResult(
                success=False,
                file_path=path_str,
                error=error_msg
            )
    
    async def ingest_directory(
        self,
        directory_path: str | Path,
        recursive: bool = True,
        pattern: str = "*"
    ) -> List[IngestionResult]:
        """
        Ingest all supported files from a directory.
        
        Args:
            directory_path: Path to the directory.
            recursive: Whether to search subdirectories.
            pattern: Glob pattern for filtering files (default: "*").
            
        Returns:
            List of IngestionResult objects, one per file.
            
        Example:
            >>> pipeline = IngestionPipeline(loader, chunker, vectorstore)
            >>> results = await pipeline.ingest_directory("documents/")
            >>> successful = [r for r in results if r.success]
            >>> len(successful)
            10
        """
        directory = Path(directory_path)
        
        if not directory.exists():
            self.logger.error("Directory not found", directory=str(directory))
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        if not directory.is_dir():
            self.logger.error("Path is not a directory", path=str(directory))
            raise ValueError(f"Path is not a directory: {directory}")
        
        self.logger.info(
            "Starting directory ingestion",
            directory=str(directory),
            recursive=recursive,
            pattern=pattern
        )
        
        # Find all supported files
        if recursive:
            all_files = list(directory.rglob(pattern))
        else:
            all_files = list(directory.glob(pattern))
        
        # Filter to only supported formats
        supported_files = [
            f for f in all_files
            if f.is_file() and self.loader_factory.is_supported(f)
        ]
        
        self.logger.info(
            "Found supported files",
            total_files=len(all_files),
            supported_files=len(supported_files)
        )
        
        if not supported_files:
            self.logger.warning("No supported files found in directory")
            return []
        
        # Ingest each file
        results = []
        for file_path in supported_files:
            result = await self.ingest_file(file_path)
            results.append(result)
        
        # Summary statistics
        successful = sum(1 for r in results if r.success)
        failed = len(results) - successful
        total_chunks = sum(r.num_chunks_stored for r in results if r.success)
        
        self.logger.info(
            "Directory ingestion complete",
            directory=str(directory),
            total_files=len(results),
            successful=successful,
            failed=failed,
            total_chunks_stored=total_chunks
        )
        
        return results
    
    async def ingest_bytes(
        self,
        content: bytes,
        file_name: str,
        metadata: dict | None = None
    ) -> IngestionResult:
        """
        Ingest document content from bytes (for API uploads).
        
        Writes bytes to a temporary file, ingests it, then cleans up.
        
        Args:
            content: Raw file content as bytes.
            file_name: Name of the file (used to determine format).
            metadata: Optional metadata to attach to all chunks.
            
        Returns:
            IngestionResult with details of the operation.
            
        Example:
            >>> pipeline = IngestionPipeline(loader, chunker, vectorstore)
            >>> with open("doc.pdf", "rb") as f:
            ...     content = f.read()
            >>> result = await pipeline.ingest_bytes(content, "doc.pdf")
            >>> result.success
            True
        """
        import tempfile
        
        self.logger.info("Starting bytes ingestion", file_name=file_name)
        
        # Create temporary file
        suffix = Path(file_name).suffix
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp_file:
            tmp_path = Path(tmp_file.name)
            tmp_file.write(content)
        
        try:
            # Ingest the temporary file
            result = await self.ingest_file(tmp_path)
            
            # Update file_path in result to use original name
            result.file_path = file_name
            
            # Optionally add custom metadata to stored chunks
            # (This would require modifying the vectorstore or tracking chunk IDs)
            
            return result
            
        finally:
            # Clean up temporary file
            if tmp_path.exists():
                tmp_path.unlink()
                self.logger.debug("Temporary file deleted", tmp_path=str(tmp_path))
    
    async def ingest_text(
        self,
        text: str,
        source_name: str = "text_input",
        metadata: dict | None = None
    ) -> IngestionResult:
        """
        Ingest raw text directly (for API text input).
        
        Args:
            text: Raw text content to ingest.
            source_name: Name to use as source in metadata.
            metadata: Optional metadata to attach to all chunks.
            
        Returns:
            IngestionResult with details of the operation.
            
        Example:
            >>> pipeline = IngestionPipeline(loader, chunker, vectorstore)
            >>> text = "This is a long document..." * 100
            >>> result = await pipeline.ingest_text(text, source_name="user_input")
            >>> result.success
            True
        """
        self.logger.info("Starting text ingestion", source_name=source_name, text_length=len(text))
        
        try:
            # Create a document from the text
            base_metadata = {
                "source": source_name,
                "file_name": source_name,
                "file_type": "text",
            }
            
            if metadata:
                base_metadata.update(metadata)
            
            document = Document(page_content=text, metadata=base_metadata)
            
            # Chunk the document
            self.logger.debug("Chunking text", source_name=source_name)
            chunks = self.chunker.chunk_documents([document])
            
            if not chunks:
                error_msg = "No chunks created from text"
                self.logger.warning(error_msg, source_name=source_name)
                return IngestionResult(
                    success=False,
                    file_path=source_name,
                    num_documents_loaded=1,
                    error=error_msg
                )
            
            # Store in vector database
            self.logger.debug("Storing chunks in vector database", source_name=source_name)
            ids = await self.vectorstore.add_documents(chunks)
            
            self.logger.info(
                "Text ingestion complete",
                source_name=source_name,
                num_chunks=len(chunks),
                num_stored=len(ids)
            )
            
            return IngestionResult(
                success=True,
                file_path=source_name,
                num_documents_loaded=1,
                num_chunks_created=len(chunks),
                num_chunks_stored=len(ids)
            )
            
        except Exception as e:
            error_msg = str(e)
            self.logger.error(
                "Text ingestion failed",
                source_name=source_name,
                error=error_msg,
                exc_info=True
            )
            return IngestionResult(
                success=False,
                file_path=source_name,
                error=error_msg
            )
    
    async def get_stats(self) -> dict:
        """
        Get statistics about the current vector store state.
        
        Returns:
            Dictionary with vector store statistics.
            
        Example:
            >>> pipeline = IngestionPipeline(loader, chunker, vectorstore)
            >>> stats = await pipeline.get_stats()
            >>> stats["total_documents"]
            150
        """
        try:
            count = await self.vectorstore.get_collection_count()
            
            stats = {
                "collection_name": self.vectorstore.collection_name,
                "total_documents": count,
                "chunk_size": self.chunker.chunk_size,
                "chunk_overlap": self.chunker.chunk_overlap,
            }
            
            self.logger.debug("Pipeline statistics retrieved", **stats)
            
            return stats
            
        except Exception as e:
            self.logger.error("Failed to get pipeline statistics", error=str(e))
            raise

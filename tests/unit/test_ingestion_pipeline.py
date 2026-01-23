"""
Unit tests for ingestion pipeline module.

Tests the complete ingestion workflow including:
- Single file ingestion
- Directory ingestion
- Bytes ingestion
- Text ingestion
- Error handling
"""

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest
from langchain_core.documents import Document

from src.ingestion.chunker import TextChunker
from src.ingestion.loader import DocumentLoaderFactory
from src.ingestion.pipeline import IngestionPipeline, IngestionResult


@pytest.fixture
def mock_loader_factory():
    """Create a mock loader factory."""
    factory = Mock(spec=DocumentLoaderFactory)
    factory.is_supported = Mock(return_value=True)
    return factory


@pytest.fixture
def mock_chunker():
    """Create a mock chunker."""
    chunker = Mock(spec=TextChunker)
    chunker.chunk_size = 1000
    chunker.chunk_overlap = 200
    return chunker


@pytest.fixture
def mock_vectorstore():
    """Create a mock vectorstore."""
    vectorstore = AsyncMock()
    vectorstore.collection_name = "test_collection"
    return vectorstore


@pytest.fixture
def pipeline(mock_loader_factory, mock_chunker, mock_vectorstore):
    """Create a pipeline instance with mocks."""
    return IngestionPipeline(
        loader_factory=mock_loader_factory,
        chunker=mock_chunker,
        vectorstore=mock_vectorstore,
    )


class TestIngestionResult:
    """Test IngestionResult dataclass."""

    def test_create_successful_result(self):
        """Should create a successful ingestion result."""
        result = IngestionResult(
            success=True,
            file_path="/data/test.pdf",
            num_documents_loaded=5,
            num_chunks_created=25,
            num_chunks_stored=25,
        )

        assert result.success is True
        assert result.file_path == "/data/test.pdf"
        assert result.num_documents_loaded == 5
        assert result.num_chunks_created == 25
        assert result.num_chunks_stored == 25
        assert result.error is None

    def test_create_failed_result(self):
        """Should create a failed ingestion result with error."""
        result = IngestionResult(
            success=False,
            file_path="/data/test.pdf",
            error="File not found",
        )

        assert result.success is False
        assert result.file_path == "/data/test.pdf"
        assert result.error == "File not found"
        assert result.num_documents_loaded == 0
        assert result.num_chunks_created == 0
        assert result.num_chunks_stored == 0

    def test_str_representation_success(self):
        """Should generate correct string for successful result."""
        result = IngestionResult(
            success=True,
            file_path="/data/test.pdf",
            num_documents_loaded=5,
            num_chunks_created=25,
            num_chunks_stored=25,
        )

        result_str = str(result)
        assert "✓" in result_str
        assert "/data/test.pdf" in result_str
        assert "5 docs" in result_str
        assert "25 chunks" in result_str
        assert "25 stored" in result_str

    def test_str_representation_failure(self):
        """Should generate correct string for failed result."""
        result = IngestionResult(
            success=False,
            file_path="/data/test.pdf",
            error="File not found",
        )

        result_str = str(result)
        assert "✗" in result_str
        assert "/data/test.pdf" in result_str
        assert "File not found" in result_str


class TestIngestionPipelineInitialization:
    """Test pipeline initialization."""

    def test_initialize_pipeline(self, mock_loader_factory, mock_chunker, mock_vectorstore):
        """Should initialize pipeline with components."""
        pipeline = IngestionPipeline(
            loader_factory=mock_loader_factory,
            chunker=mock_chunker,
            vectorstore=mock_vectorstore,
        )

        assert pipeline.loader_factory == mock_loader_factory
        assert pipeline.chunker == mock_chunker
        assert pipeline.vectorstore == mock_vectorstore


class TestIngestFile:
    """Test single file ingestion."""

    @pytest.mark.asyncio
    async def test_successful_file_ingestion(self, pipeline, mock_loader_factory, mock_chunker, mock_vectorstore):
        """Should successfully ingest a file."""
        # Setup mocks
        test_docs = [
            Document(page_content="content 1", metadata={"source": "test.txt"}),
            Document(page_content="content 2", metadata={"source": "test.txt"}),
        ]
        test_chunks = [
            Document(page_content="chunk 1", metadata={"source": "test.txt"}),
            Document(page_content="chunk 2", metadata={"source": "test.txt"}),
            Document(page_content="chunk 3", metadata={"source": "test.txt"}),
        ]
        mock_loader_factory.load = Mock(return_value=test_docs)
        mock_chunker.chunk_documents = Mock(return_value=test_chunks)
        mock_vectorstore.add_documents = AsyncMock(return_value=["id1", "id2", "id3"])

        # Test ingestion
        result = await pipeline.ingest_file("test.txt")

        # Verify result
        assert result.success is True
        assert result.num_documents_loaded == 2
        assert result.num_chunks_created == 3
        assert result.num_chunks_stored == 3
        assert result.error is None

        # Verify calls
        mock_loader_factory.load.assert_called_once()
        mock_chunker.chunk_documents.assert_called_once_with(test_docs)
        mock_vectorstore.add_documents.assert_called_once_with(test_chunks)

    @pytest.mark.asyncio
    async def test_file_ingestion_no_documents_loaded(self, pipeline, mock_loader_factory):
        """Should handle case where no documents are loaded."""
        mock_loader_factory.load = Mock(return_value=[])

        result = await pipeline.ingest_file("empty.txt")

        assert result.success is False
        assert result.error == "No documents loaded from file"
        assert result.num_documents_loaded == 0

    @pytest.mark.asyncio
    async def test_file_ingestion_no_chunks_created(self, pipeline, mock_loader_factory, mock_chunker):
        """Should handle case where no chunks are created."""
        test_docs = [Document(page_content="", metadata={})]
        mock_loader_factory.load = Mock(return_value=test_docs)
        mock_chunker.chunk_documents = Mock(return_value=[])

        result = await pipeline.ingest_file("empty_content.txt")

        assert result.success is False
        assert result.error == "No chunks created from documents"
        assert result.num_documents_loaded == 1
        assert result.num_chunks_created == 0

    @pytest.mark.asyncio
    async def test_file_ingestion_with_exception(self, pipeline, mock_loader_factory):
        """Should handle exceptions during ingestion."""
        mock_loader_factory.load = Mock(side_effect=ValueError("Test error"))

        result = await pipeline.ingest_file("error.txt")

        assert result.success is False
        assert "Test error" in result.error

    @pytest.mark.asyncio
    async def test_file_ingestion_accepts_path_object(self, pipeline, mock_loader_factory, mock_chunker, mock_vectorstore):
        """Should accept Path object as input."""
        test_docs = [Document(page_content="content", metadata={})]
        test_chunks = [Document(page_content="chunk", metadata={})]
        mock_loader_factory.load = Mock(return_value=test_docs)
        mock_chunker.chunk_documents = Mock(return_value=test_chunks)
        mock_vectorstore.add_documents = AsyncMock(return_value=["id1"])

        path = Path("test.txt")
        result = await pipeline.ingest_file(path)

        assert result.success is True
        assert result.file_path == str(path)


class TestIngestDirectory:
    """Test directory ingestion."""

    @pytest.mark.asyncio
    async def test_successful_directory_ingestion(self, pipeline, mock_loader_factory, mock_chunker, mock_vectorstore):
        """Should successfully ingest all files in a directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test files
            test_dir = Path(tmpdir)
            file1 = test_dir / "test1.txt"
            file2 = test_dir / "test2.txt"
            file1.write_text("content 1")
            file2.write_text("content 2")

            # Setup mocks
            test_docs = [Document(page_content="content", metadata={})]
            test_chunks = [Document(page_content="chunk", metadata={})]
            mock_loader_factory.load = Mock(return_value=test_docs)
            mock_chunker.chunk_documents = Mock(return_value=test_chunks)
            mock_vectorstore.add_documents = AsyncMock(return_value=["id1"])

            # Test ingestion
            results = await pipeline.ingest_directory(test_dir)

            # Verify results
            assert len(results) == 2
            assert all(r.success for r in results)

    @pytest.mark.asyncio
    async def test_directory_not_found(self, pipeline):
        """Should raise FileNotFoundError for non-existent directory."""
        with pytest.raises(FileNotFoundError, match="Directory not found"):
            await pipeline.ingest_directory("/nonexistent/directory")

    @pytest.mark.asyncio
    async def test_path_is_not_directory(self, pipeline):
        """Should raise ValueError if path is not a directory."""
        with tempfile.NamedTemporaryFile() as tmpfile:
            with pytest.raises(ValueError, match="Path is not a directory"):
                await pipeline.ingest_directory(tmpfile.name)

    @pytest.mark.asyncio
    async def test_directory_ingestion_recursive(self, pipeline, mock_loader_factory, mock_chunker, mock_vectorstore):
        """Should ingest files recursively in subdirectories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create nested structure
            test_dir = Path(tmpdir)
            subdir = test_dir / "subdir"
            subdir.mkdir()
            
            file1 = test_dir / "test1.txt"
            file2 = subdir / "test2.txt"
            file1.write_text("content 1")
            file2.write_text("content 2")

            # Setup mocks
            test_docs = [Document(page_content="content", metadata={})]
            test_chunks = [Document(page_content="chunk", metadata={})]
            mock_loader_factory.load = Mock(return_value=test_docs)
            mock_chunker.chunk_documents = Mock(return_value=test_chunks)
            mock_vectorstore.add_documents = AsyncMock(return_value=["id1"])

            # Test recursive ingestion
            results = await pipeline.ingest_directory(test_dir, recursive=True)

            assert len(results) == 2

    @pytest.mark.asyncio
    async def test_directory_ingestion_non_recursive(self, pipeline, mock_loader_factory, mock_chunker, mock_vectorstore):
        """Should only ingest files in top-level directory when not recursive."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create nested structure
            test_dir = Path(tmpdir)
            subdir = test_dir / "subdir"
            subdir.mkdir()
            
            file1 = test_dir / "test1.txt"
            file2 = subdir / "test2.txt"
            file1.write_text("content 1")
            file2.write_text("content 2")

            # Setup mocks
            test_docs = [Document(page_content="content", metadata={})]
            test_chunks = [Document(page_content="chunk", metadata={})]
            mock_loader_factory.load = Mock(return_value=test_docs)
            mock_chunker.chunk_documents = Mock(return_value=test_chunks)
            mock_vectorstore.add_documents = AsyncMock(return_value=["id1"])

            # Test non-recursive ingestion
            results = await pipeline.ingest_directory(test_dir, recursive=False)

            # Should only find file1, not file2 in subdirectory
            assert len(results) == 1

    @pytest.mark.asyncio
    async def test_directory_ingestion_with_pattern(self, pipeline, mock_loader_factory, mock_chunker, mock_vectorstore):
        """Should filter files by pattern."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create files with different extensions
            test_dir = Path(tmpdir)
            txt_file = test_dir / "test.txt"
            md_file = test_dir / "test.md"
            txt_file.write_text("content")
            md_file.write_text("content")

            # Setup mocks
            test_docs = [Document(page_content="content", metadata={})]
            test_chunks = [Document(page_content="chunk", metadata={})]
            mock_loader_factory.load = Mock(return_value=test_docs)
            mock_chunker.chunk_documents = Mock(return_value=test_chunks)
            mock_vectorstore.add_documents = AsyncMock(return_value=["id1"])

            # Test with pattern
            results = await pipeline.ingest_directory(test_dir, pattern="*.txt")

            # Should only match .txt file
            assert len(results) == 1
            assert results[0].file_path == str(txt_file)

    @pytest.mark.asyncio
    async def test_directory_ingestion_no_supported_files(self, pipeline, mock_loader_factory):
        """Should return empty list when no supported files found."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_dir = Path(tmpdir)
            # Create unsupported file
            unsupported = test_dir / "test.xyz"
            unsupported.write_text("content")

            mock_loader_factory.is_supported = Mock(return_value=False)

            results = await pipeline.ingest_directory(test_dir)

            assert len(results) == 0


class TestIngestBytes:
    """Test bytes ingestion."""

    @pytest.mark.asyncio
    async def test_successful_bytes_ingestion(self, pipeline, mock_loader_factory, mock_chunker, mock_vectorstore):
        """Should successfully ingest bytes content."""
        content = b"Test content"
        
        # Setup mocks
        test_docs = [Document(page_content="content", metadata={})]
        test_chunks = [Document(page_content="chunk", metadata={})]
        mock_loader_factory.load = Mock(return_value=test_docs)
        mock_chunker.chunk_documents = Mock(return_value=test_chunks)
        mock_vectorstore.add_documents = AsyncMock(return_value=["id1"])

        result = await pipeline.ingest_bytes(content, "test.txt")

        assert result.success is True
        assert result.file_path == "test.txt"  # Should use original filename

    @pytest.mark.asyncio
    async def test_bytes_ingestion_with_metadata(self, pipeline, mock_loader_factory, mock_chunker, mock_vectorstore):
        """Should handle optional metadata parameter."""
        content = b"Test content"
        metadata = {"custom_field": "value"}
        
        # Setup mocks
        test_docs = [Document(page_content="content", metadata={})]
        test_chunks = [Document(page_content="chunk", metadata={})]
        mock_loader_factory.load = Mock(return_value=test_docs)
        mock_chunker.chunk_documents = Mock(return_value=test_chunks)
        mock_vectorstore.add_documents = AsyncMock(return_value=["id1"])

        result = await pipeline.ingest_bytes(content, "test.txt", metadata=metadata)

        assert result.success is True


class TestIngestText:
    """Test text ingestion."""

    @pytest.mark.asyncio
    async def test_successful_text_ingestion(self, pipeline, mock_chunker, mock_vectorstore):
        """Should successfully ingest raw text."""
        text = "This is test content"
        
        # Setup mocks
        test_chunks = [Document(page_content="chunk", metadata={})]
        mock_chunker.chunk_documents = Mock(return_value=test_chunks)
        mock_vectorstore.add_documents = AsyncMock(return_value=["id1"])

        result = await pipeline.ingest_text(text)

        assert result.success is True
        assert result.file_path == "text_input"  # Default source name
        assert result.num_documents_loaded == 1
        assert result.num_chunks_created == 1
        assert result.num_chunks_stored == 1

    @pytest.mark.asyncio
    async def test_text_ingestion_with_source_name(self, pipeline, mock_chunker, mock_vectorstore):
        """Should use custom source name."""
        text = "Test content"
        source_name = "custom_input"
        
        # Setup mocks
        test_chunks = [Document(page_content="chunk", metadata={})]
        mock_chunker.chunk_documents = Mock(return_value=test_chunks)
        mock_vectorstore.add_documents = AsyncMock(return_value=["id1"])

        result = await pipeline.ingest_text(text, source_name=source_name)

        assert result.success is True
        assert result.file_path == source_name

    @pytest.mark.asyncio
    async def test_text_ingestion_with_metadata(self, pipeline, mock_chunker, mock_vectorstore):
        """Should attach custom metadata."""
        text = "Test content"
        metadata = {"user_id": "123", "category": "test"}
        
        # Setup mocks to capture document passed to chunker
        test_chunks = [Document(page_content="chunk", metadata={})]
        mock_chunker.chunk_documents = Mock(return_value=test_chunks)
        mock_vectorstore.add_documents = AsyncMock(return_value=["id1"])

        result = await pipeline.ingest_text(text, metadata=metadata)

        assert result.success is True
        
        # Verify metadata was added to document
        call_args = mock_chunker.chunk_documents.call_args[0][0]
        doc_metadata = call_args[0].metadata
        assert doc_metadata["user_id"] == "123"
        assert doc_metadata["category"] == "test"

    @pytest.mark.asyncio
    async def test_text_ingestion_no_chunks_created(self, pipeline, mock_chunker):
        """Should handle case where no chunks are created from text."""
        text = ""
        
        # Setup mock to return empty chunks
        mock_chunker.chunk_documents = Mock(return_value=[])

        result = await pipeline.ingest_text(text)

        assert result.success is False
        assert result.error == "No chunks created from text"

    @pytest.mark.asyncio
    async def test_text_ingestion_with_exception(self, pipeline, mock_chunker):
        """Should handle exceptions during text ingestion."""
        text = "Test content"
        
        # Setup mock to raise exception
        mock_chunker.chunk_documents = Mock(side_effect=ValueError("Chunking failed"))

        result = await pipeline.ingest_text(text)

        assert result.success is False
        assert "Chunking failed" in result.error


class TestGetStats:
    """Test pipeline statistics."""

    @pytest.mark.asyncio
    async def test_get_stats_successful(self, pipeline, mock_vectorstore):
        """Should return statistics about the pipeline."""
        mock_vectorstore.get_collection_count = AsyncMock(return_value=150)

        stats = await pipeline.get_stats()

        assert stats["collection_name"] == "test_collection"
        assert stats["total_documents"] == 150
        assert stats["chunk_size"] == 1000
        assert stats["chunk_overlap"] == 200

    @pytest.mark.asyncio
    async def test_get_stats_with_exception(self, pipeline, mock_vectorstore):
        """Should raise exception if stats retrieval fails."""
        mock_vectorstore.get_collection_count = AsyncMock(side_effect=Exception("Connection failed"))

        with pytest.raises(Exception, match="Connection failed"):
            await pipeline.get_stats()

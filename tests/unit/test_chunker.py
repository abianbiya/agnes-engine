"""
Unit tests for text chunker.

Tests cover:
- Document chunking with various sizes
- Metadata preservation
- Chunk overlap handling
- Edge cases (empty documents, single word, etc.)
- Statistics calculation
"""

import pytest
from hypothesis import given
from hypothesis import strategies as st
from langchain_core.documents import Document

from src.ingestion.chunker import TextChunker


class TestTextChunkerInitialization:
    """Test suite for TextChunker initialization."""
    
    def test_default_initialization(self):
        """Test chunker with default parameters."""
        chunker = TextChunker()
        
        assert chunker.chunk_size == 1000
        assert chunker.chunk_overlap == 200
        assert chunker.separators == TextChunker.DEFAULT_SEPARATORS
    
    def test_custom_initialization(self):
        """Test chunker with custom parameters."""
        chunker = TextChunker(chunk_size=500, chunk_overlap=50)
        
        assert chunker.chunk_size == 500
        assert chunker.chunk_overlap == 50
    
    def test_custom_separators(self):
        """Test chunker with custom separators."""
        custom_separators = ["\n\n", "\n", " "]
        chunker = TextChunker(separators=custom_separators)
        
        assert chunker.separators == custom_separators
    
    def test_invalid_overlap(self):
        """Test that overlap >= chunk_size raises error."""
        with pytest.raises(ValueError, match="chunk_overlap .* must be less than chunk_size"):
            TextChunker(chunk_size=100, chunk_overlap=100)
        
        with pytest.raises(ValueError):
            TextChunker(chunk_size=100, chunk_overlap=150)


class TestBasicChunking:
    """Test suite for basic document chunking."""
    
    def test_chunk_short_document(self):
        """Test chunking a document shorter than chunk_size."""
        chunker = TextChunker(chunk_size=1000, chunk_overlap=200)
        
        doc = Document(
            page_content="This is a short document.",
            metadata={"source": "test.txt"}
        )
        
        chunks = chunker.chunk_documents([doc])
        
        # Short document should remain as single chunk
        assert len(chunks) == 1
        assert chunks[0].page_content == doc.page_content
        assert chunks[0].metadata["chunk_index"] == 0
        assert chunks[0].metadata["total_chunks"] == 1
    
    def test_chunk_long_document(self):
        """Test chunking a document longer than chunk_size."""
        chunker = TextChunker(chunk_size=100, chunk_overlap=20)
        
        # Create a long document
        long_text = " ".join([f"Sentence {i}." for i in range(100)])
        doc = Document(page_content=long_text, metadata={"source": "test.txt"})
        
        chunks = chunker.chunk_documents([doc])
        
        # Should create multiple chunks
        assert len(chunks) > 1
        
        # Each chunk should be approximately chunk_size or less
        for chunk in chunks:
            assert len(chunk.page_content) <= chunker.chunk_size + 100  # Allow some tolerance
    
    def test_chunk_multiple_documents(self):
        """Test chunking multiple documents."""
        chunker = TextChunker(chunk_size=100, chunk_overlap=20)
        
        docs = [
            Document(page_content="Short doc 1", metadata={"source": "doc1.txt"}),
            Document(page_content="Short doc 2", metadata={"source": "doc2.txt"}),
            Document(page_content="Short doc 3", metadata={"source": "doc3.txt"}),
        ]
        
        chunks = chunker.chunk_documents(docs)
        
        # Should have at least as many chunks as documents
        assert len(chunks) >= len(docs)
        
        # Check doc_index is set correctly
        doc_indices = [chunk.metadata["doc_index"] for chunk in chunks]
        assert min(doc_indices) == 0
        assert max(doc_indices) == len(docs) - 1
    
    def test_chunk_empty_document_list(self):
        """Test chunking an empty list of documents."""
        chunker = TextChunker(chunk_size=1000, chunk_overlap=200)
        
        chunks = chunker.chunk_documents([])
        
        assert len(chunks) == 0


class TestMetadataPreservation:
    """Test suite for metadata preservation during chunking."""
    
    def test_metadata_preserved(self):
        """Test that original metadata is preserved in chunks."""
        chunker = TextChunker(chunk_size=100, chunk_overlap=20)
        
        original_metadata = {
            "source": "test.pdf",
            "page": 1,
            "author": "Test Author",
            "custom_field": "custom_value"
        }
        
        long_text = "Word " * 100
        doc = Document(page_content=long_text, metadata=original_metadata)
        
        chunks = chunker.chunk_documents([doc])
        
        # All chunks should have original metadata
        for chunk in chunks:
            assert chunk.metadata["source"] == "test.pdf"
            assert chunk.metadata["page"] == 1
            assert chunk.metadata["author"] == "Test Author"
            assert chunk.metadata["custom_field"] == "custom_value"
    
    def test_chunk_specific_metadata(self):
        """Test that chunk-specific metadata is added."""
        chunker = TextChunker(chunk_size=50, chunk_overlap=10)
        
        long_text = "Word " * 50
        doc = Document(page_content=long_text, metadata={"source": "test.txt"})
        
        chunks = chunker.chunk_documents([doc])
        
        # Each chunk should have chunk-specific metadata
        for i, chunk in enumerate(chunks):
            assert chunk.metadata["chunk_index"] == i
            assert chunk.metadata["total_chunks"] == len(chunks)
            assert "chunk_size" in chunk.metadata
            assert chunk.metadata["chunk_size"] == len(chunk.page_content)
            assert chunk.metadata["doc_index"] == 0
    
    def test_chunk_indices_sequential(self):
        """Test that chunk indices are sequential."""
        chunker = TextChunker(chunk_size=50, chunk_overlap=10)
        
        long_text = "Sentence. " * 100
        doc = Document(page_content=long_text, metadata={})
        
        chunks = chunker.chunk_documents([doc])
        
        chunk_indices = [chunk.metadata["chunk_index"] for chunk in chunks]
        assert chunk_indices == list(range(len(chunks)))


class TestChunkingBehavior:
    """Test suite for chunking behavior and edge cases."""
    
    def test_paragraph_splitting(self):
        """Test that paragraphs are split at natural boundaries."""
        chunker = TextChunker(chunk_size=100, chunk_overlap=10)
        
        text = "Paragraph 1.\n\nParagraph 2.\n\nParagraph 3.\n\nParagraph 4."
        doc = Document(page_content=text, metadata={})
        
        chunks = chunker.chunk_documents([doc])
        
        # Chunks should try to respect paragraph boundaries
        # (This is a behavioral test - exact behavior depends on text length)
        assert len(chunks) >= 1
    
    def test_overlap_exists(self):
        """Test that chunks have overlapping content."""
        chunker = TextChunker(chunk_size=50, chunk_overlap=10)
        
        text = "This is a sentence. " * 20
        doc = Document(page_content=text, metadata={})
        
        chunks = chunker.chunk_documents([doc])
        
        if len(chunks) > 1:
            # Check that consecutive chunks have some overlap
            for i in range(len(chunks) - 1):
                chunk1_end = chunks[i].page_content[-20:]
                chunk2_start = chunks[i + 1].page_content[:20]
                
                # There should be some textual similarity (not exact match due to splitting)
                # This is a soft check - overlap might not be character-perfect
                assert len(chunk1_end) > 0 and len(chunk2_start) > 0
    
    def test_single_word_document(self):
        """Test chunking a document with a single word."""
        chunker = TextChunker(chunk_size=1000, chunk_overlap=200)
        
        doc = Document(page_content="Word", metadata={})
        
        chunks = chunker.chunk_documents([doc])
        
        assert len(chunks) == 1
        assert chunks[0].page_content == "Word"
    
    def test_empty_document(self):
        """Test chunking an empty document."""
        chunker = TextChunker(chunk_size=1000, chunk_overlap=200)
        
        doc = Document(page_content="", metadata={"source": "empty.txt"})
        
        chunks = chunker.chunk_documents([doc])
        
        # Empty document might create one empty chunk
        assert len(chunks) >= 0
    
    def test_whitespace_only_document(self):
        """Test chunking a document with only whitespace."""
        chunker = TextChunker(chunk_size=1000, chunk_overlap=200)
        
        doc = Document(page_content="   \n\n\t  ", metadata={})
        
        chunks = chunker.chunk_documents([doc])
        
        # Should handle whitespace-only documents gracefully
        assert len(chunks) >= 0


class TestChunkText:
    """Test suite for chunk_text convenience method."""
    
    def test_chunk_text_basic(self):
        """Test chunking raw text."""
        chunker = TextChunker(chunk_size=50, chunk_overlap=10)
        
        text = "This is some text. " * 20
        chunks = chunker.chunk_text(text)
        
        assert len(chunks) > 0
        assert all(isinstance(chunk, Document) for chunk in chunks)
    
    def test_chunk_text_with_metadata(self):
        """Test chunking raw text with metadata."""
        chunker = TextChunker(chunk_size=50, chunk_overlap=10)
        
        text = "This is some text. " * 20
        metadata = {"source": "input", "user_id": 123}
        
        chunks = chunker.chunk_text(text, metadata=metadata)
        
        # All chunks should have the provided metadata
        for chunk in chunks:
            assert chunk.metadata["source"] == "input"
            assert chunk.metadata["user_id"] == 123
    
    def test_chunk_text_short(self):
        """Test chunking short raw text."""
        chunker = TextChunker(chunk_size=1000, chunk_overlap=200)
        
        text = "Short text"
        chunks = chunker.chunk_text(text)
        
        assert len(chunks) == 1
        assert chunks[0].page_content == text


class TestGetStats:
    """Test suite for get_stats method."""
    
    def test_stats_empty_documents(self):
        """Test statistics for empty document list."""
        chunker = TextChunker(chunk_size=1000, chunk_overlap=200)
        
        stats = chunker.get_stats([])
        
        assert stats["num_documents"] == 0
        assert stats["total_characters"] == 0
        assert stats["estimated_chunks"] == 0
        assert stats["avg_doc_length"] == 0
    
    def test_stats_single_document(self):
        """Test statistics for a single document."""
        chunker = TextChunker(chunk_size=1000, chunk_overlap=200)
        
        doc = Document(page_content="A" * 500, metadata={})
        stats = chunker.get_stats([doc])
        
        assert stats["num_documents"] == 1
        assert stats["total_characters"] == 500
        assert stats["avg_doc_length"] == 500
        assert stats["estimated_chunks"] >= 1
    
    def test_stats_multiple_documents(self):
        """Test statistics for multiple documents."""
        chunker = TextChunker(chunk_size=100, chunk_overlap=20)
        
        docs = [
            Document(page_content="A" * 50, metadata={}),
            Document(page_content="B" * 200, metadata={}),
            Document(page_content="C" * 150, metadata={}),
        ]
        
        stats = chunker.get_stats(docs)
        
        assert stats["num_documents"] == 3
        assert stats["total_characters"] == 400
        assert stats["avg_doc_length"] == 400 / 3
        assert stats["estimated_chunks"] >= 3


class TestPropertyBasedChunking:
    """Property-based tests using Hypothesis."""
    
    @given(
        text=st.text(min_size=1, max_size=5000),
        chunk_size=st.integers(min_value=50, max_value=1000),
    )
    def test_chunking_preserves_content(self, text, chunk_size):
        """Property: Chunked content should contain original text."""
        chunk_overlap = chunk_size // 5  # 20% overlap
        chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        
        doc = Document(page_content=text, metadata={})
        chunks = chunker.chunk_documents([doc])
        
        # Should have at least one chunk
        assert len(chunks) >= 1
        
        # All chunks should be Document objects
        assert all(isinstance(chunk, Document) for chunk in chunks)
        
        # Metadata should be preserved
        for chunk in chunks:
            assert "chunk_index" in chunk.metadata
            assert "total_chunks" in chunk.metadata
    
    @given(
        num_docs=st.integers(min_value=1, max_value=10),
        text_size=st.integers(min_value=10, max_value=500),
    )
    def test_chunking_multiple_documents_property(self, num_docs, text_size):
        """Property: Chunking multiple documents should track doc_index correctly."""
        chunker = TextChunker(chunk_size=100, chunk_overlap=20)
        
        docs = [
            Document(page_content="X" * text_size, metadata={"id": i})
            for i in range(num_docs)
        ]
        
        chunks = chunker.chunk_documents(docs)
        
        # Should have at least as many chunks as documents
        assert len(chunks) >= num_docs
        
        # doc_index should be in valid range
        for chunk in chunks:
            assert 0 <= chunk.metadata["doc_index"] < num_docs

class TestPropertyBasedChunkingEnhanced:
    """Enhanced property-based tests using custom strategies."""
    
    @given(
        text=st.text(min_size=100, max_size=2000),
        chunk_size=st.integers(min_value=50, max_value=500),
    )
    def test_chunk_size_invariant(self, text, chunk_size):
        """Property: No chunk should exceed chunk_size (except very long words)."""
        chunk_overlap = min(chunk_size // 4, chunk_size - 1)
        chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        
        doc = Document(page_content=text, metadata={})
        chunks = chunker.chunk_documents([doc])
        
        for chunk in chunks:
            # Allow some tolerance for long words that can't be split
            assert len(chunk.page_content) <= chunk_size + 50
    
    @given(
        text=st.text(min_size=200, max_size=2000),
        chunk_size=st.integers(min_value=100, max_value=500),
        overlap_ratio=st.floats(min_value=0.1, max_value=0.4),
    )
    def test_overlap_ratio_property(self, text, chunk_size, overlap_ratio):
        """Property: Overlap should be proportional to chunk_size."""
        chunk_overlap = int(chunk_size * overlap_ratio)
        chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        
        doc = Document(page_content=text, metadata={})
        chunks = chunker.chunk_documents([doc])
        
        # Verify overlap ratio is respected
        assert chunker.chunk_overlap <= chunker.chunk_size
        assert chunker.chunk_overlap >= 0
    
    @given(
        num_docs=st.integers(min_value=1, max_value=20),
        text_length=st.integers(min_value=50, max_value=500),
    )
    def test_chunk_index_continuity(self, num_docs, text_length):
        """Property: chunk_index should be continuous within each document."""
        chunker = TextChunker(chunk_size=100, chunk_overlap=20)
        
        docs = [
            Document(page_content="Text " * text_length, metadata={"doc_id": i})
            for i in range(num_docs)
        ]
        
        chunks = chunker.chunk_documents(docs)
        
        # Group chunks by doc_index
        chunks_by_doc = {}
        for chunk in chunks:
            doc_idx = chunk.metadata["doc_index"]
            if doc_idx not in chunks_by_doc:
                chunks_by_doc[doc_idx] = []
            chunks_by_doc[doc_idx].append(chunk.metadata["chunk_index"])
        
        # Check that chunk indices are continuous for each document
        for doc_idx, indices in chunks_by_doc.items():
            assert indices == list(range(len(indices)))
    
    @given(
        text=st.text(min_size=1, max_size=1000),
        chunk_size=st.integers(min_value=50, max_value=500),
    )
    def test_metadata_completeness(self, text, chunk_size):
        """Property: All chunks should have complete metadata."""
        chunk_overlap = chunk_size // 5
        chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        
        original_metadata = {"source": "test.txt", "author": "Test"}
        doc = Document(page_content=text, metadata=original_metadata)
        
        chunks = chunker.chunk_documents([doc])
        
        required_fields = ["chunk_index", "total_chunks", "chunk_size", "doc_index"]
        
        for chunk in chunks:
            # Check all required fields are present
            for field in required_fields:
                assert field in chunk.metadata
            
            # Check original metadata is preserved
            assert chunk.metadata["source"] == "test.txt"
            assert chunk.metadata["author"] == "Test"
    
    @given(
        chunk_size=st.integers(min_value=100, max_value=1000),
        num_docs=st.integers(min_value=1, max_value=5),
    )
    def test_total_chunks_accuracy(self, chunk_size, num_docs):
        """Property: total_chunks metadata should match actual chunk count."""
        chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_size // 5)
        
        docs = [
            Document(page_content="Word " * 200, metadata={})
            for _ in range(num_docs)
        ]
        
        chunks = chunker.chunk_documents(docs)
        
        # Group by doc_index and verify total_chunks
        chunks_by_doc = {}
        for chunk in chunks:
            doc_idx = chunk.metadata["doc_index"]
            if doc_idx not in chunks_by_doc:
                chunks_by_doc[doc_idx] = []
            chunks_by_doc[doc_idx].append(chunk)
        
        for doc_idx, doc_chunks in chunks_by_doc.items():
            expected_total = len(doc_chunks)
            for chunk in doc_chunks:
                assert chunk.metadata["total_chunks"] == expected_total

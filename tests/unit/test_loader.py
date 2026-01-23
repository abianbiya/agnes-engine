"""
Unit tests for document loader factory.

Tests cover:
- PDF loading
- Text loading with encoding detection
- Markdown loading
- Error handling
- Format validation
"""

import tempfile
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st
from langchain_core.documents import Document

from src.ingestion.loader import DocumentLoaderFactory, load_document


class TestDocumentLoaderFactory:
    """Test suite for DocumentLoaderFactory."""
    
    def test_initialization(self):
        """Test factory initialization."""
        factory = DocumentLoaderFactory()
        assert factory.SUPPORTED_FORMATS == {".pdf", ".txt", ".md"}
    
    def test_is_supported_formats(self):
        """Test format support checking."""
        factory = DocumentLoaderFactory()
        
        # Supported formats
        assert factory.is_supported(Path("doc.pdf")) is True
        assert factory.is_supported(Path("doc.txt")) is True
        assert factory.is_supported(Path("doc.md")) is True
        assert factory.is_supported(Path("DOC.PDF")) is True  # Case insensitive
        
        # Unsupported formats
        assert factory.is_supported(Path("doc.docx")) is False
        assert factory.is_supported(Path("doc.html")) is False
        assert factory.is_supported(Path("doc")) is False
    
    def test_load_nonexistent_file(self):
        """Test loading a file that doesn't exist."""
        factory = DocumentLoaderFactory()
        
        with pytest.raises(FileNotFoundError):
            factory.load("/path/to/nonexistent.pdf")
    
    def test_load_unsupported_format(self):
        """Test loading an unsupported file format."""
        factory = DocumentLoaderFactory()
        
        with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as tmp:
            tmp_path = Path(tmp.name)
            tmp.write(b"fake content")
        
        try:
            with pytest.raises(ValueError, match="Unsupported file format"):
                factory.load(tmp_path)
        finally:
            tmp_path.unlink()


class TestPDFLoading:
    """Test suite for PDF document loading."""
    
    def test_load_simple_pdf(self, sample_pdf_path):
        """Test loading a simple PDF file."""
        factory = DocumentLoaderFactory()
        documents = factory.load_pdf(sample_pdf_path)
        
        assert len(documents) > 0
        assert all(isinstance(doc, Document) for doc in documents)
        
        # Check first page metadata
        first_doc = documents[0]
        assert "source" in first_doc.metadata
        assert "file_name" in first_doc.metadata
        assert first_doc.metadata["file_type"] == "pdf"
        assert first_doc.metadata["page"] == 1
        assert "total_pages" in first_doc.metadata
        
        # Check content is not empty
        assert len(first_doc.page_content) > 0
    
    def test_load_multipage_pdf(self, sample_pdf_path):
        """Test loading a multi-page PDF (reuses sample PDF)."""
        factory = DocumentLoaderFactory()
        documents = factory.load_pdf(sample_pdf_path)
        
        # Just verify it can load and has proper structure
        assert len(documents) >= 1
        
        # Check page numbers are sequential
        page_numbers = [doc.metadata["page"] for doc in documents]
        assert page_numbers == list(range(1, len(documents) + 1))
        
        # Check all pages have the same total_pages
        if len(documents) > 0:
            total_pages = documents[0].metadata["total_pages"]
            assert all(doc.metadata["total_pages"] == total_pages for doc in documents)
    
    def test_pdf_empty_pages_skipped(self, sample_pdf_path):
        """Test that PDF pages are loaded correctly."""
        factory = DocumentLoaderFactory()
        documents = factory.load_pdf(sample_pdf_path)
        
        # All loaded pages should have content
        assert all(len(doc.page_content.strip()) > 0 for doc in documents)


class TestTextLoading:
    """Test suite for text document loading."""
    
    def test_load_utf8_text(self):
        """Test loading UTF-8 encoded text file."""
        factory = DocumentLoaderFactory()
        
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            suffix=".txt",
            delete=False
        ) as tmp:
            tmp_path = Path(tmp.name)
            test_content = "Hello, world!\nThis is a test file.\nWith multiple lines."
            tmp.write(test_content)
        
        try:
            documents = factory.load_text(tmp_path)
            
            assert len(documents) == 1
            assert documents[0].page_content == test_content
            assert documents[0].metadata["file_type"] == "txt"
            assert documents[0].metadata["encoding"] == "utf-8"
            assert documents[0].metadata["file_name"] == tmp_path.name
        finally:
            tmp_path.unlink()
    
    def test_load_latin1_text(self):
        """Test loading Latin-1 encoded text file."""
        factory = DocumentLoaderFactory()
        
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="latin-1",
            suffix=".txt",
            delete=False
        ) as tmp:
            tmp_path = Path(tmp.name)
            test_content = "Café résumé naïve"
            tmp.write(test_content)
        
        try:
            documents = factory.load_text(tmp_path)
            
            assert len(documents) == 1
            assert test_content in documents[0].page_content
            assert documents[0].metadata["file_type"] == "txt"
            # Encoding should be one of the tried encodings
            assert documents[0].metadata["encoding"] in ["utf-8", "latin-1", "cp1252"]
        finally:
            tmp_path.unlink()
    
    def test_load_empty_text_file(self):
        """Test loading an empty text file."""
        factory = DocumentLoaderFactory()
        
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            suffix=".txt",
            delete=False
        ) as tmp:
            tmp_path = Path(tmp.name)
            # Write nothing
        
        try:
            documents = factory.load_text(tmp_path)
            
            assert len(documents) == 1
            assert documents[0].page_content == ""
        finally:
            tmp_path.unlink()
    
    @given(st.text(min_size=1, max_size=1000))
    def test_load_text_property_based(self, text_content):
        """Property-based test: any valid text should be loadable."""
        factory = DocumentLoaderFactory()
        
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            suffix=".txt",
            delete=False
        ) as tmp:
            tmp_path = Path(tmp.name)
            tmp.write(text_content)
        
        try:
            documents = factory.load_text(tmp_path)
            
            assert len(documents) == 1
            # Normalize line endings for comparison (Windows vs Unix)
            expected_content = text_content.replace('\r\n', '\n').replace('\r', '\n')
            actual_content = documents[0].page_content.replace('\r\n', '\n').replace('\r', '\n')
            assert actual_content == expected_content
        finally:
            tmp_path.unlink()


class TestMarkdownLoading:
    """Test suite for Markdown document loading."""
    
    def test_load_simple_markdown(self):
        """Test loading a simple Markdown file."""
        factory = DocumentLoaderFactory()
        
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            suffix=".md",
            delete=False
        ) as tmp:
            tmp_path = Path(tmp.name)
            test_content = "# Header\n\nThis is **bold** text.\n\n- Item 1\n- Item 2"
            tmp.write(test_content)
        
        try:
            documents = factory.load_markdown(tmp_path)
            
            assert len(documents) == 1
            assert documents[0].page_content == test_content
            assert documents[0].metadata["file_type"] == "markdown"
            assert documents[0].metadata["encoding"] == "utf-8"
            
            # HTML conversion should be in metadata
            assert "html" in documents[0].metadata
            assert "<h1" in documents[0].metadata["html"]  # Header tag (may have attributes)
            assert "<strong>" in documents[0].metadata["html"]
        finally:
            tmp_path.unlink()
    
    def test_load_markdown_with_code_blocks(self):
        """Test loading Markdown with code blocks."""
        factory = DocumentLoaderFactory()
        
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            suffix=".md",
            delete=False
        ) as tmp:
            tmp_path = Path(tmp.name)
            test_content = "# Code Example\n\n```python\ndef hello():\n    print('world')\n```"
            tmp.write(test_content)
        
        try:
            documents = factory.load_markdown(tmp_path)
            
            assert len(documents) == 1
            assert "```python" in documents[0].page_content
            assert "def hello()" in documents[0].page_content
        finally:
            tmp_path.unlink()
    
    def test_load_markdown_preserves_structure(self):
        """Test that raw markdown is preserved for chunking."""
        factory = DocumentLoaderFactory()
        
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            suffix=".md",
            delete=False
        ) as tmp:
            tmp_path = Path(tmp.name)
            test_content = "## Section 1\n\nText.\n\n## Section 2\n\nMore text."
            tmp.write(test_content)
        
        try:
            documents = factory.load_markdown(tmp_path)
            
            # Raw markdown should be preserved
            assert documents[0].page_content == test_content
            # Not HTML-converted version
            assert "<h2>" not in documents[0].page_content
        finally:
            tmp_path.unlink()


class TestConvenienceFunction:
    """Test the convenience function."""
    
    def test_load_document_function(self):
        """Test the load_document convenience function."""
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            suffix=".txt",
            delete=False
        ) as tmp:
            tmp_path = Path(tmp.name)
            tmp.write("Test content")
        
        try:
            documents = load_document(tmp_path)
            
            assert len(documents) == 1
            assert documents[0].page_content == "Test content"
        finally:
            tmp_path.unlink()



class TestPropertyBasedLoadersEnhanced:
    """Enhanced property-based tests for document loaders."""
    
    @given(
        text_content=st.text(min_size=10, max_size=2000),
        encoding=st.sampled_from(["utf-8", "latin-1", "ascii"]),
    )
    def test_text_loader_encoding_preservation(self, text_content, encoding):
        """Property: Text loader should preserve content across different encodings."""
        # Filter out characters that can't be encoded
        try:
            encoded_text = text_content.encode(encoding)
        except UnicodeEncodeError:
            return  # Skip this test case
        
        factory = DocumentLoaderFactory()
        
        with tempfile.NamedTemporaryFile(
            mode="wb",
            suffix=".txt",
            delete=False
        ) as tmp:
            tmp_path = Path(tmp.name)
            tmp.write(encoded_text)
        
        try:
            documents = factory.load_text(tmp_path)
            
            assert len(documents) >= 1
            # Content should be loaded (may have encoding transformations)
            assert len(documents[0].page_content) > 0
        finally:
            tmp_path.unlink(missing_ok=True)
    
    @given(
        num_lines=st.integers(min_value=1, max_value=100),
        line_length=st.integers(min_value=1, max_value=200),
    )
    def test_text_loader_line_count_preservation(self, num_lines, line_length):
        """Property: Text loader should preserve line structure."""
        factory = DocumentLoaderFactory()
        
        lines = [f"Line {i}: " + "X" * line_length for i in range(num_lines)]
        text_content = "\n".join(lines)
        
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            suffix=".txt",
            delete=False
        ) as tmp:
            tmp_path = Path(tmp.name)
            tmp.write(text_content)
        
        try:
            documents = factory.load_text(tmp_path)
            
            assert len(documents) == 1
            loaded_content = documents[0].page_content
            
            # Count newlines (allowing for platform differences)
            loaded_lines = loaded_content.count('\n')
            # Should have approximately the same number of lines
            assert loaded_lines >= num_lines - 1
        finally:
            tmp_path.unlink(missing_ok=True)
    
    @given(
        header_level=st.integers(min_value=1, max_value=6),
        num_sections=st.integers(min_value=1, max_value=10),
    )
    def test_markdown_loader_structure_preservation(self, header_level, num_sections):
        """Property: Markdown loader should preserve document structure."""
        factory = DocumentLoaderFactory()
        
        # Create markdown with headers
        content_parts = []
        for i in range(num_sections):
            header = "#" * header_level + f" Section {i}\n\n"
            body = f"This is the content of section {i}.\n\n"
            content_parts.append(header + body)
        
        md_content = "".join(content_parts)
        
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            suffix=".md",
            delete=False
        ) as tmp:
            tmp_path = Path(tmp.name)
            tmp.write(md_content)
        
        try:
            documents = factory.load_markdown(tmp_path)
            
            assert len(documents) >= 1
            # Markdown headers should be preserved
            loaded_content = documents[0].page_content
            assert "Section" in loaded_content
        finally:
            tmp_path.unlink(missing_ok=True)
    
    @given(
        filename=st.text(
            min_size=5,
            max_size=50,
            alphabet=st.characters(
                whitelist_categories=("Lu", "Ll", "Nd"),
                whitelist_characters="_-",
            )
        ).filter(lambda x: len(x.strip("_-")) > 0)
    )
    def test_metadata_source_preservation(self, filename):
        """Property: Loaders should preserve source filename in metadata."""
        factory = DocumentLoaderFactory()
        
        text_content = "Test content for metadata preservation."
        
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            suffix=".txt",
            delete=False,
            prefix=filename + "_",
        ) as tmp:
            tmp_path = Path(tmp.name)
            tmp.write(text_content)
        
        try:
            documents = factory.load_text(tmp_path)
            
            assert len(documents) >= 1
            assert "source" in documents[0].metadata
            # Source should reference the file
            assert str(tmp_path) in documents[0].metadata["source"]
        finally:
            tmp_path.unlink(missing_ok=True)
    
    @given(
        chunk_count=st.integers(min_value=1, max_value=20),
        chunk_size=st.integers(min_value=10, max_value=500),
    )
    def test_loader_output_consistency(self, chunk_count, chunk_size):
        """Property: Loading the same file multiple times should give identical results."""
        factory = DocumentLoaderFactory()
        
        # Create deterministic content
        text_content = "\n".join([f"Chunk {i}: " + "X" * chunk_size for i in range(chunk_count)])
        
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            suffix=".txt",
            delete=False
        ) as tmp:
            tmp_path = Path(tmp.name)
            tmp.write(text_content)
        
        try:
            # Load the file twice
            documents1 = factory.load_text(tmp_path)
            documents2 = factory.load_text(tmp_path)
            
            # Results should be identical
            assert len(documents1) == len(documents2)
            for doc1, doc2 in zip(documents1, documents2):
                assert doc1.page_content == doc2.page_content
                # Metadata keys should be the same
                assert set(doc1.metadata.keys()) == set(doc2.metadata.keys())
        finally:
            tmp_path.unlink(missing_ok=True)
    
    @given(
        text_size=st.integers(min_value=1, max_value=5000),
    )
    def test_loader_non_empty_output(self, text_size):
        """Property: Loading a non-empty file should produce non-empty documents."""
        factory = DocumentLoaderFactory()
        
        text_content = "A" * text_size
        
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            suffix=".txt",
            delete=False
        ) as tmp:
            tmp_path = Path(tmp.name)
            tmp.write(text_content)
        
        try:
            documents = factory.load_text(tmp_path)
            
            assert len(documents) >= 1
            assert len(documents[0].page_content) > 0
            # Content length should be preserved (approximately)
            total_length = sum(len(doc.page_content) for doc in documents)
            assert total_length >= text_size * 0.9  # Allow 10% tolerance
        finally:
            tmp_path.unlink(missing_ok=True)

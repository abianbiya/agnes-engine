"""
Document loader factory for RAG chatbot.

This module provides a factory for loading documents from various file formats (PDF, TXT, MD)
and converting them to LangChain Document objects with appropriate metadata.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import fitz  # PyMuPDF
import markdown
from langchain_core.documents import Document

from src.utils.logging import LoggerMixin


class DocumentLoaderFactory(LoggerMixin):
    """
    Factory for loading documents from various file formats.
    
    Supported formats:
    - PDF (.pdf): Using PyMuPDF (fitz) for robust PDF parsing
    - Text (.txt): Plain text files with encoding detection
    - Markdown (.md): Markdown files with HTML conversion
    
    Example:
        >>> loader = DocumentLoaderFactory()
        >>> documents = loader.load("path/to/document.pdf")
        >>> for doc in documents:
        ...     print(doc.page_content)
        ...     print(doc.metadata)
    """
    
    SUPPORTED_FORMATS = {".pdf", ".txt", ".md"}
    
    def __init__(self) -> None:
        """Initialize the document loader factory."""
        super().__init__()
        self.logger.info("DocumentLoaderFactory initialized", supported_formats=list(self.SUPPORTED_FORMATS))
    
    def is_supported(self, file_path: Path) -> bool:
        """
        Check if the file format is supported.
        
        Args:
            file_path: Path to the file to check.
            
        Returns:
            True if file format is supported, False otherwise.
            
        Example:
            >>> factory = DocumentLoaderFactory()
            >>> factory.is_supported(Path("doc.pdf"))
            True
            >>> factory.is_supported(Path("doc.docx"))
            False
        """
        return file_path.suffix.lower() in self.SUPPORTED_FORMATS
    
    def load(self, file_path: str | Path) -> List[Document]:
        """
        Load a document from file path.
        
        Args:
            file_path: Path to the document file.
            
        Returns:
            List of LangChain Document objects.
            
        Raises:
            ValueError: If file format is not supported.
            FileNotFoundError: If file does not exist.
            
        Example:
            >>> loader = DocumentLoaderFactory()
            >>> documents = loader.load("path/to/document.pdf")
            >>> len(documents)
            5
        """
        path = Path(file_path)
        
        if not path.exists():
            self.logger.error("File not found", file_path=str(path))
            raise FileNotFoundError(f"File not found: {path}")
        
        if not self.is_supported(path):
            self.logger.error(
                "Unsupported file format",
                file_path=str(path),
                format=path.suffix,
                supported=list(self.SUPPORTED_FORMATS)
            )
            raise ValueError(
                f"Unsupported file format: {path.suffix}. "
                f"Supported formats: {', '.join(self.SUPPORTED_FORMATS)}"
            )
        
        self.logger.info("Loading document", file_path=str(path), format=path.suffix)
        
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            documents = self.load_pdf(path)
        elif suffix == ".txt":
            documents = self.load_text(path)
        elif suffix == ".md":
            documents = self.load_markdown(path)
        else:
            raise ValueError(f"Unsupported format: {suffix}")
        
        self.logger.info(
            "Document loaded successfully",
            file_path=str(path),
            num_documents=len(documents)
        )
        
        return documents
    
    def load_pdf(self, file_path: Path) -> List[Document]:
        """
        Load PDF document using PyMuPDF.
        
        Args:
            file_path: Path to the PDF file.
            
        Returns:
            List of Document objects, one per page.
            
        Example:
            >>> loader = DocumentLoaderFactory()
            >>> docs = loader.load_pdf(Path("document.pdf"))
            >>> docs[0].metadata["page"]
            1
        """
        self.logger.debug("Loading PDF", file_path=str(file_path))
        
        documents = []
        
        try:
            with fitz.open(file_path) as pdf:
                total_pages = len(pdf)
                
                for page_num in range(total_pages):
                    page = pdf[page_num]
                    text = page.get_text()
                    
                    # Skip empty pages
                    if not text.strip():
                        self.logger.debug(
                            "Skipping empty page",
                            file_path=str(file_path),
                            page=page_num + 1
                        )
                        continue
                    
                    metadata = {
                        "source": str(file_path),
                        "file_name": file_path.name,
                        "file_type": "pdf",
                        "page": page_num + 1,
                        "total_pages": total_pages,
                    }
                    
                    documents.append(Document(page_content=text, metadata=metadata))
                    
                    self.logger.debug(
                        "Loaded PDF page",
                        file_path=str(file_path),
                        page=page_num + 1,
                        content_length=len(text)
                    )
        
        except Exception as e:
            self.logger.error("Failed to load PDF", file_path=str(file_path), error=str(e))
            raise
        
        self.logger.info(
            "PDF loaded",
            file_path=str(file_path),
            num_pages=len(documents)
        )
        
        return documents
    
    def load_text(self, file_path: Path) -> List[Document]:
        """
        Load plain text document.
        
        Tries multiple encodings (utf-8, latin-1, cp1252) to handle various text files.
        
        Args:
            file_path: Path to the text file.
            
        Returns:
            List containing a single Document object.
            
        Example:
            >>> loader = DocumentLoaderFactory()
            >>> docs = loader.load_text(Path("document.txt"))
            >>> docs[0].metadata["file_type"]
            'txt'
        """
        self.logger.debug("Loading text file", file_path=str(file_path))
        
        # Try multiple encodings
        encodings = ["utf-8", "latin-1", "cp1252"]
        text = None
        encoding_used = None
        
        for encoding in encodings:
            try:
                with open(file_path, "r", encoding=encoding) as f:
                    text = f.read()
                encoding_used = encoding
                break
            except (UnicodeDecodeError, UnicodeError):
                continue
        
        if text is None:
            self.logger.error(
                "Failed to decode text file with any encoding",
                file_path=str(file_path),
                encodings_tried=encodings
            )
            raise ValueError(f"Could not decode file {file_path} with encodings: {encodings}")
        
        self.logger.debug(
            "Text file decoded",
            file_path=str(file_path),
            encoding=encoding_used,
            content_length=len(text)
        )
        
        metadata = {
            "source": str(file_path),
            "file_name": file_path.name,
            "file_type": "txt",
            "encoding": encoding_used,
        }
        
        documents = [Document(page_content=text, metadata=metadata)]
        
        self.logger.info("Text file loaded", file_path=str(file_path))
        
        return documents
    
    def load_markdown(self, file_path: Path) -> List[Document]:
        """
        Load Markdown document.
        
        Preserves the raw markdown for better chunking, but also stores
        HTML-rendered version in metadata for reference.
        
        Args:
            file_path: Path to the markdown file.
            
        Returns:
            List containing a single Document object.
            
        Example:
            >>> loader = DocumentLoaderFactory()
            >>> docs = loader.load_markdown(Path("README.md"))
            >>> docs[0].metadata["file_type"]
            'markdown'
        """
        self.logger.debug("Loading markdown file", file_path=str(file_path))
        
        # Try multiple encodings
        encodings = ["utf-8", "latin-1", "cp1252"]
        text = None
        encoding_used = None
        
        for encoding in encodings:
            try:
                with open(file_path, "r", encoding=encoding) as f:
                    text = f.read()
                encoding_used = encoding
                break
            except (UnicodeDecodeError, UnicodeError):
                continue
        
        if text is None:
            self.logger.error(
                "Failed to decode markdown file with any encoding",
                file_path=str(file_path),
                encodings_tried=encodings
            )
            raise ValueError(f"Could not decode file {file_path} with encodings: {encodings}")
        
        # Convert to HTML for metadata (optional, can be useful for preview)
        try:
            html = markdown.markdown(text, extensions=["extra", "codehilite", "toc"])
        except Exception as e:
            self.logger.warning(
                "Failed to convert markdown to HTML",
                file_path=str(file_path),
                error=str(e)
            )
            html = None
        
        self.logger.debug(
            "Markdown file decoded",
            file_path=str(file_path),
            encoding=encoding_used,
            content_length=len(text)
        )
        
        metadata = {
            "source": str(file_path),
            "file_name": file_path.name,
            "file_type": "markdown",
            "encoding": encoding_used,
        }
        
        if html:
            metadata["html"] = html
        
        # Use raw markdown as content for better chunking
        documents = [Document(page_content=text, metadata=metadata)]
        
        self.logger.info("Markdown file loaded", file_path=str(file_path))
        
        return documents


def load_document(file_path: str | Path) -> List[Document]:
    """
    Convenience function to load a document.
    
    Args:
        file_path: Path to the document file.
        
    Returns:
        List of LangChain Document objects.
        
    Example:
        >>> from src.ingestion.loader import load_document
        >>> documents = load_document("path/to/document.pdf")
        >>> len(documents)
        5
    """
    loader = DocumentLoaderFactory()
    return loader.load(file_path)

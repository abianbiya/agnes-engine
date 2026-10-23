"""
Text chunker for RAG chatbot.

This module provides intelligent document chunking functionality using LangChain's
RecursiveCharacterTextSplitter to split documents into optimal chunks while preserving
context and metadata.
"""

from __future__ import annotations

import re
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.utils.logging import LoggerMixin


# "### Question:\nApa ...?" -> label takes the line after a heading ending in ':'
_HEADING = re.compile(r"^(#{2,6})[ \t]+(.+?)[ \t]*$(?=\n?(.*))", re.MULTILINE)


def _heading_path(headings: list, start: int) -> str:
    """Headings enclosing `start`, outermost first, e.g. "BAB II › Pasal 2".

    A heading the chunk itself starts with is left out (it's already in the text),
    so a chunk opening on "### Question:" doesn't inherit the previous question.
    """
    stack: list = []
    for pos, level, text in headings:
        if pos > start:
            break
        stack = [h for h in stack if h[1] < level] + [(pos, level, text)]
    if stack and stack[-1][0] >= start - 1:
        stack.pop()
    return " › ".join(text for _, _, text in stack)


class TextChunker(LoggerMixin):
    """
    Intelligent text chunker for documents.
    
    Uses RecursiveCharacterTextSplitter to split documents at natural boundaries
    (paragraphs, sentences, words) while maintaining overlap for context continuity.
    Preserves all metadata from source documents and adds chunk-specific metadata.
    
    Args:
        chunk_size: Maximum size of each chunk in characters.
        chunk_overlap: Number of characters to overlap between chunks.
        separators: List of separators to use for splitting (in order of priority).
        
    Example:
        >>> chunker = TextChunker(chunk_size=1000, chunk_overlap=200)
        >>> documents = [Document(page_content="long text...", metadata={"source": "doc.pdf"})]
        >>> chunks = chunker.chunk_documents(documents)
        >>> len(chunks)
        5
        >>> chunks[0].metadata["chunk_index"]
        0
    """
    
    DEFAULT_SEPARATORS = [
        "\n### ",    # Markdown sections (FAQ "### Question:", peraturan BAB)
        "\n#### ",   # Pasal
        "\n##### ",  # Ayat
        "\n\n",  # Paragraphs
        "\n",    # Lines
        ". ",    # Sentences
        "! ",    # Sentences
        "? ",    # Sentences
        "; ",    # Clauses
        ", ",    # Phrases
        " ",     # Words
        "",      # Characters
    ]
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: List[str] | None = None,
    ) -> None:
        """
        Initialize the text chunker.
        
        Args:
            chunk_size: Maximum size of each chunk in characters (default: 1000).
            chunk_overlap: Number of characters to overlap between chunks (default: 200).
            separators: Custom list of separators (default: None, uses DEFAULT_SEPARATORS).
        """
        super().__init__()
        
        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) must be less than chunk_size ({chunk_size})"
            )
        
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators if separators is not None else self.DEFAULT_SEPARATORS
        
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=self.separators,
            length_function=len,
            is_separator_regex=False,
            add_start_index=True,
        )
        
        self.logger.info(
            "TextChunker initialized",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            num_separators=len(self.separators)
        )
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Chunk a list of documents into smaller pieces.
        
        Splits each document using RecursiveCharacterTextSplitter while preserving
        all original metadata. Adds chunk-specific metadata including:
        - chunk_index: Index of chunk within the source document
        - total_chunks: Total number of chunks from the source document
        
        Args:
            documents: List of LangChain Document objects to chunk.
            
        Returns:
            List of chunked Document objects with preserved and enhanced metadata.
            
        Example:
            >>> chunker = TextChunker(chunk_size=500, chunk_overlap=50)
            >>> docs = [Document(page_content="long text" * 200, metadata={"source": "doc.pdf"})]
            >>> chunks = chunker.chunk_documents(docs)
            >>> all("chunk_index" in chunk.metadata for chunk in chunks)
            True
        """
        if not documents:
            self.logger.warning("No documents provided for chunking")
            return []
        
        self.logger.info("Starting document chunking", num_documents=len(documents))
        
        all_chunks = []
        
        for doc_index, document in enumerate(documents):
            self.logger.debug(
                "Chunking document",
                doc_index=doc_index,
                content_length=len(document.page_content),
                metadata=document.metadata
            )
            
            # Split the document
            chunks = self.splitter.split_documents([document])
            
            headings = [
                (m.start(), len(m.group(1)),
                 f"{m.group(2)} {m.group(3).strip()}" if m.group(2).endswith(":") else m.group(2))
                for m in _HEADING.finditer(document.page_content)
            ]

            # Preserve and enhance metadata for each chunk
            for chunk_index, chunk in enumerate(chunks):
                path = _heading_path(headings, chunk.metadata.get("start_index", 0))
                if path:
                    chunk.metadata["section"] = path
                enhanced_chunk = self._preserve_metadata(
                    chunk=chunk,
                    source_document=document,
                    chunk_index=chunk_index,
                    total_chunks=len(chunks),
                    doc_index=doc_index
                )
                all_chunks.append(enhanced_chunk)
            
            self.logger.debug(
                "Document chunked",
                doc_index=doc_index,
                num_chunks=len(chunks)
            )
        
        self.logger.info(
            "Document chunking complete",
            num_input_documents=len(documents),
            num_output_chunks=len(all_chunks)
        )
        
        return all_chunks
    
    def chunk_text(self, text: str, metadata: dict | None = None) -> List[Document]:
        """
        Chunk a raw text string into documents.
        
        Convenience method for chunking text without creating a Document first.
        
        Args:
            text: Raw text to chunk.
            metadata: Optional metadata dictionary to attach to all chunks.
            
        Returns:
            List of chunked Document objects.
            
        Example:
            >>> chunker = TextChunker(chunk_size=500)
            >>> chunks = chunker.chunk_text("long text" * 100, {"source": "input"})
            >>> len(chunks) > 1
            True
        """
        self.logger.debug("Chunking raw text", text_length=len(text))
        
        document = Document(
            page_content=text,
            metadata=metadata if metadata is not None else {}
        )
        
        return self.chunk_documents([document])
    
    def _preserve_metadata(
        self,
        chunk: Document,
        source_document: Document,
        chunk_index: int,
        total_chunks: int,
        doc_index: int
    ) -> Document:
        """
        Preserve original metadata and add chunk-specific information.
        
        Args:
            chunk: The chunked document.
            source_document: The original source document.
            chunk_index: Index of this chunk within the source document.
            total_chunks: Total number of chunks from the source document.
            doc_index: Index of the source document in the batch.
            
        Returns:
            Document with enhanced metadata.
        """
        # Start with original metadata from source document
        enhanced_metadata = source_document.metadata.copy()
        
        # Add chunk-specific metadata
        enhanced_metadata.update({
            "chunk_index": chunk_index,
            "total_chunks": total_chunks,
            "chunk_size": len(chunk.page_content),
            "doc_index": doc_index,
        })
        
        content = chunk.page_content
        if section := chunk.metadata.get("section"):
            enhanced_metadata["section"] = section
        label = " — ".join(filter(None, [enhanced_metadata.get("doc_header"), section]))
        if label:
            content = f"[{label}]\n{content}"

        # Create new document with enhanced metadata
        return Document(
            page_content=content,
            metadata=enhanced_metadata
        )
    
    def get_stats(self, documents: List[Document]) -> dict:
        """
        Get statistics about how documents would be chunked.
        
        Useful for previewing chunking results without actually performing the split.
        
        Args:
            documents: List of documents to analyze.
            
        Returns:
            Dictionary containing chunking statistics.
            
        Example:
            >>> chunker = TextChunker(chunk_size=1000, chunk_overlap=200)
            >>> docs = [Document(page_content="text" * 500, metadata={})]
            >>> stats = chunker.get_stats(docs)
            >>> "estimated_chunks" in stats
            True
        """
        if not documents:
            return {
                "num_documents": 0,
                "total_characters": 0,
                "estimated_chunks": 0,
                "avg_doc_length": 0,
            }
        
        total_chars = sum(len(doc.page_content) for doc in documents)
        avg_doc_length = total_chars / len(documents)
        
        # Rough estimate of chunks (actual may vary due to separator logic)
        estimated_chunks = sum(
            max(1, len(doc.page_content) // (self.chunk_size - self.chunk_overlap))
            for doc in documents
        )
        
        stats = {
            "num_documents": len(documents),
            "total_characters": total_chars,
            "estimated_chunks": estimated_chunks,
            "avg_doc_length": avg_doc_length,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
        }
        
        self.logger.debug("Chunking statistics calculated", **stats)
        
        return stats

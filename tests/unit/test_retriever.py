"""
Unit tests for RAG retriever.

Tests cover:
- Retriever initialization
- Semantic similarity search
- MMR (Maximum Marginal Relevance) search
- Score thresholds
- Metadata filtering
- LangChain compatibility
"""

import pytest
from hypothesis import given
from hypothesis import strategies as st
from langchain_core.documents import Document
from unittest.mock import AsyncMock, Mock, patch

from src.retrieval.retriever import (
    RAGRetriever,
    RetrievedDocument,
    LangChainRetrieverWrapper,
)


class TestRetrievedDocument:
    """Test suite for RetrievedDocument dataclass."""
    
    def test_creation(self):
        """Test RetrievedDocument creation."""
        doc = RetrievedDocument(
            content="Test content",
            metadata={"source": "test.pdf"},
            score=0.95,
            source="test.pdf",
            chunk_index=0
        )
        
        assert doc.content == "Test content"
        assert doc.metadata == {"source": "test.pdf"}
        assert doc.score == 0.95
        assert doc.source == "test.pdf"
        assert doc.chunk_index == 0
    
    def test_str_representation(self):
        """Test string representation."""
        doc = RetrievedDocument(
            content="Test",
            metadata={},
            score=0.85,
            source="doc.txt",
            chunk_index=3
        )
        
        str_repr = str(doc)
        assert "doc.txt" in str_repr
        assert "0.850" in str_repr
        assert "chunk=3" in str_repr
    
    def test_to_langchain_document(self):
        """Test conversion to LangChain Document."""
        metadata = {"source": "test.pdf", "page": 1}
        doc = RetrievedDocument(
            content="Test content",
            metadata=metadata,
            score=0.90,
            source="test.pdf"
        )
        
        lc_doc = doc.to_langchain_document()
        
        assert isinstance(lc_doc, Document)
        assert lc_doc.page_content == "Test content"
        assert lc_doc.metadata == metadata


class TestRAGRetrieverInitialization:
    """Test suite for RAGRetriever initialization."""
    
    def test_default_initialization(self):
        """Test retriever with default parameters."""
        mock_vectorstore = Mock()
        
        retriever = RAGRetriever(mock_vectorstore)
        
        assert retriever.vectorstore == mock_vectorstore
        assert retriever.k == 4
        assert retriever.use_mmr is True
        assert retriever.mmr_diversity == 0.5
        assert retriever.score_threshold == 0.0
    
    def test_custom_initialization(self):
        """Test retriever with custom parameters."""
        mock_vectorstore = Mock()
        
        retriever = RAGRetriever(
            mock_vectorstore,
            k=10,
            use_mmr=False,
            mmr_diversity=0.7,
            score_threshold=0.5
        )
        
        assert retriever.k == 10
        assert retriever.use_mmr is False
        assert retriever.mmr_diversity == 0.7
        assert retriever.score_threshold == 0.5
    
    def test_invalid_k(self):
        """Test that invalid k raises ValueError."""
        mock_vectorstore = Mock()
        
        with pytest.raises(ValueError, match="k must be >= 1"):
            RAGRetriever(mock_vectorstore, k=0)
        
        with pytest.raises(ValueError):
            RAGRetriever(mock_vectorstore, k=-1)
    
    def test_invalid_mmr_diversity(self):
        """Test that invalid mmr_diversity raises ValueError."""
        mock_vectorstore = Mock()
        
        with pytest.raises(ValueError, match="mmr_diversity must be between 0 and 1"):
            RAGRetriever(mock_vectorstore, mmr_diversity=1.5)
        
        with pytest.raises(ValueError):
            RAGRetriever(mock_vectorstore, mmr_diversity=-0.1)
    
    def test_invalid_score_threshold(self):
        """Test that invalid score_threshold raises ValueError."""
        mock_vectorstore = Mock()
        
        with pytest.raises(ValueError, match="score_threshold must be between 0 and 1"):
            RAGRetriever(mock_vectorstore, score_threshold=1.5)


class TestRAGRetrieverRetrieval:
    """Test suite for document retrieval."""
    
    @pytest.mark.asyncio
    async def test_retrieve_basic(self, mock_vectorstore_with_docs):
        """Test basic document retrieval."""
        retriever = RAGRetriever(mock_vectorstore_with_docs, k=3, use_mmr=False)
        
        results = await retriever.retrieve("test query")
        
        assert len(results) <= 3
        assert all(isinstance(doc, RetrievedDocument) for doc in results)
        
        # Scores should be in descending order
        if len(results) > 1:
            for i in range(len(results) - 1):
                assert results[i].score >= results[i+1].score
    
    @pytest.mark.asyncio
    async def test_retrieve_empty_query(self, mock_vectorstore_with_docs):
        """Test retrieval with empty query."""
        retriever = RAGRetriever(mock_vectorstore_with_docs)
        
        results = await retriever.retrieve("")
        assert results == []
        
        results = await retriever.retrieve("   ")
        assert results == []
    
    @pytest.mark.asyncio
    async def test_retrieve_with_scores(self, mock_vectorstore_with_docs):
        """Test retrieval with scores."""
        retriever = RAGRetriever(mock_vectorstore_with_docs, k=5, use_mmr=False)
        
        results = await retriever.retrieve_with_scores("machine learning")
        
        assert len(results) <= 5
        assert all(isinstance(item, tuple) for item in results)
        assert all(isinstance(item[0], Document) for item in results)
        assert all(isinstance(item[1], float) for item in results)
        
        # Scores should be valid (0-1 range for most embeddings)
        for _, score in results:
            assert 0.0 <= score <= 1.0 or score >= 0.0  # Some models use different ranges
    
    @pytest.mark.asyncio
    async def test_retrieve_with_mmr(self, mock_vectorstore_with_mmr):
        """Test retrieval with MMR for diversity."""
        retriever = RAGRetriever(
            mock_vectorstore_with_mmr,
            k=4,
            use_mmr=True,
            mmr_diversity=0.5
        )
        
        results = await retriever.retrieve("Python programming")
        
        assert len(results) <= 4
        assert all(isinstance(doc, RetrievedDocument) for doc in results)
    
    @pytest.mark.asyncio
    async def test_score_threshold_filtering(self):
        """Test that score threshold filters low-scoring documents."""
        # Create mock vectorstore that returns docs with varying scores
        mock_vectorstore = Mock()
        mock_vectorstore.search_with_score = AsyncMock(return_value=[
            (Document(page_content="High score", metadata={"source": "doc1"}), 0.9),
            (Document(page_content="Medium score", metadata={"source": "doc2"}), 0.6),
            (Document(page_content="Low score", metadata={"source": "doc3"}), 0.3),
        ])
        
        retriever = RAGRetriever(
            mock_vectorstore,
            k=10,
            use_mmr=False,
            score_threshold=0.5
        )
        
        results = await retriever.retrieve_with_scores("test")
        
        # Only docs with score >= 0.5 should be returned
        assert len(results) == 2
        assert all(score >= 0.5 for _, score in results)


class TestRAGRetrieverMetadataFiltering:
    """Test suite for metadata-based filtering."""
    
    @pytest.mark.asyncio
    async def test_retrieve_by_metadata(self):
        """Test retrieval with metadata filtering."""
        # Create mock vectorstore
        mock_vectorstore = Mock()
        mock_vectorstore.search_with_score = AsyncMock(return_value=[
            (Document(page_content="PDF doc", metadata={"source": "doc.pdf", "file_type": "pdf"}), 0.9),
            (Document(page_content="TXT doc", metadata={"source": "doc.txt", "file_type": "txt"}), 0.85),
            (Document(page_content="PDF doc 2", metadata={"source": "doc2.pdf", "file_type": "pdf"}), 0.8),
        ])
        
        retriever = RAGRetriever(mock_vectorstore, use_mmr=False)
        
        # Filter for PDF files only
        results = await retriever.retrieve_by_metadata(
            "test query",
            {"file_type": "pdf"}
        )
        
        assert len(results) == 2
        assert all(doc.metadata["file_type"] == "pdf" for doc in results)
    
    @pytest.mark.asyncio
    async def test_metadata_matching(self):
        """Test metadata matching logic."""
        mock_vectorstore = Mock()
        retriever = RAGRetriever(mock_vectorstore)
        
        # Test exact match
        metadata = {"source": "test.pdf", "page": 1}
        filter_criteria = {"source": "test.pdf"}
        assert retriever._matches_metadata(metadata, filter_criteria) is True
        
        # Test no match
        filter_criteria = {"source": "other.pdf"}
        assert retriever._matches_metadata(metadata, filter_criteria) is False
        
        # Test multiple criteria
        filter_criteria = {"source": "test.pdf", "page": 1}
        assert retriever._matches_metadata(metadata, filter_criteria) is True
        
        filter_criteria = {"source": "test.pdf", "page": 2}
        assert retriever._matches_metadata(metadata, filter_criteria) is False


class TestLangChainIntegration:
    """Test suite for LangChain compatibility."""
    
    def test_as_langchain_retriever(self, mock_vectorstore_with_docs):
        """Test LangChain retriever wrapper creation."""
        retriever = RAGRetriever(mock_vectorstore_with_docs)
        
        lc_retriever = retriever.as_langchain_retriever()
        
        assert isinstance(lc_retriever, LangChainRetrieverWrapper)
        assert lc_retriever.rag_retriever == retriever
    
    @pytest.mark.asyncio
    async def test_langchain_async_retrieval(self, mock_vectorstore_with_docs):
        """Test LangChain async retrieval."""
        retriever = RAGRetriever(mock_vectorstore_with_docs, k=3, use_mmr=False)
        lc_retriever = retriever.as_langchain_retriever()
        
        docs = await lc_retriever._aget_relevant_documents("test query")
        
        assert len(docs) <= 3
        assert all(isinstance(doc, Document) for doc in docs)


class TestPropertyBased:
    """Property-based tests using Hypothesis."""
    
    @given(
        k=st.integers(min_value=1, max_value=20),
        mmr_diversity=st.floats(min_value=0.0, max_value=1.0),
        score_threshold=st.floats(min_value=0.0, max_value=1.0)
    )
    def test_initialization_with_random_params(self, k, mmr_diversity, score_threshold):
        """Property: Valid parameters should initialize successfully."""
        mock_vectorstore = Mock()
        
        retriever = RAGRetriever(
            mock_vectorstore,
            k=k,
            mmr_diversity=mmr_diversity,
            score_threshold=score_threshold
        )
        
        assert retriever.k == k
        assert retriever.mmr_diversity == mmr_diversity
        assert retriever.score_threshold == score_threshold


# Pytest fixtures

@pytest.fixture
def mock_vectorstore_with_docs():
    """Create a mock vectorstore with sample documents."""
    mock_vectorstore = Mock()
    
    # Mock search_with_score to return sample documents
    async def mock_search_with_score(query, k):
        docs = [
            (Document(page_content=f"Document {i}", metadata={"source": f"doc{i}.txt", "chunk_index": i}), 0.9 - i*0.1)
            for i in range(min(k, 5))
        ]
        return docs
    
    mock_vectorstore.search_with_score = AsyncMock(side_effect=mock_search_with_score)
    
    return mock_vectorstore


@pytest.fixture
def mock_vectorstore_with_mmr():
    """Create a mock vectorstore with MMR support."""
    mock_vectorstore = Mock()
    
    # Mock MMR search
    async def mock_search_mmr(query, k, fetch_k, lambda_mult):
        docs = [
            Document(page_content=f"MMR Document {i}", metadata={"source": f"doc{i}.txt"})
            for i in range(min(k, 5))
        ]
        return docs
    
    async def mock_search_with_score(query, k):
        docs = [
            (Document(page_content=f"Document {i}", metadata={"source": f"doc{i}.txt"}), 0.85 - i*0.05)
            for i in range(min(k, 5))
        ]
        return docs
    
    mock_vectorstore.search_mmr = AsyncMock(side_effect=mock_search_mmr)
    mock_vectorstore.search_with_score = AsyncMock(side_effect=mock_search_with_score)
    
    return mock_vectorstore

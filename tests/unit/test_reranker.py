"""
Unit tests for document reranker module.

Tests reranking functionality including:
- Initialization and configuration
- Document reranking with cross-encoder models
- Score computation
- Error handling
- Configuration presets
"""

from unittest.mock import Mock, PropertyMock, patch

import pytest

from src.retrieval.reranker import Reranker, RerankerConfig
from src.retrieval.retriever import RetrievedDocument


@pytest.fixture
def sample_documents():
    """Create sample retrieved documents."""
    return [
        RetrievedDocument(
            content="Python is a programming language",
            metadata={"source": "doc1.txt"},
            score=0.8,
            source="doc1.txt",
            chunk_index=0,
        ),
        RetrievedDocument(
            content="Java is also a programming language",
            metadata={"source": "doc2.txt"},
            score=0.7,
            source="doc2.txt",
            chunk_index=0,
        ),
        RetrievedDocument(
            content="Machine learning uses Python",
            metadata={"source": "doc3.txt"},
            score=0.6,
            source="doc3.txt",
            chunk_index=0,
        ),
    ]


class TestRerankerInitialization:
    """Test reranker initialization."""

    def test_initialize_with_defaults(self):
        """Should initialize with default values."""
        reranker = Reranker()

        assert reranker.model_name == Reranker.DEFAULT_MODEL
        assert reranker.top_k == 4
        assert reranker.score_threshold is None

    def test_initialize_with_custom_params(self):
        """Should initialize with custom parameters."""
        reranker = Reranker(
            model_name="custom-model",
            top_k=10,
            score_threshold=0.5,
        )

        assert reranker.model_name == "custom-model"
        assert reranker.top_k == 10
        assert reranker.score_threshold == 0.5

    def test_initialize_with_invalid_top_k(self):
        """Should raise ValueError for invalid top_k."""
        with pytest.raises(ValueError, match="top_k must be >= 1"):
            Reranker(top_k=0)

        with pytest.raises(ValueError, match="top_k must be >= 1"):
            Reranker(top_k=-1)

    def test_model_lazy_loading(self):
        """Should lazy load the cross-encoder model."""
        reranker = Reranker()
        
        # Model should not be loaded yet
        assert reranker._model is None


class TestRerank:
    """Test document reranking."""

    def test_rerank_with_empty_query(self, sample_documents):
        """Should return original documents for empty query."""
        reranker = Reranker()
        
        result = reranker.rerank("", sample_documents)
        
        assert result == sample_documents

    def test_rerank_with_whitespace_query(self, sample_documents):
        """Should return original documents for whitespace-only query."""
        reranker = Reranker()
        
        result = reranker.rerank("   ", sample_documents)
        
        assert result == sample_documents

    def test_rerank_with_no_documents(self):
        """Should return empty list when no documents provided."""
        reranker = Reranker()
        
        result = reranker.rerank("test query", [])
        
        assert result == []

    @patch("src.retrieval.reranker.Reranker.model", new_callable=PropertyMock)
    def test_rerank_successful(self, mock_model_prop, sample_documents):
        """Should successfully rerank documents."""
        # Create mock model
        mock_model = Mock()
        mock_model.predict = Mock(return_value=[0.9, 0.7, 0.95])  # Scores
        mock_model_prop.return_value = mock_model
        
        reranker = Reranker(top_k=2)
        result = reranker.rerank("Python programming", sample_documents)
        
        # Should return top 2 documents sorted by new scores
        assert len(result) == 2
        assert result[0].score == 0.95  # doc3 has highest score
        assert result[1].score == 0.9   # doc1 has second highest

    @patch("src.retrieval.reranker.Reranker.model", new_callable=PropertyMock)
    def test_rerank_with_custom_top_k(self, mock_model_prop, sample_documents):
        """Should respect custom top_k parameter."""
        mock_model = Mock()
        mock_model.predict = Mock(return_value=[0.9, 0.7, 0.95])
        mock_model_prop.return_value = mock_model
        
        reranker = Reranker(top_k=5)  # Default top_k=5
        result = reranker.rerank("Python programming", sample_documents, top_k=1)
        
        # Should use provided top_k (1), not default (5)
        assert len(result) == 1

    @patch("src.retrieval.reranker.Reranker.model", new_callable=PropertyMock)
    def test_rerank_with_score_threshold(self, mock_model_prop, sample_documents):
        """Should filter results by score threshold."""
        mock_model = Mock()
        mock_model.predict = Mock(return_value=[0.9, 0.4, 0.95])
        mock_model_prop.return_value = mock_model
        
        reranker = Reranker(top_k=3, score_threshold=0.5)
        result = reranker.rerank("Python programming", sample_documents)
        
        # Should only return documents with score >= 0.5
        assert len(result) == 2  # Only 0.9 and 0.95, not 0.4
        assert all(doc.score >= 0.5 for doc in result)

    @patch("src.retrieval.reranker.Reranker.model", new_callable=PropertyMock)
    def test_rerank_preserves_document_content(self, mock_model_prop, sample_documents):
        """Should preserve document content and metadata."""
        mock_model = Mock()
        mock_model.predict = Mock(return_value=[0.9, 0.7, 0.95])
        mock_model_prop.return_value = mock_model
        
        reranker = Reranker()
        result = reranker.rerank("Python programming", sample_documents)
        
        # Check content is preserved
        original_contents = {doc.content for doc in sample_documents}
        result_contents = {doc.content for doc in result}
        assert result_contents.issubset(original_contents)
        
        # Check metadata is preserved
        for doc in result:
            assert "source" in doc.metadata

    @patch("src.retrieval.reranker.Reranker.model", new_callable=PropertyMock)
    def test_rerank_with_exception(self, mock_model_prop, sample_documents):
        """Should return original documents on exception."""
        mock_model = Mock()
        mock_model.predict = Mock(side_effect=Exception("Model error"))
        mock_model_prop.return_value = mock_model
        
        reranker = Reranker(top_k=2)
        result = reranker.rerank("Python programming", sample_documents)
        
        # Should return first 2 of original documents as fallback
        assert len(result) == 2
        assert result == sample_documents[:2]


class TestComputeScores:
    """Test score computation."""

    def test_compute_scores_empty_query(self, sample_documents):
        """Should return empty list for empty query."""
        reranker = Reranker()
        
        scores = reranker.compute_scores("", sample_documents)
        
        assert scores == []

    def test_compute_scores_no_documents(self):
        """Should return empty list when no documents provided."""
        reranker = Reranker()
        
        scores = reranker.compute_scores("test query", [])
        
        assert scores == []

    @patch("src.retrieval.reranker.Reranker.model", new_callable=PropertyMock)
    def test_compute_scores_successful(self, mock_model_prop, sample_documents):
        """Should compute scores for all documents."""
        mock_model = Mock()
        mock_model.predict = Mock(return_value=[0.85, 0.72, 0.91])
        mock_model_prop.return_value = mock_model
        
        reranker = Reranker()
        scores = reranker.compute_scores("Python programming", sample_documents)
        
        assert len(scores) == 3
        assert scores == [0.85, 0.72, 0.91]
        assert all(isinstance(s, float) for s in scores)

    @patch("src.retrieval.reranker.Reranker.model", new_callable=PropertyMock)
    def test_compute_scores_with_exception(self, mock_model_prop, sample_documents):
        """Should return zeros on exception."""
        mock_model = Mock()
        mock_model.predict = Mock(side_effect=Exception("Model error"))
        mock_model_prop.return_value = mock_model
        
        reranker = Reranker()
        scores = reranker.compute_scores("Python programming", sample_documents)
        
        # Should return all zeros as fallback
        assert scores == [0.0, 0.0, 0.0]


class TestModelProperty:
    """Test lazy model loading."""

    def test_model_lazy_loads_cross_encoder(self):
        """Should lazy load CrossEncoder on first access."""
        with patch("sentence_transformers.CrossEncoder") as mock_cross_encoder_class:
            mock_model_instance = Mock()
            mock_cross_encoder_class.return_value = mock_model_instance
            
            reranker = Reranker(model_name="test-model")
            
            # Access model for the first time
            model = reranker.model
            
            # Should have created the model
            mock_cross_encoder_class.assert_called_once_with("test-model")
            assert model == mock_model_instance

    def test_model_returns_cached_instance(self):
        """Should return cached model instance on subsequent accesses."""
        with patch("sentence_transformers.CrossEncoder") as mock_cross_encoder_class:
            mock_model_instance = Mock()
            mock_cross_encoder_class.return_value = mock_model_instance
            
            reranker = Reranker(model_name="test-model")
            
            # Access model multiple times
            model1 = reranker.model
            model2 = reranker.model
            
            # Should only create once
            mock_cross_encoder_class.assert_called_once()
            assert model1 is model2

    def test_model_import_error(self):
        """Should raise ImportError if sentence-transformers not installed."""
        reranker = Reranker()
        
        with patch.dict("sys.modules", {"sentence_transformers": None}):
            with pytest.raises(ImportError, match="sentence-transformers is required"):
                _ = reranker.model


class TestRerankerConfig:
    """Test reranker configuration presets."""

    def test_fast_config_values(self):
        """Should have correct values for FAST configuration."""
        assert RerankerConfig.FAST["model_name"] == "cross-encoder/ms-marco-TinyBERT-L-2-v2"
        assert RerankerConfig.FAST["top_k"] == 4

    def test_balanced_config_values(self):
        """Should have correct values for BALANCED configuration."""
        assert RerankerConfig.BALANCED["model_name"] == "cross-encoder/ms-marco-MiniLM-L-6-v2"
        assert RerankerConfig.BALANCED["top_k"] == 4

    def test_high_quality_config_values(self):
        """Should have correct values for HIGH_QUALITY configuration."""
        assert RerankerConfig.HIGH_QUALITY["model_name"] == "cross-encoder/ms-marco-MiniLM-L-12-v2"
        assert RerankerConfig.HIGH_QUALITY["top_k"] == 4

    def test_get_fast_reranker(self):
        """Should create fast reranker instance."""
        reranker = RerankerConfig.get_fast_reranker()
        
        assert isinstance(reranker, Reranker)
        assert reranker.model_name == RerankerConfig.FAST["model_name"]
        assert reranker.top_k == RerankerConfig.FAST["top_k"]

    def test_get_balanced_reranker(self):
        """Should create balanced reranker instance."""
        reranker = RerankerConfig.get_balanced_reranker()
        
        assert isinstance(reranker, Reranker)
        assert reranker.model_name == RerankerConfig.BALANCED["model_name"]
        assert reranker.top_k == RerankerConfig.BALANCED["top_k"]

    def test_get_high_quality_reranker(self):
        """Should create high-quality reranker instance."""
        reranker = RerankerConfig.get_high_quality_reranker()
        
        assert isinstance(reranker, Reranker)
        assert reranker.model_name == RerankerConfig.HIGH_QUALITY["model_name"]
        assert reranker.top_k == RerankerConfig.HIGH_QUALITY["top_k"]

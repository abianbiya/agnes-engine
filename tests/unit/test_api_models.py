"""
Unit tests for API models module.

Tests all Pydantic models for validation, field constraints,
and serialization/deserialization.
"""

import json

import pytest
from hypothesis import given, strategies as st
from pydantic import ValidationError

from src.api.models import (
    ChatRequest,
    ChatResponse,
    ErrorResponse,
    HealthResponse,
    IngestResponse,
    SearchRequest,
    SearchResult,
    SearchResponse,
    SessionDeleteResponse,
    SourceDocument,
)
from tests.strategies import (
    chat_question,
    content_type,
    correlation_id,
    http_status_code,
    session_id,
    similarity_score,
    valid_filename,
)


class TestSourceDocument:
    """Test SourceDocument model."""

    def test_valid_creation_with_all_fields(self):
        """Should create SourceDocument with all fields."""
        doc = SourceDocument(
            filename="test.pdf",
            page=5,
            section="Introduction",
            relevance_score=0.95,
            content_preview="This is a preview",
        )

        assert doc.filename == "test.pdf"
        assert doc.page == 5
        assert doc.section == "Introduction"
        assert doc.relevance_score == 0.95
        assert doc.content_preview == "This is a preview"

    def test_valid_creation_minimal_fields(self):
        """Should create SourceDocument with only required fields."""
        doc = SourceDocument(
            filename="test.pdf",
            relevance_score=0.8,
        )

        assert doc.filename == "test.pdf"
        assert doc.relevance_score == 0.8
        assert doc.page is None
        assert doc.section is None
        assert doc.content_preview is None

    @pytest.mark.parametrize(
        "score",
        [0.0, 0.5, 1.0],
    )
    def test_relevance_score_valid_range(self, score):
        """Should accept relevance_score in valid range [0, 1]."""
        doc = SourceDocument(filename="test.pdf", relevance_score=score)
        assert doc.relevance_score == score

    @pytest.mark.parametrize(
        "score",
        [-0.1, 1.1, 2.0, -1.0],
    )
    def test_relevance_score_invalid_range(self, score):
        """Should reject relevance_score outside [0, 1]."""
        with pytest.raises(ValidationError) as exc_info:
            SourceDocument(filename="test.pdf", relevance_score=score)

        errors = exc_info.value.errors()
        assert any(e["loc"] == ("relevance_score",) for e in errors)

    def test_missing_required_fields(self):
        """Should reject creation without required fields."""
        with pytest.raises(ValidationError) as exc_info:
            SourceDocument()  # type: ignore

        errors = exc_info.value.errors()
        assert len(errors) == 2  # filename and relevance_score
        field_names = {e["loc"][0] for e in errors}
        assert "filename" in field_names
        assert "relevance_score" in field_names


class TestChatRequest:
    """Test ChatRequest model."""

    def test_valid_creation_with_session(self):
        """Should create ChatRequest with question and session_id."""
        req = ChatRequest(
            question="What is AI?",
            session_id="test-session-123",
        )

        assert req.question == "What is AI?"
        assert req.session_id == "test-session-123"

    def test_valid_creation_without_session(self):
        """Should create ChatRequest with only question."""
        req = ChatRequest(question="What is machine learning?")

        assert req.question == "What is machine learning?"
        assert req.session_id is None

    def test_question_whitespace_stripped(self):
        """Should strip leading/trailing whitespace from question."""
        req = ChatRequest(question="  What is AI?  ")
        assert req.question == "What is AI?"

    def test_question_empty_string(self):
        """Should reject empty question string."""
        with pytest.raises(ValidationError) as exc_info:
            ChatRequest(question="")

        errors = exc_info.value.errors()
        assert any("question" in str(e["loc"]) for e in errors)

    def test_question_whitespace_only(self):
        """Should reject whitespace-only question."""
        with pytest.raises(ValidationError) as exc_info:
            ChatRequest(question="   ")

        errors = exc_info.value.errors()
        assert any("question" in str(e["loc"]) for e in errors)

    def test_question_too_long(self):
        """Should reject question exceeding max length."""
        long_question = "a" * 2001

        with pytest.raises(ValidationError) as exc_info:
            ChatRequest(question=long_question)

        errors = exc_info.value.errors()
        assert any(e["loc"] == ("question",) for e in errors)

    def test_question_at_max_length(self):
        """Should accept question at exactly max length."""
        max_question = "a" * 2000
        req = ChatRequest(question=max_question)
        assert len(req.question) == 2000

    def test_missing_question(self):
        """Should reject creation without question."""
        with pytest.raises(ValidationError) as exc_info:
            ChatRequest()  # type: ignore

        errors = exc_info.value.errors()
        assert any(e["loc"] == ("question",) for e in errors)


class TestChatResponse:
    """Test ChatResponse model."""

    def test_valid_creation_with_all_fields(self):
        """Should create ChatResponse with all fields."""
        resp = ChatResponse(
            answer="AI is artificial intelligence.",
            sources=[
                SourceDocument(filename="ai.pdf", relevance_score=0.9),
            ],
            session_id="session-123",
            metadata={"processing_time": 1.5},
        )

        assert resp.answer == "AI is artificial intelligence."
        assert len(resp.sources) == 1
        assert resp.session_id == "session-123"
        assert resp.metadata == {"processing_time": 1.5}

    def test_valid_creation_without_metadata(self):
        """Should create ChatResponse without metadata."""
        resp = ChatResponse(
            answer="Test answer",
            sources=[],
            session_id="session-123",
        )

        assert resp.answer == "Test answer"
        assert resp.sources == []
        assert resp.session_id == "session-123"
        assert resp.metadata is None

    def test_empty_sources_list(self):
        """Should accept empty sources list."""
        resp = ChatResponse(
            answer="Test",
            sources=[],
            session_id="test",
        )
        assert resp.sources == []

    def test_multiple_sources(self):
        """Should handle multiple source documents."""
        sources = [
            SourceDocument(filename=f"doc{i}.pdf", relevance_score=0.9)
            for i in range(3)
        ]
        resp = ChatResponse(
            answer="Test",
            sources=sources,
            session_id="test",
        )
        assert len(resp.sources) == 3

    def test_missing_required_fields(self):
        """Should reject creation without required fields."""
        with pytest.raises(ValidationError) as exc_info:
            ChatResponse()  # type: ignore

        errors = exc_info.value.errors()
        field_names = {e["loc"][0] for e in errors}
        assert "answer" in field_names
        assert "sources" in field_names
        assert "session_id" in field_names


class TestSearchRequest:
    """Test SearchRequest model."""

    def test_valid_creation_with_defaults(self):
        """Should create SearchRequest with default values."""
        req = SearchRequest(query="test query")

        assert req.query == "test query"
        assert req.limit == 4  # default
        assert req.use_mmr is False  # default

    def test_valid_creation_with_all_fields(self):
        """Should create SearchRequest with all fields specified."""
        req = SearchRequest(
            query="neural networks",
            limit=10,
            use_mmr=True,
        )

        assert req.query == "neural networks"
        assert req.limit == 10
        assert req.use_mmr is True

    def test_query_whitespace_stripped(self):
        """Should strip leading/trailing whitespace from query."""
        req = SearchRequest(query="  test query  ")
        assert req.query == "test query"

    def test_query_empty_string(self):
        """Should reject empty query string."""
        with pytest.raises(ValidationError):
            SearchRequest(query="")

    def test_query_whitespace_only(self):
        """Should reject whitespace-only query."""
        with pytest.raises(ValidationError):
            SearchRequest(query="   ")

    def test_query_too_long(self):
        """Should reject query exceeding max length."""
        long_query = "a" * 1001

        with pytest.raises(ValidationError) as exc_info:
            SearchRequest(query=long_query)

        errors = exc_info.value.errors()
        assert any(e["loc"] == ("query",) for e in errors)

    @pytest.mark.parametrize("limit", [1, 4, 10, 20])
    def test_limit_valid_range(self, limit):
        """Should accept limit in valid range [1, 20]."""
        req = SearchRequest(query="test", limit=limit)
        assert req.limit == limit

    @pytest.mark.parametrize("limit", [0, -1, 21, 100])
    def test_limit_invalid_range(self, limit):
        """Should reject limit outside [1, 20]."""
        with pytest.raises(ValidationError) as exc_info:
            SearchRequest(query="test", limit=limit)

        errors = exc_info.value.errors()
        assert any(e["loc"] == ("limit",) for e in errors)

    def test_use_mmr_boolean(self):
        """Should accept boolean values for use_mmr."""
        req_true = SearchRequest(query="test", use_mmr=True)
        req_false = SearchRequest(query="test", use_mmr=False)

        assert req_true.use_mmr is True
        assert req_false.use_mmr is False


class TestSearchResult:
    """Test SearchResult model."""

    def test_valid_creation(self):
        """Should create SearchResult with all fields."""
        result = SearchResult(
            content="Test content",
            metadata={"source": "test.pdf", "page": 5},
            score=0.85,
            source="test.pdf",
        )

        assert result.content == "Test content"
        assert result.metadata == {"source": "test.pdf", "page": 5}
        assert result.score == 0.85
        assert result.source == "test.pdf"

    @pytest.mark.parametrize("score", [0.0, 0.5, 1.0])
    def test_score_valid_range(self, score):
        """Should accept score in valid range [0, 1]."""
        result = SearchResult(
            content="test",
            metadata={},
            score=score,
            source="test.pdf",
        )
        assert result.score == score

    @pytest.mark.parametrize("score", [-0.1, 1.1, 2.0])
    def test_score_invalid_range(self, score):
        """Should reject score outside [0, 1]."""
        with pytest.raises(ValidationError) as exc_info:
            SearchResult(
                content="test",
                metadata={},
                score=score,
                source="test.pdf",
            )

        errors = exc_info.value.errors()
        assert any(e["loc"] == ("score",) for e in errors)

    def test_missing_required_fields(self):
        """Should reject creation without required fields."""
        with pytest.raises(ValidationError) as exc_info:
            SearchResult()  # type: ignore

        errors = exc_info.value.errors()
        field_names = {e["loc"][0] for e in errors}
        assert "content" in field_names
        assert "metadata" in field_names
        assert "score" in field_names
        assert "source" in field_names


class TestSearchResponse:
    """Test SearchResponse model."""

    def test_valid_creation_with_results(self):
        """Should create SearchResponse with results."""
        results = [
            SearchResult(
                content="test",
                metadata={},
                score=0.9,
                source="test.pdf",
            )
        ]
        resp = SearchResponse(
            results=results,
            count=1,
            query="test query",
        )

        assert len(resp.results) == 1
        assert resp.count == 1
        assert resp.query == "test query"

    def test_valid_creation_empty_results(self):
        """Should create SearchResponse with empty results."""
        resp = SearchResponse(
            results=[],
            count=0,
            query="test query",
        )

        assert resp.results == []
        assert resp.count == 0
        assert resp.query == "test query"

    def test_count_non_negative(self):
        """Should enforce count is non-negative."""
        with pytest.raises(ValidationError) as exc_info:
            SearchResponse(
                results=[],
                count=-1,
                query="test",
            )

        errors = exc_info.value.errors()
        assert any(e["loc"] == ("count",) for e in errors)

    def test_multiple_results(self):
        """Should handle multiple search results."""
        results = [
            SearchResult(
                content=f"content {i}",
                metadata={},
                score=0.9,
                source=f"doc{i}.pdf",
            )
            for i in range(5)
        ]
        resp = SearchResponse(
            results=results,
            count=5,
            query="test",
        )
        assert len(resp.results) == 5
        assert resp.count == 5


class TestIngestResponse:
    """Test IngestResponse model."""

    def test_valid_creation_success(self):
        """Should create successful IngestResponse."""
        resp = IngestResponse(
            success=True,
            filename="test.pdf",
            file_path="/data/test.pdf",
            chunks_created=10,
            chunks_stored=10,
            documents_loaded=1,
            file_type="pdf",
        )

        assert resp.success is True
        assert resp.filename == "test.pdf"
        assert resp.file_path == "/data/test.pdf"
        assert resp.chunks_created == 10
        assert resp.chunks_stored == 10
        assert resp.documents_loaded == 1
        assert resp.file_type == "pdf"
        assert resp.error_message is None

    def test_valid_creation_failure(self):
        """Should create failed IngestResponse with error message."""
        resp = IngestResponse(
            success=False,
            filename="test.pdf",
            chunks_created=0,
            chunks_stored=0,
            documents_loaded=0,
            file_type="pdf",
            error_message="File parsing failed",
        )

        assert resp.success is False
        assert resp.error_message == "File parsing failed"
        assert resp.chunks_created == 0

    def test_chunks_non_negative(self):
        """Should enforce non-negative chunk counts."""
        with pytest.raises(ValidationError):
            IngestResponse(
                success=True,
                filename="test.pdf",
                chunks_created=-1,
                chunks_stored=0,
                documents_loaded=0,
                file_type="pdf",
            )

    def test_missing_required_fields(self):
        """Should reject creation without required fields."""
        with pytest.raises(ValidationError) as exc_info:
            IngestResponse()  # type: ignore

        errors = exc_info.value.errors()
        field_names = {e["loc"][0] for e in errors}
        assert "success" in field_names
        assert "filename" in field_names
        assert "chunks_created" in field_names
        assert "chunks_stored" in field_names
        assert "documents_loaded" in field_names
        assert "file_type" in field_names


class TestHealthResponse:
    """Test HealthResponse model."""

    def test_valid_creation_healthy(self):
        """Should create healthy HealthResponse."""
        resp = HealthResponse(
            status="healthy",
            services={
                "vectorstore": True,
                "llm": True,
                "embeddings": True,
            },
            version="1.0.0",
        )

        assert resp.status == "healthy"
        assert resp.services == {
            "vectorstore": True,
            "llm": True,
            "embeddings": True,
        }
        assert resp.version == "1.0.0"

    def test_valid_creation_degraded(self):
        """Should create degraded HealthResponse."""
        resp = HealthResponse(
            status="degraded",
            services={
                "vectorstore": False,
                "llm": True,
                "embeddings": True,
            },
            version="1.0.0",
        )

        assert resp.status == "degraded"
        assert resp.services["vectorstore"] is False

    def test_valid_creation_unhealthy(self):
        """Should create unhealthy HealthResponse."""
        resp = HealthResponse(
            status="unhealthy",
            services={
                "vectorstore": False,
                "llm": False,
                "embeddings": False,
            },
            version="1.0.0",
        )

        assert resp.status == "unhealthy"

    def test_missing_required_fields(self):
        """Should reject creation without required fields."""
        with pytest.raises(ValidationError) as exc_info:
            HealthResponse()  # type: ignore

        errors = exc_info.value.errors()
        field_names = {e["loc"][0] for e in errors}
        assert "status" in field_names
        assert "services" in field_names
        assert "version" in field_names


class TestErrorResponse:
    """Test ErrorResponse model."""

    def test_valid_creation_with_all_fields(self):
        """Should create ErrorResponse with all fields."""
        resp = ErrorResponse(
            error="ValidationError",
            message="Invalid input",
            detail="Question field is required",
            correlation_id="test-correlation-123",
        )

        assert resp.error == "ValidationError"
        assert resp.message == "Invalid input"
        assert resp.detail == "Question field is required"
        assert resp.correlation_id == "test-correlation-123"

    def test_valid_creation_minimal_fields(self):
        """Should create ErrorResponse with only required fields."""
        resp = ErrorResponse(
            error="InternalError",
            message="Something went wrong",
        )

        assert resp.error == "InternalError"
        assert resp.message == "Something went wrong"
        assert resp.detail is None
        assert resp.correlation_id is None

    def test_missing_required_fields(self):
        """Should reject creation without required fields."""
        with pytest.raises(ValidationError) as exc_info:
            ErrorResponse()  # type: ignore

        errors = exc_info.value.errors()
        field_names = {e["loc"][0] for e in errors}
        assert "error" in field_names
        assert "message" in field_names


class TestSessionDeleteResponse:
    """Test SessionDeleteResponse model."""

    def test_valid_creation(self):
        """Should create SessionDeleteResponse."""
        resp = SessionDeleteResponse(
            message="Session cleared successfully",
            session_id="session-123",
        )

        assert resp.message == "Session cleared successfully"
        assert resp.session_id == "session-123"

    def test_missing_required_fields(self):
        """Should reject creation without required fields."""
        with pytest.raises(ValidationError) as exc_info:
            SessionDeleteResponse()  # type: ignore

        errors = exc_info.value.errors()
        field_names = {e["loc"][0] for e in errors}
        assert "message" in field_names
        assert "session_id" in field_names


# ==============================================================================
# Property-Based Tests
# ==============================================================================


@pytest.mark.property
class TestPropertyBasedSourceDocument:
    """Property-based tests for SourceDocument model."""

    @given(
        filename=valid_filename(),
        score=similarity_score(),
    )
    def test_serialization_roundtrip(self, filename, score):
        """SourceDocument should survive JSON serialization roundtrip."""
        doc = SourceDocument(filename=filename, relevance_score=score)
        
        # Serialize to JSON
        json_str = doc.model_dump_json()
        
        # Deserialize back
        doc_dict = json.loads(json_str)
        doc_restored = SourceDocument(**doc_dict)
        
        assert doc_restored.filename == doc.filename
        assert doc_restored.relevance_score == doc.relevance_score

    @given(
        filename=valid_filename(),
        score=similarity_score(),
        page=st.integers(min_value=1, max_value=1000) | st.none(),
        section=st.text(min_size=1, max_size=100) | st.none(),
    )
    def test_optional_fields_preservation(self, filename, score, page, section):
        """Optional fields should be preserved correctly."""
        doc = SourceDocument(
            filename=filename,
            relevance_score=score,
            page=page,
            section=section,
        )
        
        assert doc.page == page
        assert doc.section == section

    @given(
        filename=valid_filename(),
        score=st.floats(min_value=-10.0, max_value=-0.001)
        | st.floats(min_value=1.001, max_value=10.0),
    )
    def test_invalid_score_rejection(self, filename, score):
        """Should reject scores outside valid range."""
        with pytest.raises(ValidationError):
            SourceDocument(filename=filename, relevance_score=score)


@pytest.mark.property
class TestPropertyBasedChatRequest:
    """Property-based tests for ChatRequest model."""

    @given(
        question=chat_question(),
        sess_id=session_id() | st.none(),
    )
    def test_serialization_roundtrip(self, question, sess_id):
        """ChatRequest should survive JSON serialization roundtrip."""
        req = ChatRequest(question=question, session_id=sess_id)
        
        # Serialize to JSON
        json_str = req.model_dump_json()
        
        # Deserialize back
        req_dict = json.loads(json_str)
        req_restored = ChatRequest(**req_dict)
        
        assert req_restored.question == req.question
        assert req_restored.session_id == req.session_id

    @given(
        question=st.text(min_size=1, max_size=2000).filter(lambda x: x.strip()),
    )
    def test_question_whitespace_handling(self, question):
        """Question whitespace should be handled consistently."""
        padded_question = f"  {question}  "
        req = ChatRequest(question=padded_question)
        
        # Whitespace should be stripped
        assert req.question == question.strip()
        assert not req.question.startswith(" ")
        assert not req.question.endswith(" ")

    @given(
        question=st.text(min_size=2001, max_size=3000),
    )
    def test_question_length_validation(self, question):
        """Should reject questions exceeding max length."""
        with pytest.raises(ValidationError) as exc_info:
            ChatRequest(question=question)
        
        errors = exc_info.value.errors()
        assert any(e["loc"] == ("question",) for e in errors)


@pytest.mark.property
class TestPropertyBasedChatResponse:
    """Property-based tests for ChatResponse model."""

    @given(
        answer=st.text(min_size=1, max_size=1000),
        sess_id=session_id(),
        num_sources=st.integers(min_value=0, max_value=10),
    )
    def test_serialization_roundtrip(self, answer, sess_id, num_sources):
        """ChatResponse should survive JSON serialization roundtrip."""
        sources = [
            SourceDocument(
                filename=f"doc{i}.pdf",
                relevance_score=0.8,
            )
            for i in range(num_sources)
        ]
        
        resp = ChatResponse(
            answer=answer,
            sources=sources,
            session_id=sess_id,
        )
        
        # Serialize to JSON
        json_str = resp.model_dump_json()
        
        # Deserialize back
        resp_dict = json.loads(json_str)
        resp_restored = ChatResponse(**resp_dict)
        
        assert resp_restored.answer == resp.answer
        assert resp_restored.session_id == resp.session_id
        assert len(resp_restored.sources) == len(resp.sources)

    @given(
        answer=st.text(min_size=1, max_size=500),
        sess_id=session_id(),
        metadata=st.dictionaries(
            keys=st.text(min_size=1, max_size=20),
            values=st.one_of(
                st.text(max_size=50),
                st.floats(allow_nan=False, allow_infinity=False),
                st.integers(),
                st.booleans(),
            ),
            max_size=5,
        )
        | st.none(),
    )
    def test_metadata_preservation(self, answer, sess_id, metadata):
        """Metadata should be preserved correctly."""
        resp = ChatResponse(
            answer=answer,
            sources=[],
            session_id=sess_id,
            metadata=metadata,
        )
        
        assert resp.metadata == metadata


@pytest.mark.property
class TestPropertyBasedSearchRequest:
    """Property-based tests for SearchRequest model."""

    @given(
        query=st.text(min_size=1, max_size=1000).filter(lambda x: x.strip()),
        limit=st.integers(min_value=1, max_value=20),
        use_mmr=st.booleans(),
    )
    def test_serialization_roundtrip(self, query, limit, use_mmr):
        """SearchRequest should survive JSON serialization roundtrip."""
        req = SearchRequest(query=query, limit=limit, use_mmr=use_mmr)
        
        # Serialize to JSON
        json_str = req.model_dump_json()
        
        # Deserialize back
        req_dict = json.loads(json_str)
        req_restored = SearchRequest(**req_dict)
        
        assert req_restored.query == req.query
        assert req_restored.limit == req.limit
        assert req_restored.use_mmr == req.use_mmr

    @given(
        query=st.text(min_size=1, max_size=1000).filter(lambda x: x.strip()),
        limit=st.integers(min_value=-100, max_value=0)
        | st.integers(min_value=21, max_value=100),
    )
    def test_limit_validation(self, query, limit):
        """Should reject invalid limit values."""
        with pytest.raises(ValidationError) as exc_info:
            SearchRequest(query=query, limit=limit)
        
        errors = exc_info.value.errors()
        assert any(e["loc"] == ("limit",) for e in errors)


@pytest.mark.property
class TestPropertyBasedSearchResult:
    """Property-based tests for SearchResult model."""

    @given(
        content=st.text(min_size=1, max_size=500),
        source=valid_filename(),
        score=similarity_score(),
        metadata=st.dictionaries(
            keys=st.text(min_size=1, max_size=20),
            values=st.one_of(st.text(max_size=50), st.integers(), st.booleans()),
            max_size=5,
        ),
    )
    def test_serialization_roundtrip(self, content, source, score, metadata):
        """SearchResult should survive JSON serialization roundtrip."""
        result = SearchResult(
            content=content,
            metadata=metadata,
            score=score,
            source=source,
        )
        
        # Serialize to JSON
        json_str = result.model_dump_json()
        
        # Deserialize back
        result_dict = json.loads(json_str)
        result_restored = SearchResult(**result_dict)
        
        assert result_restored.content == result.content
        assert result_restored.source == result.source
        assert result_restored.score == result.score
        assert result_restored.metadata == result.metadata


@pytest.mark.property
class TestPropertyBasedIngestResponse:
    """Property-based tests for IngestResponse model."""

    @given(
        success=st.booleans(),
        filename=valid_filename(),
        chunks_created=st.integers(min_value=0, max_value=1000),
        chunks_stored=st.integers(min_value=0, max_value=1000),
        documents_loaded=st.integers(min_value=0, max_value=100),
        file_type=st.sampled_from(["txt", "pdf", "md", "json", "csv"]),
    )
    def test_serialization_roundtrip(
        self,
        success,
        filename,
        chunks_created,
        chunks_stored,
        documents_loaded,
        file_type,
    ):
        """IngestResponse should survive JSON serialization roundtrip."""
        resp = IngestResponse(
            success=success,
            filename=filename,
            chunks_created=chunks_created,
            chunks_stored=chunks_stored,
            documents_loaded=documents_loaded,
            file_type=file_type,
        )
        
        # Serialize to JSON
        json_str = resp.model_dump_json()
        
        # Deserialize back
        resp_dict = json.loads(json_str)
        resp_restored = IngestResponse(**resp_dict)
        
        assert resp_restored.success == resp.success
        assert resp_restored.filename == resp.filename
        assert resp_restored.chunks_created == resp.chunks_created
        assert resp_restored.chunks_stored == resp.chunks_stored

    @given(
        filename=valid_filename(),
        chunks_created=st.integers(min_value=-100, max_value=-1),
    )
    def test_negative_chunks_rejection(self, filename, chunks_created):
        """Should reject negative chunk counts."""
        with pytest.raises(ValidationError):
            IngestResponse(
                success=True,
                filename=filename,
                chunks_created=chunks_created,
                chunks_stored=0,
                documents_loaded=0,
                file_type="txt",
            )


@pytest.mark.property
class TestPropertyBasedErrorResponse:
    """Property-based tests for ErrorResponse model."""

    @given(
        error=st.text(min_size=1, max_size=50),
        message=st.text(min_size=1, max_size=200),
        detail=st.text(min_size=1, max_size=500) | st.none(),
        corr_id=correlation_id() | st.none(),
    )
    def test_serialization_roundtrip(self, error, message, detail, corr_id):
        """ErrorResponse should survive JSON serialization roundtrip."""
        resp = ErrorResponse(
            error=error,
            message=message,
            detail=detail,
            correlation_id=corr_id,
        )
        
        # Serialize to JSON
        json_str = resp.model_dump_json()
        
        # Deserialize back
        resp_dict = json.loads(json_str)
        resp_restored = ErrorResponse(**resp_dict)
        
        assert resp_restored.error == resp.error
        assert resp_restored.message == resp.message
        assert resp_restored.detail == resp.detail
        assert resp_restored.correlation_id == resp.correlation_id


@pytest.mark.property
class TestPropertyBasedHealthResponse:
    """Property-based tests for HealthResponse model."""

    @given(
        status=st.sampled_from(["healthy", "degraded", "unhealthy"]),
        version=st.text(min_size=1, max_size=20),
        services=st.dictionaries(
            keys=st.sampled_from(["vectorstore", "llm", "embeddings", "api"]),
            values=st.booleans(),
            min_size=1,
            max_size=4,
        ),
    )
    def test_serialization_roundtrip(self, status, version, services):
        """HealthResponse should survive JSON serialization roundtrip."""
        resp = HealthResponse(
            status=status,
            services=services,
            version=version,
        )
        
        # Serialize to JSON
        json_str = resp.model_dump_json()
        
        # Deserialize back
        resp_dict = json.loads(json_str)
        resp_restored = HealthResponse(**resp_dict)
        
        assert resp_restored.status == resp.status
        assert resp_restored.version == resp.version
        assert resp_restored.services == resp.services


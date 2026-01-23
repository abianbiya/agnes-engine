"""
Unit tests for RAG chat chain module.

Tests RAG chain functionality, source document handling, and response generation
with mocked LLM and retriever components.
"""

from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from src.chat.chain import (
    ChatResponse,
    RAGChatChain,
    SimpleRAGChain,
    SourceDocument,
)
from src.chat.memory import ConversationMemory
from src.retrieval.retriever import RetrievedDocument


class TestSourceDocument:
    """Test SourceDocument dataclass."""
    
    def test_source_document_creation(self):
        """Should create SourceDocument with required fields."""
        doc = SourceDocument(
            filename="test.pdf",
            page=1,
            section="Introduction",
            relevance_score=0.95,
        )
        
        assert doc.filename == "test.pdf"
        assert doc.page == 1
        assert doc.section == "Introduction"
        assert doc.relevance_score == 0.95
    
    def test_source_document_from_retrieved_doc(self):
        """Should create from RetrievedDocument."""
        retrieved = RetrievedDocument(
            content="This is test content.",
            metadata={"source": "doc.txt", "page": 5, "section": "Chapter 1"},
            score=0.88,
            source="doc.txt",
        )
        
        source_doc = SourceDocument.from_retrieved_doc(retrieved)
        
        assert source_doc.filename == "doc.txt"
        assert source_doc.page == 5
        assert source_doc.section == "Chapter 1"
        assert source_doc.relevance_score == 0.88
        assert "This is test content" in source_doc.content_preview
    
    def test_source_document_from_retrieved_doc_truncates_preview(self):
        """Should truncate long content in preview."""
        long_content = "x" * 200
        retrieved = RetrievedDocument(
            content=long_content,
            metadata={"source": "long.txt"},
            score=0.5,
            source="long.txt",
        )
        
        source_doc = SourceDocument.from_retrieved_doc(retrieved)
        
        assert len(source_doc.content_preview) <= 104  # 100 + "..."
        assert source_doc.content_preview.endswith("...")
    
    def test_source_document_to_dict(self):
        """Should convert to dictionary."""
        doc = SourceDocument(
            filename="test.pdf",
            page=2,
            relevance_score=0.87654321,
        )
        
        result = doc.to_dict()
        
        assert result["filename"] == "test.pdf"
        assert result["page"] == 2
        assert result["relevance_score"] == 0.8765  # Rounded


class TestChatResponse:
    """Test ChatResponse dataclass."""
    
    def test_chat_response_creation(self):
        """Should create ChatResponse with required fields."""
        sources = [
            SourceDocument(filename="doc1.txt", relevance_score=0.9),
            SourceDocument(filename="doc2.txt", relevance_score=0.8),
        ]
        
        response = ChatResponse(
            answer="This is the answer.",
            sources=sources,
            session_id="test-session",
        )
        
        assert response.answer == "This is the answer."
        assert len(response.sources) == 2
        assert response.session_id == "test-session"
    
    def test_chat_response_to_dict(self):
        """Should convert to dictionary."""
        sources = [SourceDocument(filename="doc.txt", relevance_score=0.9)]
        response = ChatResponse(
            answer="Answer text",
            sources=sources,
            session_id="session-123",
            metadata={"tokens": 150},
        )
        
        result = response.to_dict()
        
        assert result["answer"] == "Answer text"
        assert len(result["sources"]) == 1
        assert result["session_id"] == "session-123"
        assert result["metadata"]["tokens"] == 150


@pytest.fixture
def mock_llm():
    """Create a fake LLM for testing that works with LangChain."""
    from langchain_core.language_models.fake_chat_models import FakeChatModel
    
    # FakeChatModel that returns predictable responses
    responses = ["This is a test response."]
    return FakeChatModel(responses=responses)


@pytest.fixture
def mock_retriever():
    """Create a mock retriever for testing."""
    retriever = MagicMock()
    
    # Mock search results
    retriever.search.return_value = [
        RetrievedDocument(
            content="Python is a programming language.",
            metadata={"source": "python_guide.txt", "page": 1},
            score=0.95,
            source="python_guide.txt",
        ),
        RetrievedDocument(
            content="Python is easy to learn.",
            metadata={"source": "python_basics.txt"},
            score=0.87,
            source="python_basics.txt",
        ),
    ]
    
    retriever.mmr_search.return_value = retriever.search.return_value
    
    return retriever


@pytest.fixture
def mock_memory():
    """Create a real ConversationMemory instance for testing."""
    return ConversationMemory(window_size=5)


class TestRAGChatChainInit:
    """Test RAGChatChain initialization."""
    
    def test_init_with_required_params(self, mock_llm, mock_retriever, mock_memory):
        """Should initialize with required parameters."""
        chain = RAGChatChain(
            llm=mock_llm,
            retriever=mock_retriever,
            memory=mock_memory,
        )
        
        assert chain.llm == mock_llm
        assert chain.retriever == mock_retriever
        assert chain.memory == mock_memory
        assert chain.use_mmr is False
        assert chain.chain is not None
    
    def test_init_with_mmr_enabled(self, mock_llm, mock_retriever, mock_memory):
        """Should initialize with MMR enabled."""
        chain = RAGChatChain(
            llm=mock_llm,
            retriever=mock_retriever,
            memory=mock_memory,
            use_mmr=True,
        )
        
        assert chain.use_mmr is True


class TestRAGChatChainChat:
    """Test RAG chat chain chat method."""
    
    @pytest.mark.asyncio
    async def test_chat_with_new_session(self, mock_llm, mock_retriever, mock_memory):
        """Should handle chat in new session."""
        chain = RAGChatChain(
            llm=mock_llm,
            retriever=mock_retriever,
            memory=mock_memory,
        )
        
        session_id = mock_memory.create_session()
        
        response = await chain.chat(
            question="What is Python?",
            session_id=session_id,
        )
        
        assert isinstance(response, ChatResponse)
        assert len(response.answer) > 0
        assert response.session_id == session_id
        assert len(response.sources) > 0
    
    @pytest.mark.asyncio
    async def test_chat_retrieves_documents(self, mock_llm, mock_retriever, mock_memory):
        """Should retrieve documents for context."""
        chain = RAGChatChain(
            llm=mock_llm,
            retriever=mock_retriever,
            memory=mock_memory,
        )
        
        session_id = mock_memory.create_session()
        
        await chain.chat(
            question="What is Python?",
            session_id=session_id,
        )
        
        # Verify retriever was called
        mock_retriever.search.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_chat_adds_to_memory(self, mock_llm, mock_retriever, mock_memory):
        """Should add exchange to conversation memory."""
        chain = RAGChatChain(
            llm=mock_llm,
            retriever=mock_retriever,
            memory=mock_memory,
        )
        
        session_id = mock_memory.create_session()
        
        await chain.chat(
            question="Test question",
            session_id=session_id,
        )
        
        # Check memory has the exchange
        messages = mock_memory.get_messages(session_id)
        assert len(messages) == 2  # Human + AI
        assert isinstance(messages[0], HumanMessage)
        assert isinstance(messages[1], AIMessage)
        assert messages[0].content == "Test question"
    
    @pytest.mark.asyncio
    async def test_chat_with_mmr(self, mock_llm, mock_retriever, mock_memory):
        """Should use MMR when enabled."""
        chain = RAGChatChain(
            llm=mock_llm,
            retriever=mock_retriever,
            memory=mock_memory,
            use_mmr=True,
        )
        
        session_id = mock_memory.create_session()
        
        await chain.chat(
            question="Test",
            session_id=session_id,
        )
        
        # Verify MMR was used
        mock_retriever.mmr_search.assert_called_once()
        mock_retriever.search.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_chat_handles_errors(self, mock_llm, mock_retriever, mock_memory):
        """Should handle and log errors during chat."""
        chain = RAGChatChain(
            llm=mock_llm,
            retriever=mock_retriever,
            memory=mock_memory,
        )
        
        session_id = mock_memory.create_session()
        
        # Mock the retriever to raise an error
        mock_retriever.search.side_effect = Exception("Test error")
        
        with pytest.raises(Exception, match="Test error"):
            await chain.chat(
                question="Test",
                session_id=session_id,
            )


class TestRAGChatChainStreamChat:
    """Test RAG chat chain streaming."""
    
    @pytest.mark.asyncio
    async def test_stream_chat_yields_chunks(self, mock_llm, mock_retriever, mock_memory):
        """Should stream response chunks."""
        chain = RAGChatChain(
            llm=mock_llm,
            retriever=mock_retriever,
            memory=mock_memory,
        )
        
        session_id = mock_memory.create_session()
        
        chunks = []
        async for chunk in chain.stream_chat(
            question="Test question",
            session_id=session_id,
        ):
            chunks.append(chunk)
        
        assert len(chunks) > 0
        full_response = "".join(chunks)
        assert len(full_response) > 0
    
    @pytest.mark.asyncio
    async def test_stream_chat_adds_to_memory(self, mock_llm, mock_retriever, mock_memory):
        """Should add completed response to memory."""
        chain = RAGChatChain(
            llm=mock_llm,
            retriever=mock_retriever,
            memory=mock_memory,
        )
        
        session_id = mock_memory.create_session()
        
        # Consume the stream
        async for _ in chain.stream_chat(
            question="Test question",
            session_id=session_id,
        ):
            pass
        
        # Check memory
        messages = mock_memory.get_messages(session_id)
        assert len(messages) == 2


class TestRAGChatChainMethods:
    """Test additional RAG chain methods."""
    
    def test_get_sources_returns_last_sources(self, mock_llm, mock_retriever, mock_memory):
        """Should return sources from last retrieval."""
        chain = RAGChatChain(
            llm=mock_llm,
            retriever=mock_retriever,
            memory=mock_memory,
        )
        
        # Simulate retrieval by setting _last_retrieved_docs
        chain._last_retrieved_docs = [
            RetrievedDocument(
                content="Test",
                metadata={"source": "test.txt"},
                score=0.9,
                source="test.txt",
            )
        ]
        
        sources = chain.get_sources()
        
        assert len(sources) == 1
        assert isinstance(sources[0], SourceDocument)
        assert sources[0].filename == "test.txt"
    
    def test_clear_session_clears_memory(self, mock_llm, mock_retriever, mock_memory):
        """Should clear session from memory."""
        chain = RAGChatChain(
            llm=mock_llm,
            retriever=mock_retriever,
            memory=mock_memory,
        )
        
        session_id = mock_memory.create_session()
        mock_memory.add_exchange(session_id, "Q", "A")
        
        chain.clear_session(session_id)
        
        messages = mock_memory.get_messages(session_id)
        assert len(messages) == 0


class TestSimpleRAGChainInit:
    """Test SimpleRAGChain initialization."""
    
    def test_init_without_memory(self, mock_llm, mock_retriever):
        """Should initialize without conversation memory."""
        chain = SimpleRAGChain(
            llm=mock_llm,
            retriever=mock_retriever,
        )
        
        assert chain.llm == mock_llm
        assert chain.retriever == mock_retriever
        assert chain.use_mmr is False
        assert chain.chain is not None
    
    def test_init_with_mmr(self, mock_llm, mock_retriever):
        """Should initialize with MMR enabled."""
        chain = SimpleRAGChain(
            llm=mock_llm,
            retriever=mock_retriever,
            use_mmr=True,
        )
        
        assert chain.use_mmr is True


class TestSimpleRAGChainAsk:
    """Test SimpleRAGChain ask method."""
    
    @pytest.mark.asyncio
    async def test_ask_returns_response(self, mock_llm, mock_retriever):
        """Should return ChatResponse for question."""
        chain = SimpleRAGChain(
            llm=mock_llm,
            retriever=mock_retriever,
        )
        
        response = await chain.ask("What is Python?")
        
        assert isinstance(response, ChatResponse)
        assert len(response.answer) > 0
        assert response.session_id == "one-off"
        assert len(response.sources) > 0
    
    @pytest.mark.asyncio
    async def test_ask_retrieves_documents(self, mock_llm, mock_retriever):
        """Should retrieve documents for context."""
        chain = SimpleRAGChain(
            llm=mock_llm,
            retriever=mock_retriever,
        )
        
        await chain.ask("Test question")
        
        mock_retriever.search.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_ask_with_mmr(self, mock_llm, mock_retriever):
        """Should use MMR when enabled."""
        chain = SimpleRAGChain(
            llm=mock_llm,
            retriever=mock_retriever,
            use_mmr=True,
        )
        
        await chain.ask("Test question")
        
        mock_retriever.mmr_search.assert_called_once()
        mock_retriever.search.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_ask_handles_errors(self, mock_llm, mock_retriever):
        """Should handle errors during question processing."""
        chain = SimpleRAGChain(
            llm=mock_llm,
            retriever=mock_retriever,
        )
        
        # Mock the retriever to raise an error
        mock_retriever.search.side_effect = Exception("Test error")
        
        with pytest.raises(Exception, match="Test error"):
            await chain.ask("Test question")


@pytest.mark.parametrize(
    "use_mmr,expected_method",
    [
        (False, "search"),
        (True, "mmr_search"),
    ],
)
@pytest.mark.asyncio
async def test_retrieval_method_selection_parametrized(
    use_mmr, expected_method, mock_llm, mock_retriever, mock_memory
):
    """Test that correct retrieval method is used based on use_mmr flag."""
    chain = RAGChatChain(
        llm=mock_llm,
        retriever=mock_retriever,
        memory=mock_memory,
        use_mmr=use_mmr,
    )
    
    session_id = mock_memory.create_session()
    
    await chain.chat(
        question="Test",
        session_id=session_id,
    )
    
    if expected_method == "search":
        mock_retriever.search.assert_called_once()
        mock_retriever.mmr_search.assert_not_called()
    else:
        mock_retriever.mmr_search.assert_called_once()
        mock_retriever.search.assert_not_called()

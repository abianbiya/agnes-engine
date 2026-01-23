"""
RAG chain implementation for conversational question-answering.

This module implements the complete RAG (Retrieval-Augmented Generation) chain
using LangChain Expression Language (LCEL) with support for conversation history,
source citations, and streaming responses.
"""

import re
from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, List, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    Runnable,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)

from src.chat.memory import ConversationMemory
from src.chat.prompts import (
    CONDENSE_QUESTION_PROMPT,
    RAG_CHAT_PROMPT,
    format_chat_history,
    format_docs_for_context,
)
from src.retrieval.retriever import RAGRetriever, RetrievedDocument
from src.utils.logging import LoggerMixin


@dataclass
class SourceDocument:
    """
    Represents a source document cited in the response.
    
    Attributes:
        filename: Name of the source file
        page: Page number (if applicable)
        section: Section name (if applicable)
        relevance_score: Retrieval relevance score
        content_preview: Preview of the referenced content
        retrieval_method: Method used to retrieve this document
            ("semantic", "bm25", or "hybrid" if found by both)
    """
    filename: str
    page: Optional[int] = None
    section: Optional[str] = None
    relevance_score: float = 0.0
    content_preview: Optional[str] = None
    retrieval_method: str = "semantic"
    
    @classmethod
    def from_retrieved_doc(cls, doc: RetrievedDocument) -> "SourceDocument":
        """
        Create SourceDocument from RetrievedDocument.
        
        Args:
            doc: Retrieved document from RAGRetriever
            
        Returns:
            SourceDocument instance
        """
        # Extract metadata
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page")
        section = doc.metadata.get("section")
        
        # Create preview (first 100 chars)
        preview = doc.content[:100] + "..." if len(doc.content) > 100 else doc.content
        
        return cls(
            filename=source,
            page=page,
            section=section,
            relevance_score=doc.score,
            content_preview=preview,
            retrieval_method=getattr(doc, "retrieval_method", "semantic"),
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "filename": self.filename,
            "page": self.page,
            "section": self.section,
            "relevance_score": round(self.relevance_score, 4),
            "content_preview": self.content_preview,
            "retrieval_method": self.retrieval_method,
        }


@dataclass
class ChatResponse:
    """
    Response from the RAG chat chain.
    
    Attributes:
        answer: Generated answer text
        sources: List of source documents cited
        session_id: Session identifier
        metadata: Additional metadata (e.g., token usage, timing)
    """
    answer: str
    sources: List[SourceDocument]
    session_id: str
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "answer": self.answer,
            "sources": [src.to_dict() for src in self.sources],
            "session_id": self.session_id,
            "metadata": self.metadata or {},
        }


class RAGChatChain(LoggerMixin):
    """
    Conversational RAG chain with memory and source citation.
    
    Implements the complete RAG workflow:
    1. Condense follow-up questions using chat history
    2. Retrieve relevant documents
    3. Generate answer with source citations
    4. Track conversation history
    
    Attributes:
        llm: Language model for generation
        retriever: Document retriever
        memory: Conversation memory manager
        chain: Compiled LangChain LCEL chain
    """
    
    def __init__(
        self,
        llm: BaseChatModel,
        retriever: RAGRetriever,
        memory: ConversationMemory,
        use_mmr: bool = False,
    ):
        """
        Initialize RAG chat chain.
        
        Args:
            llm: LangChain chat model
            retriever: RAG retriever for document search
            memory: Conversation memory manager
            use_mmr: Whether to use MMR for diverse results (default: False)
        """
        super().__init__()
        
        self.llm = llm
        self.retriever = retriever
        self.memory = memory
        self.use_mmr = use_mmr
        self.chain = self._build_chain()
        
        self.logger.info(
            "initialized_rag_chat_chain",
            llm_model=getattr(llm, "model_name", "unknown"),
            use_mmr=use_mmr,
        )
    
    def _build_chain(self) -> Runnable:
        """
        Build the RAG chain using LangChain Expression Language (LCEL).
        
        Chain flow:
        1. Check if chat history exists
        2. If yes, condense the question to standalone form
        3. Retrieve relevant documents
        4. Format context from documents
        5. Generate answer with LLM
        6. Parse output
        
        Returns:
            Compiled LangChain runnable
        """
        # Chain to condense follow-up questions
        condense_chain = (
            {
                "chat_history": lambda x: format_chat_history(x["chat_history"]),
                "question": lambda x: x["question"],
            }
            | CONDENSE_QUESTION_PROMPT
            | self.llm
            | StrOutputParser()
        )
        
        # Function to handle conditional condensing
        def get_standalone_question(inputs: Dict[str, Any]) -> str:
            """Get standalone question (condensed if history exists)."""
            chat_history = inputs.get("chat_history", [])
            question = inputs["question"]
            
            # If no history, use original question
            if not chat_history:
                return question
            
            # Otherwise, condense the question
            return condense_chain.invoke(inputs)
        
        # Main RAG chain
        # Use RunnableLambda with afunc for async retrieval
        async def retrieve_context(x: Dict[str, Any]) -> str:
            return await self._retrieve_and_format(x["standalone_question"])
        
        rag_chain = (
            RunnablePassthrough.assign(
                standalone_question=get_standalone_question
            )
            | RunnableParallel({
                "context": RunnableLambda(func=lambda x: "", afunc=retrieve_context),
                "chat_history": lambda x: format_chat_history(x.get("chat_history", [])),
                "question": lambda x: x["standalone_question"],
            })
            | RAG_CHAT_PROMPT
            | self.llm
            | StrOutputParser()
        )
        
        return rag_chain
    
    def _simple_expand_query(self, query: str) -> str:
        """
        Simple keyword-based query expansion without LLM.
        
        This is more reliable than LLM-based expansion which can produce
        conversational responses instead of expanded queries.
        
        Args:
            query: Original search query
            
        Returns:
            Expanded query with synonyms and related terms
        """
        # Keyword expansions for common Indonesian academic terms
        # Key insight: include phrases that appear in source documents
        expansions = {
            "rektor": "rektor rector dipimpin pemimpin pimpinan universitas terkemuka",
            "unnes": "unnes UNNES Universitas Negeri Semarang universitas terkemuka Indonesia",
            "biaya": "biaya UKT SPP uang kuliah tunggal tarif pembayaran",
            "kuliah": "kuliah studi belajar pendidikan akademik",
            "dosen": "dosen pengajar guru besar lecturer profesor",
            "fakultas": "fakultas jurusan prodi program studi",
            "mahasiswa": "mahasiswa siswa pelajar student calon",
            "beasiswa": "beasiswa bantuan scholarship financial aid",
            "pendaftaran": "pendaftaran daftar registrasi admission seleksi",
            "jurusan": "jurusan prodi program studi departemen",
            "akreditasi": "akreditasi akred accreditation peringkat",
            "visi": "visi misi tujuan goal vision mission",
            "sejarah": "sejarah history didirikan founded awalnya",
            "alamat": "alamat lokasi address location kampus",
            "kontak": "kontak contact telepon email phone",
            "siapa": "siapa who profil tentang",
        }
        
        query_lower = query.lower()
        expanded = query
        
        for key, expansion in expansions.items():
            if key in query_lower:
                expanded += " " + expansion
        
        return expanded
    
    async def _retrieve_and_format(self, query: str) -> str:
        """
        Retrieve documents and format them for context.
        
        Uses simple keyword-based query expansion for Indonesian queries
        to improve semantic search reliability.
        
        Args:
            query: Search query
            
        Returns:
            Formatted context string
        """
        # Use original query directly - expansion was hurting retrieval quality
        # due to diluting semantic similarity
        search_query = query
        
        self.logger.info(
            "query_search",
            original=query,
            search_query=search_query[:200],
        )
        
        # Retrieve documents using the query
        docs = await self.retriever.retrieve(search_query)
        
        # Convert to LangChain Document format for formatting
        from langchain_core.documents import Document
        lc_docs = [
            Document(page_content=doc.content, metadata=doc.metadata)
            for doc in docs
        ]
        
        # Store docs for later source extraction
        self._last_retrieved_docs = docs
        
        return format_docs_for_context(lc_docs)
    
    async def chat(
        self,
        question: str,
        session_id: Optional[str] = None,
    ) -> ChatResponse:
        """
        Process a chat question and return a response with sources.
        
        Args:
            question: User's question
            session_id: Session identifier for conversation history.
                If None, a new session will be created.
            
        Returns:
            ChatResponse with answer and source citations
        """
        # Create session if not provided
        if session_id is None:
            session_id = self.memory.create_session()
        
        self.logger.info(
            "processing_chat_request",
            session_id=session_id,
            question_length=len(question),
        )
        
        # Get conversation history
        chat_history = self.memory.get_messages(session_id)
        
        # Prepare input
        chain_input = {
            "question": question,
            "chat_history": chat_history,
        }
        
        # Invoke chain
        try:
            answer = await self.chain.ainvoke(chain_input)
            
            # Extract sources from retrieved documents
            sources = [
                SourceDocument.from_retrieved_doc(doc)
                for doc in getattr(self, "_last_retrieved_docs", [])
            ]
            
            # Add exchange to memory
            self.memory.add_exchange(session_id, question, answer)
            
            response = ChatResponse(
                answer=answer,
                sources=sources,
                session_id=session_id,
            )
            
            self.logger.info(
                "chat_response_generated",
                session_id=session_id,
                answer_length=len(answer),
                source_count=len(sources),
            )
            
            return response
            
        except Exception as e:
            self.logger.error(
                "chat_processing_failed",
                session_id=session_id,
                error=str(e),
                exc_info=True,
            )
            raise
    
    async def stream_chat(
        self,
        question: str,
        session_id: Optional[str] = None,
    ) -> AsyncIterator[str]:
        """
        Stream chat response tokens in real-time.
        
        Args:
            question: User's question
            session_id: Session identifier. If None, a new session will be created.
            
        Yields:
            Response text chunks as they are generated
        """
        # Create session if not provided
        if session_id is None:
            session_id = self.memory.create_session()
        
        self.logger.info(
            "processing_streaming_chat",
            session_id=session_id,
            question_length=len(question),
        )
        
        # Get conversation history
        chat_history = self.memory.get_messages(session_id)
        
        # Prepare input
        chain_input = {
            "question": question,
            "chat_history": chat_history,
        }
        
        # Stream response
        full_response = ""
        
        try:
            async for chunk in self.chain.astream(chain_input):
                full_response += chunk
                yield chunk
            
            # After streaming completes, add to memory
            self.memory.add_exchange(session_id, question, full_response)
            
            self.logger.info(
                "streaming_chat_completed",
                session_id=session_id,
                answer_length=len(full_response),
            )
            
        except Exception as e:
            self.logger.error(
                "streaming_chat_failed",
                session_id=session_id,
                error=str(e),
                exc_info=True,
            )
            raise
    
    def get_sources(self) -> List[SourceDocument]:
        """
        Get sources from the last retrieval.
        
        Returns:
            List of source documents from last chat invocation
        """
        docs = getattr(self, "_last_retrieved_docs", [])
        return [SourceDocument.from_retrieved_doc(doc) for doc in docs]
    
    def clear_session(self, session_id: str) -> None:
        """
        Clear conversation history for a session.
        
        Args:
            session_id: Session identifier
        """
        self.memory.clear_session(session_id)
        self.logger.info("cleared_session", session_id=session_id)


class SimpleRAGChain(LoggerMixin):
    """
    Simple RAG chain without conversation memory.
    
    Use this for one-off questions that don't require conversation history.
    """
    
    def __init__(
        self,
        llm: BaseChatModel,
        retriever: RAGRetriever,
        use_mmr: bool = False,
    ):
        """
        Initialize simple RAG chain.
        
        Args:
            llm: LangChain chat model
            retriever: RAG retriever
            use_mmr: Whether to use MMR (default: False)
        """
        super().__init__()
        
        self.llm = llm
        self.retriever = retriever
        self.use_mmr = use_mmr
        self.chain = self._build_chain()
        
        self.logger.info(
            "initialized_simple_rag_chain",
            llm_model=getattr(llm, "model_name", "unknown"),
            use_mmr=use_mmr,
        )
    
    def _build_chain(self) -> Runnable:
        """Build simple RAG chain without history."""
        from src.chat.prompts import RAG_PROMPT
        
        # Use RunnableLambda with afunc for async retrieval
        async def retrieve_context(x: Dict[str, Any]) -> str:
            return await self._retrieve_and_format(x["question"])
        
        chain = (
            RunnableParallel({
                "context": RunnableLambda(func=lambda x: "", afunc=retrieve_context),
                "question": lambda x: x["question"],
            })
            | RAG_PROMPT
            | self.llm
            | StrOutputParser()
        )
        
        return chain
    
    async def _retrieve_and_format(self, query: str) -> str:
        """Retrieve and format documents."""
        # Use the async retrieve method - handles MMR internally
        docs = await self.retriever.retrieve(query)
        
        self._last_retrieved_docs = docs
        
        from langchain_core.documents import Document
        lc_docs = [
            Document(page_content=doc.content, metadata=doc.metadata)
            for doc in docs
        ]
        
        return format_docs_for_context(lc_docs)
    
    async def ask(self, question: str) -> ChatResponse:
        """
        Ask a question without conversation history.
        
        Args:
            question: User's question
            
        Returns:
            ChatResponse with answer and sources
        """
        self.logger.info(
            "processing_simple_question",
            question_length=len(question),
        )
        
        try:
            answer = await self.chain.ainvoke({"question": question})
            
            sources = [
                SourceDocument.from_retrieved_doc(doc)
                for doc in getattr(self, "_last_retrieved_docs", [])
            ]
            
            response = ChatResponse(
                answer=answer,
                sources=sources,
                session_id="one-off",
            )
            
            self.logger.info(
                "simple_response_generated",
                answer_length=len(answer),
                source_count=len(sources),
            )
            
            return response
            
        except Exception as e:
            self.logger.error(
                "simple_question_failed",
                error=str(e),
                exc_info=True,
            )
            raise


__all__ = [
    "RAGChatChain",
    "SimpleRAGChain",
    "ChatResponse",
    "SourceDocument",
]

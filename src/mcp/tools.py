"""
MCP tool definitions and handlers for the RAG chatbot.

This module defines the Model Context Protocol (MCP) tools that expose
RAG chatbot functionality to AI assistants like Claude Desktop.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

from mcp.types import Tool

from src.chat.chain import RAGChatChain
from src.core.vectorstore import VectorStoreManager
from src.ingestion.pipeline import IngestionPipeline
from src.retrieval.retriever import RAGRetriever
from src.utils.logging import LoggerMixin


# MCP Tool Schemas
RAG_CHAT_TOOL = Tool(
    name="rag_chat",
    description=(
        "Chat with documents using Retrieval-Augmented Generation (RAG). "
        "Ask questions and get answers based on ingested documents with source citations. "
        "Supports conversation continuity with session IDs."
    ),
    inputSchema={
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "The question to ask about the documents"
            },
            "session_id": {
                "type": "string",
                "description": (
                    "Optional session ID for conversation continuity. "
                    "Use the same session_id for follow-up questions."
                )
            },
        },
        "required": ["question"]
    }
)


DOCUMENT_SEARCH_TOOL = Tool(
    name="document_search",
    description=(
        "Search documents for relevant content using semantic similarity. "
        "Returns the most relevant document chunks with metadata and relevance scores. "
        "Use this to find specific information without generating an answer."
    ),
    inputSchema={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query to find relevant documents"
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of results to return (default: 4)",
                "default": 4,
                "minimum": 1,
                "maximum": 20
            },
            "use_mmr": {
                "type": "boolean",
                "description": (
                    "Use Maximum Marginal Relevance for diverse results (default: false)"
                ),
                "default": False
            }
        },
        "required": ["query"]
    }
)


INGEST_DOCUMENT_TOOL = Tool(
    name="ingest_document",
    description=(
        "Ingest a document file into the knowledge base. "
        "Supports PDF, TXT, and Markdown files. "
        "The document will be chunked and embedded for retrieval."
    ),
    inputSchema={
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Absolute or relative path to the document file"
            },
        },
        "required": ["file_path"]
    }
)


LIST_DOCUMENTS_TOOL = Tool(
    name="list_documents",
    description=(
        "List all documents in the knowledge base with metadata. "
        "Shows document names, types, and ingestion information."
    ),
    inputSchema={
        "type": "object",
        "properties": {},
        "required": []
    }
)


# Tool list for registration
TOOLS = [
    RAG_CHAT_TOOL,
    DOCUMENT_SEARCH_TOOL,
    INGEST_DOCUMENT_TOOL,
    LIST_DOCUMENTS_TOOL,
]


class MCPToolHandler(LoggerMixin):
    """
    Handler for MCP tool invocations.
    
    Processes tool calls and delegates to appropriate components
    (chat chain, retriever, ingestion pipeline).
    
    Attributes:
        chat_chain: RAG chat chain for question answering
        retriever: Document retriever for search
        ingestion_pipeline: Pipeline for document ingestion
        vectorstore: Vector store manager for document listing
    """
    
    def __init__(
        self,
        chat_chain: RAGChatChain,
        retriever: RAGRetriever,
        ingestion_pipeline: IngestionPipeline,
        vectorstore: Optional[VectorStoreManager] = None,
    ):
        """
        Initialize MCP tool handler.
        
        Args:
            chat_chain: Initialized RAG chat chain
            retriever: Document retriever
            ingestion_pipeline: Document ingestion pipeline
            vectorstore: Optional vector store manager for document listing
        """
        super().__init__()
        
        self.chat_chain = chat_chain
        self.retriever = retriever
        self.ingestion_pipeline = ingestion_pipeline
        self.vectorstore = vectorstore
        
        self.logger.info(
            "initialized_mcp_tool_handler",
            available_tools=len(TOOLS),
        )
    
    async def handle_rag_chat(
        self,
        question: str,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Handle RAG chat tool invocation.
        
        Args:
            question: User's question
            session_id: Optional session ID for conversation continuity
            
        Returns:
            Dictionary with answer, sources, and session_id
            
        Raises:
            Exception: If chat processing fails
        """
        self.logger.info(
            "handling_rag_chat",
            question_length=len(question),
            has_session_id=session_id is not None,
        )
        
        try:
            # Generate session ID if not provided
            if session_id is None:
                session_id = self.chat_chain.memory.create_session()
            else:
                # Ensure session exists
                self.chat_chain.memory.get_session_history(session_id)
            
            # Process question
            response = await self.chat_chain.chat(
                question=question,
                session_id=session_id,
            )
            
            # Convert to MCP-compatible dict
            result = response.to_dict()
            
            self.logger.info(
                "rag_chat_completed",
                session_id=session_id,
                source_count=len(response.sources),
            )
            
            return result
            
        except Exception as e:
            self.logger.error(
                "rag_chat_failed",
                error=str(e),
                question_length=len(question),
                exc_info=True,
            )
            raise
    
    async def handle_document_search(
        self,
        query: str,
        limit: int = 4,
        use_mmr: bool = False,
    ) -> Dict[str, Any]:
        """
        Handle document search tool invocation.
        
        Args:
            query: Search query
            limit: Maximum number of results
            use_mmr: Whether to use MMR for diversity
            
        Returns:
            Dictionary with search results
            
        Raises:
            Exception: If search fails
        """
        self.logger.info(
            "handling_document_search",
            query_length=len(query),
            limit=limit,
            use_mmr=use_mmr,
        )
        
        try:
            # Perform search
            if use_mmr:
                documents = self.retriever.mmr_search(query, k=limit)
            else:
                documents = self.retriever.search(query, k=limit)
            
            # Convert to MCP-compatible format
            results = [
                {
                    "content": doc.content,
                    "metadata": doc.metadata,
                    "score": round(doc.score, 4),
                    "source": doc.source,
                }
                for doc in documents
            ]
            
            self.logger.info(
                "document_search_completed",
                result_count=len(results),
            )
            
            return {
                "results": results,
                "count": len(results),
                "query": query,
            }
            
        except Exception as e:
            self.logger.error(
                "document_search_failed",
                error=str(e),
                query=query,
                exc_info=True,
            )
            raise
    
    async def handle_ingest_document(
        self,
        file_path: str,
    ) -> Dict[str, Any]:
        """
        Handle document ingestion tool invocation.
        
        Args:
            file_path: Path to document file
            
        Returns:
            Dictionary with ingestion results
            
        Raises:
            FileNotFoundError: If file doesn't exist
            Exception: If ingestion fails
        """
        self.logger.info(
            "handling_ingest_document",
            file_path=file_path,
        )
        
        try:
            # Validate file exists
            path = Path(file_path)
            if not path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            
            if not path.is_file():
                raise ValueError(f"Path is not a file: {file_path}")
            
            # Ingest document
            result = await self.ingestion_pipeline.ingest_file(file_path)
            
            # Convert to MCP-compatible format
            # Extract filename from file_path
            filename = Path(result.file_path).name
            file_ext = Path(result.file_path).suffix.lstrip(".")
            
            response = {
                "success": result.success,
                "filename": filename,
                "file_path": result.file_path,
                "chunks_created": result.num_chunks_created,
                "chunks_stored": result.num_chunks_stored,
                "documents_loaded": result.num_documents_loaded,
                "file_type": file_ext if file_ext else "unknown",
            }
            
            if result.error:
                response["error_message"] = result.error
            
            self.logger.info(
                "ingest_document_completed",
                filename=filename,
                chunks_created=result.num_chunks_created,
                success=result.success,
            )
            
            return response
            
        except FileNotFoundError as e:
            self.logger.error(
                "ingest_document_file_not_found",
                file_path=file_path,
                error=str(e),
            )
            raise
            
        except Exception as e:
            self.logger.error(
                "ingest_document_failed",
                file_path=file_path,
                error=str(e),
                exc_info=True,
            )
            raise
    
    async def handle_list_documents(self) -> Dict[str, Any]:
        """
        Handle list documents tool invocation.
        
        Returns:
            Dictionary with list of documents and metadata
            
        Raises:
            Exception: If listing fails
        """
        self.logger.info("handling_list_documents")
        
        try:
            if self.vectorstore is None:
                self.logger.warning("vectorstore_not_available_for_list_documents")
                return {
                    "documents": [],
                    "count": 0,
                    "total_chunks": 0,
                    "error": "VectorStore not available for document listing",
                }
            
            # Get all documents from vectorstore
            doc_list = self.vectorstore.list_documents()
            
            # Calculate total chunks
            total_chunks = sum(doc["chunk_count"] for doc in doc_list)
            
            self.logger.info(
                "list_documents_completed",
                document_count=len(doc_list),
                total_chunks=total_chunks,
            )
            
            return {
                "documents": doc_list,
                "count": len(doc_list),
                "total_chunks": total_chunks,
            }
            
        except Exception as e:
            self.logger.error(
                "list_documents_failed",
                error=str(e),
                exc_info=True,
            )
            raise
    
    def get_available_tools(self) -> List[Tool]:
        """
        Get list of available MCP tools.
        
        Returns:
            List of Tool objects
        """
        return TOOLS


__all__ = [
    "MCPToolHandler",
    "TOOLS",
    "RAG_CHAT_TOOL",
    "DOCUMENT_SEARCH_TOOL",
    "INGEST_DOCUMENT_TOOL",
    "LIST_DOCUMENTS_TOOL",
]

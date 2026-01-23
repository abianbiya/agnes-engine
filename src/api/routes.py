"""
FastAPI routes for the RAG chatbot API.

This module defines all REST API endpoints for the RAG system.
"""

import tempfile
from pathlib import Path
from typing import AsyncIterator

from fastapi import APIRouter, File, HTTPException, UploadFile, status
from fastapi.responses import StreamingResponse

from src.api.dependencies import (
    ChatChainDep,
    IngestionPipelineDep,
    MemoryDep,
    RetrieverDep,
    SettingsDep,
    VectorStoreDep,
    create_retriever_for_method,
    get_conversation_memory,
    get_settings,
    get_vectorstore,
)
from src.api.models import (
    ChatRequest,
    ChatResponse,
    CollectionClearResponse,
    DocumentInfo,
    DocumentListResponse,
    ErrorResponse,
    HealthResponse,
    IngestResponse,
    RetrievalMethod,
    SearchRequest,
    SearchResponse,
    SearchResult,
    SessionDeleteResponse,
    SourceDocument,
)
from src.utils.exceptions import (
    ChatGenerationError,
    ChatMemoryError,
    DocumentIngestionError,
    DocumentLoadError,
    DocumentParseError,
    EmptyRetrievalError,
    RetrievalError,
    SessionNotFoundError,
    ValidationError,
)
from src.utils.logging import LoggerMixin, get_correlation_id

# Create API router
router = APIRouter()


class RouteHandlers(LoggerMixin):
    """Handler class for API routes with logging support."""
    
    pass


# Instantiate handler for logging
handler = RouteHandlers()


@router.post(
    "/chat",
    response_model=ChatResponse,
    status_code=status.HTTP_200_OK,
    summary="Chat with RAG system",
    description="Send a question and get an AI-generated answer based on ingested documents.",
    responses={
        200: {"description": "Successful response with answer and sources"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def chat(
    request: ChatRequest,
    settings: SettingsDep,
    vectorstore: VectorStoreDep,
    memory: MemoryDep,
) -> ChatResponse:
    """
    Chat with the RAG system.
    
    Args:
        request: Chat request with question, optional session_id, and retrieval_method
        settings: Application settings
        vectorstore: Vector store manager
        memory: Conversation memory
        
    Returns:
        ChatResponse with answer, sources, and session_id
        
    Raises:
        HTTPException: If chat processing fails
    """
    from src.chat.chain import RAGChatChain
    from src.core.llm import get_llm
    
    correlation_id = get_correlation_id()
    
    handler.logger.info(
        "chat_request_received",
        question_length=len(request.question),
        has_session_id=request.session_id is not None,
        retrieval_method=request.retrieval_method.value,
        correlation_id=correlation_id,
    )
    
    try:
        # Create retriever based on requested method
        retriever = create_retriever_for_method(
            method=request.retrieval_method,
            settings=settings,
            vectorstore=vectorstore,
        )
        
        # Create chat chain with the selected retriever
        llm = get_llm(settings)
        chat_chain = RAGChatChain(
            llm=llm,
            retriever=retriever,
            memory=memory,
            use_mmr=settings.retrieval.use_mmr,
        )
        
        # Process chat request
        response = await chat_chain.chat(
            question=request.question,
            session_id=request.session_id,
        )
        
        # Convert to API response format
        api_response = ChatResponse(
            answer=response.answer,
            sources=[
                SourceDocument(
                    filename=source.filename,
                    page=source.page,
                    section=source.section,
                    relevance_score=source.relevance_score,
                    content_preview=source.content_preview,
                    retrieval_method=source.retrieval_method,
                )
                for source in response.sources
            ],
            session_id=response.session_id,
            metadata={
                **(response.metadata or {}),
                "retrieval_method": request.retrieval_method.value,
            },
        )
        
        handler.logger.info(
            "chat_request_completed",
            session_id=response.session_id,
            source_count=len(response.sources),
            retrieval_method=request.retrieval_method.value,
            correlation_id=correlation_id,
        )
        
        return api_response
        
    except SessionNotFoundError as e:
        # Session not found - let the global handler convert this
        raise
    except ChatMemoryError as e:
        # Memory-related error
        raise
    except EmptyRetrievalError as e:
        # No documents found for query
        raise ChatGenerationError(
            message="Unable to generate answer: no relevant documents found",
            details={"question": request.question[:100]},
            cause=e,
        )
    except RetrievalError as e:
        # Retrieval system failed
        raise ChatGenerationError(
            message="Failed to retrieve relevant documents",
            details={"question": request.question[:100]},
            cause=e,
        )
    except ValueError as e:
        # Invalid input data
        raise ValidationError(
            message="Invalid chat request",
            details={"question": request.question[:100]},
            cause=e,
        )
    except Exception as e:
        # Catch-all for unexpected errors
        raise ChatGenerationError(
            message="Chat processing failed",
            details={"question": request.question[:100]},
            cause=e,
        )


@router.post(
    "/chat/stream",
    summary="Stream chat response",
    description="Send a question and receive a streaming response with SSE.",
    responses={
        200: {"description": "Streaming response"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def chat_stream(
    request: ChatRequest,
    chat_chain: ChatChainDep,
) -> StreamingResponse:
    """
    Stream chat response using Server-Sent Events.
    
    Args:
        request: Chat request with question and optional session_id
        chat_chain: RAG chat chain dependency
        
    Returns:
        StreamingResponse with SSE
        
    Raises:
        HTTPException: If streaming fails
    """
    correlation_id = get_correlation_id()
    
    handler.logger.info(
        "chat_stream_request_received",
        question_length=len(request.question),
        correlation_id=correlation_id,
    )
    
    async def generate() -> AsyncIterator[str]:
        """Generate SSE events from chat stream."""
        try:
            async for chunk in chat_chain.stream_chat(
                question=request.question,
                session_id=request.session_id,
            ):
                # Send chunk as SSE event
                yield f"data: {chunk}\n\n"
                
        except ChatGenerationError as e:
            error_msg = e.message
            yield f"data: {{\"error\": \"{error_msg}\"}}\n\n"
        except Exception as e:
            handler.logger.error(
                "chat_stream_failed",
                error=str(e),
                correlation_id=correlation_id,
                exc_info=True,
            )
            yield f"data: {{\"error\": \"Streaming failed\"}}\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


@router.post(
    "/search",
    response_model=SearchResponse,
    status_code=status.HTTP_200_OK,
    summary="Search documents",
    description="Search for relevant documents using the specified retrieval method.",
    responses={
        200: {"description": "Search results"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def search(
    request: SearchRequest,
    settings: SettingsDep,
    vectorstore: VectorStoreDep,
) -> SearchResponse:
    """
    Search documents for relevant content.
    
    Args:
        request: Search request with query, parameters, and retrieval_method
        settings: Application settings
        vectorstore: Vector store manager
        
    Returns:
        SearchResponse with matching documents
        
    Raises:
        HTTPException: If search fails
    """
    correlation_id = get_correlation_id()
    
    handler.logger.info(
        "search_request_received",
        query_length=len(request.query),
        limit=request.limit,
        use_mmr=request.use_mmr,
        retrieval_method=request.retrieval_method.value,
        correlation_id=correlation_id,
    )
    
    try:
        # Create retriever based on requested method with the requested limit
        retriever = create_retriever_for_method(
            method=request.retrieval_method,
            settings=settings,
            vectorstore=vectorstore,
            k=request.limit,
        )
        
        # Perform search using the async retrieve method
        documents = await retriever.retrieve(request.query)
        
        # Convert to API response format
        results = [
            SearchResult(
                content=doc.content,
                metadata=doc.metadata,
                score=doc.score,
                source=doc.source,
            )
            for doc in documents
        ]
        
        handler.logger.info(
            "search_request_completed",
            result_count=len(results),
            retrieval_method=request.retrieval_method.value,
            correlation_id=correlation_id,
        )
        
        return SearchResponse(
            results=results,
            count=len(results),
            query=request.query,
        )
        
    except EmptyRetrievalError as e:
        # No results found - return empty results (not an error)
        handler.logger.info(
            "search_no_results",
            query=request.query[:100],
            correlation_id=correlation_id,
        )
        return SearchResponse(
            results=[],
            count=0,
            query=request.query,
        )
    except RetrievalError as e:
        # Retrieval system failed
        raise
    except ValueError as e:
        # Invalid search parameters
        raise ValidationError(
            message="Invalid search request",
            details={"query": request.query[:100], "limit": request.limit},
            cause=e,
        )
    except Exception as e:
        # Catch-all for unexpected errors
        raise RetrievalError(
            message="Search failed",
            details={"query": request.query[:100]},
            cause=e,
        )


@router.post(
    "/ingest",
    response_model=IngestResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Ingest document",
    description="Upload and ingest a document file into the knowledge base.",
    responses={
        201: {"description": "Document ingested successfully"},
        400: {"model": ErrorResponse, "description": "Invalid file or request"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def ingest_document(
    file: UploadFile = File(..., description="Document file to ingest"),
    pipeline: IngestionPipelineDep = None,
) -> IngestResponse:
    """
    Ingest a document file.
    
    Args:
        file: Uploaded document file
        pipeline: Ingestion pipeline dependency
        
    Returns:
        IngestResponse with ingestion results
        
    Raises:
        HTTPException: If ingestion fails
    """
    correlation_id = get_correlation_id()
    
    handler.logger.info(
        "ingest_request_received",
        filename=file.filename,
        content_type=file.content_type,
        correlation_id=correlation_id,
    )
    
    # Validate file type
    allowed_extensions = {".pdf", ".txt", ".md"}
    file_ext = Path(file.filename).suffix.lower() if file.filename else ""
    
    if file_ext not in allowed_extensions:
        handler.logger.warning(
            "ingest_request_invalid_file_type",
            filename=file.filename,
            extension=file_ext,
            correlation_id=correlation_id,
        )
        raise ValidationError(
            message=f"Unsupported file type: {file_ext}",
            details={
                "filename": file.filename,
                "extension": file_ext,
                "allowed_extensions": list(allowed_extensions),
            },
        )
    
    # Save uploaded file to temporary location
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=file_ext,
        ) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_path = temp_file.name
        
        handler.logger.debug(
            "file_saved_temporarily",
            temp_path=temp_path,
            size_bytes=len(content),
            correlation_id=correlation_id,
        )
        
        # Ingest document
        result = await pipeline.ingest_file(temp_path)
        
        # Clean up temp file
        Path(temp_path).unlink(missing_ok=True)
        
        # Convert to API response format
        response = IngestResponse(
            success=result.success,
            filename=file.filename or "unknown",
            file_path=result.file_path,
            chunks_created=result.num_chunks_created,
            chunks_stored=result.num_chunks_stored,
            documents_loaded=result.num_documents_loaded,
            file_type=file_ext.lstrip("."),
            error_message=result.error,
        )
        
        handler.logger.info(
            "ingest_request_completed",
            filename=file.filename,
            success=result.success,
            chunks_created=result.num_chunks_created,
            correlation_id=correlation_id,
        )
        
        return response
        
    except FileNotFoundError as e:
        # File doesn't exist
        raise DocumentLoadError(
            message=f"Document file not found: {file.filename}",
            details={"filename": file.filename, "path": temp_path},
            cause=e,
        )
    except PermissionError as e:
        # Permission denied reading file
        raise DocumentLoadError(
            message=f"Permission denied reading document: {file.filename}",
            details={"filename": file.filename, "path": temp_path},
            cause=e,
        )
    except DocumentLoadError as e:
        # Document loading failed
        raise
    except DocumentParseError as e:
        # Document parsing failed
        raise
    except Exception as e:
        # Catch-all for unexpected errors
        raise DocumentIngestionError(
            message=f"Failed to ingest document: {file.filename}",
            details={"filename": file.filename},
            cause=e,
        )
    finally:
        # Always clean up temp file
        if temp_path is not None:
            try:
                Path(temp_path).unlink(missing_ok=True)
            except Exception:
                pass


@router.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Health check",
    description="Check the health status of the RAG system and its components.",
    responses={
        200: {"description": "System health status"},
    },
)
async def health_check(
    vectorstore: VectorStoreDep,
    settings: SettingsDep,
) -> HealthResponse:
    """
    Health check endpoint.
    
    Args:
        vectorstore: Vector store dependency
        settings: Settings dependency
        
    Returns:
        HealthResponse with system status
    """
    correlation_id = get_correlation_id()
    
    handler.logger.debug(
        "health_check_requested",
        correlation_id=correlation_id,
    )
    
    # Check vectorstore health
    try:
        vectorstore_healthy = await vectorstore.health_check()
    except Exception:
        vectorstore_healthy = False
    
    # Determine overall status
    services = {
        "vectorstore": vectorstore_healthy,
        "llm": True,  # Assume LLM is available (could add actual check)
        "embeddings": True,  # Assume embeddings are available
    }
    
    all_healthy = all(services.values())
    status_str = "healthy" if all_healthy else "degraded"
    
    return HealthResponse(
        status=status_str,
        services=services,
        version="1.0.0",
    )


@router.get(
    "/documents",
    response_model=DocumentListResponse,
    status_code=status.HTTP_200_OK,
    summary="List ingested documents",
    description="List all documents that have been ingested into the knowledge base with metadata.",
    responses={
        200: {"description": "List of ingested documents"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def list_documents(
    vectorstore: VectorStoreDep,
) -> DocumentListResponse:
    """
    List all ingested documents.
    
    Returns a list of unique documents grouped by source file,
    including metadata like chunk count, file type, and content preview.
    
    Args:
        vectorstore: Vector store dependency
        
    Returns:
        DocumentListResponse with list of documents
        
    Raises:
        HTTPException: If listing fails
    """
    correlation_id = get_correlation_id()
    
    handler.logger.info(
        "list_documents_requested",
        correlation_id=correlation_id,
    )
    
    try:
        # Get documents from vectorstore
        doc_list = vectorstore.list_documents()
        
        # Convert to response format
        documents = [
            DocumentInfo(
                source=doc["source"],
                file_name=doc["file_name"],
                file_type=doc["file_type"],
                chunk_count=doc["chunk_count"],
                sample_content=doc.get("sample_content"),
                pages=doc.get("pages"),
                page_count=doc.get("page_count"),
            )
            for doc in doc_list
        ]
        
        # Calculate total chunks
        total_chunks = sum(doc.chunk_count for doc in documents)
        
        handler.logger.info(
            "list_documents_completed",
            document_count=len(documents),
            total_chunks=total_chunks,
            correlation_id=correlation_id,
        )
        
        return DocumentListResponse(
            documents=documents,
            count=len(documents),
            total_chunks=total_chunks,
        )
        
    except Exception as e:
        handler.logger.error(
            "list_documents_failed",
            error=str(e),
            correlation_id=correlation_id,
            exc_info=True,
        )
        raise RetrievalError(
            message="Failed to list documents",
            details={},
            cause=e,
        )


@router.delete(
    "/session/{session_id}",
    response_model=SessionDeleteResponse,
    status_code=status.HTTP_200_OK,
    summary="Clear session",
    description="Clear a conversation session and its history.",
    responses={
        200: {"description": "Session cleared successfully"},
        404: {"model": ErrorResponse, "description": "Session not found"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def clear_session(
    session_id: str,
    memory: MemoryDep,
) -> SessionDeleteResponse:
    """
    Clear a conversation session.
    
    Args:
        session_id: Session ID to clear
        memory: Conversation memory dependency
        
    Returns:
        SessionDeleteResponse confirming deletion
        
    Raises:
        HTTPException: If session not found or deletion fails
    """
    correlation_id = get_correlation_id()
    
    handler.logger.info(
        "session_delete_requested",
        session_id=session_id,
        correlation_id=correlation_id,
    )
    
    try:
        # Check if session exists
        if session_id not in memory.list_sessions():
            handler.logger.warning(
                "session_not_found",
                session_id=session_id,
                correlation_id=correlation_id,
            )
            raise SessionNotFoundError(
                message=f"Session not found: {session_id}",
                details={"session_id": session_id},
            )
        
        # Clear session
        memory.clear_session(session_id)
        
        handler.logger.info(
            "session_deleted",
            session_id=session_id,
            correlation_id=correlation_id,
        )
        
        return SessionDeleteResponse(
            message="Session cleared successfully",
            session_id=session_id,
        )
        
    except SessionNotFoundError:
        # Re-raise to let global handler convert
        raise
    except ChatMemoryError as e:
        # Memory operation failed
        raise
    except Exception as e:
        # Catch-all for unexpected errors
        raise ChatMemoryError(
            message=f"Failed to clear session: {session_id}",
            details={"session_id": session_id},
            cause=e,
        )


@router.delete(
    "/collection",
    response_model=CollectionClearResponse,
    status_code=status.HTTP_200_OK,
    summary="Clear collection",
    description="Delete all documents from the vector store collection.",
    responses={
        200: {"description": "Collection cleared successfully"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def clear_collection(
    vectorstore: VectorStoreDep,
) -> CollectionClearResponse:
    """
    Clear all documents from the vector store collection.
    
    Warning: This operation is irreversible!
    
    Args:
        vectorstore: Vector store dependency
        
    Returns:
        CollectionClearResponse with deletion count
        
    Raises:
        HTTPException: If deletion fails
    """
    correlation_id = get_correlation_id()
    
    handler.logger.info(
        "collection_clear_requested",
        correlation_id=correlation_id,
    )
    
    try:
        # Get current count
        count = vectorstore.get_collection_count()
        
        # Delete collection
        await vectorstore.delete_collection()
        
        handler.logger.info(
            "collection_cleared",
            documents_deleted=count,
            correlation_id=correlation_id,
        )
        
        return CollectionClearResponse(
            message="Collection cleared successfully",
            documents_deleted=count,
        )
        
    except Exception as e:
        handler.logger.error(
            "collection_clear_failed",
            error=str(e),
            correlation_id=correlation_id,
            exc_info=True,
        )
        raise DocumentIngestionError(
            message="Failed to clear collection",
            details={},
            cause=e,
        )


__all__ = ["router"]

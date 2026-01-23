"""
MCP server entry point.

This module provides the main entry point for running the MCP server
as a standalone application via `python -m src.mcp`.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add project root to path to ensure imports work
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.chat.chain import RAGChatChain
from src.config.settings import get_settings
from src.core.embeddings import get_embeddings
from src.core.llm import get_llm
from src.core.vectorstore import VectorStoreManager
from src.ingestion.chunker import TextChunker
from src.ingestion.loader import DocumentLoaderFactory
from src.ingestion.pipeline import IngestionPipeline
from src.mcp.server import RAGMCPServer
from src.retrieval.retriever import RAGRetriever
from src.utils.logging import setup_logging


async def main() -> None:
    """
    Main entry point for MCP server.
    
    Initializes all required components and starts the MCP server
    with the configured transport (stdio or SSE).
    """
    # Setup logging
    setup_logging()
    
    # Load settings
    settings = get_settings()
    
    print(f"Starting MCP Server: {settings.mcp.server_name}", file=sys.stderr)
    print(f"Transport: {settings.mcp.transport}", file=sys.stderr)
    print(f"LLM Provider: {settings.llm.llm_provider}", file=sys.stderr)
    print(f"LLM Model: {settings.llm.llm_model}", file=sys.stderr)
    
    # Initialize components
    try:
        # Embeddings
        embeddings = get_embeddings(settings)
        
        # Vector store
        vectorstore = VectorStoreManager(
            settings=settings.chroma,
            embeddings=embeddings,
        )
        
        # Retriever
        retriever = RAGRetriever(
            vectorstore=vectorstore,
            k=settings.retrieval.retrieval_k,
            score_threshold=0.7,
            mmr_diversity=settings.retrieval.mmr_diversity,
        )
        
        # LLM
        llm = get_llm(settings)
        
        # Ingestion pipeline
        loader_factory = DocumentLoaderFactory()
        chunker = TextChunker(
            chunk_size=settings.chunking.chunk_size,
            chunk_overlap=settings.chunking.chunk_overlap,
        )
        ingestion_pipeline = IngestionPipeline(
            loader_factory=loader_factory,
            chunker=chunker,
            vectorstore=vectorstore,
        )
        
        # Chat chain
        from src.chat.memory import ConversationMemory
        memory = ConversationMemory(
            window_size=5,
            session_timeout=60,
        )
        
        chat_chain = RAGChatChain(
            llm=llm,
            retriever=retriever,
            memory=memory,
            use_mmr=settings.retrieval.use_mmr,
        )
        
        # Create and run MCP server
        mcp_server = RAGMCPServer(
            chat_chain=chat_chain,
            retriever=retriever,
            ingestion_pipeline=ingestion_pipeline,
            vectorstore=vectorstore,
            server_name=settings.mcp.server_name,
        )
        
        print(f"MCP Server initialized successfully", file=sys.stderr)
        print(f"Ready to accept connections...", file=sys.stderr)
        
        # Run with configured transport
        if settings.mcp.transport == "stdio":
            await mcp_server.run_stdio()
        elif settings.mcp.transport == "sse":
            await mcp_server.run_sse(
                host="0.0.0.0",
                port=settings.mcp.server_port,
            )
        else:
            raise ValueError(f"Unsupported transport: {settings.mcp.transport}")
            
    except Exception as e:
        print(f"Error starting MCP server: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())

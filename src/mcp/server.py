"""
MCP server implementation for the RAG chatbot.

This module implements the Model Context Protocol server that exposes
RAG chatbot capabilities to AI assistants like Claude Desktop.
"""

from typing import Any, Dict, List, Optional

from mcp import types
from mcp.server import Server
from mcp.server.sse import SseServerTransport
from mcp.server.stdio import stdio_server

from src.chat.chain import RAGChatChain
from src.core.vectorstore import VectorStoreManager
from src.ingestion.pipeline import IngestionPipeline
from src.mcp.resources import MCPResourceHandler
from src.mcp.tools import TOOLS, MCPToolHandler
from src.retrieval.retriever import RAGRetriever
from src.utils.logging import LoggerMixin


class RAGMCPServer(LoggerMixin):
    """
    MCP server for the RAG chatbot.
    
    Exposes tools and resources through the Model Context Protocol,
    allowing AI assistants to interact with the RAG system.
    
    Attributes:
        server: MCP Server instance
        tool_handler: Handler for tool invocations
        resource_handler: Handler for resource requests
        chat_chain: RAG chat chain
        retriever: Document retriever
        ingestion_pipeline: Document ingestion pipeline
        vectorstore: Vector store manager
    """
    
    def __init__(
        self,
        chat_chain: RAGChatChain,
        retriever: RAGRetriever,
        ingestion_pipeline: IngestionPipeline,
        vectorstore: VectorStoreManager,
        server_name: str = "rag-chatbot",
        server_version: str = "1.0.0",
    ):
        """
        Initialize RAG MCP server.
        
        Args:
            chat_chain: Initialized RAG chat chain
            retriever: Document retriever
            ingestion_pipeline: Document ingestion pipeline
            vectorstore: Vector store manager
            server_name: MCP server name
            server_version: MCP server version
        """
        super().__init__()
        
        # Store components
        self.chat_chain = chat_chain
        self.retriever = retriever
        self.ingestion_pipeline = ingestion_pipeline
        self.vectorstore = vectorstore
        
        # Initialize handlers
        self.tool_handler = MCPToolHandler(
            chat_chain=chat_chain,
            retriever=retriever,
            ingestion_pipeline=ingestion_pipeline,
        )
        
        self.resource_handler = MCPResourceHandler(
            vectorstore=vectorstore,
        )
        
        # Create MCP server
        self.server = Server(server_name)
        
        # Register handlers
        self._register_handlers()
        
        self.logger.info(
            "initialized_rag_mcp_server",
            server_name=server_name,
            server_version=server_version,
            tool_count=len(TOOLS),
        )
    
    def _register_handlers(self) -> None:
        """Register MCP protocol handlers."""
        
        # Register tool list handler
        @self.server.list_tools()
        async def list_tools() -> List[types.Tool]:
            """List available tools."""
            self.logger.debug("mcp_list_tools_called")
            return self.tool_handler.get_available_tools()
        
        # Register tool call handler
        @self.server.call_tool()
        async def call_tool(
            name: str,
            arguments: Dict[str, Any]
        ) -> List[types.TextContent]:
            """Handle tool invocation."""
            self.logger.info(
                "mcp_call_tool",
                tool_name=name,
                arguments_keys=list(arguments.keys()),
            )
            
            try:
                # Route to appropriate handler
                if name == "rag_chat":
                    result = await self.tool_handler.handle_rag_chat(
                        question=arguments["question"],
                        session_id=arguments.get("session_id"),
                    )
                    
                elif name == "document_search":
                    result = await self.tool_handler.handle_document_search(
                        query=arguments["query"],
                        limit=arguments.get("limit", 4),
                        use_mmr=arguments.get("use_mmr", False),
                    )
                    
                elif name == "ingest_document":
                    result = await self.tool_handler.handle_ingest_document(
                        file_path=arguments["file_path"],
                    )
                    
                elif name == "list_documents":
                    result = await self.tool_handler.handle_list_documents()
                    
                else:
                    raise ValueError(f"Unknown tool: {name}")
                
                # Convert result to TextContent
                import json
                return [
                    types.TextContent(
                        type="text",
                        text=json.dumps(result, indent=2),
                    )
                ]
                
            except Exception as e:
                self.logger.error(
                    "mcp_tool_call_failed",
                    tool_name=name,
                    error=str(e),
                    exc_info=True,
                )
                # Return error as text content
                import json
                error_response = {
                    "error": str(e),
                    "tool": name,
                    "success": False,
                }
                return [
                    types.TextContent(
                        type="text",
                        text=json.dumps(error_response, indent=2),
                    )
                ]
        
        # Register resource list handler
        @self.server.list_resources()
        async def list_resources() -> List[types.Resource]:
            """List available resources."""
            self.logger.debug("mcp_list_resources_called")
            return await self.resource_handler.list_resources()
        
        # Register resource read handler
        @self.server.read_resource()
        async def read_resource(uri: str) -> str:
            """Read resource contents."""
            self.logger.info("mcp_read_resource", uri=uri)
            
            try:
                contents = await self.resource_handler.get_resource(uri)
                return contents.text
                
            except Exception as e:
                self.logger.error(
                    "mcp_read_resource_failed",
                    uri=uri,
                    error=str(e),
                    exc_info=True,
                )
                raise
        
        # Register resource templates handler
        @self.server.list_resource_templates()
        async def list_resource_templates() -> List[types.ResourceTemplate]:
            """List resource templates."""
            self.logger.debug("mcp_list_resource_templates_called")
            return await self.resource_handler.list_resource_templates()
        
        self.logger.info("mcp_handlers_registered")
    
    async def run_stdio(self) -> None:
        """
        Run MCP server with stdio transport.
        
        This is the primary transport method for integration with
        Claude Desktop and other AI assistants.
        
        The server will read requests from stdin and write responses
        to stdout, allowing it to be used as a subprocess.
        """
        self.logger.info("starting_mcp_server_stdio")
        
        try:
            async with stdio_server() as (read_stream, write_stream):
                self.logger.info("mcp_server_running_stdio")
                
                await self.server.run(
                    read_stream,
                    write_stream,
                    self.server.create_initialization_options(),
                )
                
        except Exception as e:
            self.logger.error(
                "mcp_server_stdio_failed",
                error=str(e),
                exc_info=True,
            )
            raise
        finally:
            self.logger.info("mcp_server_stopped")
    
    async def run_sse(
        self,
        host: str = "0.0.0.0",
        port: int = 8000,
    ) -> None:
        """
        Run MCP server with SSE (Server-Sent Events) transport.
        
        This transport is useful for web-based integrations and
        allows the server to push updates to connected clients.
        
        Args:
            host: Host to bind to (default: 0.0.0.0)
            port: Port to listen on (default: 8000)
        """
        from starlette.applications import Starlette
        from starlette.routing import Mount, Route
        from starlette.responses import Response
        import uvicorn
        
        self.logger.info(
            "starting_mcp_server_sse",
            host=host,
            port=port,
        )
        
        # Create SSE transport
        sse = SseServerTransport("/messages/")
        
        async def handle_sse(request):
            async with sse.connect_sse(
                request.scope, request.receive, request._send
            ) as streams:
                await self.server.run(
                    streams[0], 
                    streams[1], 
                    self.server.create_initialization_options()
                )
            return Response()
        
        # Create Starlette routes
        routes = [
            Route("/sse", endpoint=handle_sse, methods=["GET"]),
            Mount("/messages/", app=sse.handle_post_message),
        ]
        
        # Create Starlette app
        starlette_app = Starlette(routes=routes)
        
        self.logger.info("mcp_server_running_sse", host=host, port=port)
        
        # Run with uvicorn
        config = uvicorn.Config(
            starlette_app, 
            host=host, 
            port=port, 
            log_level="info"
        )
        server = uvicorn.Server(config)
        await server.serve()
    
    def get_server_info(self) -> Dict[str, Any]:
        """
        Get server information and status.
        
        Returns:
            Dictionary with server metadata
        """
        return {
            "name": self.server.name,
            "version": "1.0.0",
            "protocol_version": "2024-11-05",
            "tools": [tool.name for tool in self.tool_handler.get_available_tools()],
            "capabilities": {
                "tools": True,
                "resources": True,
                "prompts": False,
                "logging": True,
            },
        }


async def create_mcp_server(
    chat_chain: RAGChatChain,
    retriever: RAGRetriever,
    ingestion_pipeline: IngestionPipeline,
    vectorstore: VectorStoreManager,
) -> RAGMCPServer:
    """
    Factory function to create and initialize MCP server.
    
    Args:
        chat_chain: Initialized RAG chat chain
        retriever: Document retriever
        ingestion_pipeline: Document ingestion pipeline
        vectorstore: Vector store manager
        
    Returns:
        Initialized RAGMCPServer instance
    """
    server = RAGMCPServer(
        chat_chain=chat_chain,
        retriever=retriever,
        ingestion_pipeline=ingestion_pipeline,
        vectorstore=vectorstore,
    )
    
    return server


__all__ = [
    "RAGMCPServer",
    "create_mcp_server",
]

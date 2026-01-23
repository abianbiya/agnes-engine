"""
MCP resource definitions and handlers for the RAG chatbot.

This module exposes document collections and vectorstore contents
as MCP resources that can be accessed by AI assistants.
"""

from typing import Any, Dict, List, Optional

from mcp.types import Resource, ResourceContents, ResourceTemplate

from src.core.vectorstore import VectorStoreManager
from src.utils.logging import LoggerMixin


# Resource URI schemes
DOCUMENTS_URI_SCHEME = "documents://"
COLLECTION_URI_SCHEME = "collection://"


class MCPResourceHandler(LoggerMixin):
    """
    Handler for MCP resource requests.
    
    Exposes document collections and vectorstore contents as resources
    that can be read by AI assistants through the MCP protocol.
    
    Attributes:
        vectorstore: Vector store manager for accessing documents
    """
    
    def __init__(
        self,
        vectorstore: VectorStoreManager,
    ):
        """
        Initialize MCP resource handler.
        
        Args:
            vectorstore: Initialized vector store manager
        """
        super().__init__()
        
        self.vectorstore = vectorstore
        
        self.logger.info("initialized_mcp_resource_handler")
    
    async def list_resources(self) -> List[Resource]:
        """
        List available document resources.
        
        Returns resources representing document collections and
        individual documents in the vectorstore.
        
        Returns:
            List of Resource objects
            
        Raises:
            Exception: If listing fails
        """
        self.logger.info("listing_resources")
        
        try:
            resources = []
            
            # Add main documents collection resource
            resources.append(
                Resource(
                    uri=f"{COLLECTION_URI_SCHEME}all",
                    name="All Documents",
                    description="Complete collection of all ingested documents",
                    mimeType="application/json",
                )
            )
            
            # In a production system, you'd enumerate individual documents
            # from the vectorstore metadata and create resources for each
            
            self.logger.info(
                "resources_listed",
                resource_count=len(resources),
            )
            
            return resources
            
        except Exception as e:
            self.logger.error(
                "list_resources_failed",
                error=str(e),
                exc_info=True,
            )
            raise
    
    async def get_resource(self, uri: str) -> ResourceContents:
        """
        Get resource contents by URI.
        
        Args:
            uri: Resource URI (e.g., "documents://filename" or "collection://all")
            
        Returns:
            ResourceContents with the requested data
            
        Raises:
            ValueError: If URI format is invalid
            Exception: If retrieval fails
        """
        self.logger.info("getting_resource", uri=uri)
        
        try:
            # Parse URI
            if uri.startswith(COLLECTION_URI_SCHEME):
                # Collection resource
                collection_id = uri[len(COLLECTION_URI_SCHEME):]
                return await self._get_collection_resource(collection_id)
                
            elif uri.startswith(DOCUMENTS_URI_SCHEME):
                # Individual document resource
                doc_id = uri[len(DOCUMENTS_URI_SCHEME):]
                return await self._get_document_resource(doc_id)
                
            else:
                raise ValueError(f"Invalid resource URI scheme: {uri}")
                
        except Exception as e:
            self.logger.error(
                "get_resource_failed",
                uri=uri,
                error=str(e),
                exc_info=True,
            )
            raise
    
    async def _get_collection_resource(
        self,
        collection_id: str
    ) -> ResourceContents:
        """
        Get collection resource contents.
        
        Args:
            collection_id: Collection identifier
            
        Returns:
            ResourceContents with collection data
        """
        self.logger.info(
            "getting_collection_resource",
            collection_id=collection_id,
        )
        
        # For "all" collection, return summary of all documents
        if collection_id == "all":
            # In production, query vectorstore for actual document metadata
            contents = {
                "collection": "all",
                "description": "All ingested documents in the knowledge base",
                "document_count": 0,  # Would query vectorstore
                "documents": [],  # Would list all document metadata
            }
            
            return ResourceContents(
                uri=f"{COLLECTION_URI_SCHEME}{collection_id}",
                mimeType="application/json",
                text=str(contents),
            )
        
        raise ValueError(f"Unknown collection: {collection_id}")
    
    async def _get_document_resource(
        self,
        doc_id: str
    ) -> ResourceContents:
        """
        Get individual document resource contents.
        
        Args:
            doc_id: Document identifier
            
        Returns:
            ResourceContents with document data
        """
        self.logger.info(
            "getting_document_resource",
            doc_id=doc_id,
        )
        
        # In production, query vectorstore for document chunks
        # and assemble the complete document content
        
        raise NotImplementedError(
            "Individual document resources not yet implemented"
        )
    
    async def list_resource_templates(self) -> List[ResourceTemplate]:
        """
        List available resource templates for dynamic resources.
        
        Resource templates define URI patterns for resources that
        can be dynamically generated based on parameters.
        
        Returns:
            List of ResourceTemplate objects
        """
        self.logger.info("listing_resource_templates")
        
        templates = [
            ResourceTemplate(
                uriTemplate=f"{DOCUMENTS_URI_SCHEME}{{document_id}}",
                name="Document by ID",
                description="Access individual documents by their unique identifier",
                mimeType="text/plain",
            ),
            ResourceTemplate(
                uriTemplate=f"{COLLECTION_URI_SCHEME}{{collection_name}}",
                name="Document Collection",
                description="Access document collections (e.g., 'all', 'recent', 'by-type')",
                mimeType="application/json",
            ),
        ]
        
        self.logger.info(
            "resource_templates_listed",
            template_count=len(templates),
        )
        
        return templates
    
    async def get_resource_metadata(self, uri: str) -> Dict[str, Any]:
        """
        Get metadata for a resource without fetching full contents.
        
        Args:
            uri: Resource URI
            
        Returns:
            Dictionary with resource metadata
            
        Raises:
            ValueError: If URI is invalid
            Exception: If metadata retrieval fails
        """
        self.logger.info("getting_resource_metadata", uri=uri)
        
        try:
            if uri.startswith(COLLECTION_URI_SCHEME):
                collection_id = uri[len(COLLECTION_URI_SCHEME):]
                return {
                    "type": "collection",
                    "id": collection_id,
                    "uri": uri,
                    "mime_type": "application/json",
                }
            
            elif uri.startswith(DOCUMENTS_URI_SCHEME):
                doc_id = uri[len(DOCUMENTS_URI_SCHEME):]
                return {
                    "type": "document",
                    "id": doc_id,
                    "uri": uri,
                    "mime_type": "text/plain",
                }
            
            else:
                raise ValueError(f"Invalid resource URI: {uri}")
                
        except Exception as e:
            self.logger.error(
                "get_resource_metadata_failed",
                uri=uri,
                error=str(e),
                exc_info=True,
            )
            raise


__all__ = [
    "MCPResourceHandler",
    "DOCUMENTS_URI_SCHEME",
    "COLLECTION_URI_SCHEME",
]

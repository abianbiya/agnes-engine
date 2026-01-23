"""
Unit tests for MCP resources module.

Tests MCP resource definitions, URI handling, and MCPResourceHandler with mocked vectorstore.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest
from mcp.types import Resource, ResourceContents, ResourceTemplate

from src.mcp.resources import (
    COLLECTION_URI_SCHEME,
    DOCUMENTS_URI_SCHEME,
    MCPResourceHandler,
)


@pytest.fixture
def mock_vectorstore():
    """Create a mock vectorstore for testing."""
    vectorstore = MagicMock()
    # Add any necessary mock methods here
    return vectorstore


@pytest.fixture
def resource_handler(mock_vectorstore):
    """Create MCPResourceHandler with mocked vectorstore."""
    return MCPResourceHandler(vectorstore=mock_vectorstore)


class TestURISchemes:
    """Test URI scheme constants."""
    
    def test_documents_uri_scheme_defined(self):
        """Documents URI scheme should be defined."""
        assert isinstance(DOCUMENTS_URI_SCHEME, str)
        assert DOCUMENTS_URI_SCHEME == "documents://"
    
    def test_collection_uri_scheme_defined(self):
        """Collection URI scheme should be defined."""
        assert isinstance(COLLECTION_URI_SCHEME, str)
        assert COLLECTION_URI_SCHEME == "collection://"


class TestMCPResourceHandlerInitialization:
    """Test MCPResourceHandler initialization."""
    
    def test_initialization_stores_vectorstore(self, mock_vectorstore):
        """Should store the provided vectorstore."""
        handler = MCPResourceHandler(vectorstore=mock_vectorstore)
        
        assert handler.vectorstore is mock_vectorstore
    
    def test_initialization_succeeds(self, mock_vectorstore):
        """Should successfully initialize handler."""
        handler = MCPResourceHandler(vectorstore=mock_vectorstore)
        
        assert handler is not None
        assert hasattr(handler, "vectorstore")
        assert hasattr(handler, "logger")


class TestListResources:
    """Test list_resources method."""
    
    @pytest.mark.asyncio
    async def test_list_resources_returns_list(self, resource_handler):
        """Should return a list of resources."""
        resources = await resource_handler.list_resources()
        
        assert isinstance(resources, list)
    
    @pytest.mark.asyncio
    async def test_list_resources_includes_all_collection(self, resource_handler):
        """Should include 'all' collection resource."""
        resources = await resource_handler.list_resources()
        
        # Should have at least one resource
        assert len(resources) > 0
        
        # Check for 'all' collection (uri might be AnyUrl type)
        collection_uris = [str(r.uri) for r in resources]
        assert f"{COLLECTION_URI_SCHEME}all" in collection_uris
    
    @pytest.mark.asyncio
    async def test_list_resources_all_are_resource_type(self, resource_handler):
        """Should return Resource instances."""
        resources = await resource_handler.list_resources()
        
        for resource in resources:
            assert isinstance(resource, Resource)
    
    @pytest.mark.asyncio
    async def test_list_resources_have_required_fields(self, resource_handler):
        """Each resource should have required fields."""
        resources = await resource_handler.list_resources()
        
        for resource in resources:
            assert hasattr(resource, "uri")
            assert hasattr(resource, "name")
            assert hasattr(resource, "description")
            assert hasattr(resource, "mimeType")
            
            # Check types (uri might be AnyUrl type from MCP)
            assert resource.uri is not None
            assert isinstance(resource.name, str)
            assert isinstance(resource.description, str)
            assert isinstance(resource.mimeType, str)
    
    @pytest.mark.asyncio
    async def test_list_resources_all_collection_has_correct_properties(
        self,
        resource_handler,
    ):
        """'All' collection resource should have correct properties."""
        resources = await resource_handler.list_resources()
        
        # Find the 'all' collection resource (using str() to handle AnyUrl)
        all_collection = None
        for r in resources:
            if str(r.uri) == f"{COLLECTION_URI_SCHEME}all":
                all_collection = r
                break
        
        assert all_collection is not None
        assert all_collection.name == "All Documents"
        assert "collection" in all_collection.description.lower()
        assert all_collection.mimeType == "application/json"


class TestGetResource:
    """Test get_resource method."""
    
    @pytest.mark.asyncio
    async def test_get_resource_with_collection_uri(self, resource_handler):
        """Should get resource with collection URI."""
        uri = f"{COLLECTION_URI_SCHEME}all"
        
        contents = await resource_handler.get_resource(uri)
        
        assert isinstance(contents, ResourceContents)
        # Compare as string since MCP might use AnyUrl type
        assert str(contents.uri) == uri
        assert contents.mimeType == "application/json"
        assert isinstance(contents.text, str)
    
    @pytest.mark.asyncio
    async def test_get_resource_collection_all_contains_expected_fields(
        self,
        resource_handler,
    ):
        """Collection 'all' resource should contain expected fields."""
        uri = f"{COLLECTION_URI_SCHEME}all"
        
        contents = await resource_handler.get_resource(uri)
        
        # The text should be a string representation of a dict
        text = contents.text
        assert "collection" in text
        assert "all" in text
    
    @pytest.mark.asyncio
    async def test_get_resource_with_invalid_uri_scheme(self, resource_handler):
        """Should raise ValueError for invalid URI scheme."""
        invalid_uri = "invalid://test"
        
        with pytest.raises(ValueError, match="Invalid resource URI scheme"):
            await resource_handler.get_resource(invalid_uri)
    
    @pytest.mark.asyncio
    async def test_get_resource_with_unknown_collection(self, resource_handler):
        """Should raise ValueError for unknown collection."""
        unknown_uri = f"{COLLECTION_URI_SCHEME}unknown_collection"
        
        with pytest.raises(ValueError, match="Unknown collection"):
            await resource_handler.get_resource(unknown_uri)
    
    @pytest.mark.asyncio
    async def test_get_resource_with_document_uri_not_implemented(
        self,
        resource_handler,
    ):
        """Should raise NotImplementedError for document URIs."""
        doc_uri = f"{DOCUMENTS_URI_SCHEME}some_doc_id"
        
        with pytest.raises(NotImplementedError):
            await resource_handler.get_resource(doc_uri)


class TestGetCollectionResource:
    """Test _get_collection_resource method."""
    
    @pytest.mark.asyncio
    async def test_get_collection_resource_all(self, resource_handler):
        """Should get 'all' collection resource."""
        contents = await resource_handler._get_collection_resource("all")
        
        assert isinstance(contents, ResourceContents)
        # Compare as string since MCP might use AnyUrl type
        assert str(contents.uri) == f"{COLLECTION_URI_SCHEME}all"
        assert contents.mimeType == "application/json"
    
    @pytest.mark.asyncio
    async def test_get_collection_resource_unknown_raises_error(
        self,
        resource_handler,
    ):
        """Should raise ValueError for unknown collection."""
        with pytest.raises(ValueError, match="Unknown collection"):
            await resource_handler._get_collection_resource("nonexistent")


class TestGetDocumentResource:
    """Test _get_document_resource method."""
    
    @pytest.mark.asyncio
    async def test_get_document_resource_not_implemented(self, resource_handler):
        """Should raise NotImplementedError."""
        with pytest.raises(NotImplementedError):
            await resource_handler._get_document_resource("test_doc_id")


class TestListResourceTemplates:
    """Test list_resource_templates method."""
    
    @pytest.mark.asyncio
    async def test_list_resource_templates_returns_list(self, resource_handler):
        """Should return a list of resource templates."""
        templates = await resource_handler.list_resource_templates()
        
        assert isinstance(templates, list)
        assert len(templates) > 0
    
    @pytest.mark.asyncio
    async def test_list_resource_templates_all_are_template_type(
        self,
        resource_handler,
    ):
        """Should return ResourceTemplate instances."""
        templates = await resource_handler.list_resource_templates()
        
        for template in templates:
            assert isinstance(template, ResourceTemplate)
    
    @pytest.mark.asyncio
    async def test_list_resource_templates_have_required_fields(
        self,
        resource_handler,
    ):
        """Each template should have required fields."""
        templates = await resource_handler.list_resource_templates()
        
        for template in templates:
            assert hasattr(template, "uriTemplate")
            assert hasattr(template, "name")
            assert hasattr(template, "description")
            assert hasattr(template, "mimeType")
            
            # Check types
            assert isinstance(template.uriTemplate, str)
            assert isinstance(template.name, str)
            assert isinstance(template.description, str)
            assert isinstance(template.mimeType, str)
    
    @pytest.mark.asyncio
    async def test_list_resource_templates_includes_document_template(
        self,
        resource_handler,
    ):
        """Should include document by ID template."""
        templates = await resource_handler.list_resource_templates()
        
        # Find document template
        doc_template = next(
            (t for t in templates if DOCUMENTS_URI_SCHEME in t.uriTemplate),
            None,
        )
        
        assert doc_template is not None
        assert "{document_id}" in doc_template.uriTemplate
        assert doc_template.mimeType == "text/plain"
    
    @pytest.mark.asyncio
    async def test_list_resource_templates_includes_collection_template(
        self,
        resource_handler,
    ):
        """Should include collection template."""
        templates = await resource_handler.list_resource_templates()
        
        # Find collection template
        coll_template = next(
            (t for t in templates if COLLECTION_URI_SCHEME in t.uriTemplate),
            None,
        )
        
        assert coll_template is not None
        assert "{collection_name}" in coll_template.uriTemplate
        assert coll_template.mimeType == "application/json"


class TestGetResourceMetadata:
    """Test get_resource_metadata method."""
    
    @pytest.mark.asyncio
    async def test_get_metadata_for_collection_uri(self, resource_handler):
        """Should get metadata for collection URI."""
        uri = f"{COLLECTION_URI_SCHEME}all"
        
        metadata = await resource_handler.get_resource_metadata(uri)
        
        assert isinstance(metadata, dict)
        assert metadata["type"] == "collection"
        assert metadata["id"] == "all"
        assert metadata["uri"] == uri
        assert metadata["mime_type"] == "application/json"
    
    @pytest.mark.asyncio
    async def test_get_metadata_for_document_uri(self, resource_handler):
        """Should get metadata for document URI."""
        uri = f"{DOCUMENTS_URI_SCHEME}doc123"
        
        metadata = await resource_handler.get_resource_metadata(uri)
        
        assert isinstance(metadata, dict)
        assert metadata["type"] == "document"
        assert metadata["id"] == "doc123"
        assert metadata["uri"] == uri
        assert metadata["mime_type"] == "text/plain"
    
    @pytest.mark.asyncio
    async def test_get_metadata_with_invalid_uri(self, resource_handler):
        """Should raise ValueError for invalid URI."""
        invalid_uri = "invalid://test"
        
        with pytest.raises(ValueError, match="Invalid resource URI"):
            await resource_handler.get_resource_metadata(invalid_uri)
    
    @pytest.mark.asyncio
    async def test_get_metadata_returns_required_fields(self, resource_handler):
        """Should return all required metadata fields."""
        uri = f"{COLLECTION_URI_SCHEME}all"
        
        metadata = await resource_handler.get_resource_metadata(uri)
        
        # Check required fields
        assert "type" in metadata
        assert "id" in metadata
        assert "uri" in metadata
        assert "mime_type" in metadata

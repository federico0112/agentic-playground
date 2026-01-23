"""Unit tests for semantic_search_agent.py."""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document


class TestSemanticSearchTool:
    """Tests for the semantic_search tool."""

    @patch("semantic_search_agent.vector_store")
    def test_semantic_search_returns_results(self, mock_vector_store):
        """Test that semantic search returns results."""
        from semantic_search_agent import semantic_search

        mock_results = [
            (Document(page_content="Dragons are mythical creatures.", metadata={"source": "book.pdf", "page": 1}), 0.95),
            (Document(page_content="Magic is powerful.", metadata={"source": "book.pdf", "page": 2}), 0.85),
        ]
        mock_vector_store.similarity_search_with_score.return_value = mock_results

        result = semantic_search.invoke({"query": "What are dragons?", "top_k": 5})

        assert len(result) == 2
        mock_vector_store.similarity_search_with_score.assert_called_once_with(query="What are dragons?", k=5)

    @patch("semantic_search_agent.vector_store")
    def test_semantic_search_uses_default_top_k(self, mock_vector_store):
        """Test that semantic search uses default top_k value."""
        from semantic_search_agent import semantic_search, TOP_K

        mock_vector_store.similarity_search_with_score.return_value = []

        semantic_search.invoke({"query": "test query"})

        mock_vector_store.similarity_search_with_score.assert_called_once_with(query="test query", k=TOP_K)

    @patch("semantic_search_agent.vector_store")
    def test_semantic_search_handles_empty_results(self, mock_vector_store):
        """Test that semantic search handles empty results gracefully."""
        from semantic_search_agent import semantic_search

        mock_vector_store.similarity_search_with_score.return_value = []

        result = semantic_search.invoke({"query": "nonexistent topic"})

        assert result == []

    @patch("semantic_search_agent.vector_store")
    def test_semantic_search_handles_exception(self, mock_vector_store):
        """Test that semantic search handles exceptions gracefully."""
        from semantic_search_agent import semantic_search

        mock_vector_store.similarity_search_with_score.side_effect = Exception("Connection error")

        result = semantic_search.invoke({"query": "test"})

        assert result == []


class TestGetSourceFilenamesTool:
    """Tests for the get_source_filenames tool."""

    @patch("semantic_search_agent.MongoClient")
    def test_get_source_filenames_returns_sources(self, mock_client):
        """Test that get_source_filenames returns list of sources."""
        from semantic_search_agent import get_source_filenames

        mock_collection = MagicMock()
        mock_collection.aggregate.return_value.to_list.return_value = [
            {"_id": "book1.pdf"},
            {"_id": "book2.pdf"},
        ]
        mock_client.return_value.__getitem__.return_value.__getitem__.return_value = mock_collection

        result = get_source_filenames.invoke({})

        assert "book1.pdf" in result
        assert "book2.pdf" in result

    @patch("semantic_search_agent.MongoClient")
    def test_get_source_filenames_handles_error(self, mock_client):
        """Test that get_source_filenames handles errors gracefully."""
        from semantic_search_agent import get_source_filenames

        mock_client.side_effect = Exception("Connection failed")

        result = get_source_filenames.invoke({})

        assert "Error" in result


class TestAgentConfiguration:
    """Tests for agent configuration."""

    def test_system_prompt_exists(self):
        """Test that system prompt is defined."""
        from semantic_search_agent import SYSTEM_PROMPT

        assert SYSTEM_PROMPT is not None
        assert len(SYSTEM_PROMPT) > 0
        assert "Semantic Search" in SYSTEM_PROMPT

    def test_system_prompt_contains_instructions(self):
        """Test that system prompt contains key instructions."""
        from semantic_search_agent import SYSTEM_PROMPT

        assert "semantic_search" in SYSTEM_PROMPT
        assert "cite" in SYSTEM_PROMPT.lower()
        assert "source" in SYSTEM_PROMPT.lower()

    def test_agent_has_tools(self):
        """Test that agent is configured with tools."""
        from semantic_search_agent import agent

        # Agent should be a compiled graph
        assert agent is not None


class TestMongoDBConfiguration:
    """Tests for MongoDB configuration."""

    def test_mongodb_uri_default(self):
        """Test default MongoDB URI."""
        from semantic_search_agent import MONGODB_URI

        assert MONGODB_URI is not None
        assert "mongodb" in MONGODB_URI.lower()

    def test_db_name_configured(self):
        """Test database name is configured."""
        from semantic_search_agent import DB_NAME

        assert DB_NAME is not None
        assert len(DB_NAME) > 0

    def test_collection_name_configured(self):
        """Test collection name is configured."""
        from semantic_search_agent import COLLECTION_NAME

        assert COLLECTION_NAME is not None
        assert len(COLLECTION_NAME) > 0

    def test_index_name_configured(self):
        """Test index name is configured."""
        from semantic_search_agent import INDEX_NAME

        assert INDEX_NAME is not None
        assert len(INDEX_NAME) > 0


class TestLLMConfiguration:
    """Tests for LLM configuration."""

    def test_llm_is_configured(self):
        """Test that LLM is configured."""
        from semantic_search_agent import llm

        assert llm is not None

    def test_embeddings_is_configured(self):
        """Test that embeddings is configured."""
        from semantic_search_agent import embeddings

        assert embeddings is not None


class TestVectorStoreInitialization:
    """Tests for vector store initialization."""

    @patch("semantic_search_agent.MongoClient")
    @patch("semantic_search_agent.MongoDBAtlasVectorSearch")
    def test_initialize_vector_store_connects(self, mock_vector_search, mock_client):
        """Test that initialize_vector_store connects to MongoDB."""
        from semantic_search_agent import initialize_vector_store

        mock_collection = MagicMock()
        mock_collection.count_documents.return_value = 100
        mock_client.return_value.__getitem__.return_value.__getitem__.return_value = mock_collection

        mock_store = MagicMock()
        mock_vector_search.return_value = mock_store

        result = initialize_vector_store()

        mock_client.assert_called()
        assert result == mock_store

    @patch("semantic_search_agent.MongoClient")
    def test_initialize_vector_store_handles_connection_error(self, mock_client):
        """Test that connection errors are handled."""
        from semantic_search_agent import initialize_vector_store

        mock_client.side_effect = Exception("Connection refused")

        with pytest.raises(SystemExit):
            initialize_vector_store()

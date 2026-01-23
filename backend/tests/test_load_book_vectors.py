"""Unit tests for load_book_vectors.py."""

from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open

import pytest
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class TestCreateEmbeddings:
    """Tests for create_embeddings function."""

    @patch("backend.vector_storage_helpers.GoogleGenerativeAIEmbeddings")
    def test_create_embeddings_returns_instance(self, mock_embeddings_class):
        """Test that create_embeddings returns an embeddings instance."""
        from backend.vector_storage_helpers import create_embeddings

        mock_instance = MagicMock()
        mock_embeddings_class.return_value = mock_instance

        result = create_embeddings()

        assert result == mock_instance
        mock_embeddings_class.assert_called_once_with(
            model="models/gemini-embedding-001"
        )

    @patch("backend.vector_storage_helpers.GoogleGenerativeAIEmbeddings")
    def test_create_embeddings_with_correct_model(self, mock_embeddings_class):
        """Test that the correct embedding model is used."""
        from backend.vector_storage_helpers import create_embeddings

        create_embeddings()

        call_kwargs = mock_embeddings_class.call_args[1]
        assert call_kwargs["model"] == "models/gemini-embedding-001"


class TestCreateMongoVectorStore:
    """Tests for create_mongo_vector_store function."""

    @patch("backend.vector_storage_helpers.MongoDBAtlasVectorSearch")
    @patch("backend.vector_storage_helpers.MongoClient")
    def test_create_mongo_vector_store_connects(self, mock_client, mock_vector_search, mock_embeddings):
        """Test that vector store connects to MongoDB."""
        from backend.vector_storage_helpers import create_mongo_vector_store

        mock_collection = MagicMock()
        mock_client.return_value.__getitem__.return_value.__getitem__.return_value = mock_collection

        result = create_mongo_vector_store(
            mongodb_uri="mongodb://localhost:27017",
            db_name="test_db",
            collection_name="test_collection",
            index_name="test_index",
            embeddings=mock_embeddings,
            create_index=False,
        )

        mock_client.assert_called_once_with("mongodb://localhost:27017")
        mock_vector_search.assert_called_once()

    @patch("backend.vector_storage_helpers.MongoDBAtlasVectorSearch")
    @patch("backend.vector_storage_helpers.MongoClient")
    def test_create_mongo_vector_store_creates_index(self, mock_client, mock_vector_search, mock_embeddings):
        """Test that vector index is created when requested."""
        from backend.vector_storage_helpers import create_mongo_vector_store

        mock_collection = MagicMock()
        mock_client.return_value.__getitem__.return_value.__getitem__.return_value = mock_collection
        mock_store_instance = MagicMock()
        mock_vector_search.return_value = mock_store_instance

        create_mongo_vector_store(
            mongodb_uri="mongodb://localhost:27017",
            db_name="test_db",
            collection_name="test_collection",
            index_name="test_index",
            embeddings=mock_embeddings,
            create_index=True,
        )

        mock_store_instance.create_vector_search_index.assert_called_once_with(dimensions=3072)

    @patch("backend.vector_storage_helpers.MongoDBAtlasVectorSearch")
    @patch("backend.vector_storage_helpers.MongoClient")
    def test_create_mongo_vector_store_skips_index(self, mock_client, mock_vector_search, mock_embeddings):
        """Test that vector index creation is skipped when not requested."""
        from backend.vector_storage_helpers import create_mongo_vector_store

        mock_collection = MagicMock()
        mock_client.return_value.__getitem__.return_value.__getitem__.return_value = mock_collection
        mock_store_instance = MagicMock()
        mock_vector_search.return_value = mock_store_instance

        create_mongo_vector_store(
            mongodb_uri="mongodb://localhost:27017",
            db_name="test_db",
            collection_name="test_collection",
            index_name="test_index",
            embeddings=mock_embeddings,
            create_index=False,
        )

        mock_store_instance.create_vector_search_index.assert_not_called()


class TestStoreDocuments:
    """Tests for store_documents function."""

    def test_store_documents_chunks_and_stores(self, sample_documents, mock_vector_store):
        """Test that documents are chunked and stored."""
        from backend.vector_storage_helpers import store_documents

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=100,
            chunk_overlap=10,
        )

        documents = {Path("test.pdf"): sample_documents}

        result = store_documents(
            documents=documents,
            vector_store=mock_vector_store,
            text_splitter=text_splitter,
        )

        assert Path("test.pdf") in result
        mock_vector_store.add_documents.assert_called()

    def test_store_documents_generates_uuids(self, sample_documents, mock_vector_store):
        """Test that UUIDs are generated for each chunk."""
        from backend.vector_storage_helpers import store_documents

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
        )

        documents = {Path("test.pdf"): sample_documents}

        result = store_documents(
            documents=documents,
            vector_store=mock_vector_store,
            text_splitter=text_splitter,
        )

        # Check that add_documents was called with ids
        call_kwargs = mock_vector_store.add_documents.call_args[1]
        assert "ids" in call_kwargs
        assert len(call_kwargs["ids"]) > 0

    def test_store_documents_empty_input(self, mock_vector_store):
        """Test behavior with empty documents dict."""
        from backend.vector_storage_helpers import store_documents

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=100,
            chunk_overlap=10,
        )

        result = store_documents(
            documents={},
            vector_store=mock_vector_store,
            text_splitter=text_splitter,
        )

        assert result == {}
        mock_vector_store.add_documents.assert_not_called()


class TestTextSplitting:
    """Tests for text splitting behavior."""

    def test_text_splitter_respects_chunk_size(self, sample_documents):
        """Test that text splitter respects chunk size configuration."""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=50,
            chunk_overlap=10,
        )

        chunks = text_splitter.split_documents(sample_documents)

        for chunk in chunks:
            # Allow some flexibility for word boundaries
            assert len(chunk.page_content) <= 100

    def test_text_splitter_preserves_metadata(self, sample_documents):
        """Test that metadata is preserved after splitting."""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=50,
            chunk_overlap=10,
        )

        chunks = text_splitter.split_documents(sample_documents)

        for chunk in chunks:
            assert "source" in chunk.metadata

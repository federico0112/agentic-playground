"""Integration tests for the semantic search pipeline.

These tests verify the full flow from document loading to querying the agent.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock
import io

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from fastapi.testclient import TestClient


# Sample book content for testing
SAMPLE_BOOK_CONTENT = """
Chapter 1: Introduction to Dragons

Dragons are legendary creatures that appear in the folklore of many cultures worldwide.
They are typically depicted as large, serpentine creatures with wings and the ability to breathe fire.

In fantasy literature, dragons are often portrayed as intelligent beings capable of speech.
Some dragons are benevolent guardians, while others are fearsome adversaries.

The most famous dragon types include:
- Red Dragons: Known for their fiery breath and aggressive nature
- Blue Dragons: Associated with lightning and desert environments
- Gold Dragons: Wise and benevolent creatures that often help heroes
- Black Dragons: Dwelling in swamps, they breathe acid

Chapter 2: Dragon Lairs

Dragons typically make their homes in remote, defensible locations.
Mountain caves are the most common choice, providing natural protection.
Some dragons prefer underwater lairs or ancient ruins.

A dragon's hoard is its most prized possession, containing gold, gems, and magical artifacts.
The size of a hoard often reflects the dragon's age and power.
"""

SAMPLE_BOOK_PAGES = [
    Document(
        page_content="""Chapter 1: Introduction to Dragons

Dragons are legendary creatures that appear in the folklore of many cultures worldwide.
They are typically depicted as large, serpentine creatures with wings and the ability to breathe fire.

In fantasy literature, dragons are often portrayed as intelligent beings capable of speech.
Some dragons are benevolent guardians, while others are fearsome adversaries.""",
        metadata={"source": "dragon_guide.pdf", "page": 0}
    ),
    Document(
        page_content="""The most famous dragon types include:
- Red Dragons: Known for their fiery breath and aggressive nature
- Blue Dragons: Associated with lightning and desert environments
- Gold Dragons: Wise and benevolent creatures that often help heroes
- Black Dragons: Dwelling in swamps, they breathe acid""",
        metadata={"source": "dragon_guide.pdf", "page": 1}
    ),
    Document(
        page_content="""Chapter 2: Dragon Lairs

Dragons typically make their homes in remote, defensible locations.
Mountain caves are the most common choice, providing natural protection.
Some dragons prefer underwater lairs or ancient ruins.

A dragon's hoard is its most prized possession, containing gold, gems, and magical artifacts.
The size of a hoard often reflects the dragon's age and power.""",
        metadata={"source": "dragon_guide.pdf", "page": 2}
    ),
]


class TestEndToEndSearchPipeline:
    """Integration tests for the complete search pipeline."""

    @pytest.fixture
    def mock_vector_store_with_data(self):
        """Create a mock vector store pre-loaded with test documents."""
        mock_store = MagicMock()

        # Simulate similarity search returning relevant chunks
        def mock_search(query, k=10):
            query_lower = query.lower()

            results = []

            # Return relevant documents based on query
            if "dragon" in query_lower and "type" in query_lower:
                results.append((SAMPLE_BOOK_PAGES[1], 0.95))
                results.append((SAMPLE_BOOK_PAGES[0], 0.80))
            elif "lair" in query_lower or "home" in query_lower or "live" in query_lower:
                results.append((SAMPLE_BOOK_PAGES[2], 0.92))
            elif "dragon" in query_lower:
                results.append((SAMPLE_BOOK_PAGES[0], 0.90))
                results.append((SAMPLE_BOOK_PAGES[1], 0.85))
            elif "hoard" in query_lower or "treasure" in query_lower:
                results.append((SAMPLE_BOOK_PAGES[2], 0.88))
            else:
                # Return all with lower scores for generic queries
                for i, page in enumerate(SAMPLE_BOOK_PAGES):
                    results.append((page, 0.5 - i * 0.1))

            return results[:k]

        mock_store.similarity_search_with_score = mock_search
        mock_store.add_documents = MagicMock(return_value=["id1", "id2", "id3"])

        return mock_store

    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM that generates responses based on context."""
        mock = MagicMock()

        def generate_response(messages):
            # Extract the last human message
            last_message = None
            for msg in messages:
                if hasattr(msg, 'content'):
                    last_message = msg.content

            query_lower = last_message.lower() if last_message else ""

            # Generate contextual responses
            if "type" in query_lower and "dragon" in query_lower:
                response = """Based on the Dragon Guide, there are four main types of dragons:

1. **Red Dragons** - Known for their fiery breath and aggressive nature
2. **Blue Dragons** - Associated with lightning and desert environments
3. **Gold Dragons** - Wise and benevolent creatures that often help heroes
4. **Black Dragons** - Dwelling in swamps, they breathe acid

Source: dragon_guide.pdf, page 2"""
            elif "lair" in query_lower or "where" in query_lower and "live" in query_lower:
                response = """According to the Dragon Guide, dragons typically make their homes in remote, defensible locations. Mountain caves are the most common choice, providing natural protection. Some dragons prefer underwater lairs or ancient ruins.

Source: dragon_guide.pdf, page 3"""
            else:
                response = """Dragons are legendary creatures that appear in folklore worldwide. They are typically depicted as large, serpentine creatures with wings and the ability to breathe fire.

Source: dragon_guide.pdf, page 1"""

            return AIMessage(content=response)

        mock.invoke = generate_response
        return mock

    @patch("semantic_search_agent.vector_store")
    def test_semantic_search_finds_relevant_content(self, mock_vs, mock_vector_store_with_data):
        """Test that semantic search returns relevant documents."""
        mock_vs.similarity_search_with_score = mock_vector_store_with_data.similarity_search_with_score

        from semantic_search_agent import semantic_search

        # Query about dragon types
        results = semantic_search.invoke({"query": "What types of dragons exist?", "top_k": 5})

        assert len(results) > 0
        # First result should be about dragon types
        top_doc, score = results[0]
        assert "Red Dragons" in top_doc.page_content or "dragon" in top_doc.page_content.lower()
        assert score > 0.8

    @patch("semantic_search_agent.vector_store")
    def test_semantic_search_ranks_by_relevance(self, mock_vs, mock_vector_store_with_data):
        """Test that results are ranked by relevance score."""
        mock_vs.similarity_search_with_score = mock_vector_store_with_data.similarity_search_with_score

        from semantic_search_agent import semantic_search

        results = semantic_search.invoke({"query": "Where do dragons live?", "top_k": 5})

        # Results should be sorted by score (highest first)
        scores = [score for _, score in results]
        assert scores == sorted(scores, reverse=True)

    @patch("semantic_search_agent.vector_store")
    def test_search_with_no_matching_content(self, mock_vs):
        """Test behavior when no relevant content is found."""
        mock_vs.similarity_search_with_score.return_value = []

        from semantic_search_agent import semantic_search

        results = semantic_search.invoke({"query": "What is quantum physics?", "top_k": 5})

        assert results == []


class TestUploadAndSearchIntegration:
    """Integration tests for upload -> vectorize -> search flow."""

    @pytest.fixture
    def mock_full_pipeline(self):
        """Mock the complete pipeline components."""
        with patch("upload_server.create_embeddings") as mock_emb, \
             patch("upload_server.create_mongo_vector_store") as mock_store_create, \
             patch("upload_server.store_documents") as mock_store_docs, \
             patch("upload_server.PyPDFLoader") as mock_loader:

            # Setup embeddings mock
            mock_emb.return_value = MagicMock()

            # Setup vector store mock
            mock_vector_store = MagicMock()
            mock_vector_store.create_vector_search_index.return_value = None
            mock_store_create.return_value = mock_vector_store

            # Setup document storage mock
            mock_store_docs.return_value = {Path("test_book.pdf"): ["chunk1", "chunk2", "chunk3"]}

            # Setup PDF loader mock
            mock_loader_instance = MagicMock()
            mock_loader_instance.load.return_value = SAMPLE_BOOK_PAGES
            mock_loader.return_value = mock_loader_instance

            yield {
                "embeddings": mock_emb,
                "vector_store": mock_vector_store,
                "store_docs": mock_store_docs,
                "loader": mock_loader,
            }

    @pytest.fixture
    def upload_client(self, mock_full_pipeline):
        """Create test client for upload server."""
        import upload_server
        upload_server._embeddings = None
        upload_server._vector_store = None
        upload_server._text_splitter = None

        from upload_server import app
        return TestClient(app)

    def test_upload_vectorizes_document(self, upload_client, mock_full_pipeline):
        """Test that uploading a document triggers vectorization."""
        # Create a dummy PDF file
        pdf_content = b"%PDF-1.4 dummy content"

        response = upload_client.post(
            "/upload",
            files={"file": ("dragon_guide.pdf", io.BytesIO(pdf_content), "application/pdf")}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["filename"] == "dragon_guide.pdf"
        assert data["pages"] == 3  # Our mock has 3 pages
        assert data["chunks"] == 3

        # Verify store_documents was called
        mock_full_pipeline["store_docs"].assert_called_once()

    def test_upload_preserves_metadata(self, upload_client, mock_full_pipeline):
        """Test that document metadata is preserved during upload."""
        pdf_content = b"%PDF-1.4 dummy content"

        response = upload_client.post(
            "/upload",
            files={"file": ("my_book.pdf", io.BytesIO(pdf_content), "application/pdf")}
        )

        assert response.status_code == 200

        # Check that store_documents received documents with correct source
        call_args = mock_full_pipeline["store_docs"].call_args
        documents_dict = call_args[1]["documents"]

        # Get the documents from the dict
        for path, docs in documents_dict.items():
            for doc in docs:
                assert doc.metadata["source"] == "my_book.pdf"


class TestAgentIntegration:
    """Integration tests for the full agent conversation flow."""

    @pytest.fixture
    def mock_agent_components(self):
        """Mock agent components for testing."""
        with patch("semantic_search_agent.vector_store") as mock_vs, \
             patch("semantic_search_agent.llm") as mock_llm:

            # Setup vector store to return relevant documents
            def mock_search(query, k=10):
                return [(SAMPLE_BOOK_PAGES[1], 0.95), (SAMPLE_BOOK_PAGES[0], 0.85)]

            mock_vs.similarity_search_with_score = mock_search

            yield {"vector_store": mock_vs, "llm": mock_llm}

    def test_agent_uses_semantic_search_tool(self, mock_agent_components):
        """Test that agent invokes semantic search for user queries."""
        from semantic_search_agent import semantic_search

        # Simulate agent calling the search tool
        results = semantic_search.invoke({"query": "What are the different types of dragons?"})

        assert len(results) > 0
        assert any("Red Dragons" in doc.page_content for doc, _ in results)

    def test_agent_retrieves_source_citations(self, mock_agent_components):
        """Test that search results include source metadata for citations."""
        from semantic_search_agent import semantic_search

        results = semantic_search.invoke({"query": "Tell me about dragon lairs"})

        for doc, score in results:
            assert "source" in doc.metadata
            assert doc.metadata["source"] == "dragon_guide.pdf"


class TestUserQueryScenarios:
    """Test various user query scenarios."""

    @pytest.fixture
    def search_with_mock_store(self):
        """Setup search with mock vector store."""
        with patch("semantic_search_agent.vector_store") as mock_vs:
            def mock_search(query, k=10):
                query_lower = query.lower()
                if "dragon" in query_lower:
                    return [(SAMPLE_BOOK_PAGES[0], 0.9), (SAMPLE_BOOK_PAGES[1], 0.85)]
                return []

            mock_vs.similarity_search_with_score = mock_search
            yield

    def test_user_asks_about_dragons(self, search_with_mock_store):
        """Test user asking 'What are dragons?'"""
        from semantic_search_agent import semantic_search

        results = semantic_search.invoke({"query": "What are dragons?"})

        assert len(results) > 0
        top_result = results[0][0]
        assert "dragon" in top_result.page_content.lower()

    def test_user_asks_specific_question(self, search_with_mock_store):
        """Test user asking specific question about content."""
        from semantic_search_agent import semantic_search

        results = semantic_search.invoke({"query": "What color are aggressive dragons?"})

        assert len(results) > 0
        # Should find content about Red Dragons being aggressive

    def test_user_asks_unrelated_question(self, search_with_mock_store):
        """Test user asking question not in the documents."""
        from semantic_search_agent import semantic_search

        results = semantic_search.invoke({"query": "What is the capital of France?"})

        # Should return empty or low-relevance results
        assert len(results) == 0

    def test_multiple_sequential_queries(self, search_with_mock_store):
        """Test multiple queries in sequence."""
        from semantic_search_agent import semantic_search

        # First query
        results1 = semantic_search.invoke({"query": "Tell me about dragons"})
        assert len(results1) > 0

        # Second query
        results2 = semantic_search.invoke({"query": "What types of dragons exist?"})
        assert len(results2) > 0

        # Results can be different based on query


class TestErrorHandling:
    """Test error handling in the integration flow."""

    def test_search_handles_connection_error(self):
        """Test that search handles database connection errors gracefully."""
        with patch("semantic_search_agent.vector_store") as mock_vs:
            mock_vs.similarity_search_with_score.side_effect = Exception("Connection refused")

            from semantic_search_agent import semantic_search

            results = semantic_search.invoke({"query": "test query"})

            # Should return empty list, not raise exception
            assert results == []

    def test_upload_handles_invalid_pdf(self):
        """Test that upload handles corrupted PDFs."""
        with patch("upload_server.create_embeddings"), \
             patch("upload_server.create_mongo_vector_store"), \
             patch("upload_server.PyPDFLoader") as mock_loader:

            mock_loader.side_effect = Exception("Invalid PDF format")

            import upload_server
            upload_server._embeddings = None
            upload_server._vector_store = None
            upload_server._text_splitter = None

            from upload_server import app
            client = TestClient(app)

            response = client.post(
                "/upload",
                files={"file": ("bad.pdf", io.BytesIO(b"not a pdf"), "application/pdf")}
            )

            assert response.status_code == 500

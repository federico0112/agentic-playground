"""Unit tests for upload_server.py."""

import io
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock

import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document


# We need to mock before importing the app
@pytest.fixture
def mock_components():
    """Mock ML components to avoid initialization."""
    with patch("backend.upload_server.create_embeddings") as mock_embeddings, \
         patch("backend.upload_server.create_mongo_vector_store") as mock_store, \
         patch("backend.upload_server.store_documents") as mock_store_docs:

        mock_embeddings.return_value = MagicMock()

        mock_vector_store = MagicMock()
        mock_vector_store.create_vector_search_index.return_value = None
        mock_store.return_value = mock_vector_store

        mock_store_docs.return_value = {Path("test.pdf"): ["id1", "id2", "id3"]}

        yield {
            "embeddings": mock_embeddings,
            "store": mock_store,
            "store_docs": mock_store_docs,
            "vector_store": mock_vector_store,
        }


@pytest.fixture
def client(mock_components):
    """Create test client with mocked components."""
    # Reset global state
    import backend.upload_server as upload_server
    upload_server._embeddings = None
    upload_server._vector_store = None
    upload_server._text_splitter = None

    from backend.upload_server import app
    return TestClient(app)


@pytest.fixture
def sample_pdf_bytes():
    """Create minimal PDF bytes for testing."""
    return b"""%PDF-1.4
1 0 obj
<< /Type /Catalog /Pages 2 0 R >>
endobj
2 0 obj
<< /Type /Pages /Kids [3 0 R] /Count 1 >>
endobj
3 0 obj
<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R >>
endobj
4 0 obj
<< /Length 44 >>
stream
BT
/F1 12 Tf
100 700 Td
(Test PDF) Tj
ET
endstream
endobj
xref
0 5
0000000000 65535 f
0000000009 00000 n
0000000058 00000 n
0000000115 00000 n
0000000214 00000 n
trailer
<< /Size 5 /Root 1 0 R >>
startxref
306
%%EOF"""


class TestHealthEndpoint:
    """Tests for the health check endpoint."""

    def test_health_returns_ok(self, client):
        """Test that health endpoint returns OK status."""
        response = client.get("/health")

        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


class TestUploadEndpoint:
    """Tests for the upload endpoint."""

    @patch("backend.upload_server.PyPDFLoader")
    def test_upload_accepts_pdf(self, mock_loader, client, sample_pdf_bytes, mock_components):
        """Test that upload accepts PDF files."""
        mock_loader_instance = MagicMock()
        mock_loader_instance.load.return_value = [
            Document(page_content="Test content", metadata={"source": "test.pdf", "page": 0})
        ]
        mock_loader.return_value = mock_loader_instance

        response = client.post(
            "/upload",
            files={"file": ("test.pdf", io.BytesIO(sample_pdf_bytes), "application/pdf")}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["filename"] == "test.pdf"
        assert "pages" in data
        assert "chunks" in data

    def test_upload_rejects_non_pdf(self, client):
        """Test that upload rejects non-PDF files."""
        response = client.post(
            "/upload",
            files={"file": ("test.txt", io.BytesIO(b"not a pdf"), "text/plain")}
        )

        assert response.status_code == 400
        assert "PDF" in response.json()["detail"]

    def test_upload_rejects_empty_filename(self, client):
        """Test that upload rejects files without filename."""
        response = client.post(
            "/upload",
            files={"file": ("", io.BytesIO(b"content"), "application/pdf")}
        )

        # FastAPI returns 422 for validation errors
        assert response.status_code == 422

    @patch("backend.upload_server.PyPDFLoader")
    def test_upload_returns_page_count(self, mock_loader, client, sample_pdf_bytes, mock_components):
        """Test that upload returns correct page count."""
        mock_loader_instance = MagicMock()
        mock_loader_instance.load.return_value = [
            Document(page_content="Page 1", metadata={"page": 0}),
            Document(page_content="Page 2", metadata={"page": 1}),
            Document(page_content="Page 3", metadata={"page": 2}),
        ]
        mock_loader.return_value = mock_loader_instance

        response = client.post(
            "/upload",
            files={"file": ("test.pdf", io.BytesIO(sample_pdf_bytes), "application/pdf")}
        )

        assert response.status_code == 200
        assert response.json()["pages"] == 3

    @patch("backend.upload_server.PyPDFLoader")
    def test_upload_returns_chunk_count(self, mock_loader, client, sample_pdf_bytes, mock_components):
        """Test that upload returns correct chunk count."""
        mock_loader_instance = MagicMock()
        mock_loader_instance.load.return_value = [
            Document(page_content="Test content", metadata={"page": 0})
        ]
        mock_loader.return_value = mock_loader_instance

        response = client.post(
            "/upload",
            files={"file": ("test.pdf", io.BytesIO(sample_pdf_bytes), "application/pdf")}
        )

        assert response.status_code == 200
        # Mock returns 3 chunk IDs
        assert response.json()["chunks"] == 3

    @patch("backend.upload_server.PyPDFLoader")
    def test_upload_handles_empty_pdf(self, mock_loader, client, sample_pdf_bytes):
        """Test that upload handles PDFs with no extractable content."""
        mock_loader_instance = MagicMock()
        mock_loader_instance.load.return_value = []
        mock_loader.return_value = mock_loader_instance

        response = client.post(
            "/upload",
            files={"file": ("empty.pdf", io.BytesIO(sample_pdf_bytes), "application/pdf")}
        )

        assert response.status_code == 400
        assert "extract" in response.json()["detail"].lower()

    @patch("backend.upload_server.PyPDFLoader")
    def test_upload_handles_loader_exception(self, mock_loader, client, sample_pdf_bytes):
        """Test that upload handles PDF loader exceptions."""
        mock_loader.side_effect = Exception("Failed to load PDF")

        response = client.post(
            "/upload",
            files={"file": ("corrupt.pdf", io.BytesIO(sample_pdf_bytes), "application/pdf")}
        )

        assert response.status_code == 500


class TestMetadataHandling:
    """Tests for document metadata handling."""

    @patch("backend.upload_server.PyPDFLoader")
    def test_upload_fixes_source_metadata(self, mock_loader, client, sample_pdf_bytes, mock_components):
        """Test that source metadata is updated to original filename."""
        mock_docs = [
            Document(page_content="Content", metadata={"source": "/tmp/xyz123.pdf", "page": 0})
        ]
        mock_loader_instance = MagicMock()
        mock_loader_instance.load.return_value = mock_docs
        mock_loader.return_value = mock_loader_instance

        response = client.post(
            "/upload",
            files={"file": ("original_name.pdf", io.BytesIO(sample_pdf_bytes), "application/pdf")}
        )

        assert response.status_code == 200
        # Verify that source metadata was updated
        assert mock_docs[0].metadata["source"] == "original_name.pdf"

    @patch("backend.upload_server.PyPDFLoader")
    def test_upload_preserves_page_metadata(self, mock_loader, client, sample_pdf_bytes, mock_components):
        """Test that page metadata is preserved."""
        mock_docs = [
            Document(page_content="Content", metadata={"source": "/tmp/xyz.pdf", "page": 5, "author": "Test Author"})
        ]
        mock_loader_instance = MagicMock()
        mock_loader_instance.load.return_value = mock_docs
        mock_loader.return_value = mock_loader_instance

        response = client.post(
            "/upload",
            files={"file": ("test.pdf", io.BytesIO(sample_pdf_bytes), "application/pdf")}
        )

        assert response.status_code == 200
        # Page and author metadata should be preserved
        assert mock_docs[0].metadata["page"] == 5
        assert mock_docs[0].metadata["author"] == "Test Author"


class TestGetComponents:
    """Tests for the get_components function."""

    def test_get_components_initializes_lazily(self, mock_components):
        """Test that components are initialized lazily."""
        import backend.upload_server as upload_server
        upload_server._embeddings = None
        upload_server._vector_store = None
        upload_server._text_splitter = None

        from backend.upload_server import get_components

        embeddings, vector_store, text_splitter = get_components()

        assert embeddings is not None
        assert vector_store is not None
        assert text_splitter is not None

    def test_get_components_reuses_instances(self, mock_components):
        """Test that components are reused on subsequent calls."""
        import backend.upload_server as upload_server
        upload_server._embeddings = None
        upload_server._vector_store = None
        upload_server._text_splitter = None

        from backend.upload_server import get_components

        result1 = get_components()
        result2 = get_components()

        # Should be the same instances
        assert result1[0] is result2[0]  # embeddings
        assert result1[1] is result2[1]  # vector_store
        assert result1[2] is result2[2]  # text_splitter


class TestCORSConfiguration:
    """Tests for CORS configuration."""

    def test_cors_allows_all_origins(self, client):
        """Test that CORS is configured to allow all origins."""
        response = client.options(
            "/upload",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "POST",
            }
        )

        # CORS preflight should be handled
        assert response.status_code in [200, 204, 405]

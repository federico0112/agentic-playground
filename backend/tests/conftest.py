"""Shared pytest fixtures for all tests."""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document


@pytest.fixture
def sample_documents():
    """Create sample documents for testing."""
    return [
        Document(
            page_content="This is page 1 content about dragons and magic.",
            metadata={"source": "test_book.pdf", "page": 0}
        ),
        Document(
            page_content="This is page 2 content about wizards and spells.",
            metadata={"source": "test_book.pdf", "page": 1}
        ),
        Document(
            page_content="This is page 3 content about dungeons and treasure.",
            metadata={"source": "test_book.pdf", "page": 2}
        ),
    ]


@pytest.fixture
def temp_pdf_file():
    """Create a temporary PDF file for testing."""
    # Create a minimal valid PDF
    pdf_content = b"""%PDF-1.4
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

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        f.write(pdf_content)
        temp_path = f.name

    yield temp_path

    # Cleanup
    Path(temp_path).unlink(missing_ok=True)


@pytest.fixture
def mock_embeddings():
    """Mock embeddings for testing without API calls."""
    mock = MagicMock()
    mock.embed_documents.return_value = [[0.1] * 3072 for _ in range(3)]
    mock.embed_query.return_value = [0.1] * 3072
    return mock


@pytest.fixture
def mock_vector_store():
    """Mock vector store for testing without MongoDB."""
    mock = MagicMock()
    mock.add_documents.return_value = ["id1", "id2", "id3"]
    mock.similarity_search_with_score.return_value = [
        (Document(page_content="Result 1", metadata={"source": "test.pdf"}), 0.95),
        (Document(page_content="Result 2", metadata={"source": "test.pdf"}), 0.85),
    ]
    return mock


@pytest.fixture
def mock_mongo_client():
    """Mock MongoDB client."""
    with patch("pymongo.MongoClient") as mock_client:
        mock_db = MagicMock()
        mock_collection = MagicMock()
        mock_collection.count_documents.return_value = 100
        mock_db.__getitem__.return_value = mock_collection
        mock_client.return_value.__getitem__.return_value = mock_db
        yield mock_client


@pytest.fixture
def env_vars():
    """Set up test environment variables."""
    original_env = os.environ.copy()

    os.environ["MONGODB_URI"] = "mongodb://localhost:27017"
    os.environ["DB_NAME"] = "test_db"
    os.environ["COLLECTION_NAME"] = "test_collection"
    os.environ["INDEX_NAME"] = "test_index"
    os.environ["GOOGLE_API_KEY"] = "test-api-key"

    yield

    os.environ.clear()
    os.environ.update(original_env)

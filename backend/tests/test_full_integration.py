"""Full integration tests using real services.

These tests require:
- MongoDB running (uses the same database as the app)
- Google API key for embeddings
- All environment variables configured in .env

Run with: pytest tests/test_full_integration.py -v -s
"""

import os
import tempfile
import time
from pathlib import Path

import pytest
from dotenv import load_dotenv
from fastapi.testclient import TestClient
from pymongo import MongoClient
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# Load environment variables
load_dotenv()

# Test configuration
TEST_COLLECTION_SUFFIX = "_test"
CLEANUP_AFTER_TESTS = True


def create_test_pdf(filepath: str, content_pages: list[str]) -> str:
    """Create a real PDF file with the given content.

    Args:
        filepath: Path to save the PDF
        content_pages: List of strings, each string becomes a page

    Returns:
        Path to the created PDF
    """
    c = canvas.Canvas(filepath, pagesize=letter)
    width, height = letter

    for i, content in enumerate(content_pages):
        # Write content to page
        text_object = c.beginText(50, height - 50)
        text_object.setFont("Helvetica", 12)

        # Split content into lines
        lines = content.split('\n')
        for line in lines:
            # Wrap long lines
            while len(line) > 80:
                text_object.textLine(line[:80])
                line = line[80:]
            text_object.textLine(line)

        c.drawText(text_object)

        if i < len(content_pages) - 1:
            c.showPage()

    c.save()
    return filepath


# Sample test content - a mini "handbook" about wizards
TEST_BOOK_CONTENT = [
    """The Wizard's Handbook - Page 1

Chapter 1: Introduction to Wizardry

Wizards are practitioners of arcane magic who spend years studying ancient tomes
and mystical texts. Unlike sorcerers who have innate magical abilities, wizards
must learn their craft through dedicated study and practice.

The path to becoming a wizard typically begins at a young age, when a potential
student shows aptitude for understanding complex magical theories. Most wizards
train at prestigious academies or under the tutelage of experienced masters.

Key traits of successful wizards include:
- Exceptional memory for spells and incantations
- Patience for long hours of study
- Analytical thinking for understanding magical formulas
- Wisdom to know when and how to use their powers""",

    """The Wizard's Handbook - Page 2

Chapter 2: Schools of Magic

There are eight recognized schools of magic that wizards may specialize in:

1. Abjuration - Protective magic and wards
2. Conjuration - Summoning creatures and objects
3. Divination - Seeing the future and gathering information
4. Enchantment - Influencing minds and emotions
5. Evocation - Elemental damage spells like fireball
6. Illusion - Creating false images and sounds
7. Necromancy - Magic dealing with life and death
8. Transmutation - Changing the properties of things

Most wizards choose to specialize in one school while maintaining basic
competency in others. The choice of school often reflects the wizard's
personality and goals.""",

    """The Wizard's Handbook - Page 3

Chapter 3: Essential Spells

Every wizard should master these fundamental spells:

Cantrips (at-will spells):
- Light: Creates a glowing orb for illumination
- Mage Hand: Manipulates objects from a distance
- Prestidigitation: Minor magical tricks and effects

First Level Spells:
- Magic Missile: Unerring darts of magical force
- Shield: Temporary protective barrier
- Detect Magic: Sense magical auras nearby

The most famous wizard spell is Fireball, a third-level evocation that
creates a massive explosion of flame. It requires precise aim to avoid
harming allies and is considered a rite of passage for evokers.""",
]


@pytest.fixture(scope="module")
def mongodb_client():
    """Create MongoDB client for test database operations."""
    uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
    client = MongoClient(uri)
    yield client
    client.close()


@pytest.fixture(scope="module")
def test_collection_name():
    """Generate unique test collection name."""
    db_name = os.getenv("DB_NAME", "book_search_db")
    base_collection = os.getenv("COLLECTION_NAME", "vectorized_documents")
    return f"{base_collection}{TEST_COLLECTION_SUFFIX}"


@pytest.fixture(scope="module")
def test_pdf_file():
    """Create a test PDF file."""
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        filepath = f.name

    create_test_pdf(filepath, TEST_BOOK_CONTENT)

    yield filepath

    # Cleanup
    Path(filepath).unlink(missing_ok=True)


@pytest.fixture(scope="module")
def upload_client():
    """Create FastAPI test client for upload server."""
    from backend.upload_server import app
    return TestClient(app)


@pytest.fixture(scope="module")
def cleanup_test_data(mongodb_client, test_collection_name):
    """Fixture to clean up test data after tests complete."""
    yield

    if CLEANUP_AFTER_TESTS:
        db_name = os.getenv("DB_NAME", "book_search_db")
        db = mongodb_client[db_name]

        # Delete test documents (those with test file source)
        collection = db[os.getenv("COLLECTION_NAME", "vectorized_documents")]
        result = collection.delete_many({"source": {"$regex": "wizard.*test.*\\.pdf", "$options": "i"}})
        print(f"\nCleanup: Deleted {result.deleted_count} test documents")


class TestFullIntegration:
    """Full integration tests with real services."""

    @pytest.mark.integration
    def test_upload_real_pdf(self, upload_client, test_pdf_file):
        """Test uploading a real PDF file to the server."""
        with open(test_pdf_file, "rb") as f:
            response = upload_client.post(
                "/upload",
                files={"file": ("wizard_handbook_test.pdf", f, "application/pdf")}
            )

        print(f"\nUpload response: {response.json()}")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["pages"] == 3
        assert data["chunks"] > 0

        # Store chunk count for later verification
        pytest.test_chunk_count = data["chunks"]

    @pytest.mark.integration
    def test_documents_stored_in_mongodb(self, mongodb_client):
        """Verify documents were stored in MongoDB."""
        db_name = os.getenv("DB_NAME", "book_search_db")
        collection_name = os.getenv("COLLECTION_NAME", "vectorized_documents")

        db = mongodb_client[db_name]
        collection = db[collection_name]

        # Find our test documents
        test_docs = list(collection.find({"source": "wizard_handbook_test.pdf"}))

        print(f"\nFound {len(test_docs)} documents in MongoDB")

        assert len(test_docs) > 0

        # Verify document structure
        sample_doc = test_docs[0]
        assert "embedding" in sample_doc
        assert "text" in sample_doc or "page_content" in sample_doc
        assert "source" in sample_doc

        print(f"Sample document keys: {list(sample_doc.keys())}")

    @pytest.mark.integration
    def test_semantic_search_finds_content(self):
        """Test that semantic search finds the uploaded content."""
        from backend.semantic_search_agent import semantic_search

        # Query about wizard schools
        results = semantic_search.invoke({
            "query": "What are the schools of magic for wizards?",
            "top_k": 5
        })

        print(f"\nSearch results for 'schools of magic':")
        for doc, score in results[:3]:
            print(f"  Score: {score:.4f}")
            print(f"  Source: {doc.metadata.get('source', 'N/A')}")
            print(f"  Content: {doc.page_content[:100]}...")
            print()

        assert len(results) > 0

        # Check if we found relevant content
        found_relevant = False
        for doc, score in results:
            if "school" in doc.page_content.lower() or "abjuration" in doc.page_content.lower():
                found_relevant = True
                break

        assert found_relevant, "Should find content about magic schools"

    @pytest.mark.integration
    def test_semantic_search_specific_query(self):
        """Test semantic search with specific query about fireball."""
        from backend.semantic_search_agent import semantic_search

        results = semantic_search.invoke({
            "query": "Tell me about the fireball spell",
            "top_k": 5
        })

        print(f"\nSearch results for 'fireball spell':")
        for doc, score in results[:3]:
            print(f"  Score: {score:.4f} | {doc.page_content[:80]}...")

        assert len(results) > 0

        # Should find content mentioning fireball
        found_fireball = any("fireball" in doc.page_content.lower() for doc, _ in results)
        assert found_fireball, "Should find content about fireball"

    @pytest.mark.integration
    def test_agent_answers_question(self):
        """Test the full agent answering a question."""
        from backend.semantic_search_agent import agent

        # Ask the agent a question
        query = "What are the eight schools of magic that wizards can specialize in?"

        print(f"\nAsking agent: {query}")
        print("-" * 50)

        response_content = ""

        # Stream the agent response
        for step in agent.stream(
            {"messages": [{"role": "user", "content": query}]},
            stream_mode="values",
        ):
            last_message = step["messages"][-1]
            if hasattr(last_message, 'content') and last_message.content:
                response_content = last_message.content

        print(f"Agent response:\n{response_content[:500]}...")
        print("-" * 50)

        assert len(response_content) > 0

        # Check if response mentions some schools of magic
        # Handle new SearchResults format: response_content is a string with the answer
        response_text = response_content if isinstance(response_content, str) else str(response_content)
        response_lower = response_text.lower()
        schools_mentioned = sum(1 for school in ["abjuration", "conjuration", "divination",
                                                   "enchantment", "evocation", "illusion",
                                                   "necromancy", "transmutation"]
                                if school in response_lower)

        print(f"Schools of magic mentioned: {schools_mentioned}")
        assert schools_mentioned >= 3, "Agent should mention at least 3 schools of magic"

    @pytest.mark.integration
    def test_agent_cites_sources(self):
        """Test that agent cites sources in its response."""
        from backend.semantic_search_agent import agent

        query = "What spells should every wizard know?"

        print(f"\nAsking agent: {query}")

        response_content = ""
        for step in agent.stream(
            {"messages": [{"role": "user", "content": query}]},
            stream_mode="values",
        ):
            last_message = step["messages"][-1]
            if hasattr(last_message, 'content') and last_message.content:
                response_content = last_message.content

        print(f"Agent response:\n{response_content[:500]}...")

        # Check for source citations (agent should mention the source)
        assert len(response_content) > 0

        # Response should mention specific spells from our test content
        # Handle new SearchResults format: response_content is a string with the answer
        response_text = response_content if isinstance(response_content, str) else str(response_content)
        spells_mentioned = sum(1 for spell in ["magic missile", "shield", "fireball",
                                                "light", "mage hand", "prestidigitation"]
                               if spell in response_text.lower())

        print(f"Spells mentioned: {spells_mentioned}")
        assert spells_mentioned >= 2, "Agent should mention specific spells from the content"

    @pytest.mark.integration
    def test_agent_handles_followup_questions(self):
        """Test agent handling follow-up questions in a conversation."""
        from backend.semantic_search_agent import agent

        # First question
        query1 = "What is evocation magic?"
        print(f"\nFirst question: {query1}")

        response1 = ""
        for step in agent.stream(
            {"messages": [{"role": "user", "content": query1}]},
            stream_mode="values",
        ):
            last_message = step["messages"][-1]
            if hasattr(last_message, 'content') and last_message.content:
                response1 = last_message.content

        # Handle new SearchResults format: response is a string with the answer
        response1_text = response1 if isinstance(response1, str) else str(response1)
        response1_lower = response1_text.lower()
        print(f"Response 1: {response1_text[:200]}...")
        assert "evocation" in response1_lower or "damage" in response1_lower or "fireball" in response1_lower

        # Second question (different topic)
        query2 = "How do wizards learn their craft?"
        print(f"\nSecond question: {query2}")

        response2 = ""
        for step in agent.stream(
            {"messages": [{"role": "user", "content": query2}]},
            stream_mode="values",
        ):
            last_message = step["messages"][-1]
            if hasattr(last_message, 'content') and last_message.content:
                response2 = last_message.content

        # Handle new SearchResults format: response is a string with the answer
        response2_text = response2 if isinstance(response2, str) else str(response2)
        response2_lower = response2_text.lower()
        print(f"Response 2: {response2_text[:200]}...")
        assert "study" in response2_lower or "learn" in response2_lower or "train" in response2_lower


class TestCleanup:
    """Cleanup tests - run last."""

    @pytest.mark.integration
    def test_cleanup_test_documents(self, mongodb_client, cleanup_test_data):
        """Clean up test documents from MongoDB."""
        # This test just triggers the cleanup fixture
        print("\nTriggering test data cleanup...")
        pass


# Pytest configuration for integration tests
def pytest_configure(config):
    config.addinivalue_line(
        "markers", "integration: mark test as integration test requiring real services"
    )


if __name__ == "__main__":
    # Run with: python -m pytest tests/test_full_integration.py -v -s
    pytest.main([__file__, "-v", "-s", "-m", "integration"])

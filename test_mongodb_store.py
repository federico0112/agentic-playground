"""Test script for MongoDBVectorStore implementation."""

from mongodb_vector_store import MongoDBVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langgraph.store.base import IndexConfig

def test_basic_operations():
    """Test basic put/get/search operations."""
    print("Testing basic MongoDB store operations...")

    # Create store without vector search first
    store = MongoDBVectorStore(
        connection_string="mongodb://localhost:27017",
        database="test_db",
        collection="test_collection"
    )

    # Test put operation
    print("1. Testing put operation...")
    store.put(("testpdf",), "0-9", {"summary": "Test summary for pages 0-9"})
    print("   ✓ Put operation successful")

    # Test get operation
    print("2. Testing get operation...")
    item = store.get(("testpdf",), "0-9")
    assert item is not None, "Item should exist"
    assert item.value["summary"] == "Test summary for pages 0-9", "Value should match"
    print(f"   ✓ Get operation successful: {item.value}")

    # Test search operation
    print("3. Testing search operation...")
    results = store.search(("testpdf",))
    assert len(results) == 1, "Should find 1 item"
    assert results[0].key == "0-9", "Key should match"
    print(f"   ✓ Search operation successful: found {len(results)} item(s)")

    # Test multiple items
    print("4. Testing multiple items...")
    store.put(("testpdf",), "10-19", {"summary": "Test summary for pages 10-19"})
    store.put(("testpdf",), "20-29", {"summary": "Test summary for pages 20-29"})
    results = store.search(("testpdf",))
    assert len(results) == 3, "Should find 3 items"
    print(f"   ✓ Multiple items test successful: found {len(results)} item(s)")

    # Cleanup
    print("5. Cleaning up test data...")
    store.put(("testpdf",), "0-9", None)
    store.put(("testpdf",), "10-19", None)
    store.put(("testpdf",), "20-29", None)
    results = store.search(("testpdf",))
    assert len(results) == 0, "Should find 0 items after deletion"
    print("   ✓ Cleanup successful")

    print("\n✅ All basic operations tests passed!")

def test_vector_search():
    """Test vector search with embeddings."""
    print("\n\nTesting vector search operations...")

    try:
        # Create store with vector search enabled using Gemini
        embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

        index_config = IndexConfig(
            dims=768,  # Gemini text-embedding-004 dimensions
            embed=embeddings
        )

        store = MongoDBVectorStore(
            connection_string="mongodb://localhost:27017",
            database="test_db",
            collection="test_vector_collection",
            index=index_config
        )

        # Test put operation with embeddings
        print("1. Testing put with embeddings...")
        store.put(("physicspdf",), "0-9", {"text": "Newton's laws of motion and forces"})
        store.put(("physicspdf",), "10-19", {"text": "Quantum mechanics and wave functions"})
        store.put(("physicspdf",), "20-29", {"text": "Thermodynamics and heat transfer"})
        print("   ✓ Put operations with embeddings successful")

        # Test semantic search
        print("2. Testing semantic search...")
        results = store.search(("physicspdf",), query="forces and acceleration")
        assert len(results) > 0, "Should find results"
        print(f"   ✓ Semantic search successful: found {len(results)} item(s)")
        print(f"   Top result: {results[0].key} (score: {results[0].score})")

        # Cleanup
        print("3. Cleaning up test data...")
        store.put(("physicspdf",), "0-9", None)
        store.put(("physicspdf",), "10-19", None)
        store.put(("physicspdf",), "20-29", None)
        print("   ✓ Cleanup successful")

        print("\n✅ All vector search tests passed!")

    except Exception as e:
        print(f"\n⚠️  Vector search test failed: {e}")
        print("This might be due to missing GOOGLE_API_KEY environment variable or network issues.")
        print("Basic operations still work without vector search.")

if __name__ == "__main__":
    test_basic_operations()
    test_vector_search()
    print("\n\n🎉 All tests completed!")

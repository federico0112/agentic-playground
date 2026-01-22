"""
Book Vectorization Module.

Loads PDF books, splits them into chunks, generates embeddings using Google's
Gemini embeddings, and stores them in MongoDB for semantic search.

Usage:
    from load_book_vectors import (
        create_embeddings,
        create_mongo_vector_store,
        load_documents,
        store_documents,
        test_vector_search
    )

    # Create components
    embeddings = create_embeddings()
    vector_store = create_mongo_vector_store(uri, db, coll, index, embeddings)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    # Load and store documents
    documents = load_documents()
    stored_ids = store_documents(documents, vector_store, text_splitter)

    # Or use all-in-one test function
    results = test_vector_search(vector_store, text_splitter, "search query")
"""

import os
from pathlib import Path
from typing import List, Dict, Tuple
from uuid import uuid4

from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mongodb import MongoDBAtlasVectorSearch
from pymongo import MongoClient


# Load environment variables
load_dotenv()

# ============================================================================
# CONFIGURATION - Edit these constants as needed or set in .env file
# ============================================================================

BOOKS_DIR = "books"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100

# MongoDB Configuration - loaded from environment variables with defaults
MONGODB_URI = os.getenv(
    "MONGODB_URI",
    "mongodb://localhost:61213/?directConnection=true&serverSelectionTimeoutMS=2000&appName=mongosh+2.6.0"
)
DB_NAME = os.getenv("DB_NAME", "book_search_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "vectorized_documents")
MONGODB_INDEX_NAME = os.getenv("INDEX_NAME", "vector_index")

# ============================================================================
# FUNCTIONS
# ============================================================================


def create_embeddings() -> GoogleGenerativeAIEmbeddings:
    """Create and return Gemini embeddings instance with rate limiting.

    Returns:
        GoogleGenerativeAIEmbeddings: Configured embeddings instance using gemini-embedding-001 model
                                      with rate limiting to avoid API quota exhaustion
    """
    return GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        requests_per_minute=2500,  # Set below the 3000 limit to avoid rate limiting
        task_type="retrieval_document"
    )


def create_mongo_vector_store(
    mongodb_uri: str,
    db_name: str,
    collection_name: str,
    index_name: str,
    embeddings: GoogleGenerativeAIEmbeddings,
    create_index: bool = True,
) -> MongoDBAtlasVectorSearch:
    """Create MongoDB Atlas Vector Store.

    Args:
        mongodb_uri: MongoDB connection string
        db_name: Database name
        collection_name: Collection name
        index_name: Vector search index name
        embeddings: Embeddings instance
        create_index: Whether to create the vector search index (default: True)

    Returns:
        MongoDBAtlasVectorSearch instance
    """
    # Connect to MongoDB
    client = MongoClient(mongodb_uri)
    collection = client[db_name][collection_name]

    # Create vector store
    vector_store = MongoDBAtlasVectorSearch(
        collection=collection,
        embedding=embeddings,
        index_name=index_name,
        relevance_score_fn="cosine"
    )

    # Create vector search index if requested
    if create_index:
        try:
            vector_store.create_vector_search_index(dimensions=3072)
            print(f"✓ Created vector search index: {index_name}")
        except Exception as e:
            print(f"Note: Vector index may already exist or creation skipped: {e}")

    return vector_store


def load_documents() -> Dict[Path, List[Document]]:
    """Load all PDF documents from the books directory.

    Returns:
        dict: Dictionary where key is filename and value is list of LangChain Document objects
    """
    books_path = Path(BOOKS_DIR)

    if not books_path.exists():
        print(f"Error: Books directory not found: {books_path}")
        return {}

    pdf_files = list(books_path.glob("*.pdf"))

    if not pdf_files:
        print(f"No PDF files found in {books_path}")
        return {}

    documents = {}

    print(f"Loading {len(pdf_files)} PDF file(s) from {BOOKS_DIR}...")

    for pdf_path in sorted(pdf_files):
        try:
            print(f"  Loading: {pdf_path.name}...", end=" ")
            loader = PyPDFLoader(str(pdf_path))
            document = loader.load()
            documents[pdf_path] = document
            print(f"✓ ({len(document)} pages)")
        except Exception as e:
            print(f"✗ Error: {e}")

    return documents


def store_documents(
    documents: Dict[Path, List[Document]],
    vector_store: MongoDBAtlasVectorSearch,
    text_splitter: RecursiveCharacterTextSplitter
) -> Dict[Path, List[str]]:
    """Split documents into chunks and store them in the vector store.

    Args:
        documents: Dictionary mapping PDF file paths to their loaded Document objects
        vector_store: Configured MongoDB Atlas Vector Search instance
        text_splitter: Text splitter for chunking documents

    Returns:
        Dict[Path, List[str]]: Dictionary mapping file paths to lists of stored document UUIDs

    Note:
        - Automatically generates UUIDs for each chunk
        - Embeddings are generated automatically by the vector store
        - Prints progress for each document processed
    """
    total_chunks_stored = 0
    stored_ids: Dict[Path, List[str]] = {}

    for doc_path, doc in documents.items():
        print(f"\nProcessing: {doc_path.name}")

        # Split documents into chunks
        split_documents = text_splitter.split_documents(doc)
        print(f"  Split into {len(split_documents)} chunks")

        # Store in vector store (embeddings are generated automatically)
        print(f"  Storing chunks in MongoDB...")
        uuids = [str(uuid4()) for _ in range(len(split_documents))]
        vector_store.add_documents(split_documents, ids=uuids, batch_size=1000)
        stored_ids[doc_path] = uuids
        print(f"  ✓ Stored {len(split_documents)} chunks")

        total_chunks_stored += len(split_documents)

    print(f"\n{'='*80}")
    print(f"✓ Vector storage complete")
    print(f"  Total chunks stored: {total_chunks_stored}")
    print(f"{'='*80}\n")

    return stored_ids


def test_vector_search(
    vector_store: MongoDBAtlasVectorSearch,
    text_splitter: RecursiveCharacterTextSplitter,
    query: str,
    with_cleanup: bool = False
) -> List[Tuple[Document, float]]:
    """Load, vectorize, and search documents in one operation (for testing).

    This is a convenience function that combines document loading, vectorization,
    and search testing. Useful for quick testing and validation.

    Args:
        vector_store: Configured MongoDB Atlas Vector Search instance
        text_splitter: Text splitter for chunking documents
        query: Search query to test with
        with_cleanup: If True, delete all stored documents after search (default: False)

    Returns:
        List[Tuple[Document, float]]: Search results as list of (document, score) tuples

    Note:
        - Prints configuration and progress information
        - Loads all PDFs from BOOKS_DIR
        - Chunks and stores documents with embeddings
        - Performs similarity search with the provided query
        - Optionally cleans up stored documents if with_cleanup=True
    """

    documents = load_documents()

    stored_ids = store_documents(documents=documents, vector_store=vector_store, text_splitter=text_splitter)

    # Test search
    print(f"\nTesting search with following query: {query}...")
    results: List[Tuple[Document, float]] = vector_store.similarity_search_with_score(query=query, k=10)
    print(f"Found {len(results)} results")

    if with_cleanup:
        for _, ids in stored_ids.items():
            vector_store.delete(ids)

    return results


if __name__ == "__main__":
    # Run vectorization when executed directly

    # Create embeddings
    print("Initializing embeddings...")
    embeddings = create_embeddings()

    # Create vector store
    print("Creating MongoDB vector store...")
    store = create_mongo_vector_store(
        mongodb_uri=MONGODB_URI,
        db_name=DB_NAME,
        collection_name=COLLECTION_NAME,
        index_name=MONGODB_INDEX_NAME,
        embeddings=embeddings,
        create_index=True
    )

    # Initialize text splitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=True
    )

    # Perform vectorization
    res = test_vector_search(store, splitter, query="whats a characters starting equipment?")
    for doc, score in res[:3]:
        print(f"  Score: {score:.4f} | {doc.page_content}...")

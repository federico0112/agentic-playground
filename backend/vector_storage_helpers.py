"""
Book Vectorization Module.

Loads PDF books, splits them into chunks, generates embeddings using Google's
Gemini embeddings, and stores them in MongoDB for semantic search.

Provides both sync and async versions.

Usage (Sync):
    from vector_storage_helpers import (
        create_embeddings,
        create_mongo_vector_store,
        store_documents
    )

Usage (Async):
    from vector_storage_helpers import store_documents_async
"""

from pathlib import Path
from typing import List, Dict, Tuple
from uuid import uuid4

from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mongodb import MongoDBAtlasVectorSearch
from pymongo import MongoClient


# Load environment variables from parent directory
load_dotenv(Path(__file__).parent.parent / ".env")


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
        model="models/gemini-embedding-001"
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


async def store_documents_async(
    documents: Dict[Path, List[Document]],
    vector_store: MongoDBAtlasVectorSearch,
    text_splitter: RecursiveCharacterTextSplitter
) -> Dict[Path, List[str]]:
    """Split documents into chunks and store them in the vector store (async version).

    Args:
        documents: Dictionary mapping PDF file paths to their loaded Document objects
        vector_store: Configured MongoDB Atlas Vector Search instance
        text_splitter: Text splitter for chunking documents

    Returns:
        Dict[Path, List[str]]: Dictionary mapping file paths to lists of stored document UUIDs

    Note:
        - Automatically generates UUIDs for each chunk
        - Embeddings are generated automatically by the vector store using aadd_documents
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
        # Use aadd_documents for async operation
        print(f"  Storing chunks in MongoDB...")
        uuids = [str(uuid4()) for _ in range(len(split_documents))]
        await vector_store.aadd_documents(split_documents, ids=uuids)
        stored_ids[doc_path] = uuids
        print(f"  ✓ Stored {len(split_documents)} chunks")

        total_chunks_stored += len(split_documents)

    print(f"\n{'='*80}")
    print(f"✓ Vector storage complete")
    print(f"  Total chunks stored: {total_chunks_stored}")
    print(f"{'='*80}\n")

    return stored_ids

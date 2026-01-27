"""
Custom FastAPI routes for LangGraph server.

This adds the upload endpoint to the LangGraph server, consolidating
the backend into a single server that handles both:
1. Agent endpoints (provided by LangGraph)
2. Custom upload endpoint (defined here)
"""

import asyncio
import logging
import os
import tempfile
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pymongo import MongoClient

from backend.vector_storage_helpers import (
    create_embeddings,
    create_mongo_vector_store,
    store_documents_async
)

# Load environment variables
load_dotenv(Path(__file__).parent.parent / ".env")

# ============================================================================
# CONFIGURATION
# ============================================================================

# Chunking configuration
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100

# MongoDB Configuration
MONGODB_URI = os.getenv(
    "MONGODB_URI",
    "mongodb://localhost:61213/?directConnection=true&serverSelectionTimeoutMS=2000&appName=mongosh+2.6.0"
)
DB_NAME = os.getenv("DB_NAME", "book_search_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "vectorized_documents")
MONGODB_INDEX_NAME = os.getenv("INDEX_NAME", "vector_index")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# ============================================================================
# FASTAPI APP
# ============================================================================

app = FastAPI(
    title="Semantic Search API Extensions",
    description="Custom routes for PDF upload and management"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# SHARED COMPONENTS (Lazy Initialization)
# ============================================================================

_embeddings = None
_vector_store = None
_text_splitter = None


def get_embeddings():
    """Get or initialize embeddings."""
    global _embeddings
    if _embeddings is None:
        logger.info("Initializing Google Generative AI embeddings...")
        _embeddings = create_embeddings()
        logger.info("Embeddings initialized")
    return _embeddings


def get_vector_store():
    """Get or initialize MongoDB vector store."""
    global _vector_store
    if _vector_store is None:
        embeddings = get_embeddings()
        logger.info(f"Connecting to MongoDB: {DB_NAME}/{COLLECTION_NAME}")
        _vector_store = create_mongo_vector_store(MONGODB_URI, DB_NAME, COLLECTION_NAME,
                                                  embeddings=embeddings, index_name=MONGODB_INDEX_NAME)

        # Ensure index exists
        try:
            logger.info(f"Ensuring vector index '{MONGODB_INDEX_NAME}' exists...")
            _vector_store.create_vector_search_index(dimensions=3072)
            logger.info("Vector index created/verified")
        except Exception as e:
            logger.debug(f"Index creation note: {e}")

        logger.info("MongoDB vector store connected")

    return _vector_store


def get_text_splitter():
    """Get or initialize text splitter."""
    global _text_splitter
    if _text_splitter is None:
        logger.info(f"Creating text splitter (chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
        _text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            add_start_index=True,
        )
    return _text_splitter


# ============================================================================
# CUSTOM ROUTES
# ============================================================================

@app.post("/upload")
async def upload_and_vectorize(file: UploadFile):
    """Upload a PDF and vectorize it into MongoDB."""
    logger.info("=" * 50)
    logger.info("Received upload request")
    logger.info(f"  Filename: {file.filename}")
    logger.info(f"  Content-Type: {file.content_type}")

    if not file.filename:
        logger.error("No filename provided")
        raise HTTPException(status_code=400, detail="No filename provided")

    if not file.filename.lower().endswith(".pdf"):
        logger.error(f"Invalid file type: {file.filename}")
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    try:
        # Save to temp file
        logger.info("Reading file content...")
        content = await file.read()
        logger.info(f"  File size: {len(content)} bytes")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            await asyncio.to_thread(tmp.write, content)
            tmp_path = tmp.name
        logger.info(f"  Saved to temp: {tmp_path}")

        # Load PDF
        logger.info("Loading PDF with PyPDFLoader...")
        loader = PyPDFLoader(tmp_path)
        documents = await loader.aload()
        logger.info(f"  Loaded {len(documents)} pages")

        # Fix metadata to use original filename and preserve info
        for doc in documents:
            # Keep original metadata but fix source
            original_metadata = doc.metadata.copy()
            doc.metadata['source'] = file.filename
            doc.metadata['original_filename'] = file.filename
            # Preserve any existing metadata from PDF
            if 'author' in original_metadata:
                doc.metadata['author'] = original_metadata['author']
            if 'title' in original_metadata:
                doc.metadata['title'] = original_metadata['title']
            if 'creator' in original_metadata:
                doc.metadata['creator'] = original_metadata['creator']
        logger.info(f"  Updated source metadata to: {file.filename}")
        logger.info(f"  Sample metadata: {documents[0].metadata if documents else 'N/A'}")

        if not documents:
            logger.error("Could not extract content from PDF")
            raise HTTPException(status_code=400, detail="Could not extract content from PDF")

        # Get components and vectorize
        logger.info("Getting ML components...")
        vector_store = get_vector_store()
        text_splitter = get_text_splitter()

        logger.info("Storing documents (chunking + embedding + MongoDB insert)...")
        stored_ids = await store_documents_async(
            documents={Path(file.filename): documents},
            vector_store=vector_store,
            text_splitter=text_splitter,
        )

        # Cleanup
        await asyncio.to_thread(Path(tmp_path).unlink, missing_ok=True)
        logger.info("  Temp file cleaned up")

        chunk_count = len(stored_ids.get(Path(file.filename), []))
        logger.info(f"SUCCESS: {file.filename}")
        logger.info(f"  Pages: {len(documents)}")
        logger.info(f"  Chunks stored: {chunk_count}")
        logger.info("=" * 50)

        return {
            "status": "ok",
            "filename": file.filename,
            "pages": len(documents),
            "chunks": chunk_count,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"ERROR processing {file.filename}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/sources")
async def get_sources():
    """Get information about available document sources in the vector store."""
    try:
        logger.info("Fetching available sources from MongoDB...")
        client = MongoClient(MONGODB_URI)
        collection = client[DB_NAME][COLLECTION_NAME]

        # Get unique sources
        sources = collection.distinct("source")
        total_chunks = collection.count_documents({})

        logger.info(f"  Found {len(sources)} unique sources")
        logger.info(f"  Total chunks: {total_chunks}")

        return {
            "status": "ok",
            "sources": sources,
            "total_chunks": total_chunks,
            "database": DB_NAME,
            "collection": COLLECTION_NAME
        }
    except Exception as e:
        logger.exception(f"ERROR fetching sources: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/upload/health")
async def upload_health():
    """Health check for upload service."""
    try:
        # Check MongoDB connection
        client = MongoClient(MONGODB_URI)
        client.server_info()
        mongodb_status = "ok"
    except Exception:
        mongodb_status = "error"

    try:
        # Check if embeddings can be initialized
        get_embeddings()
        embeddings_status = "ok"
    except Exception:
        embeddings_status = "error"

    return {
        "status": "ok",
        "service": "upload",
        "mongodb": mongodb_status,
        "embeddings": embeddings_status
    }


# ============================================================================
# STARTUP LOGGING
# ============================================================================

# Log configuration on import (LangGraph manages lifespan, so no @app.on_event)
logger.info("=" * 50)
logger.info("Custom routes initialized")
logger.info(f"  Upload endpoint: /upload")
logger.info(f"  Sources endpoint: /sources")
logger.info(f"  MongoDB: {DB_NAME}/{COLLECTION_NAME}")
logger.info("=" * 50)

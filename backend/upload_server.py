"""Upload server that vectorizes PDFs into MongoDB."""

import logging
import os
import tempfile
from pathlib import Path

from dotenv import load_dotenv
# Load environment variables from parent directory
load_dotenv(Path(__file__).parent.parent / ".env")

# ============================================================================
# CONFIGURATION
# ============================================================================

# Chunking configuration
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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from vector_storage_helpers import (
    create_embeddings,
    create_mongo_vector_store,
    store_documents,
)

app = FastAPI(title="Book Upload & Vectorization Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy initialization
_embeddings = None
_vector_store = None
_text_splitter = None


def get_components():
    """Lazy initialization of ML components."""
    global _embeddings, _vector_store, _text_splitter

    if _embeddings is None:
        logger.info("Initializing Google Generative AI embeddings...")
        _embeddings = create_embeddings()
        logger.info("Embeddings initialized")

    if _vector_store is None:
        logger.info(f"Connecting to MongoDB: {DB_NAME}/{COLLECTION_NAME}")
        _vector_store = create_mongo_vector_store(
            mongodb_uri=MONGODB_URI,
            db_name=DB_NAME,
            collection_name=COLLECTION_NAME,
            index_name=MONGODB_INDEX_NAME,
            embeddings=_embeddings,
            create_index=False,  # We'll create it explicitly below
        )
        logger.info("MongoDB vector store connected")

    # Always ensure index exists before writing
    try:
        logger.info(f"Ensuring vector index '{MONGODB_INDEX_NAME}' exists...")
        _vector_store.create_vector_search_index(dimensions=3072)
        logger.info("Vector index created/verified")
    except Exception as e:
        # Index likely already exists
        logger.debug(f"Index creation note: {e}")

    if _text_splitter is None:
        logger.info(f"Creating text splitter (chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
        _text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            add_start_index=True,
        )

    return _embeddings, _vector_store, _text_splitter


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
            tmp.write(content)
            tmp_path = tmp.name
        logger.info(f"  Saved to temp: {tmp_path}")

        # Load PDF
        logger.info("Loading PDF with PyPDFLoader...")
        loader = PyPDFLoader(tmp_path)
        documents = loader.load()
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
        _, vector_store, text_splitter = get_components()

        logger.info("Storing documents (chunking + embedding + MongoDB insert)...")
        stored_ids = store_documents(
            documents={Path(file.filename): documents},
            vector_store=vector_store,
            text_splitter=text_splitter,
        )

        # Cleanup
        Path(tmp_path).unlink(missing_ok=True)
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


@app.get("/health")
async def health():
    logger.debug("Health check")
    return {"status": "ok"}


@app.on_event("startup")
async def startup():
    logger.info("=" * 50)
    logger.info("Upload server starting...")
    logger.info(f"  MongoDB URI: {MONGODB_URI[:30]}...")
    logger.info(f"  Database: {DB_NAME}")
    logger.info(f"  Collection: {COLLECTION_NAME}")
    logger.info("=" * 50)


if __name__ == "__main__":
    import uvicorn
    logger.info("Starting server on http://0.0.0.0:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)

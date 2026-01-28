#!/usr/bin/env python3
"""
Verify MongoDB Vector Search Index Configuration.

This script checks if the vector search index is properly configured
with source metadata filtering support.
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from pymongo import MongoClient

# Load environment variables
load_dotenv(Path(__file__).parent.parent / ".env")

# MongoDB Configuration
MONGODB_URI = os.getenv(
    "MONGODB_URI",
    "mongodb://localhost:61213/?directConnection=true&serverSelectionTimeoutMS=2000&appName=mongosh+2.6.0"
)
DB_NAME = os.getenv("DB_NAME", "book_search_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "vectorized_documents")
INDEX_NAME = os.getenv("INDEX_NAME", "vector_index")


def verify_index():
    """Verify the vector search index configuration."""
    print("=" * 80)
    print("MongoDB Vector Search Index Verification")
    print("=" * 80)

    try:
        # Connect to MongoDB
        print(f"\n1. Connecting to MongoDB...")
        print(f"   URI: {MONGODB_URI.split('@')[-1] if '@' in MONGODB_URI else MONGODB_URI}")
        client = MongoClient(MONGODB_URI)

        # Test connection
        client.server_info()
        print(f"   ✓ Connected successfully")

        # Check database and collection
        print(f"\n2. Checking database and collection...")
        print(f"   Database: {DB_NAME}")
        print(f"   Collection: {COLLECTION_NAME}")

        db = client[DB_NAME]
        collection = db[COLLECTION_NAME]

        # Count documents
        doc_count = collection.count_documents({})
        print(f"   ✓ Found {doc_count} documents")

        # Get unique sources
        sources = collection.distinct("source")
        print(f"   ✓ Found {len(sources)} unique sources:")
        for source in sources[:10]:  # Show first 10
            print(f"     - {source}")
        if len(sources) > 10:
            print(f"     ... and {len(sources) - 10} more")

        # Check for vector index
        print(f"\n3. Checking vector search indexes...")
        indexes = list(collection.list_search_indexes())

        if not indexes:
            print(f"   ⚠ No search indexes found")
            print(f"\n   To create the index, the application will attempt to create it on startup.")
            print(f"   Or you can manually create it in MongoDB Atlas UI with:")
            print(f"   - Index name: {INDEX_NAME}")
            print(f"   - Dimensions: 768")
            print(f"   - Similarity: cosine")
            print(f"   - Filters: source (as filter)")
            return False

        print(f"   ✓ Found {len(indexes)} search index(es)")

        # Check specific index
        target_index = None
        for idx in indexes:
            print(f"\n   Index: {idx.get('name', 'unnamed')}")
            print(f"     Status: {idx.get('status', 'unknown')}")
            print(f"     Type: {idx.get('type', 'unknown')}")

            if idx.get('name') == INDEX_NAME:
                target_index = idx

                # Check definition
                definition = idx.get('latestDefinition', {})
                mappings = definition.get('mappings', {})

                # Check for dynamic or fields mapping
                if 'dynamic' in mappings:
                    print(f"     ✓ Dynamic mapping enabled")

                if 'fields' in mappings:
                    fields = mappings['fields']
                    print(f"     Fields configuration:")

                    # Check for embedding field
                    if 'embedding' in fields:
                        emb_config = fields['embedding']
                        print(f"       - embedding:")
                        print(f"         Type: {emb_config.get('type', 'unknown')}")
                        print(f"         Dimensions: {emb_config.get('dimensions', 'unknown')}")
                        print(f"         Similarity: {emb_config.get('similarity', 'unknown')}")

                    # Check for source filter
                    if 'source' in fields:
                        src_config = fields['source']
                        print(f"       - source:")
                        print(f"         Type: {src_config.get('type', 'unknown')}")
                        print(f"         ✓ Source metadata filtering enabled")
                    else:
                        print(f"       ⚠ 'source' field not found in index")
                        print(f"       Filtering by source may not work correctly")

        if target_index:
            print(f"\n   ✓ Index '{INDEX_NAME}' found and configured")
            if target_index.get('status') == 'READY':
                print(f"   ✓ Index is READY for queries")
            else:
                print(f"   ℹ Index status: {target_index.get('status')}")
                print(f"   Wait for it to become READY before running queries")

            return True
        else:
            print(f"\n   ⚠ Index '{INDEX_NAME}' not found")
            print(f"   Found indexes: {[idx.get('name') for idx in indexes]}")
            return False

    except Exception as e:
        print(f"\n   ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        print("\n" + "=" * 80)


if __name__ == "__main__":
    success = verify_index()
    sys.exit(0 if success else 1)

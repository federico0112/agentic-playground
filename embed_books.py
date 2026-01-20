#!/usr/bin/env python3
"""
Standalone script to embed all PDF books into MongoDB with Gemini embeddings.

This script processes all PDFs in the books directory, creates summaries for
page intervals, and stores them with vector embeddings in MongoDB for fast
semantic search.

Usage:
    python embed_books.py
    python embed_books.py --force  # Re-embed even if already processed
"""

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langgraph.store.base import IndexConfig
from pypdf import PdfReader
from tqdm import tqdm

from mongodb_vector_store import MongoDBVectorStore

# Load environment variables
load_dotenv()

# Constants
PAGE_INTERVAL = 10
BOOKS_DIR = "books"


def sanitize_filename(filename: str) -> str:
    """Convert filename to valid namespace (remove .pdf extension and replace periods)."""
    name = filename.replace('.pdf', '')
    name = name.replace('.', '_').replace('/', '_').replace('\\', '_')
    return name


def get_book_info(pdf_path: Path) -> dict:
    """Get basic information about a PDF book."""
    try:
        reader = PdfReader(str(pdf_path))
        return {
            'filename': pdf_path.name,
            'path': str(pdf_path),
            'total_pages': len(reader.pages),
            'stem': pdf_path.stem
        }
    except Exception as e:
        print(f"Error reading {pdf_path.name}: {e}")
        return None


def extract_page_interval(pdf_path: Path, start_page: int, interval: int = PAGE_INTERVAL) -> dict:
    """Extract text from a page interval."""
    try:
        reader = PdfReader(str(pdf_path))
        total_pages = len(reader.pages)
        end_page = min(start_page + interval, total_pages)

        pages_text = []
        for page_num in range(start_page, end_page):
            page_text = reader.pages[page_num].extract_text()
            pages_text.append(f"--- Page {page_num} ---\n{page_text}")

        return {
            'start_page': start_page,
            'end_page': end_page - 1,
            'total_pages': total_pages,
            'text': "\n\n".join(pages_text)
        }
    except Exception as e:
        print(f"Error extracting pages {start_page}-{start_page + interval} from {pdf_path.name}: {e}")
        return None


def summarize_text(text: str, llm: ChatGoogleGenerativeAI) -> str:
    """Generate a concise summary of the text using Gemini."""
    try:
        prompt = f"""Summarize the following text concisely but preserve all key information,
concepts, definitions, formulas, and important details. Be thorough but concise.

Text:
{text}

Summary:"""

        response = llm.invoke(prompt)
        return response.content.strip()
    except Exception as e:
        print(f"Error generating summary: {e}")
        # Fallback to truncated text if summarization fails
        return text[:500] + "..." if len(text) > 500 else text


def check_book_status(store: MongoDBVectorStore, filename: str, total_pages: int) -> dict:
    """Check if a book is already fully embedded."""
    namespace_key = sanitize_filename(filename)

    # Search for existing summaries
    items = store.search((namespace_key,))

    coverage = set()
    for item in items:
        try:
            start, end = item.key.split('-')
            coverage.update(range(int(start), int(end) + 1))
        except (ValueError, AttributeError):
            pass

    sorted_coverage = sorted(list(coverage))

    return {
        'has_summaries': len(items) > 0,
        'total_summaries': len(items),
        'last_covered_page': sorted_coverage[-1] if sorted_coverage else -1,
        'total_pages_covered': len(sorted_coverage),
        'is_complete': sorted_coverage[-1] >= (total_pages - 1) if sorted_coverage else False
    }


def embed_book(
    pdf_path: Path,
    store: MongoDBVectorStore,
    llm: ChatGoogleGenerativeAI,
    force: bool = False
) -> bool:
    """
    Embed a single book into MongoDB with summaries and embeddings.

    Args:
        pdf_path: Path to the PDF file
        store: MongoDB vector store instance
        llm: Gemini model for summarization
        force: If True, re-embed even if already complete

    Returns:
        True if successful, False otherwise
    """
    print(f"\n{'='*80}")
    print(f"Processing: {pdf_path.name}")
    print(f"{'='*80}")

    # Get book info
    book_info = get_book_info(pdf_path)
    if not book_info:
        return False

    total_pages = book_info['total_pages']
    filename = book_info['filename']
    namespace_key = sanitize_filename(filename)

    print(f"Total pages: {total_pages}")

    # Check existing status
    status = check_book_status(store, filename, total_pages)

    if status['is_complete'] and not force:
        print(f"✓ Book already fully embedded ({status['total_summaries']} summaries, {status['total_pages_covered']} pages)")
        print("  Use --force to re-embed")
        return True

    if status['has_summaries'] and not force:
        print(f"Resuming from page {status['last_covered_page'] + 1}")
        start_from = ((status['last_covered_page'] + 1) // PAGE_INTERVAL) * PAGE_INTERVAL
    else:
        print("Starting fresh embedding...")
        start_from = 0

    # Process book in intervals
    num_intervals = (total_pages + PAGE_INTERVAL - 1) // PAGE_INTERVAL

    with tqdm(total=num_intervals, desc="Embedding pages", unit="interval") as pbar:
        # Update progress bar to show already processed intervals
        if start_from > 0:
            pbar.update(start_from // PAGE_INTERVAL)

        for start_page in range(start_from, total_pages, PAGE_INTERVAL):
            # Extract page interval
            interval_data = extract_page_interval(pdf_path, start_page, PAGE_INTERVAL)
            if not interval_data:
                print(f"Failed to extract pages {start_page}-{start_page + PAGE_INTERVAL}")
                continue

            # Generate summary
            summary = summarize_text(interval_data['text'], llm)

            # Store with embeddings
            key = f"{interval_data['start_page']}-{interval_data['end_page']}"

            try:
                store.put((namespace_key,), key, {
                    "text": summary,  # Used for generating embeddings
                    "filename": filename,
                    "start_page": interval_data['start_page'],
                    "end_page": interval_data['end_page']
                })
            except Exception as e:
                print(f"Error storing summary for pages {key}: {e}")
                continue

            pbar.update(1)

    print(f"\n✓ Successfully embedded {pdf_path.name}")
    return True


def embed_all_books(
    books_dir: str = BOOKS_DIR,
    connection_string: str = "mongodb://localhost:27017",
    database: str = "book_search_db",
    collection: str = "page_summaries",
    force: bool = False
):
    """
    Embed all PDF books in the books directory into MongoDB.

    Args:
        books_dir: Directory containing PDF files
        connection_string: MongoDB connection string
        database: MongoDB database name
        collection: MongoDB collection name
        force: If True, re-embed books that are already complete
    """
    books_path = Path(books_dir)

    if not books_path.exists():
        print(f"Error: Books directory not found: {books_path}")
        return

    # Find all PDFs
    pdf_files = list(books_path.glob("*.pdf"))

    if not pdf_files:
        print(f"No PDF files found in {books_path}")
        return

    print(f"\n{'='*80}")
    print(f"BOOK EMBEDDING SCRIPT")
    print(f"{'='*80}")
    print(f"Found {len(pdf_files)} PDF file(s) in {books_path}")
    print(f"MongoDB: {connection_string}/{database}/{collection}")
    print(f"Embedding Model: Gemini text-embedding-004")
    print(f"Summarization Model: Gemini 2.5 Flash")
    print(f"Page Interval: {PAGE_INTERVAL} pages")
    print(f"{'='*80}\n")

    # Initialize Gemini embeddings
    print("Initializing Gemini embeddings...")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

    index_config = IndexConfig(
        dims=768,
        embed=embeddings
    )

    # Initialize MongoDB store
    print("Connecting to MongoDB...")
    store = MongoDBVectorStore(
        connection_string=connection_string,
        database=database,
        collection=collection,
        index=index_config
    )

    # Initialize Gemini LLM for summarization
    print("Initializing Gemini LLM...")
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0
    )

    print("\nStarting book embedding process...\n")

    # Process each book
    success_count = 0
    failed_books = []

    for pdf_path in pdf_files:
        try:
            if embed_book(pdf_path, store, llm, force):
                success_count += 1
            else:
                failed_books.append(pdf_path.name)
        except Exception as e:
            print(f"Error processing {pdf_path.name}: {e}")
            failed_books.append(pdf_path.name)

    # Final summary
    print(f"\n{'='*80}")
    print(f"EMBEDDING COMPLETE")
    print(f"{'='*80}")
    print(f"Successfully embedded: {success_count}/{len(pdf_files)} books")

    if failed_books:
        print(f"\nFailed books:")
        for book in failed_books:
            print(f"  - {book}")

    print(f"\nAll embeddings are now stored in MongoDB!")
    print(f"You can now use the agent to search through the books with semantic search.")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Embed all PDF books into MongoDB with Gemini embeddings"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-embed books even if already complete"
    )
    parser.add_argument(
        "--books-dir",
        default=BOOKS_DIR,
        help=f"Directory containing PDF files (default: {BOOKS_DIR})"
    )
    parser.add_argument(
        "--mongo-uri",
        default="mongodb://localhost:27017",
        help="MongoDB connection string (default: mongodb://localhost:27017)"
    )
    parser.add_argument(
        "--database",
        default="book_search_db",
        help="MongoDB database name (default: book_search_db)"
    )
    parser.add_argument(
        "--collection",
        default="book_embedded",
        help="MongoDB collection name (default: book_embedded)"
    )

    args = parser.parse_args()

    try:
        embed_all_books(
            books_dir=args.books_dir,
            connection_string=args.mongo_uri,
            database=args.database,
            collection=args.collection + "_interval_" + PAGE_INTERVAL,
            force=args.force
        )
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Progress has been saved to MongoDB.")
        print("You can resume by running the script again.")
        sys.exit(0)
    except Exception as e:
        print(f"\nFatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

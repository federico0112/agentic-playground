#!/usr/bin/env python3
"""
Semantic Search Agent with LangSmith Integration.

An intelligent agent that performs semantic search across vectorized documents
stored in MongoDB Atlas Vector Search. Integrated with LangSmith for tracing
and monitoring agent behavior.

Purpose:
    - Answer questions about books using semantic search
    - Retrieve relevant passages from vectorized documents
    - Provide context-aware responses based on search results
    - Track agent execution with LangSmith

Usage:
    # Set up LangSmith environment variables:
    export LANGCHAIN_TRACING_V2=true
    export LANGCHAIN_API_KEY=<your-api-key>
    export LANGCHAIN_PROJECT=semantic-search-agent

    # Run the agent:
    python semantic_search_agent.py

    # Or use in interactive mode:
    python semantic_search_agent.py --interactive
"""

import os
import sys
from typing import List, Tuple, Optional

from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.agents import create_agent
from langchain_mongodb import MongoDBAtlasVectorSearch
from pymongo import MongoClient

# Load environment variables
load_dotenv()

# ============================================================================
# LANGSMITH CONFIGURATION
# ============================================================================

# LangSmith will automatically trace if these environment variables are set:
# LANGCHAIN_TRACING_V2=true
# LANGCHAIN_API_KEY=<your-api-key>
# LANGCHAIN_PROJECT=semantic-search-agent

def setup_langsmith():
    """Configure LangSmith tracing.

    Supports both LANGCHAIN_* and LANGSMITH_* environment variable naming conventions.
    # """

    # Ensure LangSmith is properly configured
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
    os.environ["LANGCHAIN_PROJECT"] = "semantic_search_agent"  # Set a project name

    # # Check for standard LangChain naming
    # tracing_enabled = os.getenv("LANGCHAIN_TRACING_V2") == "true"
    #
    # # Fallback to LANGSMITH_* naming if LANGCHAIN_* not set
    # if not tracing_enabled and os.getenv("LANGSMITH_TRACING") == "true":
    #     os.environ["LANGCHAIN_TRACING_V2"] = "true"
    #     tracing_enabled = True
    #
    # # # Set API key from LANGSMITH_API_KEY if LANGCHAIN_API_KEY not set
    # # if not os.getenv("LANGCHAIN_API_KEY") and os.getenv("LANGSMITH_API_KEY"):
    # #     os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
    #
    # # Set project from LANGSMITH_PROJECT if LANGCHAIN_PROJECT not set
    # if not os.getenv("LANGCHAIN_PROJECT") and os.getenv("LANGSMITH_PROJECT"):
    #     os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGSMITH_PROJECT")
    #
    # if not tracing_enabled:
    #     print("⚠ LangSmith tracing not enabled. Set LANGCHAIN_TRACING_V2=true to enable.")
    # else:
    #     print("✓ LangSmith tracing enabled")
    #     print(f"  Project: {os.getenv('LANGCHAIN_PROJECT', 'default')}")
    #     print(f"  API Key: {'*' * 20}{os.getenv('LANGCHAIN_API_KEY', '')[-8:]}")

# ============================================================================
# MONGODB CONFIGURATION
# ============================================================================

MONGODB_URI = os.getenv(
    "MONGODB_URI",
    "mongodb://localhost:61213/?directConnection=true&serverSelectionTimeoutMS=2000&appName=mongosh+2.6.0"
)
DB_NAME = os.getenv("DB_NAME", "book_search_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "vectorized_documents")
INDEX_NAME = os.getenv("INDEX_NAME", "vector_index")
TOP_K = 10  # Number of results to return from semantic search

# ============================================================================
# GLOBAL VARIABLES
# ============================================================================

# Initialize embeddings with rate limiting
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    task_type="retrieval_query"  # Use retrieval_query for search queries
)

# Initialize LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-3-flash-preview",
)


def initialize_vector_store():
    """Initialize MongoDB Atlas Vector Search connection.

    Returns:
        MongoDBAtlasVectorSearch: Connected vector store
    """
    try:
        client = MongoClient(MONGODB_URI)
        collection = client[DB_NAME][COLLECTION_NAME]

        store = MongoDBAtlasVectorSearch(
            collection=collection,
            embedding=embeddings,
            index_name=INDEX_NAME,
            relevance_score_fn="cosine"
        )

        # Test connection
        total_docs = collection.count_documents({})
        print(f"✓ Connected to MongoDB")
        print(f"  Database: {DB_NAME}")
        print(f"  Collection: {COLLECTION_NAME}")
        print(f"  Total Documents: {total_docs}")

        return store

    except Exception as e:
        print(f"❌ Error connecting to MongoDB: {e}")
        sys.exit(1)


# Vector store (initialized in main)
vector_store: Optional[MongoDBAtlasVectorSearch] = initialize_vector_store()

# ============================================================================
# TOOLS
# ============================================================================

@tool(
    "semantic_search",
    description="Search the book collection using semantic similarity. "
                "Returns relevant passages from books that match the query. "
                "Use this to find information about specific topics in the books."
)
def semantic_search(query: str, top_k: int = TOP_K) -> List[Tuple[Document, float]]:
    """Perform semantic search on vectorized documents.

    Args:
        query: The search query (question or topic to search for)
        top_k: Number of results to return (default: 5)

    Returns:
        results: List of tuples where the tuple contains document relevant to query with score of how relevant document is to semantic search
    """
    try:
        # Perform similarity search with scores
        results: List[Tuple[Document, float]] = vector_store.similarity_search_with_score(
            query=query,
            k=top_k
        )

        return results

    except Exception as e:
        print(f"got exception searching query: {e}")
        return []


@tool(
    "get_source_filenames",
    description="Shows total number of files being reference by semantic search."
)
def get_source_filenames() -> str:
    """Get all the different sources semantic agent has access to.

    Returns:
        str: List of sources semantic agent has access to in search.
    """
    try:
        client = MongoClient(MONGODB_URI)
        collection = client[DB_NAME][COLLECTION_NAME]

        # Count total documents
        total_docs = collection.count_documents({})

        # Get unique filenames
        return str(collection.aggregate([{'$group': {'_id': '$source'}}]).to_list())

    except Exception as e:
        return f"Error getting source stats: {str(e)}"


# ============================================================================
# SYSTEM PROMPT
# ============================================================================

SYSTEM_PROMPT = """You are a Semantic Search Assistant for a book collection. Your role is to help users find information across vectorized documents using semantic search.

Your Capabilities:
1. semantic_search(query, top_k) - Search books for relevant content
2. get_collection_stats() - View collection statistics and available books

Workflow:
- When the user asks a question:
   1. Use semantic_search to find relevant passages from question asked
   2. Analyze the search results
   3. If search results don't contain all the information needed, with a new query to find missing info go to step 1.
   4. Synthesize a comprehensive answer from the retrieved content
   5. Cite sources (book name and page number)

- When the user wants to know what's available:
   - Use get_source_filenames to show what files are being referenced

Important Guidelines:
- Always cite your sources with book name and page number
- If results have low relevance scores (<0.5), mention that the information may not be very relevant
- Synthesize information from multiple results when appropriate
- VERY IMPORTANT: Use semantic_search multiple times with different queries if needed to gather comprehensive information

Response Format:
1. Direct answer to the user's question
2. Supporting details from search results
3. Source citations located next to source info, (e.g., "This is a fact. According to [Book Name], page X...")
4. Relevance assessment if scores are low

Remember: You're helping users explore and understand the content of books through semantic search. Focus on providing accurate, well-sourced answers."""

# ============================================================================
# AGENT SETUP
# ============================================================================

agent = create_agent(
    model=llm,
    tools=[semantic_search, get_source_filenames],
    system_prompt=SYSTEM_PROMPT,
)

# ============================================================================
# EXECUTION MODES
# ============================================================================


def run_single_query(agent, query: str):
    """Run a single query through the agent.

    Args:
        agent: The semantic search agent
        query: User's question
    """
    print(f"\n{'='*80}")
    print(f"Query: {query}")
    print(f"{'='*80}\n")

    # Stream agent execution
    for step in agent.stream(
        {"messages": [{"role": "user", "content": query}]},
        stream_mode="values",
    ):
        step["messages"][-1].pretty_print()

    print(f"\n{'='*80}\n")


def run_interactive(agent):
    """Run the agent in interactive mode.

    Args:
        agent: The semantic search agent
    """
    print(f"\n{'='*80}")
    print("INTERACTIVE SEMANTIC SEARCH")
    print(f"{'='*80}")
    print("Ask questions about your books. Type 'exit' or 'quit' to stop.")
    print(f"{'='*80}\n")

    while True:
        try:
            query = input("\nYour question: ").strip()

            if query.lower() in ['exit', 'quit', 'q']:
                print("\nGoodbye!")
                break

            if not query:
                continue

            run_single_query(agent, query)

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}\n")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main entry point for semantic search agent."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Semantic Search Agent with LangSmith Integration"
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Run in interactive mode"
    )
    parser.add_argument(
        "--query", "-q",
        type=str,
        help="Single query to execute"
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show collection statistics and exit"
    )

    args = parser.parse_args()

    # Setup LangSmith
    setup_langsmith()
    print()

    # Initialize vector store
    global vector_store
    vector_store = initialize_vector_store()
    print()

    # Run based on mode
    if args.stats:
        # Just show stats
        result = get_source_filenames.invoke({})
        print(result)
    elif args.interactive:
        # Interactive mode
        run_interactive(agent)
    elif args.query:
        # Single query mode
        run_single_query(agent, args.query)
    else:
        # Default: run example query
        example_query = "What is the role of a dungeon master in D&D?"
        print(f"Running example query (use --interactive for interactive mode)")
        run_single_query(agent, example_query)


if __name__ == "__main__":
    main()
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
from pathlib import Path
from typing import List, Tuple, Optional

from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.agents.middleware import ModelRetryMiddleware
from langchain.agents import create_agent
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain.agents.middleware import SummarizationMiddleware
from pymongo import MongoClient
from pydantic import BaseModel


# Load environment variables from parent directory
load_dotenv(Path(__file__).parent.parent / ".env")


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
    model="gemini-2.5-flash",
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
2. get_source_filenames() - View available sources

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
1. Direct answer to the user's question, in a user readable friendly format.
2. Supporting details from search results
3. Source citation list must not have duplicate sources, same source from different pages dont count as duplicate.
4. Source citations located next to source info but noted with a [1] (index starts at 1) which correspond to the index of the source in source list.
5. Dont overuse citations, have only a couple per source on really important details. Ensure source citation in text map to an index in source list.
5. Output is a structured output containing answer to user questions, list of sources, and how many times semantic_search was called.

Remember: You're helping users explore and understand the content of books through semantic search. Focus on providing accurate, well-sourced answers."""

# ============================================================================
# AGENT SETUP
# ============================================================================


class SearchResults(BaseModel):
    search_result_message: str
    times_semantic_search_tool_called: int
    sources: List[str]


agent = create_agent(
    model=llm,
    tools=[semantic_search, get_source_filenames],
    system_prompt=SYSTEM_PROMPT,
    middleware=[
        SummarizationMiddleware(
            model=llm,
            trigger=("fraction", 0.8),
            keep=("fraction", 0.3),
        ),
        ModelRetryMiddleware(
            max_retries=3,
            backoff_factor=2.0,
            initial_delay=2.0,
        ),
    ],
    response_format=SearchResults
)

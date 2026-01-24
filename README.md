# Semantic Search Agent

A semantic search system that vectorizes PDF documents and enables intelligent question-answering through an AI agent powered by LangChain and Google's Gemini models.

## Architecture

The project consists of three main components:

1. **Upload Server (FastAPI)** - Handles PDF uploads and vectorization into MongoDB
2. **Semantic Search Agent (LangGraph)** - AI agent that performs semantic search and answers questions
3. **React UI** - Web interface for uploading documents and querying the agent

## Prerequisites

- Python 3.11-3.13
- Node.js and npm
- MongoDB (local or Atlas)
- Google API Key (for Gemini embeddings and LLM)

## Setup

### 1. Clone and Navigate to Project

```bash
cd agentic-playground
```

### 2. Environment Variables

Copy the example environment file and fill in your actual values:

```bash
cp .env.example .env
```

Then edit `.env` and add your API keys:

**Required:**
- `GOOGLE_API_KEY` - Get from [Google AI Studio](https://makersuite.google.com/app/apikey)

**Optional:**
- `LANGCHAIN_API_KEY` / `LANGSMITH_API_KEY` - For tracing with [LangSmith](https://smith.langchain.com/)

See `.env.example` for all available configuration options.

### 3. Install Python Dependencies

Using pip:
```bash
pip install -e .
```

Or using uv (recommended):
```bash
uv sync
```

### 4. Install Frontend Dependencies

```bash
cd semantic-search-ui
npm install
cd ..
```

### 5. Start MongoDB

Ensure MongoDB is running on the URI specified in your `.env` file. If using a local instance:

```bash
# Using MongoDB installed locally
mongod --port 61213

# Or using Docker
docker run -d -p 61213:27017 --name mongodb mongo:latest
```

## Running the Application

You need to start three servers in separate terminal windows:

### Terminal 1: Upload Server (FastAPI)

The upload server handles PDF file uploads and vectorization.

```bash
cd backend/
uvicorn upload_server:app --port 8000
```

The server will start on `http://localhost:8000`

**Endpoints:**
- `POST /upload` - Upload and vectorize PDF files
- `GET /health` - Health check

### Terminal 2: Semantic Search Agent (LangGraph)

The LangGraph server runs the AI agent for semantic search.

```bash
# From project root
langgraph dev
```

The server will start on `http://localhost:2024`

**Configuration:**
- Defined in `langgraph.json`
- Agent implementation: `backend/semantic_search_agent.py:agent`

### Terminal 3: React UI

The web interface for interacting with the system.

```bash
# From semantic-search-ui directory
cd semantic-search-ui
npm start
```

The UI will open automatically at `http://localhost:3000`

## Usage

1. **Upload Documents**
   - Navigate to `http://localhost:3000` in your browser
   - Use the upload interface to select and upload PDF files
   - The system will chunk, embed, and store the documents in MongoDB

2. **Ask Questions**
   - Enter your question in the search interface
   - The AI agent will perform semantic search across uploaded documents
   - Receive synthesized answers with source citations

## Features

### Upload Server
- PDF parsing and text extraction
- Automatic chunking (1000 chars, 100 overlap)
- Google Generative AI embeddings (gemini-embedding-001)
- MongoDB Atlas Vector Search integration
- CORS enabled for web access

### Semantic Search Agent
- Powered by Gemini 2.5 Flash
- Tools:
  - `semantic_search` - Find relevant passages using vector similarity
  - `get_source_filenames` - List available documents
- Multi-query capability for comprehensive answers
- Source citation with page numbers
- Structured JSON responses

### React UI
- File upload with drag-and-drop
- Real-time search interface
- Markdown rendering with math support (KaTeX)
- Source citation display
- Responsive design

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest backend/tests/test_semantic_search_agent.py

# Run integration tests only
pytest -m integration
```

### Code Quality

```bash
# Run linter
ruff check .

# Auto-fix issues
ruff check --fix .

# Type checking
mypy backend/
```

## Project Structure

```
agentic-playground/
├── backend/
│   ├── semantic_search_agent.py    # LangGraph agent implementation
│   ├── upload_server.py            # FastAPI upload server
│   ├── vector_storage_helpers.py   # MongoDB vector store utilities
│   └── tests/                      # Backend tests
├── semantic-search-ui/
│   ├── src/
│   │   ├── App.js                  # Main React component
│   │   └── App.css                 # Styles
│   ├── public/                     # Static assets
│   └── package.json                # Node dependencies
├── .env                            # Environment variables (create this)
├── langgraph.json                  # LangGraph configuration
├── pyproject.toml                  # Python dependencies
└── README.md                       # This file
```

## Troubleshooting

### MongoDB Connection Issues
- Verify MongoDB is running: `mongosh --port 61213`
- Check the `MONGODB_URI` in your `.env` file
- Ensure network connectivity and firewall settings

### Upload Server Errors
- Verify Google API key is valid and has Generative AI API enabled
- Check MongoDB connection and index creation
- Review logs for specific error messages

### LangGraph Agent Issues
- Ensure `.env` file is in the project root
- Verify `langgraph.json` paths are correct
- Check LangSmith configuration if using tracing

### UI Connection Issues
- Confirm all three servers are running
- Check CORS settings if accessing from different origin
- Verify API URLs in `App.js` match your server ports

## API Reference

### Upload Server (Port 8000)

**POST /upload**
```bash
curl -X POST http://localhost:8000/upload \
  -F "file=@path/to/document.pdf"
```

Response:
```json
{
  "status": "ok",
  "filename": "document.pdf",
  "pages": 10,
  "chunks": 45
}
```

**GET /health**
```bash
curl http://localhost:8000/health
```

### LangGraph Agent (Port 2024)

The LangGraph API provides standard endpoints for agent interaction. See [LangGraph documentation](https://langchain-ai.github.io/langgraph/cloud/) for details.

## License

This project is for educational and research purposes.

## Contributing

This is a personal project. For issues or suggestions, please create an issue in the repository.

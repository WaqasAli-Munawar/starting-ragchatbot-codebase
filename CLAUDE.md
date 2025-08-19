# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Running the Application
```bash
# Quick start (recommended)
chmod +x run.sh && ./run.sh

# Manual start
cd backend && uv run uvicorn app:app --reload --port 8000
```

### Setup Commands
```bash
# Install dependencies 
uv sync

# Set up environment
cp .env.example .env
# Then edit .env with your ANTHROPIC_API_KEY
```

### Development Server
- Web Interface: http://localhost:8000
- API Documentation: http://localhost:8000/docs
- Server runs with auto-reload enabled

## Architecture Overview

This is a **Retrieval-Augmented Generation (RAG) system** for querying course materials with AI-powered responses. The system follows a tool-based architecture where Claude AI decides when to search the vector database versus using general knowledge.

### Core Architecture Pattern

**Query Flow**: Frontend → FastAPI → RAG System → AI Generator (with tools) → Vector Store → Response
- Frontend sends HTTP POST to `/api/query`  
- RAG system orchestrates components and maintains session state
- AI Generator uses Claude API with tool definitions for course search
- Claude decides whether to invoke `search_course_content` tool based on query type
- Vector store performs semantic search using ChromaDB and sentence-transformers
- Response flows back with sources and session continuity

### Key Components Integration

**RAG System (`rag_system.py`)**: Central orchestrator that coordinates all components
- Initializes and connects: DocumentProcessor, VectorStore, AIGenerator, SessionManager, ToolManager
- Handles document ingestion from `/docs` folder at startup
- Manages query processing with session-aware conversation history

**Tool-Based Search Architecture**: 
- `CourseSearchTool` in `search_tools.py` provides semantic search capability to Claude
- Claude autonomously decides when to search vs. answer from knowledge
- Supports course name filtering and lesson-specific queries
- Tool results include source attribution for transparency

**Vector Storage Strategy**:
- ChromaDB with sentence-transformers embeddings (`all-MiniLM-L6-v2`)
- Course documents chunked into 800-character segments with 100-character overlap
- Dual storage: course metadata + content chunks for optimal retrieval
- Automatic deduplication prevents re-processing existing courses

**Session Management**:
- Maintains conversation context across queries using session IDs
- Configurable history length (default: 2 exchanges)
- Frontend automatically creates sessions and maintains continuity

## Configuration System

**Environment Variables** (`.env` file):
- `ANTHROPIC_API_KEY`: Required for Claude API access
- All other settings use sensible defaults

**Key Configuration Parameters** (`backend/config.py`):
- `ANTHROPIC_MODEL`: "claude-sonnet-4-20250514" 
- `EMBEDDING_MODEL`: "all-MiniLM-L6-v2"
- `CHUNK_SIZE`: 800 characters for document segments
- `CHUNK_OVERLAP`: 100 characters between chunks
- `MAX_RESULTS`: 5 search results returned per query
- `MAX_HISTORY`: 2 conversation exchanges remembered
- `CHROMA_PATH`: "./chroma_db" for vector storage

## Document Processing

**Supported Formats**: .txt, .pdf, .docx files in `/docs` folder

**Document Structure Expected**:
```
Course Title: [Title]
Course Link: [Optional URL]
Course Instructor: [Optional Name]

Lesson X: [Lesson Title]
Lesson Link: [Optional URL]
[Lesson content...]
```

**Processing Behavior**:
- Documents automatically loaded at server startup
- Existing courses detected and skipped to prevent duplicates
- Chunking preserves lesson boundaries when possible
- Course metadata and content stored separately for optimal search

## Database and Storage

**ChromaDB Vector Database**:
- Persistent storage in `./chroma_db/` directory
- Sentence-transformer embeddings for semantic similarity
- Collections: course metadata + content chunks
- Automatic cleanup and deduplication

**No traditional database**: System uses ChromaDB for all persistence needs

## API Endpoints

**Core Endpoints**:
- `POST /api/query`: Process user queries, returns `{answer, sources, session_id}`
- `GET /api/courses`: Get course analytics `{total_courses, course_titles}`

**Frontend Serving**:
- Static files served from `/frontend` directory
- Development headers (no-cache) applied automatically

## Frontend Architecture 

**Technology Stack**: Vanilla HTML/CSS/JavaScript with marked.js for markdown parsing

**Key Frontend Features**:
- Session-aware chat interface with conversation continuity  
- Real-time loading indicators and error handling
- Course statistics sidebar with collapsible sections
- Suggested questions for user guidance
- Markdown response rendering with collapsible source attribution

## Data Models

**Core Models** (`backend/models.py`):
- `Course`: Contains title, instructor, lessons list, optional course_link
- `Lesson`: Contains lesson_number, title, optional lesson_link  
- `CourseChunk`: Text segments with course_title, lesson_number, chunk_index, content

**API Models** (Pydantic):
- `QueryRequest`: {query, session_id}
- `QueryResponse`: {answer, sources, session_id}
- `CourseStats`: {total_courses, course_titles}

## Tool System

**Tool Architecture**: Extensible system supporting multiple search tools
- `ToolManager` coordinates available tools and execution
- `CourseSearchTool` provides semantic search with course/lesson filtering
- Claude receives tool definitions and autonomously decides when to use them
- Tool results include source tracking for response attribution

## Development Patterns

**Component Isolation**: Each backend module handles a single responsibility
- Document processing, vector storage, AI generation, session management decoupled
- Configuration centralized in `config.py` with environment variable support
- Error handling and logging at component boundaries

**Frontend-Backend Communication**: 
- RESTful API with JSON requests/responses
- Session management via optional session_id parameter
- CORS enabled for development with all origins allowed

**Adding New Document Types**: Extend `DocumentProcessor.read_file()` method
**Adding New Tools**: Implement `Tool` abstract class and register with `ToolManager`
**Modifying Search**: Adjust parameters in `config.py` or extend `CourseSearchTool`
- use uv to run python files or add any dependencies
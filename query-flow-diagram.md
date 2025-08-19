# RAG System Query Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                FRONTEND                                     │
│                            (script.js)                                     │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                              1. User Input
                           ┌─────────────────┐
                           │ sendMessage()   │
                           │ - Capture query │
                           │ - Disable UI    │
                           │ - Show loading  │
                           └─────────────────┘
                                    │
                              2. HTTP POST
                           /api/query
                           {query, session_id}
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                             BACKEND API                                     │
│                              (app.py)                                      │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                           3. API Route Handler
                        ┌─────────────────────────┐
                        │ @app.post("/api/query") │
                        │ - Create session        │
                        │ - Call RAG system       │
                        └─────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            RAG SYSTEM                                       │
│                          (rag_system.py)                                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                            4. Query Processing
                        ┌─────────────────────────┐
                        │ query() method          │
                        │ - Get conversation      │
                        │   history               │
                        │ - Build prompt          │
                        └─────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          AI GENERATOR                                       │
│                        (ai_generator.py)                                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                        5. Claude API Call with Tools
                        ┌─────────────────────────┐
                        │ generate_response()     │
                        │ - System prompt         │
                        │ - Tool definitions      │
                        │ - Conversation history  │
                        └─────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    │                               │
            6a. General Knowledge          6b. Course-Specific Query
                 Answer                          │
                    │                           ▼
                    │               ┌─────────────────────────┐
                    │               │    TOOL EXECUTION       │
                    │               │   (search_tools.py)     │
                    │               └─────────────────────────┘
                    │                           │
                    │                  7. CourseSearchTool
                    │               ┌─────────────────────────┐
                    │               │ search_course_content   │
                    │               │ - Query vector store    │
                    │               │ - Filter by course/     │
                    │               │   lesson                │
                    │               └─────────────────────────┘
                    │                           │
                    │                           ▼
                    │               ┌─────────────────────────┐
                    │               │    VECTOR STORE         │
                    │               │   (vector_store.py)     │
                    │               │ - ChromaDB search       │
                    │               │ - Return relevant       │
                    │               │   chunks + sources      │
                    │               └─────────────────────────┘
                    │                           │
                    └───────────────┬───────────┘
                                    │
                        8. Claude Synthesizes Response
                        ┌─────────────────────────┐
                        │ - Combine search        │
                        │   results with prompt   │
                        │ - Generate final answer │
                        └─────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          RESPONSE FLOW                                      │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                         9. Back to RAG System
                        ┌─────────────────────────┐
                        │ - Extract sources       │
                        │ - Update session        │
                        │ - Return (answer,       │
                        │   sources)              │
                        └─────────────────────────┘
                                    │
                                    ▼
                         10. Back to API Handler
                        ┌─────────────────────────┐
                        │ QueryResponse           │
                        │ {answer, sources,       │
                        │  session_id}            │
                        └─────────────────────────┘
                                    │
                              11. HTTP Response
                                 JSON
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         FRONTEND RESPONSE                                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                        12. UI Update
                        ┌─────────────────────────┐
                        │ - Remove loading        │
                        │ - Parse markdown        │
                        │ - Display answer        │
                        │ - Show sources          │
                        │ - Re-enable input       │
                        └─────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                        SUPPORTING COMPONENTS                                │
├─────────────────────────────────────────────────────────────────────────────┤
│ Session Manager: Tracks conversation history per session                   │
│ Document Processor: Chunks course documents into searchable pieces         │
│ Vector Store: ChromaDB with sentence-transformers embeddings               │
│ Tool Manager: Manages available tools and their execution                  │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Key Data Flow Points:

1. **Frontend → Backend**: HTTP POST with query and session_id
2. **RAG Orchestration**: Coordinates all components
3. **Tool-Based Search**: Claude decides when to search vs. use general knowledge  
4. **Vector Search**: Semantic search in ChromaDB for relevant course content
5. **Response Synthesis**: Claude combines search results with query context
6. **Session Continuity**: Conversation history maintained across queries
7. **Frontend Rendering**: Markdown parsing and source attribution display
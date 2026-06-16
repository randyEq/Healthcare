# Architecture

## Directory Structure
```
/workspace/
├── app/
│   ├── main.py              # FastAPI + WebSocket server
│   ├── graph/
│   │   ├── workflow.py      # LangGraph StateGraph
│   │   ├── nodes.py         # Node wrapper functions
│   │   └── state.py         # State TypedDict
│   ├── agents/
│   │   ├── planner.py       # Planner Agent
│   │   ├── rag.py           # RAG Agent
│   │   ├── analyst.py       # Cost Analysis Agent
│   │   └── summary.py       # Summary Agent
│   ├── memory/
│   │   └── conversation.py  # Conversation memory
│   ├── retrieval/
│   │   └── retriever.py     # FAISS retriever
│   ├── config.py            # Configuration
│   └── prompts.py           # Prompt templates
├── ingestion/
│   ├── ingest.py            # FAISS ingestion pipeline
│   └── data/
│       └── medical_knowledge.json
├── templates/
│   └── index.html           # Chat UI
├── static/
│   ├── style.css
│   └── app.js
├── vector_db/               # FAISS index storage
├── requirements.txt
└── run.py                   # Entry point
```

## Data Flow
1. User sends message via WebSocket
2. Message enters LangGraph workflow state
3. Planner Agent evaluates completeness
4. If insufficient → generate follow-up, pause workflow, return question
5. If sufficient → RAG retrieves from FAISS, Analyst evaluates, Summary formats
6. Response streamed back via WebSocket

## Background Service
- Uvicorn running FastAPI on port 8000
- Caddy reverse proxy at / → :8000

## Ports
- 8000: FastAPI/Uvicorn

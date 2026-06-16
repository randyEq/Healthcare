# Task: Healthcare CDSS Implementation

## Goal
Build a complete LangGraph multi-agent Clinical Decision Support System with Azure OpenAI and FAISS.

## Acceptance Criteria
- [ ] Separate ingestion pipeline creates and saves FAISS index
- [ ] Planner Agent analyzes input and routes correctly
- [ ] RAG Agent retrieves from FAISS and performs clinical reasoning
- [ ] Cost Effective Analysis Agent evaluates treatments
- [ ] Summary Agent consolidates into structured patient-friendly output
- [ ] Human-in-the-loop: follow-up questions when info insufficient
- [ ] Conversation memory maintained across turns
- [ ] FastAPI WebSocket chat interface working
- [ ] Chat UI renders messages and follow-up questions
- [ ] Background service runs production command
- [ ] Caddy proxy routes traffic to FastAPI
- [ ] No hardcoded secrets or URLs in source
- [ ] All env vars through backend env keys

## Files to Create
- app/main.py, app/config.py, app/prompts.py
- app/graph/workflow.py, app/graph/nodes.py, app/graph/state.py
- app/agents/planner.py, app/agents/rag.py, app/agents/analyst.py, app/agents/summary.py
- app/memory/conversation.py, app/retrieval/retriever.py
- ingestion/ingest.py, ingestion/data/medical_knowledge.json
- templates/index.html, static/style.css, static/app.js
- requirements.txt, run.py

## Tests
- Unit tests per agent (mocked LLM)
- Integration test: full workflow
- Browser test: chat UI

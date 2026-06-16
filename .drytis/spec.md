# Healthcare CDSS - Technical Spec

## Overview
AI-powered Clinical Decision Support System using LangGraph multi-agent architecture.
Assists users with symptom analysis, diagnosis, treatment, medication safety, and clinical guidance.

## Tech Stack
- Python 3, FastAPI, LangGraph, LangChain
- Azure OpenAI-compatible LLM (via Drytis OpenAI gateway)
- FAISS vector DB for medical knowledge retrieval
- WebSocket for real-time chat
- MySQL for session/conversation persistence

## Agents
1. **Planner Agent** — Analyzes input, routes workflow, checks info sufficiency, generates follow-up questions
2. **RAG Agent (Patient/Diagnosis)** — Retrieves medical knowledge from FAISS, clinical reasoning, differential diagnoses
3. **Cost Effective Analysis Agent** — Treatment cost-effectiveness, medication safety, drug interactions
4. **Summary Agent** — Consolidates findings into patient-friendly structured response

## Workflow
```
User Input → Planner → [Insufficient?] → Follow-up (HITL pause)
                      → [Sufficient]   → RAG Agent → Analyst → Summary → Response
```

## Key Features
- Human-in-the-loop: pauses for follow-up questions when info insufficient
- Conversation memory across turns
- Separate FAISS ingestion pipeline
- Structured patient-friendly output

"""FastAPI application for the Healthcare CDSS.

Provides:
  - GET  /          → Chat UI
  - WS   /ws        → WebSocket for real-time chat
  - GET  /health    → Health check
  - POST /api/chat  → REST fallback for chat (non-WebSocket)
"""
from __future__ import annotations

import json
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
from loguru import logger

from app.config import settings
from app.graph.workflow import build_workflow
from app.memory.conversation import memory


# ── Lifecycle ──
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("[App] Starting Healthcare CDSS...")
    # Pre-build the workflow graph (agents are instantiated at module load)
    app.state.workflow = build_workflow()
    logger.info("[App] LangGraph workflow ready")
    yield
    logger.info("[App] Shutdown complete")


# ── App ──
app = FastAPI(title="Healthcare CDSS", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="/workspace/static"), name="static")
templates = Jinja2Templates(directory="/workspace/templates")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Serve the chat UI."""
    return templates.TemplateResponse(request=request, name="index.html")


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "service": "Healthcare CDSS"}


async def _run_workflow(user_input: str, session_id: str) -> dict:
    """Run the LangGraph workflow for a single user message.

    Manages conversation memory and feeds it as context to the graph.
    """
    # Store user message
    memory.add_message(session_id, "user", user_input)
    history = memory.get_formatted_history(session_id)

    # Build initial state
    initial_state = {
        "user_input": user_input,
        "session_id": session_id,
        "conversation_history": history,
        "awaiting_user_response": False,
    }

    # Execute the graph
    workflow = app.state.workflow
    result = await workflow.ainvoke(initial_state)

    final_response = result.get("final_response", "I was unable to process your request.")

    # Store assistant response
    memory.add_message(session_id, "assistant", final_response)

    # Determine if we're awaiting follow-up
    awaiting = result.get("awaiting_user_response", False)

    return {
        "response": final_response,
        "awaiting_followup": awaiting,
        "completeness_score": result.get("completeness_score", 100),
    }


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time chat."""
    await websocket.accept()
    session_id = str(uuid.uuid4())
    logger.info(f"[WS] Session {session_id} connected")

    try:
        while True:
            data = await websocket.receive_text()
            try:
                msg = json.loads(data)
                user_input = msg.get("message", "").strip()
            except json.JSONDecodeError:
                user_input = data.strip()

            if not user_input:
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": "Please enter a message.",
                }))
                continue

            # Send "processing" indicator
            await websocket.send_text(json.dumps({
                "type": "processing",
                "message": "Analyzing your symptoms...",
            }))

            try:
                result = await _run_workflow(user_input, session_id)
                await websocket.send_text(json.dumps({
                    "type": "response",
                    **result,
                }))
            except Exception as e:
                logger.error(f"[WS] Workflow error: {e}")
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": f"An error occurred while processing your request. Please try again.",
                }))

    except WebSocketDisconnect:
        logger.info(f"[WS] Session {session_id} disconnected")
    except Exception as e:
        logger.error(f"[WS] Unexpected error: {e}")


@app.post("/api/chat")
async def chat_api(payload: dict):
    """REST fallback endpoint for chat (non-WebSocket)."""
    user_input = payload.get("message", "").strip()
    session_id = payload.get("session_id", str(uuid.uuid4()))

    if not user_input:
        return JSONResponse({"error": "message is required"}, status_code=400)

    try:
        result = await _run_workflow(user_input, session_id)
        return result
    except Exception as e:
        logger.error(f"[API] Error: {e}")
        return JSONResponse(
            {"error": "An error occurred while processing your request."},
            status_code=500,
        )

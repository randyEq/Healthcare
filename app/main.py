"""FastAPI application for the Healthcare CDSS.

Provides:
  - GET  /          → Chat UI
  - WS   /ws        → WebSocket for real-time chat
  - GET  /health    → Health check
  - POST /api/chat  → REST fallback for chat (non-WebSocket)
"""

from __future__ import annotations

import json
import secrets
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
from loguru import logger

from app.config import settings
from app.graph.workflow import build_workflow
from app.guardrails import apply_output_guardrails, screen_input
from app.memory.conversation import memory
from app.sql_tool import (
    database_config_summary,
    execute_sql,
    execute_write,
    is_database_configured,
    test_database_connection,
)


# ── Lifecycle ──
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("[App] Starting Healthcare CDSS...")
    if is_database_configured():
        logger.info(
            "[App] Database settings detected target={}",
            database_config_summary(),
        )
        if test_database_connection():
            logger.info("[App] Database connection ready")
        else:
            logger.error("[App] Database connection failed; patient login may fail")
    else:
        logger.warning("[App] Database settings incomplete; skipping DB startup check")

    # Pre-build the workflow graph (agents are instantiated at module load)
    app.state.workflow = build_workflow()
    logger.info("[App] LangGraph workflow ready")
    yield
    logger.info("[App] Shutdown complete")


# ── App ──
app = FastAPI(title="Healthcare CDSS", lifespan=lifespan)

# Resolve project-relative directories to support different workspaces/OS
_BASE_DIR = Path(__file__).resolve().parent.parent
_STATIC_DIR = str(_BASE_DIR / "static")
_TEMPLATES_DIR = str(_BASE_DIR / "templates")

app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")
templates = Jinja2Templates(directory=_TEMPLATES_DIR)
patient_sessions: dict[str, dict[str, Any]] = {}


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Serve the chat UI."""
    return templates.TemplateResponse(request=request, name="index.html")


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "service": "Healthcare CDSS"}


def _error(message: str, status_code: int = 400) -> JSONResponse:
    return JSONResponse({"error": message}, status_code=status_code)


def _session_for_token(token: str | None) -> dict[str, Any] | None:
    if not token:
        return None
    return patient_sessions.get(token)


def _patient_history(patient_id: int) -> list[dict[str, Any]]:
    return execute_sql(
        """
        SELECT
            pd.patientDiseaseId AS patient_disease_id,
            pd.patientId AS patient_id,
            d.diseaseId AS disease_id,
            d.disease_name,
            d.severity_group,
            d.severity_level,
            d.common_symptoms,
            d.triage_recommendation,
            pd.multiDiagnosis AS multi_diagnosis
        FROM patient_disease pd
        INNER JOIN disease d ON d.diseaseId = pd.diseaseId
        WHERE pd.patientId = :patient_id
        ORDER BY
            CASE d.severity_group
                WHEN 'High' THEN 1
                WHEN 'Medium' THEN 2
                WHEN 'Low' THEN 3
                ELSE 4
            END,
            d.disease_name
        """,
        {"patient_id": patient_id},
    )


def _patient_context(patient_id: int) -> str:
    history = _patient_history(patient_id)
    if not history:
        return "Patient disease history: none recorded."

    lines = ["Patient disease history from database:"]
    for row in history:
        lines.append(
            "- {name} ({severity}; triage: {triage})".format(
                name=row.get("disease_name", "Unknown"),
                severity=row.get("severity_level") or row.get("severity_group") or "n/a",
                triage=row.get("triage_recommendation") or "n/a",
            )
        )
    return "\n".join(lines)


@app.post("/api/patient/login")
async def patient_login(payload: dict):
    """Authenticate a patient and return their disease history."""
    identifier = str(payload.get("identifier", "")).strip()
    password = str(payload.get("password", "")).strip()

    if not identifier or not password:
        return _error("Username/email and password are required.")

    try:
        rows = execute_sql(
            """
            SELECT
                l.loginId AS login_id,
                l.patientId AS patient_id,
                l.username,
                l.email,
                p.patientName AS patient_name
            FROM `login` l
            INNER JOIN patient p ON p.patientId = l.patientId
            WHERE (l.username = :identifier OR l.email = :identifier)
                AND l.passwordHash = :password
                AND l.isActive = 1
            LIMIT 1
            """,
            {"identifier": identifier, "password": password},
        )
    except Exception as exc:
        logger.error(f"[PatientLogin] Database error: {exc}")
        return _error("Unable to connect to the patient database.", 500)

    if not rows:
        return _error("Invalid credentials or inactive login.", 401)

    user = rows[0]
    token = secrets.token_urlsafe(32)
    patient_sessions[token] = {
        "patient_id": int(user["patient_id"]),
        "patient_name": user["patient_name"],
        "username": user["username"],
        "email": user["email"],
    }

    try:
        execute_write(
            "UPDATE `login` SET lastLogin = NOW() WHERE loginId = :login_id",
            {"login_id": user["login_id"]},
        )
    except Exception as exc:
        logger.warning(f"[PatientLogin] Could not update lastLogin: {exc}")

    return {
        "token": token,
        "patient": patient_sessions[token],
        "history": _patient_history(int(user["patient_id"])),
    }


@app.post("/api/patient/logout")
async def patient_logout(payload: dict):
    """End a patient UI session."""
    token = str(payload.get("token", "")).strip()
    patient_sessions.pop(token, None)
    return {"ok": True}


@app.get("/api/patient/history")
async def get_patient_history(token: str):
    """Return the logged-in patient's disease history."""
    session = _session_for_token(token)
    if not session:
        return _error("Please log in again.", 401)
    return {"history": _patient_history(int(session["patient_id"]))}


@app.get("/api/diseases")
async def get_diseases(token: str):
    """Return disease choices for patient history maintenance."""
    if not _session_for_token(token):
        return _error("Please log in again.", 401)

    rows = execute_sql(
        """
        SELECT
            diseaseId AS disease_id,
            disease_name,
            severity_group,
            severity_level,
            triage_recommendation
        FROM disease
        ORDER BY
            CASE severity_group
                WHEN 'High' THEN 1
                WHEN 'Medium' THEN 2
                WHEN 'Low' THEN 3
                ELSE 4
            END,
            disease_name
        """
    )
    return {"diseases": rows}


@app.post("/api/patient/history")
async def add_patient_history(payload: dict):
    """Add a disease row to the logged-in patient's history."""
    session = _session_for_token(str(payload.get("token", "")).strip())
    if not session:
        return _error("Please log in again.", 401)

    try:
        disease_id = int(payload.get("disease_id"))
        multi_diagnosis = 1 if payload.get("multi_diagnosis", True) else 0
    except (TypeError, ValueError):
        return _error("A valid disease is required.")

    patient_id = int(session["patient_id"])
    existing = execute_sql(
        """
        SELECT patientDiseaseId
        FROM patient_disease
        WHERE patientId = :patient_id AND diseaseId = :disease_id
        LIMIT 1
        """,
        {"patient_id": patient_id, "disease_id": disease_id},
    )
    if existing:
        return _error("That condition is already in this patient's history.", 409)

    execute_write(
        """
        INSERT INTO patient_disease (patientId, diseaseId, multiDiagnosis)
        VALUES (:patient_id, :disease_id, :multi_diagnosis)
        """,
        {
            "patient_id": patient_id,
            "disease_id": disease_id,
            "multi_diagnosis": multi_diagnosis,
        },
    )
    return {"history": _patient_history(patient_id)}


@app.post("/api/patient/history/delete")
async def delete_patient_history(payload: dict):
    """Remove a disease row from the logged-in patient's history."""
    session = _session_for_token(str(payload.get("token", "")).strip())
    if not session:
        return _error("Please log in again.", 401)

    try:
        history_id = int(payload.get("patient_disease_id"))
    except (TypeError, ValueError):
        return _error("A valid history row is required.")

    patient_id = int(session["patient_id"])
    deleted = execute_write(
        """
        DELETE FROM patient_disease
        WHERE patientDiseaseId = :history_id AND patientId = :patient_id
        """,
        {"history_id": history_id, "patient_id": patient_id},
    )
    if not deleted:
        return _error("History row was not found.", 404)

    return {"history": _patient_history(patient_id)}


async def _run_workflow(
    user_input: str,
    session_id: str,
    patient_token: str | None = None,
    emergency_detected: bool = False,
) -> dict:
    """Run the LangGraph workflow for a single user message.

    Manages conversation memory and feeds it as context to the graph.
    """
    # Store user message
    memory.add_message(session_id, "user", user_input)
    history = memory.get_formatted_history(session_id)
    patient_session = _session_for_token(patient_token)
    if patient_session:
        history = (
            "Logged-in patient disease history loaded from database.\n"
            f"{_patient_context(int(patient_session['patient_id']))}\n\n"
            f"{history}"
        )

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

    final_response = result.get(
        "final_response", "I was unable to process your request."
    )
    final_response = apply_output_guardrails(final_response, emergency_detected)

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
                patient_token = str(msg.get("patient_token", "")).strip() or None
            except json.JSONDecodeError:
                user_input = data.strip()
                patient_token = None

            if not user_input:
                await websocket.send_text(
                    json.dumps(
                        {
                            "type": "error",
                            "message": "Please enter a message.",
                        }
                    )
                )
                continue

            guardrail = screen_input(user_input)
            if not guardrail.allowed:
                logger.warning(
                    "[Guardrails][WS] blocked session={} reason={}",
                    session_id,
                    guardrail.blocked_reason,
                )
                await websocket.send_text(
                    json.dumps(
                        {
                            "type": "error",
                            "message": guardrail.blocked_reason,
                        }
                    )
                )
                continue

            # Send "processing" indicator
            await websocket.send_text(
                json.dumps(
                    {
                        "type": "processing",
                        "message": "Analyzing your symptoms...",
                    }
                )
            )

            try:
                logger.info(
                    "[Guardrails][WS] passed session={} warnings={} emergency={}",
                    session_id,
                    guardrail.warnings,
                    guardrail.emergency_detected,
                )
                result = await _run_workflow(
                    guardrail.sanitized_text,
                    session_id,
                    patient_token,
                    guardrail.emergency_detected,
                )
                result["guardrail_warnings"] = guardrail.warnings
                await websocket.send_text(
                    json.dumps(
                        {
                            "type": "response",
                            **result,
                        }
                    )
                )
            except Exception as e:
                logger.error(f"[WS] Workflow error: {e}")
                await websocket.send_text(
                    json.dumps(
                        {
                            "type": "error",
                            "message": f"An error occurred while processing your request. Please try again.",
                        }
                    )
                )

    except WebSocketDisconnect:
        logger.info(f"[WS] Session {session_id} disconnected")
    except Exception as e:
        logger.error(f"[WS] Unexpected error: {e}")


@app.post("/api/chat")
async def chat_api(payload: dict):
    """REST fallback endpoint for chat (non-WebSocket)."""
    user_input = payload.get("message", "").strip()
    session_id = payload.get("session_id", str(uuid.uuid4()))
    patient_token = str(payload.get("patient_token", "")).strip() or None

    if not user_input:
        return JSONResponse({"error": "message is required"}, status_code=400)

    guardrail = screen_input(user_input)
    if not guardrail.allowed:
        logger.warning(
            "[Guardrails][REST] blocked session={} reason={}",
            session_id,
            guardrail.blocked_reason,
        )
        return JSONResponse({"error": guardrail.blocked_reason}, status_code=400)

    try:
        logger.info(
            "[Guardrails][REST] passed session={} warnings={} emergency={}",
            session_id,
            guardrail.warnings,
            guardrail.emergency_detected,
        )
        result = await _run_workflow(
            guardrail.sanitized_text,
            session_id,
            patient_token,
            guardrail.emergency_detected,
        )
        result["guardrail_warnings"] = guardrail.warnings
        return result
    except Exception as e:
        logger.error(f"[API] Error: {e}")
        return JSONResponse(
            {"error": "An error occurred while processing your request."},
            status_code=500,
        )

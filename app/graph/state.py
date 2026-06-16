"""LangGraph state schema for the CDSS workflow."""
from __future__ import annotations

from typing import TypedDict, Literal


class CDSSState(TypedDict, total=False):
    """Shared state passed between all agents in the LangGraph workflow."""

    # ── Input ──
    user_input: str
    """The latest message from the user."""

    session_id: str
    """Unique session identifier for memory tracking."""

    # ── Conversation History ──
    conversation_history: str
    """Full conversation history as formatted text."""

    # ── Planner Output ──
    is_sufficient: bool
    """Whether we have enough info for clinical assessment."""
    completeness_score: int
    """0-100 completeness score from planner."""
    analysis_summary: str
    """What the planner understood from the user's input."""
    follow_up_questions: list[str]
    """Follow-up questions if info is insufficient."""
    next_action: Literal["proceed", "ask_followup"]
    """What the workflow should do next."""

    # ── RAG Agent Output ──
    retrieved_context: str
    """Medical knowledge retrieved from FAISS."""
    primary_assessment: str
    """Most likely diagnosis with reasoning."""
    differential_diagnoses: list[dict]
    """List of alternative diagnoses."""
    symptom_analysis: str
    """Detailed symptom breakdown."""
    severity: str
    """mild | moderate | severe."""
    urgency: str
    """routine | urgent | emergency."""
    clinical_findings: list[str]
    """Key clinical findings."""

    # ── Analyst Output ──
    diagnosis_review: str
    """Validation of primary diagnosis."""
    treatment_options: list[dict]
    """Recommended treatments with cost-effectiveness."""
    medication_safety: dict
    """Interactions, contraindications, warnings."""
    risk_assessment: dict
    """Overall risk level, emergency signs, human review recommendation."""
    alternative_options: list[str]
    """Alternative treatment options."""

    # ── Summary Output ──
    final_response: str
    """Consolidated patient-friendly response (markdown)."""

    # ── Workflow Control ──
    awaiting_user_response: bool
    """True when workflow is paused waiting for user follow-up answer."""

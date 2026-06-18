"""LangGraph workflow for the CDSS multi-agent system.

Workflow:
  User Input → Planner → [insufficient?] → follow_up (HITL pause)
                        → [sufficient]   → RAG → Analyst → Summary → END
"""

from __future__ import annotations

from langgraph.graph import StateGraph, END
from loguru import logger

from app.graph.state import CDSSState
from app.graph.nodes import (
    planner_node,
    rag_node,
    analyst_node,
    summary_node,
    followup_node,
)


def _route_after_planner(state: CDSSState) -> str:
    """Conditional edge: route to RAG if sufficient, followup if not."""
    if state.get("next_action") == "ask_followup" or not state.get(
        "is_sufficient", False
    ):
        logger.info("[Router] → followup (insufficient info)")
        return "followup"
    logger.info("[Router] → rag (proceed to assessment)")
    return "rag"


def build_workflow() -> StateGraph:
    """Build and compile the LangGraph StateGraph."""
    graph = StateGraph(CDSSState)

    # ── Add Nodes ──
    graph.add_node("planner", planner_node)
    graph.add_node("followup", followup_node)
    graph.add_node("rag", rag_node)
    graph.add_node("analyst", analyst_node)
    graph.add_node("summary", summary_node)

    # ── Add Edges ──
    graph.set_entry_point("planner")

    # Planner → conditional routing
    # graph.add_conditional_edges(
    #     "planner",
    #     _route_after_planner,
    #     {
    #         "followup": "followup",
    #         "rag": "rag",
    #     },
    # )

    # followup → END (pause for user response)
    # graph.add_edge("followup", END)
    graph.add_edge("planner", "rag")
    # Linear pipeline after RAG
    graph.add_edge("rag", "analyst")
    graph.add_edge("analyst", "summary")
    graph.add_edge("summary", END)

    compiled = graph.compile()
    logger.info("[Workflow] LangGraph compiled successfully")
    return compiled

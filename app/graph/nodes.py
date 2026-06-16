"""LangGraph node wrapper functions.

Each node function receives the CDSSState, delegates to the corresponding agent,
and returns the updated state.
"""
from __future__ import annotations

from loguru import logger

from app.graph.state import CDSSState
from app.agents.planner import PlannerAgent
from app.agents.rag import RAGAgent
from app.agents.analyst import CostEffectiveAgent
from app.agents.summary import SummaryAgent

# Singleton agent instances
_planner = PlannerAgent()
_rag = RAGAgent()
_analyst = CostEffectiveAgent()
_summary = SummaryAgent()


async def planner_node(state: CDSSState) -> CDSSState:
    """Planner agent node."""
    logger.info("[Node] ═══ Planner ═══")
    return await _planner.run(state)


async def followup_node(state: CDSSState) -> CDSSState:
    """Generate the follow-up question(s) for the user.

    This node represents the human-in-the-loop pause point. It formats
    the planner's follow-up questions into a user-facing message and then
    ends this graph invocation (LangGraph END). The calling application
    sends the questions to the user and waits for their response before
    re-invoking the workflow.
    """
    logger.info("[Node] ═══ Follow-up (HITL) ═══")
    questions = state.get("follow_up_questions", [])
    if not questions:
        questions = ["Could you provide more detail about your symptoms?"]

    lines = [
        "📋 **I need a bit more information to provide a reliable assessment.**\n",
        "To help you better, could you please answer the following:\n",
    ]
    for i, q in enumerate(questions, 1):
        lines.append(f"**{i}.** {q}")
    lines.append("\n_Please reply with your answers and I'll continue the analysis._")

    state["final_response"] = "\n".join(lines)
    state["awaiting_user_response"] = True
    return state


async def rag_node(state: CDSSState) -> CDSSState:
    """RAG / Patient Assessment node."""
    logger.info("[Node] ═══ RAG Agent (Patient Assessment) ═══")
    return await _rag.run(state)


async def analyst_node(state: CDSSState) -> CDSSState:
    """Cost-Effective Analysis + Risk Evaluation node."""
    logger.info("[Node] ═══ Analyst (Cost-Effectiveness & Risk) ═══")
    return await _analyst.run(state)


async def summary_node(state: CDSSState) -> CDSSState:
    """Summary node."""
    logger.info("[Node] ═══ Summary ═══")
    return await _summary.run(state)

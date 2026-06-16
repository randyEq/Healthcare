"""Planner Agent — analyzes user input, checks info sufficiency, generates follow-up questions."""
from __future__ import annotations

import json

from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage
from loguru import logger

from app.config import settings
from app.graph.state import CDSSState
from app.prompts import get_planner_prompt


class PlannerAgent:
    """The Planner Agent evaluates whether enough information is available."""

    def __init__(self) -> None:
        self.llm = ChatOpenAI(
            model=settings.llm_model_name,
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            temperature=0.3,
        )
        self.prompt = get_planner_prompt()

    async def run(self, state: CDSSState) -> CDSSState:
        """Execute the planner analysis."""
        user_input = state.get("user_input", "")
        conversation_history = state.get("conversation_history", "(no prior conversation)")

        logger.info(f"[Planner] Analyzing input: {user_input[:80]}...")

        chain = self.prompt | self.llm
        response: AIMessage = await chain.ainvoke({
            "conversation_history": conversation_history,
            "user_input": user_input,
        })

        raw = response.content.strip()
        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1] if "\n" in raw else raw
            raw = raw.rsplit("```", 1)[0]

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning(f"[Planner] Failed to parse JSON, raw: {raw[:200]}")
            parsed = {
                "is_sufficient": True,
                "completeness_score": 50,
                "analysis_summary": raw[:500],
                "follow_up_questions": [],
                "next_action": "proceed",
            }

        state["is_sufficient"] = parsed.get("is_sufficient", True)
        state["completeness_score"] = parsed.get("completeness_score", 50)
        state["analysis_summary"] = parsed.get("analysis_summary", "")
        state["follow_up_questions"] = parsed.get("follow_up_questions", [])
        state["next_action"] = parsed.get("next_action", "proceed")
        state["awaiting_user_response"] = not state["is_sufficient"]

        logger.info(
            f"[Planner] sufficient={state['is_sufficient']} "
            f"score={state['completeness_score']} "
            f"action={state['next_action']} "
            f"questions={len(state['follow_up_questions'])}"
        )

        return state

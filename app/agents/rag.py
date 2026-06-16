"""RAG Agent (Patient Assessment) — retrieves medical knowledge, performs clinical reasoning."""
from __future__ import annotations

import json

from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage
from loguru import logger

from app.config import settings
from app.graph.state import CDSSState
from app.prompts import get_rag_prompt
from app.retrieval.retriever import retrieve


class RAGAgent:
    """Patient Assessment Agent with FAISS-backed retrieval."""

    def __init__(self) -> None:
        self.llm = ChatOpenAI(
            model=settings.llm_model_name,
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            temperature=0.2,
        )
        self.prompt = get_rag_prompt()

    async def run(self, state: CDSSState) -> CDSSState:
        """Execute RAG-based patient assessment."""
        user_input = state.get("user_input", "")
        conversation_history = state.get("conversation_history", "")

        # Build the patient context from user input + conversation history
        patient_context = f"Current complaint: {user_input}\n\nPrior conversation:\n{conversation_history}"

        # ── Retrieve from FAISS ──
        # Use a combined query of the user input + planner summary
        query = f"{state.get('analysis_summary', '')} {user_input}"
        retrieved_context = retrieve(query, k=5)
        state["retrieved_context"] = retrieved_context

        logger.info(f"[RAG] Retrieved {len(retrieved_context)} chars of medical context")

        # ── LLM Analysis ──
        chain = self.prompt | self.llm
        response: AIMessage = await chain.ainvoke({
            "retrieved_context": retrieved_context,
            "patient_context": patient_context,
        })

        raw = response.content.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1] if "\n" in raw else raw
            raw = raw.rsplit("```", 1)[0]

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning(f"[RAG] Failed to parse JSON, raw: {raw[:200]}")
            parsed = {
                "primary_assessment": raw[:500],
                "differential_diagnoses": [],
                "symptom_analysis": "Unable to parse structured analysis.",
                "severity": "moderate",
                "urgency": "routine",
                "clinical_findings": [],
            }

        state["primary_assessment"] = parsed.get("primary_assessment", "")
        state["differential_diagnoses"] = parsed.get("differential_diagnoses", [])
        state["symptom_analysis"] = parsed.get("symptom_analysis", "")
        state["severity"] = parsed.get("severity", "moderate")
        state["urgency"] = parsed.get("urgency", "routine")
        state["clinical_findings"] = parsed.get("clinical_findings", [])

        logger.info(
            f"[RAG] Assessment complete: severity={state['severity']} "
            f"urgency={state['urgency']} "
            f"diffs={len(state['differential_diagnoses'])}"
        )

        return state

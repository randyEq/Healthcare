"""Cost-Effective Analysis Agent — treatment evaluation, medication safety, risk assessment."""
from __future__ import annotations

import json

from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage
from loguru import logger

from app.config import settings
from app.graph.state import CDSSState
from app.prompts import get_analyst_prompt


class CostEffectiveAgent:
    """Reviews diagnoses, evaluates cost-effective treatments, checks medication safety, assesses risk."""

    def __init__(self) -> None:
        self.llm = ChatOpenAI(
            model=settings.llm_model_name,
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            temperature=0.2,
        )
        self.prompt = get_analyst_prompt()

    async def run(self, state: CDSSState) -> CDSSState:
        """Execute cost-effectiveness analysis and risk evaluation."""
        # Build the RAG assessment summary for the analyst
        rag_assessment = json.dumps({
            "primary_assessment": state.get("primary_assessment", ""),
            "differential_diagnoses": state.get("differential_diagnoses", []),
            "symptom_analysis": state.get("symptom_analysis", ""),
            "severity": state.get("severity", ""),
            "urgency": state.get("urgency", ""),
            "clinical_findings": state.get("clinical_findings", []),
        }, indent=2)

        patient_context = (
            f"Current complaint: {state.get('user_input', '')}\n"
            f"History: {state.get('conversation_history', '')}"
        )

        logger.info("[Analyst] Starting cost-effectiveness and risk analysis...")

        chain = self.prompt | self.llm
        response: AIMessage = await chain.ainvoke({
            "rag_assessment": rag_assessment,
            "patient_context": patient_context,
        })

        raw = response.content.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1] if "\n" in raw else raw
            raw = raw.rsplit("```", 1)[0]

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning(f"[Analyst] Failed to parse JSON, raw: {raw[:200]}")
            parsed = {
                "diagnosis_review": raw[:500],
                "treatment_options": [],
                "medication_safety": {"interactions": [], "contraindications": [], "warnings": []},
                "risk_assessment": {
                    "overall_risk": "moderate",
                    "emergency_signs": [],
                    "human_review_recommended": True,
                    "human_review_reason": "Unable to parse structured analysis — defaulting to human review.",
                },
                "alternative_options": [],
            }

        state["diagnosis_review"] = parsed.get("diagnosis_review", "")
        state["treatment_options"] = parsed.get("treatment_options", [])
        state["medication_safety"] = parsed.get("medication_safety", {})
        state["risk_assessment"] = parsed.get("risk_assessment", {})
        state["alternative_options"] = parsed.get("alternative_options", [])

        risk = state["risk_assessment"]
        logger.info(
            f"[Analyst] Done: risk={risk.get('overall_risk', '?')} "
            f"human_review={risk.get('human_review_recommended', '?')} "
            f"treatments={len(state['treatment_options'])}"
        )

        return state

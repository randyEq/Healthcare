"""Summary Agent — consolidates all findings into a patient-friendly structured response."""
from __future__ import annotations

import json

from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage
from loguru import logger

from app.config import settings
from app.graph.state import CDSSState
from app.prompts import get_summary_prompt


class SummaryAgent:
    """Consolidates all agent outputs into a clear, structured, patient-friendly response."""

    def __init__(self) -> None:
        self.llm = ChatOpenAI(
            model=settings.llm_model_name,
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            temperature=0.4,
        )
        self.prompt = get_summary_prompt()

    async def run(self, state: CDSSState) -> CDSSState:
        """Generate the consolidated summary response."""
        rag_assessment = json.dumps({
            "primary_assessment": state.get("primary_assessment", ""),
            "differential_diagnoses": state.get("differential_diagnoses", []),
            "symptom_analysis": state.get("symptom_analysis", ""),
            "severity": state.get("severity", ""),
            "urgency": state.get("urgency", ""),
            "clinical_findings": state.get("clinical_findings", []),
        }, indent=2)

        analyst_assessment = json.dumps({
            "diagnosis_review": state.get("diagnosis_review", ""),
            "treatment_options": state.get("treatment_options", []),
            "medication_safety": state.get("medication_safety", {}),
            "risk_assessment": state.get("risk_assessment", {}),
            "alternative_options": state.get("alternative_options", []),
        }, indent=2)

        patient_context = state.get("user_input", "")

        logger.info("[Summary] Generating consolidated response...")

        chain = self.prompt | self.llm
        response: AIMessage = await chain.ainvoke({
            "rag_assessment": rag_assessment,
            "analyst_assessment": analyst_assessment,
            "patient_context": patient_context,
        })

        state["final_response"] = response.content.strip()

        logger.info(f"[Summary] Response generated ({len(state['final_response'])} chars)")
        return state

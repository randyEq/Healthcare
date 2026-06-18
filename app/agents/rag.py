"""RAG Agent (Patient Assessment) — retrieves medical knowledge, performs clinical reasoning."""

from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.messages import AIMessage, ToolMessage
from loguru import logger

from app.graph.state import CDSSState
from app.prompts import get_rag_prompt
from app.retrieval.retriever import retrieve
from app.agents.llm_provider import get_chat_llm
from app.agents.medical_api_mcp_tools import MEDICAL_API_MCP_TOOLS
from app.agents.mysql_mcp_tools import MYSQL_MCP_TOOLS


class RAGAgent:
    """Patient Assessment Agent with FAISS-backed retrieval."""

    def __init__(self) -> None:
        self.llm = get_chat_llm(temperature=0.2)
        self.prompt = get_rag_prompt()
        self.tools = [*MYSQL_MCP_TOOLS, *MEDICAL_API_MCP_TOOLS]
        self.tools_by_name = {tool.name: tool for tool in self.tools}

    @staticmethod
    def _normalize_text(value: str) -> str:
        return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()

    def _extract_symptoms(self, raw_symptoms: Any) -> list[str]:
        """Parse common_symptoms from JSON/list/comma-separated DB values."""
        if raw_symptoms is None:
            return []

        if isinstance(raw_symptoms, list):
            return [
                str(symptom).strip() for symptom in raw_symptoms if str(symptom).strip()
            ]

        symptom_text = str(raw_symptoms).strip()
        if not symptom_text:
            return []

        try:
            parsed = json.loads(symptom_text)
        except json.JSONDecodeError:
            parsed = None

        if isinstance(parsed, list):
            return [str(symptom).strip() for symptom in parsed if str(symptom).strip()]
        if isinstance(parsed, dict):
            symptoms = []
            for value in parsed.values():
                symptoms.extend(self._extract_symptoms(value))
            return symptoms

        return [
            symptom.strip(" -")
            for symptom in re.split(r"[,;|\n]+", symptom_text)
            if symptom.strip(" -")
        ]

    def _match_disease_symptoms(
        self,
        user_input: str,
        disease_rows: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Find diseases whose common symptoms appear in the user's query."""
        normalized_query = self._normalize_text(user_input)
        query_tokens = set(normalized_query.split())
        stop_words = {
            "a",
            "an",
            "and",
            "are",
            "for",
            "in",
            "of",
            "or",
            "the",
            "to",
            "with",
        }
        matches = []

        for row in disease_rows:
            symptoms = self._extract_symptoms(row.get("common_symptoms"))
            matched_symptoms = []

            for symptom in symptoms:
                normalized_symptom = self._normalize_text(symptom)
                symptom_tokens = [
                    token
                    for token in normalized_symptom.split()
                    if token not in stop_words
                ]
                phrase_match = (
                    normalized_symptom and normalized_symptom in normalized_query
                )
                token_match = symptom_tokens and all(
                    token in query_tokens for token in symptom_tokens
                )
                if phrase_match or token_match:
                    matched_symptoms.append(symptom)

            if matched_symptoms:
                matches.append(
                    {
                        "disease_name": row.get("disease_name"),
                        "severity_group": row.get("severity_group"),
                        "severity_level": row.get("severity_level"),
                        "matched_symptoms": matched_symptoms,
                        "triage_recommendation": row.get("triage_recommendation"),
                    }
                )

        return sorted(
            matches,
            key=lambda match: len(match["matched_symptoms"]),
            reverse=True,
        )

    @staticmethod
    def _format_disease_matches(matches: list[dict[str, Any]]) -> str:
        if not matches:
            return "No direct symptom matches found in patientcare.disease."

        lines = []
        for match in matches[:5]:
            symptoms = ", ".join(match["matched_symptoms"])
            lines.append(
                f"- {match['disease_name']} "
                f"(severity_group={match['severity_group']}, "
                f"severity_level={match['severity_level']}): "
                f"matched symptoms: {symptoms}; "
                f"triage: {match.get('triage_recommendation') or 'not provided'}"
            )
        return "\n".join(lines)

    def _extract_rows_from_tool_messages(
        self,
        tool_messages: list[ToolMessage],
    ) -> list[dict[str, Any]]:
        """Pull disease rows out of MCP tool result messages for app state."""
        rows: list[dict[str, Any]] = []
        for message in tool_messages:
            try:
                payload = json.loads(str(message.content))
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict) and isinstance(payload.get("rows"), list):
                rows.extend(row for row in payload["rows"] if isinstance(row, dict))
        return rows

    @staticmethod
    def _redact_tool_args(value: Any) -> Any:
        """Redact secrets before logging tool arguments."""
        if isinstance(value, dict):
            redacted = {}
            for key, item in value.items():
                key_text = str(key).lower()
                if any(secret in key_text for secret in ("api_key", "token", "secret", "password")):
                    redacted[key] = "***redacted***"
                else:
                    redacted[key] = RAGAgent._redact_tool_args(item)
            return redacted
        if isinstance(value, list):
            return [RAGAgent._redact_tool_args(item) for item in value]
        return value

    @staticmethod
    def _tool_response_preview(content: Any, limit: int = 3000) -> str:
        """Return a readable, bounded preview of a tool response for logs."""
        text = content if isinstance(content, str) else json.dumps(content, default=str)
        if len(text) <= limit:
            return text
        return f"{text[:limit]}... [truncated {len(text) - limit} chars]"

    async def _invoke_llm_with_auto_tools(
        self,
        retrieved_context: str,
        patient_context: str,
        session_id: str,
    ) -> tuple[AIMessage, list[ToolMessage]]:
        """Let the chat model decide whether to call MCP tools."""
        prompt_value = self.prompt.invoke(
            {
                "retrieved_context": retrieved_context,
                "patient_context": patient_context,
            }
        )
        messages = prompt_value.to_messages()

        if not hasattr(self.llm, "bind_tools"):
            logger.warning("[RAG] LLM does not support bind_tools; running without tools")
            chain = self.prompt | self.llm
            response: AIMessage = await chain.ainvoke(
                {
                    "retrieved_context": retrieved_context,
                    "patient_context": patient_context,
                },
                config={
                    "run_name": "rag_assessment_no_tools",
                    "tags": ["healthcare_cdss", "rag"],
                    "metadata": {"session_id": session_id},
                },
            )
            return response, []

        try:
            tool_llm = self.llm.bind_tools(self.tools)
        except Exception as exc:
            logger.warning(f"[RAG] Could not bind MCP tools to LLM: {exc}")
            chain = self.prompt | self.llm
            response: AIMessage = await chain.ainvoke(
                {
                    "retrieved_context": retrieved_context,
                    "patient_context": patient_context,
                },
                config={
                    "run_name": "rag_assessment_tool_binding_fallback",
                    "tags": ["healthcare_cdss", "rag"],
                    "metadata": {"session_id": session_id},
                },
            )
            return response, []

        response: AIMessage = await tool_llm.ainvoke(
            messages,
            config={
                "run_name": "rag_assessment_with_tools",
                "tags": ["healthcare_cdss", "rag", "tools"],
                "metadata": {"session_id": session_id},
            },
        )
        tool_messages: list[ToolMessage] = []

        for _ in range(3):
            tool_calls = getattr(response, "tool_calls", []) or []
            if not tool_calls:
                return response, tool_messages

            messages.append(response)
            logger.info(f"[RAG] LLM requested {len(tool_calls)} MCP tool call(s)")
            for tool_call in tool_calls:
                tool_name = tool_call.get("name")
                tool_args = tool_call.get("args") or {}
                safe_args = self._redact_tool_args(tool_args)
                logger.info(
                    f"[RAG] MCP tool call -> {tool_name} "
                    f"args={json.dumps(safe_args, default=str)}"
                )

                tool = self.tools_by_name.get(tool_name or "")
                if tool is None:
                    content = json.dumps({"error": f"Unknown tool: {tool_name}"})
                else:
                    try:
                        content = await tool.ainvoke(tool_args)
                    except Exception as exc:
                        content = json.dumps({"error": str(exc)})

                logger.info(
                    f"[RAG] MCP tool response <- {tool_name}: "
                    f"{self._tool_response_preview(content)}"
                )

                tool_message = ToolMessage(
                    content=content,
                    tool_call_id=tool_call.get("id", ""),
                    name=tool_name,
                )
                tool_messages.append(tool_message)
                messages.append(tool_message)

            response = await tool_llm.ainvoke(
                messages,
                config={
                    "run_name": "rag_assessment_tool_followup",
                    "tags": ["healthcare_cdss", "rag", "tools"],
                    "metadata": {"session_id": session_id},
                },
            )

        logger.warning("[RAG] Tool loop limit reached; using latest model response")
        return response, tool_messages

    async def run(self, state: CDSSState) -> CDSSState:
        """Execute RAG-based patient assessment."""
        user_input = state.get("user_input", "")
        conversation_history = state.get("conversation_history", "")

        logger.debug(f"[RAG] Running RAG agent with state keys: {list(state.keys())}")

        # Build the patient context from user input + conversation history
        patient_context = f"Current complaint: {user_input}\n\nPrior conversation:\n{conversation_history}"

        # ── Retrieve from FAISS ──
        # Use a combined query of the user input + planner summary
        query = f"{state.get('analysis_summary', '')} {user_input}"
        retrieved_context = retrieve(query, k=5)
        state["retrieved_context"] = retrieved_context

        logger.debug(f"[RAG] Patient context for LLM:\n{patient_context}")
        logger.info(
            f"[RAG] Retrieved {len(retrieved_context)} chars of medical context"
        )

        # ── LLM Analysis ──
        response, tool_messages = await self._invoke_llm_with_auto_tools(
            retrieved_context=retrieved_context,
            patient_context=patient_context,
            session_id=state.get("session_id", ""),
        )
        disease_rows = self._extract_rows_from_tool_messages(tool_messages)
        state["disease_severity_data"] = disease_rows
        state["disease_symptom_matches"] = self._match_disease_symptoms(
            user_input,
            disease_rows,
        )

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

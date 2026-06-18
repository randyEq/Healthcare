"""Unit tests for CDSS agents with mocked LLM responses."""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.graph.state import CDSSState


# ──────────────────────────────────────────────
# Planner Agent Tests
# ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_planner_sufficient_input():
    """Planner should detect when enough information is provided."""
    mock_response = MagicMock()
    mock_response.content = json.dumps(
        {
            "is_sufficient": True,
            "completeness_score": 85,
            "analysis_summary": "User reports fever and sore throat for 3 days.",
            "follow_up_questions": [],
            "next_action": "proceed",
        }
    )

    with patch("app.agents.planner.get_chat_llm") as MockLLM:
        mock_llm = MockLLM.return_value
        mock_chain = MagicMock()
        mock_chain.ainvoke = AsyncMock(return_value=mock_response)
        mock_llm.__or__ = MagicMock(return_value=mock_chain)

        from app.agents.planner import PlannerAgent

        agent = PlannerAgent()
        agent.llm = mock_llm
        agent.prompt = MagicMock()
        agent.prompt.__or__ = MagicMock(return_value=mock_chain)

        state: CDSSState = {
            "user_input": "I am 35yo male with 101F fever and sore throat for 3 days.",
            "conversation_history": "(no prior conversation)",
        }
        result = await agent.run(state)

        assert result["is_sufficient"] is True
        assert result["completeness_score"] == 85
        assert result["next_action"] == "proceed"
        assert result["awaiting_user_response"] is False


@pytest.mark.asyncio
async def test_planner_insufficient_input():
    """Planner should detect when more info is needed and generate follow-up questions."""
    mock_response = MagicMock()
    mock_response.content = json.dumps(
        {
            "is_sufficient": False,
            "completeness_score": 20,
            "analysis_summary": "User only says 'I feel sick'.",
            "follow_up_questions": [
                "What symptoms do you have?",
                "How long have you felt this way?",
            ],
            "next_action": "ask_followup",
        }
    )

    with patch("app.agents.planner.get_chat_llm"):
        mock_chain = MagicMock()
        mock_chain.ainvoke = AsyncMock(return_value=mock_response)

        from app.agents.planner import PlannerAgent

        agent = PlannerAgent()
        agent.llm = MagicMock()
        agent.prompt = MagicMock()
        agent.prompt.__or__ = MagicMock(return_value=mock_chain)

        state: CDSSState = {
            "user_input": "I feel sick",
            "conversation_history": "(no prior conversation)",
        }
        result = await agent.run(state)

        assert result["is_sufficient"] is False
        assert result["completeness_score"] == 20
        assert result["next_action"] == "ask_followup"
        assert len(result["follow_up_questions"]) == 2
        assert result["awaiting_user_response"] is True


@pytest.mark.asyncio
async def test_planner_invalid_json_fallback():
    """Planner should fall back gracefully when LLM returns non-JSON."""
    mock_response = MagicMock()
    mock_response.content = "This is not JSON at all."

    with patch("app.agents.planner.get_chat_llm"):
        from app.agents.planner import PlannerAgent

        agent = PlannerAgent()
        agent.llm = MagicMock()
        agent.prompt = MagicMock()
        mock_chain = MagicMock()
        mock_chain.ainvoke = AsyncMock(return_value=mock_response)
        agent.prompt.__or__ = MagicMock(return_value=mock_chain)

        state: CDSSState = {"user_input": "test", "conversation_history": ""}
        result = await agent.run(state)

        # Should fall back to proceeding
        assert result["is_sufficient"] is True
        assert result["next_action"] == "proceed"


# ──────────────────────────────────────────────
# RAG Agent Tests
# ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_rag_agent_retrieves_and_assesses():
    """RAG Agent should retrieve context and produce structured assessment."""
    mock_response = MagicMock()
    mock_response.content = json.dumps(
        {
            "primary_assessment": "Influenza based on fever and body aches.",
            "differential_diagnoses": [
                {
                    "condition": "Common Cold",
                    "likelihood": "moderate",
                    "reasoning": "Similar symptoms.",
                },
            ],
            "symptom_analysis": "High fever, body aches suggest flu.",
            "severity": "moderate",
            "urgency": "urgent",
            "clinical_findings": ["Fever 101F", "Body aches"],
        }
    )

    with patch(
        "app.agents.rag.retrieve", return_value="Mock medical knowledge about flu."
    ), patch("app.agents.rag.get_chat_llm"):
        from app.agents.rag import RAGAgent

        agent = RAGAgent()
        agent._invoke_llm_with_auto_tools = AsyncMock(return_value=(mock_response, []))

        state: CDSSState = {
            "user_input": "Fever and body aches",
            "conversation_history": "",
            "analysis_summary": "Flu symptoms",
        }
        result = await agent.run(state)

        assert "Influenza" in result["primary_assessment"]
        assert result["severity"] == "moderate"
        assert result["urgency"] == "urgent"
        assert len(result["differential_diagnoses"]) == 1
        assert result["retrieved_context"] == "Mock medical knowledge about flu."


@pytest.mark.asyncio
async def test_rag_agent_auto_tool_loop_executes_requested_tool():
    """RAG tool loop should execute model-requested MCP tools."""
    from langchain_core.messages import AIMessage
    from app.agents.rag import RAGAgent

    class FakeTool:
        name = "get_disease_rows"

        async def ainvoke(self, args):
            return json.dumps(
                {
                    "rows": [
                        {
                            "disease_name": "Influenza",
                            "severity_group": "moderate",
                            "severity_level": 2,
                            "common_symptoms": ["fever", "body aches"],
                            "triage_recommendation": "urgent if worsening",
                        }
                    ]
                }
            )

    class FakeToolCallingLLM:
        def __init__(self):
            self.calls = 0

        def bind_tools(self, tools):
            return self

        async def ainvoke(self, messages):
            self.calls += 1
            if self.calls == 1:
                return AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "get_disease_rows",
                            "args": {"max_rows": 1},
                            "id": "call_1",
                        }
                    ],
                )
            return AIMessage(
                content=json.dumps(
                    {
                        "primary_assessment": "Influenza",
                        "differential_diagnoses": [],
                        "symptom_analysis": "Fever and body aches.",
                        "severity": "moderate",
                        "urgency": "urgent",
                        "clinical_findings": ["fever"],
                    }
                )
            )

    prompt_value = MagicMock()
    prompt_value.to_messages.return_value = []
    prompt = MagicMock()
    prompt.invoke.return_value = prompt_value

    agent = RAGAgent.__new__(RAGAgent)
    agent.llm = FakeToolCallingLLM()
    agent.prompt = prompt
    agent.tools = [FakeTool()]
    agent.tools_by_name = {"get_disease_rows": agent.tools[0]}

    response, tool_messages = await agent._invoke_llm_with_auto_tools(
        retrieved_context="flu context",
        patient_context="Current complaint: fever and body aches",
    )

    assert "Influenza" in response.content
    assert len(tool_messages) == 1
    assert "body aches" in tool_messages[0].content


# ──────────────────────────────────────────────
# Analyst Agent Tests
# ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_analyst_evaluates_treatment_and_risk():
    """Cost-Effective Agent should produce treatment options and risk assessment."""
    mock_response = MagicMock()
    mock_response.content = json.dumps(
        {
            "diagnosis_review": "Diagnosis is valid.",
            "treatment_options": [
                {
                    "name": "Rest and fluids",
                    "cost_effectiveness": "high",
                    "estimated_cost_range": "$0-10",
                    "rationale": "Best first step.",
                    "safety_profile": "Safe.",
                },
            ],
            "medication_safety": {
                "interactions": ["NSAIDs with warfarin"],
                "contraindications": ["Allergy to NSAIDs"],
                "warnings": ["Monitor for dehydration"],
            },
            "risk_assessment": {
                "overall_risk": "moderate",
                "emergency_signs": ["Difficulty breathing"],
                "human_review_recommended": True,
                "human_review_reason": "Persistent fever",
            },
            "alternative_options": ["Herbal remedies"],
        }
    )

    with patch("app.agents.analyst.get_chat_llm"):
        from app.agents.analyst import CostEffectiveAgent

        agent = CostEffectiveAgent()
        agent.llm = MagicMock()
        agent.prompt = MagicMock()
        mock_chain = MagicMock()
        mock_chain.ainvoke = AsyncMock(return_value=mock_response)
        agent.prompt.__or__ = MagicMock(return_value=mock_chain)

        state: CDSSState = {
            "user_input": "Fever",
            "primary_assessment": "Flu",
            "differential_diagnoses": [],
            "symptom_analysis": "Fever and aches",
            "severity": "moderate",
            "urgency": "urgent",
            "clinical_findings": [],
            "conversation_history": "",
        }
        result = await agent.run(state)

        assert len(result["treatment_options"]) == 1
        assert result["treatment_options"][0]["cost_effectiveness"] == "high"
        assert result["risk_assessment"]["overall_risk"] == "moderate"
        assert result["risk_assessment"]["human_review_recommended"] is True


# ──────────────────────────────────────────────
# Summary Agent Tests
# ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_summary_produces_structured_response():
    """Summary Agent should produce a patient-friendly response with disclaimer."""
    mock_response = MagicMock()
    mock_response.content = """### 🏥 Assessment Summary
Likely condition: Flu.

### ⚕️ Disclaimer
This AI-generated information is for educational purposes only."""

    with patch("app.agents.summary.get_chat_llm"):
        from app.agents.summary import SummaryAgent

        agent = SummaryAgent()
        agent.llm = MagicMock()
        agent.prompt = MagicMock()
        mock_chain = MagicMock()
        mock_chain.ainvoke = AsyncMock(return_value=mock_response)
        agent.prompt.__or__ = MagicMock(return_value=mock_chain)

        state: CDSSState = {
            "user_input": "Fever",
            "primary_assessment": "Flu",
            "differential_diagnoses": [],
            "symptom_analysis": "",
            "severity": "moderate",
            "urgency": "routine",
            "clinical_findings": [],
            "diagnosis_review": "",
            "treatment_options": [],
            "medication_safety": {},
            "risk_assessment": {},
            "alternative_options": [],
        }
        result = await agent.run(state)

        assert "Assessment Summary" in result["final_response"]
        assert "Disclaimer" in result["final_response"]
        assert len(result["final_response"]) > 0


# ──────────────────────────────────────────────
# Memory Tests
# ──────────────────────────────────────────────


def test_conversation_memory_add_and_retrieve():
    """ConversationMemory should store and retrieve messages by session."""
    from app.memory.conversation import ConversationMemory

    mem = ConversationMemory()

    mem.add_message("s1", "user", "Hello")
    mem.add_message("s1", "assistant", "Hi there")

    history = mem.get_history("s1")
    assert len(history) == 2
    assert history[0]["role"] == "user"
    assert history[0]["content"] == "Hello"


def test_conversation_memory_formatted_history():
    """Formatted history should include speaker labels."""
    from app.memory.conversation import ConversationMemory

    mem = ConversationMemory()

    mem.add_message("s2", "user", "I have a headache")
    mem.add_message("s2", "assistant", "Tell me more")

    formatted = mem.get_formatted_history("s2")
    assert "Patient" in formatted
    assert "System" in formatted
    assert "headache" in formatted


def test_conversation_memory_isolation():
    """Different sessions should have isolated histories."""
    from app.memory.conversation import ConversationMemory

    mem = ConversationMemory()

    mem.add_message("session-a", "user", "Message A")
    mem.add_message("session-b", "user", "Message B")

    assert len(mem.get_history("session-a")) == 1
    assert len(mem.get_history("session-b")) == 1
    assert mem.get_history("session-a")[0]["content"] == "Message A"


def test_conversation_memory_clear():
    """Clearing a session should remove its history."""
    from app.memory.conversation import ConversationMemory

    mem = ConversationMemory()

    mem.add_message("s3", "user", "Test")
    mem.clear("s3")
    assert len(mem.get_history("s3")) == 0


# ──────────────────────────────────────────────
# Retriever Tests
# ──────────────────────────────────────────────


def test_retrieve_returns_context():
    """Retriever should return formatted context from FAISS."""
    from app.retrieval.retriever import retrieve

    result = retrieve("headache fever", k=2)
    assert isinstance(result, str)
    assert len(result) > 0

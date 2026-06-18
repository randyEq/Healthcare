"""LangChain tool wrappers around the MySQL MCP server."""

from __future__ import annotations

import json

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from app.mcp_mysql_client import call_mysql_tool


class DiseaseRowsInput(BaseModel):
    """Input schema for disease table lookups."""

    max_rows: int = Field(
        default=500,
        ge=1,
        le=500,
        description="Maximum disease rows to return from patientcare.disease.",
    )


def _get_disease_rows(max_rows: int = 500) -> str:
    """Fetch disease severity, symptom, and triage rows from MySQL via MCP."""
    result = call_mysql_tool("get_disease_rows", {"max_rows": max_rows})
    return json.dumps(result, default=str)


MYSQL_MCP_TOOLS = [
    StructuredTool.from_function(
        func=_get_disease_rows,
        name="get_disease_rows",
        description=(
            "Fetch disease_name, severity_group, severity_level, common_symptoms, "
            "and triage_recommendation rows from the patientcare.disease MySQL table. "
            "Use this when patient symptoms need database-backed severity or triage context."
        ),
        args_schema=DiseaseRowsInput,
    )
]

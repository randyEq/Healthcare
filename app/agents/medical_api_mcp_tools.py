"""LangChain tool wrappers for the remote Medical APIs MCP server."""

from __future__ import annotations

import json
from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from app.mcp_http_client import call_http_mcp_tool


def _json_result(value: Any) -> str:
    return json.dumps(value, default=str)


class DiseaseInfoInput(BaseModel):
    disease_id_or_name: str = Field(
        description="Disease name or identifier, for example 'influenza' or 'asthma'."
    )


class LiteratureSearchInput(BaseModel):
    query: str = Field(description="Biomedical literature search query.")
    limit: int = Field(default=5, ge=1, le=25)


class DrugSafetyInput(BaseModel):
    drug_name: str = Field(description="Drug name to search, for example 'ibuprofen'.")
    reaction: str | None = Field(default=None, description="Optional adverse reaction.")
    serious: bool | None = Field(default=None, description="Filter for serious events.")
    limit: int = Field(default=5, ge=1, le=25)


class DrugLabelInput(BaseModel):
    drug_name: str | None = Field(default=None, description="Drug name to search.")
    indication: str | None = Field(
        default=None,
        description="Condition or indication to search drug labels for.",
    )
    section: str | None = Field(
        default=None,
        description="Optional label section, such as warnings_and_precautions.",
    )
    limit: int = Field(default=5, ge=1, le=25)


class DrugIndicationInput(BaseModel):
    disease_query: str = Field(description="Disease or indication name.")
    limit: int = Field(default=10, ge=1, le=25)


class ClinicalTrialsInput(BaseModel):
    condition_query: str = Field(description="Condition or disease name.")
    status: str | None = Field(
        default=None,
        description="Optional comma-separated statuses, e.g. RECRUITING,COMPLETED.",
    )
    page_size: int = Field(default=5, ge=1, le=25)


class GenericMedicalMCPInput(BaseModel):
    tool_name: str = Field(description="Remote Medical APIs MCP tool name to call.")
    arguments: dict[str, Any] = Field(
        default_factory=dict,
        description="JSON arguments for the remote MCP tool.",
    )


def _medical_get_disease_info(disease_id_or_name: str) -> str:
    return _json_result(
        call_http_mcp_tool(
            "mydisease_get_disease",
            {"disease_id_or_name": disease_id_or_name},
        )
    )


def _medical_search_literature(query: str, limit: int = 5) -> str:
    return _json_result(
        call_http_mcp_tool(
            "pubmed_search_articles",
            {"keywords": [query], "limit": limit, "page": 1},
        )
    )


def _medical_search_drug_adverse_events(
    drug_name: str,
    reaction: str | None = None,
    serious: bool | None = None,
    limit: int = 5,
) -> str:
    return _json_result(
        call_http_mcp_tool(
            "openfda_search_adverse_events",
            {
                "drug": drug_name,
                "reaction": reaction,
                "serious": serious,
                "limit": limit,
                "page": 1,
            },
        )
    )


def _medical_search_drug_labels(
    drug_name: str | None = None,
    indication: str | None = None,
    section: str | None = None,
    limit: int = 5,
) -> str:
    return _json_result(
        call_http_mcp_tool(
            "openfda_search_drug_labels",
            {
                "drug_name": drug_name,
                "indication": indication,
                "section": section,
                "limit": limit,
                "page": 1,
            },
        )
    )


def _medical_find_drugs_by_indication(disease_query: str, limit: int = 10) -> str:
    return _json_result(
        call_http_mcp_tool(
            "chembl_find_drugs_by_indication",
            {"disease_query": disease_query, "limit": limit},
        )
    )


def _medical_search_clinical_trials(
    condition_query: str,
    status: str | None = None,
    page_size: int = 5,
) -> str:
    return _json_result(
        call_http_mcp_tool(
            "ctg_search_by_condition",
            {
                "condition_query": condition_query,
                "status": status,
                "page_size": page_size,
            },
        )
    )


def _medical_mcp_call_tool(tool_name: str, arguments: dict[str, Any]) -> str:
    return _json_result(call_http_mcp_tool(tool_name, arguments))


MEDICAL_API_MCP_TOOLS = [
    StructuredTool.from_function(
        func=_medical_get_disease_info,
        name="medical_get_disease_info",
        description=(
            "Look up disease information from the remote Medical APIs MCP server "
            "using MyDisease.info."
        ),
        args_schema=DiseaseInfoInput,
    ),
    StructuredTool.from_function(
        func=_medical_search_literature,
        name="medical_search_literature",
        description=(
            "Search biomedical literature using the remote Medical APIs MCP PubMed tool."
        ),
        args_schema=LiteratureSearchInput,
    ),
    StructuredTool.from_function(
        func=_medical_search_drug_adverse_events,
        name="medical_search_drug_adverse_events",
        description=(
            "Search FDA adverse event reports for a drug using the remote OpenFDA MCP tool."
        ),
        args_schema=DrugSafetyInput,
    ),
    StructuredTool.from_function(
        func=_medical_search_drug_labels,
        name="medical_search_drug_labels",
        description=(
            "Search FDA drug labels by drug, indication, or section using the remote "
            "OpenFDA MCP tool."
        ),
        args_schema=DrugLabelInput,
    ),
    StructuredTool.from_function(
        func=_medical_find_drugs_by_indication,
        name="medical_find_drugs_by_indication",
        description=(
            "Find drugs associated with a disease or indication using the remote "
            "ChEMBL MCP tool."
        ),
        args_schema=DrugIndicationInput,
    ),
    StructuredTool.from_function(
        func=_medical_search_clinical_trials,
        name="medical_search_clinical_trials",
        description=(
            "Search ClinicalTrials.gov trials by condition using the remote Medical APIs MCP tool."
        ),
        args_schema=ClinicalTrialsInput,
    ),
    StructuredTool.from_function(
        func=_medical_mcp_call_tool,
        name="medical_mcp_call_tool",
        description=(
            "Advanced fallback: call any remote Medical APIs MCP tool by exact tool "
            "name and JSON arguments, such as kegg_find_diseases or pubmed_get_article."
        ),
        args_schema=GenericMedicalMCPInput,
    ),
]

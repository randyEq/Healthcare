"""MCP stdio server exposing safe MySQL tools for the CDSS flow."""

from __future__ import annotations

import json
import re
import sys
from typing import Any

from app.sql_tool import execute_sql

SERVER_NAME = "healthcare-mysql-mcp"
PROTOCOL_VERSION = "2024-11-05"
MAX_ROWS_DEFAULT = 100
MAX_ROWS_LIMIT = 500

READ_ONLY_COMMANDS = {"select", "show", "describe", "desc", "explain"}
BLOCKED_KEYWORDS = {
    "alter",
    "create",
    "delete",
    "drop",
    "grant",
    "insert",
    "load",
    "replace",
    "revoke",
    "set",
    "truncate",
    "update",
}


def _json_default(value: Any) -> str:
    return str(value)


def _clamp_max_rows(value: Any) -> int:
    try:
        max_rows = int(value)
    except (TypeError, ValueError):
        max_rows = MAX_ROWS_DEFAULT
    return min(max(max_rows, 1), MAX_ROWS_LIMIT)


def _validate_read_only_query(query: str) -> str:
    query = query.strip()
    if not query:
        raise ValueError("Query is required.")

    normalized = re.sub(r"/\*.*?\*/", " ", query, flags=re.DOTALL).strip()
    normalized = re.sub(r"--.*?$", " ", normalized, flags=re.MULTILINE).strip()
    normalized = normalized.rstrip(";").strip()

    if ";" in normalized:
        raise ValueError("Multiple SQL statements are not allowed.")

    first_word_match = re.match(r"^\s*([a-zA-Z]+)\b", normalized)
    first_word = first_word_match.group(1).lower() if first_word_match else ""
    if first_word not in READ_ONLY_COMMANDS:
        allowed = ", ".join(sorted(READ_ONLY_COMMANDS))
        raise ValueError(f"Only read-only SQL is allowed: {allowed}.")

    tokens = set(re.findall(r"\b[a-zA-Z_]+\b", normalized.lower()))
    blocked = sorted(tokens & BLOCKED_KEYWORDS)
    if blocked:
        raise ValueError(f"Blocked SQL keyword(s): {', '.join(blocked)}.")

    return normalized


def _format_tool_result(rows: list[dict[str, Any]], max_rows: int) -> dict[str, Any]:
    visible_rows = rows[:max_rows]
    return {
        "row_count": len(rows),
        "returned_rows": len(visible_rows),
        "truncated": len(rows) > len(visible_rows),
        "rows": visible_rows,
    }


def query_mysql(query: str, max_rows: int = MAX_ROWS_DEFAULT) -> dict[str, Any]:
    safe_query = _validate_read_only_query(query)
    rows = execute_sql(safe_query)
    return _format_tool_result(rows, _clamp_max_rows(max_rows))


def get_disease_rows(max_rows: int = MAX_ROWS_DEFAULT) -> dict[str, Any]:
    query = """
    SELECT
        disease_name,
        severity_group,
        severity_level,
        common_symptoms,
        triage_recommendation
    FROM patientcare.disease
    """
    rows = execute_sql(query)
    return _format_tool_result(rows, _clamp_max_rows(max_rows))


def _tools() -> list[dict[str, Any]]:
    return [
        {
            "name": "query_mysql",
            "description": (
                "Run a read-only MySQL query against the configured healthcare "
                "database. Only SELECT, SHOW, DESCRIBE, DESC, and EXPLAIN are allowed."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "A single read-only MySQL statement.",
                    },
                    "max_rows": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": MAX_ROWS_LIMIT,
                        "default": MAX_ROWS_DEFAULT,
                    },
                },
                "required": ["query"],
            },
        },
        {
            "name": "get_disease_rows",
            "description": (
                "Fetch disease severity, symptom, and triage rows from "
                "patientcare.disease for clinical matching."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "max_rows": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": MAX_ROWS_LIMIT,
                        "default": MAX_ROWS_DEFAULT,
                    }
                },
            },
        },
    ]


def _call_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    if name == "query_mysql":
        result = query_mysql(
            query=str(arguments.get("query", "")),
            max_rows=_clamp_max_rows(arguments.get("max_rows")),
        )
    elif name == "get_disease_rows":
        result = get_disease_rows(max_rows=_clamp_max_rows(arguments.get("max_rows")))
    else:
        raise ValueError(f"Unknown tool: {name}")

    return {
        "content": [
            {
                "type": "text",
                "text": json.dumps(result, default=_json_default),
            }
        ],
        "isError": False,
    }


def _read_message() -> dict[str, Any] | None:
    headers: dict[str, str] = {}
    while True:
        line = sys.stdin.buffer.readline()
        if line == b"":
            return None
        line_text = line.decode("ascii", errors="replace").strip()
        if not line_text:
            break
        if ":" in line_text:
            key, value = line_text.split(":", 1)
            headers[key.lower()] = value.strip()

    content_length = int(headers.get("content-length", "0"))
    if content_length <= 0:
        return None

    payload = sys.stdin.buffer.read(content_length)
    return json.loads(payload.decode("utf-8"))


def _write_message(message: dict[str, Any]) -> None:
    payload = json.dumps(message, default=_json_default).encode("utf-8")
    sys.stdout.buffer.write(f"Content-Length: {len(payload)}\r\n\r\n".encode("ascii"))
    sys.stdout.buffer.write(payload)
    sys.stdout.buffer.flush()


def _handle_request(message: dict[str, Any]) -> dict[str, Any] | None:
    message_id = message.get("id")
    method = message.get("method")

    if message_id is None:
        return None

    try:
        if method == "initialize":
            result = {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": SERVER_NAME, "version": "0.1.0"},
            }
        elif method == "tools/list":
            result = {"tools": _tools()}
        elif method == "tools/call":
            params = message.get("params") or {}
            result = _call_tool(
                name=str(params.get("name", "")),
                arguments=params.get("arguments") or {},
            )
        else:
            return {
                "jsonrpc": "2.0",
                "id": message_id,
                "error": {"code": -32601, "message": f"Method not found: {method}"},
            }
        return {"jsonrpc": "2.0", "id": message_id, "result": result}
    except Exception as exc:
        return {
            "jsonrpc": "2.0",
            "id": message_id,
            "error": {"code": -32000, "message": str(exc)},
        }


def main() -> None:
    while True:
        message = _read_message()
        if message is None:
            break
        response = _handle_request(message)
        if response is not None:
            _write_message(response)


if __name__ == "__main__":
    main()

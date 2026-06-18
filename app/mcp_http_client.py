"""HTTP MCP client for remote Streamable HTTP MCP servers."""

from __future__ import annotations

import json
from typing import Any

import requests

from app.config import settings


class MCPHttpClientError(RuntimeError):
    """Raised when a remote HTTP MCP server returns an error."""


def _parse_mcp_response(response: requests.Response) -> dict[str, Any]:
    response.raise_for_status()
    content_type = response.headers.get("content-type", "")

    if "text/event-stream" in content_type:
        for line in response.text.splitlines():
            if line.startswith("data: "):
                data = line.removeprefix("data: ").strip()
                if data and data != "[DONE]":
                    return json.loads(data)
        raise MCPHttpClientError("MCP server returned an empty event stream.")

    return response.json()


def call_http_mcp_tool(
    name: str,
    arguments: dict[str, Any] | None = None,
    url: str | None = None,
    timeout: int = 30,
) -> Any:
    """Call a tool on a remote HTTP MCP server."""
    endpoint = url or settings.medical_apis_mcp_url
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {"name": name, "arguments": arguments or {}},
    }
    response = requests.post(
        endpoint,
        json=payload,
        headers={
            "Accept": "application/json, text/event-stream",
            "Content-Type": "application/json",
        },
        timeout=timeout,
    )
    message = _parse_mcp_response(response)

    if "error" in message:
        error = message["error"]
        raise MCPHttpClientError(error.get("message", str(error)))

    result = message.get("result", {})
    content = result.get("content") if isinstance(result, dict) else None
    if content:
        text = content[0].get("text", "")
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return text
    return result


def list_http_mcp_tools(url: str | None = None, timeout: int = 30) -> list[dict[str, Any]]:
    """Return tool metadata from a remote HTTP MCP server."""
    endpoint = url or settings.medical_apis_mcp_url
    payload = {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}
    response = requests.post(
        endpoint,
        json=payload,
        headers={
            "Accept": "application/json, text/event-stream",
            "Content-Type": "application/json",
        },
        timeout=timeout,
    )
    message = _parse_mcp_response(response)
    if "error" in message:
        error = message["error"]
        raise MCPHttpClientError(error.get("message", str(error)))
    return (message.get("result") or {}).get("tools", [])

"""Small MCP stdio client used by the LangGraph flow."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any


class MCPClientError(RuntimeError):
    """Raised when the MySQL MCP server returns an error or malformed response."""


def _encode_message(message: dict[str, Any]) -> bytes:
    payload = json.dumps(message).encode("utf-8")
    return f"Content-Length: {len(payload)}\r\n\r\n".encode("ascii") + payload


def _read_message(stream) -> dict[str, Any]:
    headers: dict[str, str] = {}
    while True:
        line = stream.readline()
        if line == b"":
            raise MCPClientError("MCP server closed the stream.")
        line_text = line.decode("ascii", errors="replace").strip()
        if not line_text:
            break
        if ":" in line_text:
            key, value = line_text.split(":", 1)
            headers[key.lower()] = value.strip()

    content_length = int(headers.get("content-length", "0"))
    if content_length <= 0:
        raise MCPClientError("MCP server returned a response without Content-Length.")

    payload = stream.read(content_length)
    return json.loads(payload.decode("utf-8"))


def _send_message(process: subprocess.Popen[bytes], message: dict[str, Any]) -> None:
    if process.stdin is None:
        raise MCPClientError("MCP process stdin is unavailable.")
    process.stdin.write(_encode_message(message))
    process.stdin.flush()


def call_mysql_tool(name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
    """Call a tool exposed by app.mcp_mysql_server and return parsed JSON text."""
    project_root = Path(__file__).resolve().parent.parent
    process = subprocess.Popen(
        [sys.executable, "-m", "app.mcp_mysql_server"],
        cwd=str(project_root),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    try:
        _send_message(
            process,
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "healthcare-cdss", "version": "0.1.0"},
                },
            },
        )
        _read_success(process)
        _send_message(
            process,
            {"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}},
        )
        _send_message(
            process,
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {"name": name, "arguments": arguments or {}},
            },
        )
        response = _read_success(process)
        content = response.get("content") or []
        if not content:
            return {}
        text = content[0].get("text", "{}")
        return json.loads(text)
    finally:
        if process.stdin is not None:
            process.stdin.close()
        process.terminate()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()


def _read_success(process: subprocess.Popen[bytes]) -> dict[str, Any]:
    if process.stdout is None:
        raise MCPClientError("MCP process stdout is unavailable.")

    message = _read_message(process.stdout)
    if "error" in message:
        error = message["error"]
        raise MCPClientError(error.get("message", str(error)))
    return message.get("result") or {}

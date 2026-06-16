"""Conversation memory management for CDSS sessions."""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone

from loguru import logger


class ConversationMemory:
    """In-memory conversation store keyed by session_id.

    Maintains a running history of user/assistant messages so that
    the planner agent can evaluate completeness across multiple turns.
    """

    def __init__(self) -> None:
        # session_id → list of {"role": "user"|"assistant", "content": "...", "ts": ...}
        self._store: dict[str, list[dict]] = defaultdict(list)

    def add_message(self, session_id: str, role: str, content: str) -> None:
        """Append a message to the session's history."""
        self._store[session_id].append({
            "role": role,
            "content": content,
            "ts": datetime.now(timezone.utc).isoformat(),
        })
        logger.debug(f"[Memory] session={session_id} role={role} added")

    def get_history(self, session_id: str) -> list[dict]:
        """Return the list of messages for a session."""
        return self._store.get(session_id, [])

    def get_formatted_history(self, session_id: str, limit: int = 20) -> str:
        """Return conversation as a readable string for prompt context."""
        messages = self._store.get(session_id, [])
        if limit:
            messages = messages[-limit:]
        lines: list[str] = []
        for msg in messages:
            speaker = "Patient" if msg["role"] == "user" else "System"
            lines.append(f"{speaker}: {msg['content']}")
        return "\n".join(lines) if lines else "(no prior conversation)"

    def clear(self, session_id: str) -> None:
        """Clear a session's history."""
        if session_id in self._store:
            del self._store[session_id]
            logger.debug(f"[Memory] session={session_id} cleared")


# Singleton instance shared across the application
memory = ConversationMemory()

"""Bounded, in-process conversation memory.

Why bounded
-----------
Every model has a finite context window. An unbounded history does not fail
gracefully - it fails as a wall of backend errors, or silently truncated
prompts, the moment a conversation gets long. This class enforces two limits at
once:

* ``max_turns`` - how many *messages* are kept (a rough proxy for depth);
* ``max_chars`` - a hard character budget (protects against one huge paste).

Trimming always removes the oldest messages first and never splits a message.

Why in-process
--------------
v1 is a local single-user app, so a dict of deques is the right amount of
machinery. The class is small and its interface is storage-shaped on purpose:
swapping it for SQLite or Redis later means implementing the same five methods.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable

from kumaru.core.types import Message, Role


class ConversationMemory:
    """Per-session message history with turn and character budgets."""

    def __init__(self, max_turns: int = 12, max_chars: int = 12000) -> None:
        if max_turns < 1:
            raise ValueError("max_turns must be >= 1")
        if max_chars < 1:
            raise ValueError("max_chars must be >= 1")
        self.max_turns = max_turns
        self.max_chars = max_chars
        self._sessions: dict[str, deque[Message]] = {}

    def _bucket(self, session_id: str) -> deque[Message]:
        return self._sessions.setdefault(session_id, deque(maxlen=self.max_turns))

    def add(self, session_id: str, message: Message) -> None:
        """Append a message, dropping system messages and re-applying budgets."""
        if message.role is Role.SYSTEM:
            return  # the system prompt is rebuilt every turn; never store it
        bucket = self._bucket(session_id)
        bucket.append(message)
        self._trim_chars(bucket)

    def extend(self, session_id: str, messages: Iterable[Message]) -> None:
        for message in messages:
            self.add(session_id, message)

    def _trim_chars(self, bucket: deque[Message]) -> None:
        total = sum(len(m.content) for m in bucket)
        while len(bucket) > 1 and total > self.max_chars:
            total -= len(bucket.popleft().content)

    def history(self, session_id: str) -> list[Message]:
        """Oldest-first copy of the session history."""
        return list(self._sessions.get(session_id, ()))

    def clear(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    def clear_all(self) -> None:
        self._sessions.clear()

    def sessions(self) -> list[str]:
        return sorted(self._sessions)

    def __len__(self) -> int:
        return len(self._sessions)

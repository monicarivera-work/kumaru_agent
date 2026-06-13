"""
kumaru/memory/conversation.py
------------------------------
Short-term (in-process) conversation memory.

Concept: Why agents need memory
---------------------------------
LLMs are stateless – they have no memory between API calls.
To hold a multi-turn conversation we keep a list of all past messages and
send the entire list on every API call.  This list IS the agent's working
memory.

Limits and production concerns
--------------------------------
* Token budgets – Every token in the history costs money and adds latency.
  Real agents trim or summarise old turns once the context window fills up.
  ``ConversationMemory.trim_to_token_budget()`` shows a simple version of
  this pattern.

* Persistence – In-memory history is lost when the process restarts.
  Enterprise agents store history in a database (Redis, Postgres, DynamoDB).
  Subclass ``ConversationMemory`` and override ``add`` / ``get_messages`` to
  do that without touching any other code.

* Long-term memory – Separate from conversation history; uses embedding
  search (RAG) to recall facts from past sessions.  That's a whole module
  on its own and is out of scope for this introductory implementation, but
  the retrieval_memory stub in this file shows where it would plug in.
"""

from __future__ import annotations

from typing import Optional

from kumaru.llm.base import Message


class ConversationMemory:
    """Manages the rolling list of conversation messages.

    Args:
        system_prompt: The very first "system" message injected into every
                       conversation.  This sets the agent's persona and
                       ground rules.
        max_messages:  When set, the memory automatically drops the *oldest*
                       non-system messages once this limit is exceeded.
                       ``None`` means unlimited.
    """

    def __init__(
        self,
        system_prompt: str,
        max_messages: Optional[int] = None,
    ) -> None:
        self._system_prompt = system_prompt
        self._max_messages = max_messages
        # We always keep the system prompt as the first entry.
        self._messages: list[Message] = [
            Message(role="system", content=system_prompt)
        ]

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def add(self, message: Message) -> None:
        """Append *message* to the conversation history.

        If a ``max_messages`` limit was set and the buffer is full, the
        oldest *non-system* message is dropped to make room.
        """
        self._messages.append(message)
        if self._max_messages and len(self._messages) > self._max_messages + 1:
            # Always keep index 0 (the system prompt).
            self._messages.pop(1)

    def add_user(self, content: str) -> None:
        """Convenience wrapper – add a user-role message."""
        self.add(Message(role="user", content=content))

    def add_assistant(self, content: Optional[str], **kwargs) -> None:
        """Convenience wrapper – add an assistant-role message."""
        self.add(Message(role="assistant", content=content, **kwargs))

    def add_tool_result(
        self, tool_call_id: str, name: str, content: str
    ) -> None:
        """Append the result of a tool execution back into the history.

        The model needs this so it can read the tool's output and decide what
        to do next.
        """
        self.add(
            Message(
                role="tool",
                content=content,
                tool_call_id=tool_call_id,
                name=name,
            )
        )

    def get_messages(self) -> list[Message]:
        """Return a *copy* of the current conversation history."""
        return list(self._messages)

    def clear(self) -> None:
        """Reset the memory, keeping only the system prompt."""
        self._messages = [Message(role="system", content=self._system_prompt)]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Return the number of messages (including the system prompt)."""
        return len(self._messages)

    def __repr__(self) -> str:
        return (
            f"ConversationMemory("
            f"messages={len(self)}, "
            f"max_messages={self._max_messages})"
        )

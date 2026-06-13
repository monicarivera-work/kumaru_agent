"""
kumaru/llm/base.py
------------------
Abstract base class for every LLM provider.

Why an abstraction layer?
  Today you might use OpenAI.  Tomorrow you might want Anthropic Claude or a
  local Ollama model.  By coding against this interface – not against the
  OpenAI SDK directly – you can swap providers with a one-line config change.

  This is the *Dependency Inversion Principle* from SOLID design.

Concept: Messages
-----------------
LLM APIs communicate via a list of "messages".  Each message has a role:
  * system    – Background instructions / persona for the model.
  * user      – What the human (or the agent itself) says.
  * assistant – What the model previously responded.
  * tool      – The result returned by a tool call.

Maintaining this list IS the agent's "short-term memory".
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class Message:
    """A single turn in a conversation.

    Attributes:
        role:         One of "system", "user", "assistant", or "tool".
        content:      The text content of the message (may be None when the
                      model is making a tool call).
        tool_calls:   Populated by the model when it wants to call a tool.
        tool_call_id: The id from the corresponding tool_call (used in tool
                      result messages so the model can correlate them).
        name:         For tool result messages, the tool's name.
    """

    role: str
    content: Optional[str] = None
    tool_calls: Optional[list[dict[str, Any]]] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Serialise to the wire format expected by most LLM APIs."""
        d: dict[str, Any] = {"role": self.role}
        if self.content is not None:
            d["content"] = self.content
        if self.tool_calls is not None:
            d["tool_calls"] = self.tool_calls
        if self.tool_call_id is not None:
            d["tool_call_id"] = self.tool_call_id
        if self.name is not None:
            d["name"] = self.name
        return d


@dataclass
class LLMResponse:
    """Structured response returned by every LLM client.

    Attributes:
        content:    The model's text response (None when it's making a tool
                    call instead of producing text).
        tool_calls: The list of tool calls the model wants to make.
        usage:      Token counts – important for cost monitoring in prod.
        raw:        The unmodified API response for debugging.
    """

    content: Optional[str]
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)
    raw: Any = None


class BaseLLMClient(ABC):
    """Interface that every LLM provider must implement.

    To add a new provider:
      1. Create a new file in ``kumaru/llm/``.
      2. Subclass ``BaseLLMClient``.
      3. Implement ``chat()``.
      4. Pass an instance to ``AgentConfig`` or the ``Agent`` constructor.
    """

    @abstractmethod
    def chat(
        self,
        messages: list[Message],
        tools: Optional[list[dict[str, Any]]] = None,
    ) -> LLMResponse:
        """Send a conversation to the model and return its response.

        Args:
            messages: The full conversation history (oldest first).
            tools:    JSON-Schema definitions of tools the model may call.

        Returns:
            An :class:`LLMResponse` with the model's reply.
        """

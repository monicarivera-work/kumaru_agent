"""
kumaru/tools/base.py
---------------------
Base class for all tools the agent can call.

Concept: Tools / Function Calling
----------------------------------
A "tool" is any Python function that the agent can invoke.  The key insight
is that the LLM does NOT run the tool directly.  Instead:

  1. We describe each tool to the model in JSON Schema format.
  2. The model responds with a *request* to call a tool (name + arguments).
  3. Our Python code executes the actual function.
  4. We send the result back to the model as a "tool" role message.

This keeps the model's role narrow (decide *what* to do) while Python handles
the *actual* execution safely.

Defining a new tool is straightforward:
  1. Subclass ``BaseTool``.
  2. Fill in ``name``, ``description``, and ``parameters``.
  3. Implement ``run()``.
  4. Register the instance with the agent.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseTool(ABC):
    """Interface that every tool must implement.

    Class attributes (override in subclasses):
        name:        The identifier the model uses to call this tool.
                     Must be unique within an agent and contain only
                     letters, numbers, and underscores.
        description: A plain-English sentence explaining what the tool does
                     and *when* to use it.  The model reads this to decide
                     whether to call the tool, so write it carefully.
        parameters:  JSON Schema object describing the tool's arguments.
                     The model uses this to construct valid argument JSON.
    """

    name: str
    description: str
    parameters: dict[str, Any]

    @abstractmethod
    def run(self, **kwargs: Any) -> str:
        """Execute the tool with the provided keyword arguments.

        Args:
            **kwargs: Arguments validated against ``self.parameters``.

        Returns:
            A string that will be sent back to the model as the tool result.
            Keep it concise – the model has a limited context window.

        Raises:
            ToolError: On expected failures (bad input, external API down, …).
        """

    def to_openai_schema(self) -> dict[str, Any]:
        """Convert this tool to the JSON format OpenAI expects.

        OpenAI tool format:
        {
            "type": "function",
            "function": {
                "name":        <str>,
                "description": <str>,
                "parameters":  <JSON Schema object>
            }
        }
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"


class ToolError(Exception):
    """Raised when a tool encounters an expected failure.

    The agent catches ``ToolError`` and sends the message back to the model
    as the tool result so it can recover gracefully.
    """

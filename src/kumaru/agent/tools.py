"""The tool seam: how Kumaru grows from "chat" into "agent".

Status in v0.1
--------------
Deliberately **inert**. `agent.enabled_tools` is empty, so the chat path never
touches this module. The machinery exists now because retrofitting tool calling
later means changing the message schema, the streaming protocol, and the agent
loop all at once — whereas building the seam up front costs one file and makes
the upgrade purely additive.

The model
--------
A tool is a typed function the model may ask to run. The loop is:

1. Describe the available tools in the request (JSON Schema).
2. The model replies with either prose (done) or a structured tool call.
3. Execute it, append the result as a `TOOL` message, and call the model again.
4. Repeat until it stops asking, or `max_tool_iterations` is reached.

Why the iteration cap is not optional
-------------------------------------
Step 4 is an unbounded loop driven by a stochastic process. Without a hard cap,
one confused model can burn your GPU and your context window indefinitely. Caps
on iterations, and timeouts on individual tools, are load-bearing safety
properties — not polish.

Adding a tool
-------------
Subclass `Tool`, decorate with ``@TOOLS.register("name")``, and add the name to
``agent.enabled_tools`` in your config. Nothing else changes. Tools shipped by
other pip packages are discovered automatically through the ``kumaru.tools``
entry-point group.
"""

from __future__ import annotations

import asyncio
import datetime as _dt
from abc import ABC, abstractmethod
from typing import Any

from kumaru.core.errors import ToolError
from kumaru.core.logging import get_logger
from kumaru.core.registry import TOOLS

logger = get_logger(__name__)


class Tool(ABC):
    """A capability the model can invoke."""

    #: Stable identifier the model emits. Keep it short and verb-like.
    name: str = "tool"
    #: Shown to the model. This *is* the API documentation the model reads, so
    #: state what it does, when to use it, and what it returns.
    description: str = ""
    #: JSON Schema for the arguments object.
    parameters: dict[str, Any] = {"type": "object", "properties": {}}
    #: Hard wall-clock limit. A hung tool must never hang the conversation.
    timeout_s: float = 30.0
    #: Tools that mutate state require confirmation before running.
    read_only: bool = True

    @abstractmethod
    async def run(self, **kwargs: Any) -> str:
        """Execute and return a string for the model to read."""

    def spec(self) -> dict[str, Any]:
        """OpenAI-style function-calling declaration."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    async def safe_run(self, **kwargs: Any) -> str:
        """Run with a timeout, converting every failure into readable text.

        Errors are returned to the *model*, not raised, on purpose: a model that
        is told "that path does not exist" can correct itself on the next turn,
        whereas a raised exception ends the conversation.
        """
        try:
            return await asyncio.wait_for(self.run(**kwargs), timeout=self.timeout_s)
        except TimeoutError:
            return f"ERROR: tool '{self.name}' timed out after {self.timeout_s}s."
        except ToolError as exc:
            return f"ERROR: {exc}"
        except Exception as exc:  # noqa: BLE001 - surfaced to the model as text
            logger.exception("tool '%s' failed", self.name)
            return f"ERROR: tool '{self.name}' failed: {type(exc).__name__}: {exc}"


@TOOLS.register("current_time")
class CurrentTimeTool(Tool):
    """Reference implementation — copy this file's shape when adding your own.

    Chosen because it is genuinely useful (models cannot know the time) and has
    zero dependencies, zero side effects, and no security surface.
    """

    name = "current_time"
    description = (
        "Return the current local date and time. Use when the user asks about "
        "today, now, or anything requiring the present date."
    )
    parameters = {
        "type": "object",
        "properties": {
            "format": {
                "type": "string",
                "description": "strftime format string.",
                "default": "%Y-%m-%d %H:%M:%S",
            }
        },
        "required": [],
    }

    async def run(self, **kwargs: Any) -> str:
        fmt = kwargs.get("format") or "%Y-%m-%d %H:%M:%S"
        return _dt.datetime.now().strftime(fmt)


class ToolBelt:
    """The set of tools enabled for one agent, resolved from config names."""

    def __init__(self, names: list[str]) -> None:
        self.tools: dict[str, Tool] = {}
        for name in names:
            self.tools[name] = TOOLS.create(name)

    def __bool__(self) -> bool:
        return bool(self.tools)

    def specs(self) -> list[dict[str, Any]]:
        return [t.spec() for t in self.tools.values()]

    def descriptions(self) -> list[str]:
        return [f"{t.name}: {t.description}" for t in self.tools.values()]

    async def invoke(self, name: str, arguments: dict[str, Any]) -> str:
        tool = self.tools.get(name)
        if tool is None:
            return f"ERROR: unknown tool '{name}'. Available: {', '.join(self.tools) or 'none'}"
        logger.info("tool call: %s(%s)", name, arguments)
        return await tool.safe_run(**arguments)

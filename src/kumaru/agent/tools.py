"""Tools: how Kumaru acts instead of only talking.

Status
------
Live. With tools enabled (``agent.enabled_tools``, or ``kumaru chat --workspace
DIR``, which adds the project-folder tools), `ChatAgent` runs the loop below on
backends that can call tools: llamacpp_server, openai_compat and echo (scripted,
for tests). With the list empty, Kumaru is a pure chat model and never runs one.

The loop
--------
1. Describe the offered tools in the request (JSON Schema).
2. The model replies with either prose (done) or structured tool calls.
3. Run each call, append its result as a `TOOL` message, and call the model again.
4. Repeat until it stops asking, or ``agent.max_tool_iterations`` rounds have run.

Approval is the safety property
-------------------------------
A tool that only reads (``read_only = True``) runs as soon as the model asks.
Any other tool runs in two steps: `prepare()` checks the arguments and describes
exactly what would happen (a diff, a command line) without doing it, and the
`Action` it returns is applied only after the approver, the person at the
terminal, says yes. Interfaces that cannot ask (the HTTP server, the web UI,
eval and benchmarks) are never offered those tools, nor the project-folder
tools: `ToolBelt` withholds them, so the model cannot even request them.

Everything a tool receives comes from the model, and everything it returns
(file contents, command output) goes back to the model. Treat both as untrusted:
a file can contain text written to steer the model ("ignore your instructions
and ..."). Tools validate their own arguments, and the approval prompt shows the
real effect rather than the model's description of it.

Why the iteration cap is not optional
-------------------------------------
Step 4 is an unbounded loop driven by a stochastic process. Without a hard cap,
one confused model can burn your GPU and your context window indefinitely. Caps
on iterations, and timeouts on individual tools, are load-bearing safety
properties — not polish.

Adding a tool
-------------
Subclass `Tool`, decorate with ``@TOOLS.register("name")``, and add the name to
``agent.enabled_tools`` in your config. A tool that changes anything sets
``read_only = False`` and overrides `prepare()`, so the summary the user approves
shows exactly what `apply` will do. Tools shipped by other pip packages are
discovered automatically through the ``kumaru.tools`` entry-point group.
"""

from __future__ import annotations

import asyncio
import datetime as _dt
import json
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeVar

from kumaru.core.errors import ConfigError, ToolError
from kumaru.core.logging import get_logger
from kumaru.core.registry import TOOLS

if TYPE_CHECKING:
    from kumaru.agent.workspace import Workspace
    from kumaru.core.config import WorkspaceConfig

logger = get_logger(__name__)
T = TypeVar("T")

#: What `kumaru chat --workspace DIR` adds to ``agent.enabled_tools``.
WORKSPACE_TOOLS = ("list_files", "read_file", "write_file", "edit_file", "run_command")


@dataclass
class Action:
    """A change that is ready to make: what the user is shown, and what makes it."""

    summary: str
    apply: Callable[[], Awaitable[str]]


@dataclass(frozen=True)
class ApprovalRequest:
    """One decision for the approver: may this tool call run?"""

    tool: str
    arguments: dict[str, Any]
    #: Exactly what will happen: the diff, or the program and its arguments.
    summary: str


#: Answers an `ApprovalRequest`: True runs the action, False declines it.
Approver = Callable[[ApprovalRequest], Awaitable[bool]]


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
    #: False for tools that change anything: they run only after approval.
    read_only: bool = True
    #: True for tools that work in the project folder (``agent.workspace``).
    #: They are built with a `Workspace`, and offered only where approval is.
    needs_workspace: bool = False

    @abstractmethod
    async def run(self, **kwargs: Any) -> str:
        """Execute and return a string for the model to read."""

    async def prepare(self, **kwargs: Any) -> Action:
        """Check the arguments and describe what running would do, without doing it.

        The default suits tools whose arguments say it all. Tools that change
        files or start programs override it to validate first and show the real
        effect: a diff, the resolved program.
        """
        shown = json.dumps(kwargs, ensure_ascii=False, sort_keys=True)
        return Action(summary=f"{self.name}({shown})", apply=lambda: self.run(**kwargs))

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
        return await self._guarded(lambda: self.run(**kwargs))

    async def safe_prepare(self, **kwargs: Any) -> Action | str:
        """`prepare` behind the same safety net: an `Action`, or an ERROR text."""
        return await self._guarded(lambda: self.prepare(**kwargs))

    async def safe_apply(self, action: Action) -> str:
        """Apply an approved action behind the same safety net."""
        return await self._guarded(action.apply)

    async def _guarded(self, call: Callable[[], Awaitable[T]]) -> T | str:
        # `call()` runs inside the try, so arguments the tool does not take (a
        # TypeError at call time) are reported like any other failure.
        try:
            return await asyncio.wait_for(call(), timeout=self.timeout_s)
        except (TimeoutError, asyncio.TimeoutError):  # two distinct classes before Python 3.11
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


def declined(name: str) -> str:
    """The result the model reads when the user says no."""
    return (
        f"DENIED: the user declined this {name} call. Nothing was done. Do not retry "
        "it unchanged; ask the user what they want instead."
    )


class ToolBelt:
    """The tools offered to the model for one agent, resolved from config names.

    Tools that change things (``read_only = False``) or work in the project
    folder (``needs_workspace``) are offered only when there is an `approver`.
    Without one they are left out and listed in `withheld`: an interface that
    cannot ask must not change anything, and must not show the folder either.
    """

    def __init__(
        self,
        names: Sequence[str],
        *,
        workspace: WorkspaceConfig | None = None,
        approver: Approver | None = None,
    ) -> None:
        self.approver = approver
        self.tools: dict[str, Tool] = {}
        self.withheld: list[str] = []
        self.workspace: Workspace | None = None
        for name in dict.fromkeys(names):
            cls = TOOLS.get(name)
            gated = not getattr(cls, "read_only", True) or getattr(cls, "needs_workspace", False)
            if gated and approver is None:
                self.withheld.append(name)
                continue
            if getattr(cls, "needs_workspace", False):
                if self.workspace is None:
                    self.workspace = _open_workspace(workspace)
                self.tools[name] = cls(self.workspace)
            else:
                self.tools[name] = cls()

    def __bool__(self) -> bool:
        return bool(self.tools)

    def specs(self) -> list[dict[str, Any]]:
        return [t.spec() for t in self.tools.values()]

    def descriptions(self) -> list[str]:
        return [f"{t.name}: {t.description}" for t in self.tools.values()]

    async def invoke(self, name: str, arguments: dict[str, Any]) -> str:
        """Run one call and return its result as text; never raises for tool failures."""
        tool = self.tools.get(name)
        if tool is None:
            return f"ERROR: unknown tool '{name}'. Available: {', '.join(self.tools) or 'none'}"
        shown = json.dumps(arguments, ensure_ascii=False, default=str)
        logger.info("tool call: %s(%s)", name, shown if len(shown) <= 200 else shown[:200] + "…")
        if tool.read_only:
            return await tool.safe_run(**arguments)
        if self.approver is None:
            # Unreachable through __init__, which withholds these tools; kept
            # so a hand-built belt still cannot change anything unasked.
            return f"ERROR: tool '{name}' needs approval, and nobody here can give it. Nothing was done."
        action = await tool.safe_prepare(**arguments)
        if isinstance(action, str):
            return action  # invalid: the user is not asked about a change that cannot happen
        request = ApprovalRequest(tool=name, arguments=dict(arguments), summary=action.summary)
        if not await self.approver(request):
            logger.info("tool call declined: %s", name)
            return declined(name)
        logger.info("tool call approved: %s", name)
        return await tool.safe_apply(action)


def _open_workspace(config: WorkspaceConfig | None) -> Workspace:
    from kumaru.agent.workspace import Workspace  # imports this module

    if config is None or not config.root:
        raise ConfigError(
            "The project-folder tools are enabled, but no project folder is set.",
            hint="Start the chat with `kumaru chat --workspace DIR`, or set agent.workspace.root.",
        )
    return Workspace(config)


# The project-folder tools import more (difflib, subprocess), and need a folder,
# so they register lazily: listed by `kumaru info`, imported when enabled.
TOOLS.register_lazy("list_files", "kumaru.agent.workspace:ListFilesTool")
TOOLS.register_lazy("read_file", "kumaru.agent.workspace:ReadFileTool")
TOOLS.register_lazy("write_file", "kumaru.agent.workspace:WriteFileTool")
TOOLS.register_lazy("edit_file", "kumaru.agent.workspace:EditFileTool")
TOOLS.register_lazy("run_command", "kumaru.agent.commands:RunCommandTool")

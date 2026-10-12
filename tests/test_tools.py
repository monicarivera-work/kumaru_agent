"""Protect the tool safety net: a tool's result, timeout or failure always comes back as text."""

import asyncio
import datetime as dt
from typing import Any

import pytest

from kumaru.agent.tools import Tool, ToolBelt
from kumaru.core.errors import ToolError


class EchoArgsTool(Tool):
    name = "echo_args"

    async def run(self, **kwargs: Any) -> str:
        return ",".join(f"{key}={value}" for key, value in sorted(kwargs.items()))


class SleepyTool(Tool):
    name = "sleepy"
    timeout_s = 0.01

    def __init__(self) -> None:
        self.cancelled = False

    async def run(self, **kwargs: Any) -> str:
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        return "too late"


class FailingTool(Tool):
    name = "failing"

    def __init__(self, exc: Exception) -> None:
        self.exc = exc

    async def run(self, **kwargs: Any) -> str:
        raise self.exc


class StrictTool(Tool):
    name = "strict"

    async def run(self, *, path: str) -> str:  # type: ignore[override]
        return path


async def test_safe_run_returns_the_tools_answer() -> None:
    assert await EchoArgsTool().safe_run(b=2, a=1) == "a=1,b=2"


async def test_safe_run_stops_a_hung_tool_and_says_so() -> None:
    tool = SleepyTool()
    assert await tool.safe_run() == "ERROR: tool 'sleepy' timed out after 0.01s."
    assert tool.cancelled  # the hung call is cancelled, not left running


@pytest.mark.parametrize(
    ("exc", "expected"),
    [
        # An expected failure: its message (and hint) is written for the model.
        (
            ToolError("no such file: notes.txt", hint="list the folder first"),
            "ERROR: no such file: notes.txt (list the folder first)",
        ),
        # Anything else is a bug in the tool, reported with its type.
        (ValueError("boom"), "ERROR: tool 'failing' failed: ValueError: boom"),
        (KeyError("path"), "ERROR: tool 'failing' failed: KeyError: 'path'"),
    ],
)
async def test_safe_run_returns_failures_as_text_instead_of_raising(
    exc: Exception, expected: str
) -> None:
    assert await FailingTool(exc).safe_run() == expected


async def test_safe_run_reports_arguments_the_tool_does_not_take() -> None:
    result = await StrictTool().safe_run(pth="notes.txt")  # a model misspelled `path`
    assert result.startswith("ERROR: tool 'strict' failed: TypeError: ")
    assert "pth" in result


async def test_invoke_runs_the_built_in_tool_with_its_arguments() -> None:
    year = dt.datetime.now().year
    result = await ToolBelt(["current_time"]).invoke("current_time", {"format": "%Y"})
    assert result in {str(year), str(year + 1)}  # in case the year turns mid-test


async def test_invoke_goes_through_the_safety_net() -> None:
    belt = ToolBelt([])
    belt.tools.update(
        {"echo_args": EchoArgsTool(), "sleepy": SleepyTool(), "failing": FailingTool(OSError("disk full"))}
    )
    assert await belt.invoke("echo_args", {"b": 2, "a": 1}) == "a=1,b=2"
    assert await belt.invoke("sleepy", {}) == "ERROR: tool 'sleepy' timed out after 0.01s."
    assert await belt.invoke("failing", {}) == "ERROR: tool 'failing' failed: OSError: disk full"


@pytest.mark.parametrize(
    ("enabled", "available"), [([], "none"), (["current_time"], "current_time")]
)
async def test_invoke_names_the_enabled_tools_when_asked_for_an_unknown_one(
    enabled: list[str], available: str
) -> None:
    result = await ToolBelt(enabled).invoke("read_file", {"path": "notes.txt"})
    assert result == f"ERROR: unknown tool 'read_file'. Available: {available}"

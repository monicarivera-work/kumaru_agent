"""Protect the tool safety net: a tool's result, timeout or failure always comes back as text,
and a tool that changes anything does so only after the approver says yes.
"""

import asyncio
import datetime as dt
from pathlib import Path
from typing import Any

import pytest

from kumaru.agent.tools import (
    WORKSPACE_TOOLS,
    Action,
    ApprovalRequest,
    Approver,
    Tool,
    ToolBelt,
    declined,
)
from kumaru.core.config import WorkspaceConfig
from kumaru.core.errors import ConfigError, ToolError
from kumaru.core.registry import TOOLS


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


# -- approval ------------------------------------------------------------------------


class ChangeTool(Tool):
    """Stands in for a tool that changes something, and records each change it makes."""

    name = "change"
    read_only = False

    def __init__(self) -> None:
        self.applied: list[str] = []

    async def prepare(self, **kwargs: Any) -> Action:
        if kwargs.get("bad"):
            raise ToolError("that change cannot be made", hint="fix the arguments")
        value = str(kwargs.get("value", ""))

        async def apply() -> str:
            self.applied.append(value)
            return f"changed to {value}"

        return Action(summary=f"Change to {value}", apply=apply)

    async def run(self, **kwargs: Any) -> str:
        raise AssertionError("a tool that changes things runs only through prepare and apply")


def approver(*answers: bool) -> tuple[Approver, list[ApprovalRequest]]:
    """An approver that gives `answers` in order; and the list of what it was asked."""
    asked: list[ApprovalRequest] = []
    replies = list(answers)

    async def answer(request: ApprovalRequest) -> bool:
        asked.append(request)
        return replies.pop(0)

    return answer, asked


def belt_with_change(answer: Approver | None) -> tuple[ToolBelt, ChangeTool]:
    belt = ToolBelt([], approver=answer)
    tool = belt.tools["change"] = ChangeTool()
    return belt, tool


async def test_a_change_is_made_only_after_approval() -> None:
    answer, asked = approver(True, False)
    belt, tool = belt_with_change(answer)
    assert await belt.invoke("change", {"value": "a"}) == "changed to a"
    assert await belt.invoke("change", {"value": "b"}) == declined("change")
    assert tool.applied == ["a"]
    assert asked == [
        ApprovalRequest(tool="change", arguments={"value": "a"}, summary="Change to a"),
        ApprovalRequest(tool="change", arguments={"value": "b"}, summary="Change to b"),
    ]


async def test_a_change_that_cannot_be_made_is_reported_without_asking() -> None:
    answer, asked = approver()
    belt, tool = belt_with_change(answer)
    assert await belt.invoke("change", {"bad": True}) == "ERROR: that change cannot be made (fix the arguments)"
    assert asked == [] and tool.applied == []


async def test_tools_that_only_read_never_ask() -> None:
    answer, asked = approver()
    assert await ToolBelt(["current_time"], approver=answer).invoke("current_time", {"format": "x"}) == "x"
    assert asked == []


async def test_a_failure_while_applying_comes_back_as_text() -> None:
    class BrokenChange(ChangeTool):
        async def prepare(self, **kwargs: Any) -> Action:
            async def apply() -> str:
                raise OSError("disk full")

            return Action(summary="Change", apply=apply)

    answer, _ = approver(True)
    belt = ToolBelt([], approver=answer)
    belt.tools["change"] = BrokenChange()
    assert await belt.invoke("change", {}) == "ERROR: tool 'change' failed: OSError: disk full"


async def test_without_an_approver_even_a_hand_built_belt_changes_nothing() -> None:
    belt, tool = belt_with_change(None)
    assert await belt.invoke("change", {"value": "a"}) == (
        "ERROR: tool 'change' needs approval, and nobody here can give it. Nothing was done."
    )
    assert tool.applied == []


async def test_the_default_prepare_shows_the_call_and_runs_it_on_apply() -> None:
    action = await EchoArgsTool().prepare(b=2, a="é")
    assert action.summary == 'echo_args({"a": "é", "b": 2})'
    assert await action.apply() == "a=é,b=2"


def test_tools_that_change_things_are_offered_only_with_an_approver(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(TOOLS._items, "change", ChangeTool)
    belt = ToolBelt(["change", "current_time", "change"])
    assert list(belt.tools) == ["current_time"]
    assert belt.withheld == ["change"]
    answer, _ = approver()
    assert list(ToolBelt(["change", "current_time"], approver=answer).tools) == ["change", "current_time"]


def test_project_folder_tools_need_a_project_folder() -> None:
    answer, _ = approver()
    for workspace in (None, WorkspaceConfig()):
        with pytest.raises(ConfigError, match="no project folder is set") as error:
            ToolBelt(["read_file"], workspace=workspace, approver=answer)
        assert "--workspace" in (error.value.hint or "")


def test_project_folder_tools_share_one_workspace(scratch_dir: Path) -> None:
    answer, _ = approver()
    belt = ToolBelt(WORKSPACE_TOOLS, workspace=WorkspaceConfig(root=str(scratch_dir)), approver=answer)
    assert list(belt.tools) == list(WORKSPACE_TOOLS)
    assert belt.workspace is not None
    assert all(tool.workspace is belt.workspace for tool in belt.tools.values())

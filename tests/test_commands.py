"""Run programs in the project folder: only allowed ones, never through a shell, never past their limit.

`run_command` is the one tool that executes code, so these tests pin down what
the approval prompt shows, which programs can start (and which never can), the
environment they get, and that a time limit or a cancelled turn ends the
program together with every program it started.
"""

import asyncio
import os
import re
import sys
import threading
import time
from pathlib import Path

import pytest
from pydantic import ValidationError

from kumaru.agent import commands
from kumaru.agent.commands import (
    RunCommandTool,
    _argv,
    _Capture,
    _check_batch_arguments,
    _clean,
    _run,
    build_env,
    display_command,
    safe_path,
    which,
)
from kumaru.agent.workspace import Workspace
from kumaru.core.config import WorkspaceConfig, program_name
from kumaru.core.errors import ToolError

windows_only = pytest.mark.skipif(os.name != "nt", reason="Windows behaviour")
posix_only = pytest.mark.skipif(os.name == "nt", reason="POSIX behaviour")

# Commands never see Kumaru's own virtual environment on PATH (so `pip install`
# in a project cannot change Kumaru), so tests put the base interpreter first.
BASE_PYTHON_DIR = os.path.dirname(getattr(sys, "_base_executable", sys.executable))

# Appends to beat.txt until stopped; ends by itself after about 20 seconds.
HEARTBEAT = (
    "import time\n"
    "for _ in range(400):\n"
    "    with open('beat.txt', 'a') as f:\n"
    "        f.write('.')\n"
    "    time.sleep(0.05)\n"
)


@pytest.fixture
def root(scratch_dir: Path) -> Path:
    path = scratch_dir / "project"
    path.mkdir()
    return path


@pytest.fixture
def python_on_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PATH", BASE_PYTHON_DIR + os.pathsep + os.environ.get("PATH", ""))


def make_tool(root: Path, **settings: object) -> RunCommandTool:
    return RunCommandTool(Workspace(WorkspaceConfig(root=str(root), **settings)))


async def run(root: Path, command: list, **arguments: object) -> str:
    action = await make_tool(root).prepare(command=command, **arguments)
    return await action.apply()


def assert_stopped(beat: Path) -> None:
    """Once every process is gone, the heartbeat file stops growing."""
    time.sleep(0.5)
    size = beat.stat().st_size
    assert size > 0
    time.sleep(0.6)
    assert beat.stat().st_size == size


# -- what may run ----------------------------------------------------------------------


async def test_run_command_runs_only_after_approval_and_says_what_it_allows(root: Path) -> None:
    tool = make_tool(root, command_timeout_s=60)
    assert (tool.read_only, tool.needs_workspace, tool.timeout_s) == (False, True, 90)
    assert tool.parameters["properties"]["timeout_s"]["maximum"] == 60
    assert "Allowed programs: python, node, npm, npx, dotnet." in tool.description
    assert await tool.safe_run(command=["python", "-V"]) == (
        "ERROR: run_command runs only through prepare() and the user's approval."
    )


@pytest.mark.parametrize(
    ("command", "message"),
    [
        ("npm test", "command must be a list, not one string"),
        ([], 'command must be a non-empty list, e.g. ["npm", "test"].'),
        (None, "command must be a non-empty list"),
        ({"program": "npm"}, "command must be a non-empty list"),
        (["npm", True], "every item in command must be a string."),
        (["npm", None], "every item in command must be a string."),
        (["npm", ["test"]], "every item in command must be a string."),
        (["npm", "te\x00st"], "command contains a NUL character."),
        ([" ", "test"], "the first item in command must name the program."),
        (["npm", *["x"] * 256], "command is too long (at most 256 items, 16000 characters)."),
        (["npm", "x" * 16_000], "command is too long"),
    ],
)
def test_a_command_is_a_short_list_of_strings(command: object, message: str) -> None:
    with pytest.raises(ToolError, match=re.escape(message)):
        _argv(command)


def test_numbers_in_a_command_become_arguments() -> None:
    assert _argv(["python", "-m", "http.server", 8000, 1.5]) == [
        "python", "-m", "http.server", "8000", "1.5",
    ]  # fmt: skip


@pytest.mark.parametrize(
    ("name", "message"),
    [
        ("C:\\Python312\\python.exe", "is not a bare program name"),
        ("bin/python", "is not a bare program name"),
        ("my tool", "is not a bare program name"),
        ("", "is not a bare program name"),
        ("cmd", "is a shell or launcher"),
        ("CMD.EXE", "is a shell or launcher"),
        ("powershell", "is a shell or launcher"),
        ("pwsh", "is a shell or launcher"),
        ("bash", "is a shell or launcher"),
        ("env", "is a shell or launcher"),
        ("wsl", "is a shell or launcher"),
        ("xargs", "is a shell or launcher"),
        ("nohup", "is a shell or launcher"),
        ("timeout", "is a shell or launcher"),
        ("schtasks", "is a shell or launcher"),
        ("WMIC.exe", "is a shell or launcher"),
        ("wt", "is a shell or launcher"),
    ],
)
def test_the_allowed_list_takes_bare_names_and_never_a_shell(name: str, message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        WorkspaceConfig(allowed_commands=["python", name])


def test_the_allowed_list_is_normalized() -> None:
    config = WorkspaceConfig(allowed_commands=["Python.EXE", "python", "npm.cmd", "cargo"])
    assert config.allowed_commands == ["python", "npm", "cargo"]
    assert [program_name(n) for n in ("node.exe", "NPX.CMD", " dotnet ", "python3")] == [
        "node", "npx", "dotnet", "python3",
    ]  # fmt: skip


async def test_prepare_refuses_before_anything_is_shown(root: Path, python_on_path: None) -> None:
    (root / "a.txt").write_text("x", encoding="utf-8")
    tool = make_tool(root)
    allowed = (
        "Allowed: python, node, npm, npx, dotnet "
        "(the user sets this list in agent.workspace.allowed_commands)."
    )
    cases = [
        ({"command": ["python"], "shell": True}, "run_command has no argument 'shell'; it takes command, cwd, timeout_s."),
        ({"command": "python -V"}, "command must be a list, not one string"),
        ({"command": ["python", "-V"], "cwd": ".."}, "uses '..'"),
        ({"command": ["python", "-V"], "cwd": "C:\\Windows"}, "which file names cannot contain"),
        ({"command": ["python", "-V"], "cwd": "missing"}, "cwd 'missing' is not a folder in the project."),
        ({"command": ["python", "-V"], "cwd": "a.txt"}, "cwd 'a.txt' is not a folder in the project."),
        ({"command": ["python", "-V"], "timeout_s": 0}, "timeout_s must be a number of seconds, at least 1."),
        ({"command": ["python", "-V"], "timeout_s": 0.5}, "timeout_s must be a number of seconds, at least 1."),
        ({"command": ["python", "-V"], "timeout_s": True}, "timeout_s must be a number of seconds, at least 1."),
        ({"command": ["python", "-V"], "timeout_s": "soon"}, "timeout_s must be a number of seconds, at least 1."),
        ({"command": ["python", "-V"], "timeout_s": float("nan")}, "timeout_s must be a number of seconds, at least 1."),
        ({"command": ["python", "-V"], "timeout_s": float("inf")}, "timeout_s must be a number of seconds, at least 1."),
        ({"command": ["cmd", "/c", "del", "*"]}, f"'cmd' is not an allowed program. {allowed}"),
        ({"command": ["powershell", "-c", "x"]}, f"'powershell' is not an allowed program. {allowed}"),
        ({"command": ["git", "push"]}, f"'git' is not an allowed program. {allowed}"),
    ]
    for arguments, message in cases:
        with pytest.raises(ToolError, match=re.escape(message)):
            await tool.prepare(**arguments)


async def test_an_allowed_program_must_be_installed(root: Path) -> None:
    tool = make_tool(root, allowed_commands=["zz-kumaru-missing"])
    with pytest.raises(ToolError, match=r"^'zz-kumaru-missing' is allowed, but it was not found on PATH\."):
        await tool.prepare(command=["zz-kumaru-missing"])


async def test_nothing_runs_when_nothing_is_allowed(root: Path) -> None:
    tool = make_tool(root, allowed_commands=[])
    assert "Allowed programs: none." in tool.description
    with pytest.raises(ToolError, match=r"^'python' is not an allowed program\. Allowed: none "):
        await tool.prepare(command=["python", "-V"])


async def test_a_program_in_the_folder_runs_by_path_only_if_its_name_is_allowed(root: Path) -> None:
    (root / "tools").mkdir()
    for name in ("node.exe", "evil.exe"):
        (root / "tools" / name).write_bytes(b"")
    tool = make_tool(root)
    cases = {
        "tools/evil.exe": r"^'evil\.exe' is not an allowed program\.",
        "tools/missing.exe": r"^'tools/missing\.exe' is not a file in the project folder\.$",
        "./tools": r"^'tools' is not a file in the project folder\.$",
        "../python.exe": r"uses '\.\.'",
        "C:\\Windows\\System32\\cmd.exe": "which file names cannot contain",
        "/usr/bin/python": "is absolute",
    }
    for program, message in cases.items():
        with pytest.raises(ToolError, match=message):
            await tool.prepare(command=[program])


@windows_only
async def test_a_program_in_the_folder_must_be_a_windows_program(root: Path) -> None:
    (root / "tools").mkdir()
    (root / "tools" / "node").write_bytes(b"")
    (root / "tools" / "node.exe").write_bytes(b"")  # empty, so Windows cannot start it
    tool = make_tool(root)
    with pytest.raises(ToolError, match=r"^'tools/node' is not a program \(\.exe, \.com, \.bat or \.cmd\)\.$"):
        await tool.prepare(command=["tools/node"])
    action = await tool.prepare(command=["tools\\node.exe", "--version"])
    assert action.summary.splitlines()[0] == "$ tools\\node.exe --version"
    assert action.summary.splitlines()[2] == f"program: {tool.workspace.root / 'tools' / 'node.exe'}"
    with pytest.raises(ToolError, match=r"^could not start .*node\.exe: "):
        await action.apply()


@posix_only
async def test_a_program_in_the_folder_must_be_executable(root: Path) -> None:
    (root / "tools").mkdir()
    (root / "tools" / "node").write_bytes(b"")
    with pytest.raises(ToolError, match=r"^'tools/node' is not executable\.$"):
        await make_tool(root).prepare(command=["tools/node"])


@pytest.mark.parametrize(
    ("value", "limit"),
    [(None, 300), (5, 5), ("5", 5), (2.7, 2), (10**6, 300), (10**400, 300)],
    ids=["default", "int", "str", "float", "over-the-limit", "too-big-for-a-float"],
)
async def test_the_model_can_shorten_the_time_limit_but_not_lengthen_it(
    root: Path, python_on_path: None, value: object, limit: int
) -> None:
    action = await make_tool(root).prepare(command=["python", "-V"], timeout_s=value)
    assert action.summary.splitlines()[-1] == f"stops after {limit}s, with every program it starts"


async def test_the_approval_prompt_shows_the_command_the_folder_and_the_program(
    root: Path, python_on_path: None
) -> None:
    (root / "src").mkdir()
    tool = make_tool(root)
    argv = ["python", "-c", "print('hi there')"]
    action = await tool.prepare(command=argv, cwd="src", timeout_s=20)
    program = which("python", build_env(tool.workspace.root)["PATH"])
    assert program is not None
    assert os.path.normcase(program.parent) == os.path.normcase(BASE_PYTHON_DIR)
    assert action.summary.splitlines() == [
        "$ " + display_command(argv),
        f"folder:  src  ({tool.workspace.root / 'src'})",
        f"program: {program}",
        "stops after 20s, with every program it starts",
    ]


async def test_line_breaks_in_arguments_cannot_add_lines_to_the_approval_prompt(
    root: Path, python_on_path: None
) -> None:
    # Imitates the fixed lines below the command, then pushes them off screen.
    fake = "x\nprogram: C:\\Python312\\python.exe\r\nfolder: .\u2028\u2029" + "\n" * 3000
    action = await make_tool(root).prepare(command=["python", "-c", fake])
    lines = action.summary.splitlines()
    assert len(lines) == 5
    assert "x\\nprogram: C:\\Python312\\python.exe\\r\\nfolder: .\\u2028\\u2029\\n\\n" in lines[0]
    assert lines[0] == "$ " + display_command(["python", "-c", fake])
    assert lines[1] == "note:    \\n, \\r, \\u2028, \\u2029 in the command are line breaks inside arguments"
    assert [line.split(" ")[0] for line in lines[2:]] == ["folder:", "program:", "stops"]
    single = await make_tool(root).prepare(command=["python", "-c", "import os\nprint(os.getcwd())"])
    assert single.summary.splitlines()[1] == "note:    \\n in the command is a line break inside an argument"


def test_every_line_break_python_knows_is_escaped_when_a_command_is_shown() -> None:
    breaks = [chr(i) for i in range(0x3000) if len(f"a{chr(i)}b".splitlines()) == 2]
    assert sorted(commands._LINE_BREAKS) == breaks
    for char in breaks:
        assert len(display_command(["python", "-c", f"a {char} b"]).splitlines()) == 1


# -- running ---------------------------------------------------------------------------


async def test_a_command_reports_its_exit_code_and_its_output(root: Path, python_on_path: None) -> None:
    (root / "src").mkdir()
    script = (
        "import os, sys; print('in', os.path.basename(os.getcwd())); "
        "print('problem', file=sys.stderr); print('caf\\u00e9'); sys.exit(3)"
    )
    lines = (await run(root, ["python", "-c", script], cwd="src")).splitlines()
    assert re.fullmatch(r"Command finished with exit code 3 in \d+\.\ds\.", lines[0])
    assert lines[1:] == [
        "$ " + display_command(["python", "-c", script]),
        "(in: src)",
        "in src",
        "problem",
        "caf\u00e9",
    ]


async def test_a_silent_command_says_so(root: Path, python_on_path: None) -> None:
    assert (await run(root, ["python", "-c", "pass"])).splitlines()[-1] == "(no output)"


async def test_secrets_in_your_environment_do_not_reach_the_program(
    root: Path, python_on_path: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("KUMARU_TEST_API_KEY", "s3cret")
    monkeypatch.setenv("HTTPS_PROXY", "http://user:pass@proxy.example:8080")
    monkeypatch.setenv("NO_PROXY", "localhost")
    script = (
        "import os; print([os.environ.get(k) for k in "
        "('KUMARU_TEST_API_KEY', 'HTTPS_PROXY', 'NO_PROXY', 'NO_COLOR')])"
    )
    output = (await run(root, ["python", "-c", script])).splitlines()[-1]
    assert output == "[None, None, 'localhost', '1']"


@windows_only
async def test_program_names_match_whatever_their_case_or_extension(root: Path, python_on_path: None) -> None:
    assert (await run(root, ["Python.EXE", "-c", "print('ok')"])).splitlines()[-1] == "ok"


@windows_only
async def test_a_batch_file_runs_but_cmd_never_sees_command_characters(
    root: Path, scratch_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bin_dir = scratch_dir / "bin"
    bin_dir.mkdir()
    (bin_dir / "npm.cmd").write_bytes(b"@echo off\r\necho fake npm %1\r\n")
    monkeypatch.setenv("PATH", str(bin_dir) + os.pathsep + os.environ["PATH"])
    tool = make_tool(root)
    refused = (
        "npm is run by cmd.exe, which would treat '&' in the argument 'a&calc' "
        "as a command character. Leave it out."
    )
    with pytest.raises(ToolError, match=f"^{re.escape(refused)}$"):
        await tool.prepare(command=["npm", "a&calc"])
    action = await tool.prepare(command=["npm", "test"])
    assert action.summary.splitlines()[2] == f"program: {bin_dir / 'npm.cmd'}  (batch script, runs in cmd.exe)"
    lines = (await action.apply()).splitlines()
    assert re.fullmatch(r"Command finished with exit code 0 in \d+\.\ds\.", lines[0])
    assert lines[1:] == ["$ npm test", "(in: .)", "fake npm test"]


@windows_only
async def test_a_batch_file_in_the_folder_is_labelled_as_one(root: Path) -> None:
    # Its name passes the allowed list ("python"), but cmd.exe runs it, not Python.
    (root / "tools").mkdir()
    (root / "tools" / "python.cmd").write_bytes(b"@echo off\r\necho not python\r\n")
    tool = make_tool(root)
    action = await tool.prepare(command=["tools/python.cmd", "-V"])
    assert action.summary.splitlines()[2] == (
        f"program: {tool.workspace.root / 'tools' / 'python.cmd'}  (batch script, runs in cmd.exe)"
    )


@pytest.mark.parametrize("arg", ["a&calc", "a|b", "%PATH%", "!x!", "a^b", "<in", ">out", "(x)", '"q"', "a\nb", "a\rb"])
def test_batch_files_refuse_characters_cmd_would_act_on(arg: str) -> None:
    with pytest.raises(ToolError, match=r"^npm\.cmd is run by cmd\.exe, which would treat "):
        _check_batch_arguments(["npm.cmd", "run", arg])


def test_batch_files_take_ordinary_arguments_up_to_cmds_line_limit() -> None:
    _check_batch_arguments(["npm", "run", "build", "--", "--mode=production", "src/app.js", "@scope/pkg"])
    with pytest.raises(ToolError, match=r"^npm is run by cmd\.exe, which cannot take a command this long\.$"):
        _check_batch_arguments(["npm", *["x" * 100] * 80])


async def test_a_timeout_stops_the_program_and_every_program_it_started(
    root: Path, python_on_path: None
) -> None:
    parent = (
        "import subprocess, sys, time\n"
        f"subprocess.Popen([sys.executable, '-c', {HEARTBEAT!r}])\n"
        "print('started', flush=True)\n"
        "time.sleep(30)\n"
    )
    lines = (await run(root, ["python", "-c", parent], timeout_s=2)).splitlines()
    assert lines[0] == "Command timed out after 2s and was stopped, with every program it started."
    assert lines[-1] == "started"
    assert_stopped(root / "beat.txt")


async def test_cancelling_the_turn_stops_the_command(root: Path, python_on_path: None) -> None:
    action = await make_tool(root).prepare(command=["python", "-c", HEARTBEAT])
    task = asyncio.create_task(action.apply())
    beat = root / "beat.txt"
    for _ in range(200):
        if beat.exists():
            break
        await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert_stopped(beat)


def test_exiting_mid_command_stops_it(root: Path, python_on_path: None) -> None:
    # How a closed terminal or a `kill` plays out in `kumaru chat`: SystemExit
    # leaves the event loop, and the cleanup on the way out stops the program.
    beat = root / "beat.txt"

    async def turn() -> None:
        action = await make_tool(root).prepare(command=["python", "-c", HEARTBEAT])
        loop = asyncio.get_running_loop()

        def exit_once_running() -> None:
            if beat.exists():
                raise SystemExit(143)
            loop.call_later(0.05, exit_once_running)

        loop.call_later(0.05, exit_once_running)
        await action.apply()

    with pytest.raises(SystemExit):
        asyncio.run(turn())
    assert_stopped(beat)


@pytest.mark.parametrize("end", ["close", "exit"])
async def test_a_turn_that_ends_without_being_cancelled_stops_the_command(
    root: Path, python_on_path: None, end: str
) -> None:
    # A coroutine dropped unfinished is closed, and an exit raised in it propagates, without a cancel.
    action = await make_tool(root).prepare(command=["python", "-c", HEARTBEAT])
    turn = action.apply()
    runner = turn.send(None)  # started: the program runs, and the turn waits for it
    beat = root / "beat.txt"
    for _ in range(200):
        if beat.exists():
            break
        await asyncio.sleep(0.05)
    if end == "close":
        turn.close()
    else:
        with pytest.raises(SystemExit):
            turn.throw(SystemExit(143))
    for _ in range(200):  # not awaited: no task took charge of the future the turn handed out
        if runner.done():
            break
        await asyncio.sleep(0.05)
    assert runner.result().status == "cancelled"
    assert_stopped(beat)


def test_the_runner_stops_as_soon_as_it_is_cancelled(root: Path, python_on_path: None) -> None:
    env = build_env(root)
    cancel = threading.Event()
    cancel.set()
    outcome = _run([str(which("python", env["PATH"])), "-c", "import time; time.sleep(30)"], root, env, 60, cancel, 1000)
    assert (outcome.status, outcome.exit_code, outcome.output) == ("cancelled", None, "")
    assert outcome.seconds < 10


def test_the_runner_reports_a_program_that_cannot_start(root: Path) -> None:
    with pytest.raises(ToolError, match=r"^could not start .*nope"):
        _run([str(root / "nope.exe")], root, build_env(root), 5, threading.Event(), 1000)


def test_the_report_cuts_a_very_long_command_short() -> None:
    text = commands._report(commands._Outcome("cancelled", None, 0.2, ""), "x" * 400, "src", 9)
    assert text.splitlines() == [
        "Command was cancelled and stopped, with every program it started.",
        "$ " + "x" * 300 + " …",
        "(in: src)",
        "(no output)",
    ]


# -- output ----------------------------------------------------------------------------


async def test_long_output_keeps_its_start_and_its_end(root: Path, python_on_path: None) -> None:
    action = await make_tool(root, max_output_chars=1000).prepare(
        command=["python", "-c", "print('x' * 5000); print('END')"]
    )
    output = (await action.apply()).split("\n", 3)[3]
    assert output.startswith("x" * 400 + "\n… ")
    assert "characters omitted" in output
    assert output.endswith("x\nEND")
    assert len(output) <= 1100


def test_capture_keeps_the_first_and_last_bytes_and_counts_the_rest() -> None:
    capture = _Capture(5)
    for chunk in (b"aaa", b"aa" + b"x" * 10, b"bbbbb"):
        capture.feed(chunk)
    assert capture.text(1000) == "aaaaa\n… 10 bytes of output omitted …\nbbbbb"


def test_capture_returns_short_output_whole_and_cuts_long_text_keeping_more_of_the_end() -> None:
    short = _Capture(100)
    short.feed(b"hello\r\n")
    short.feed(b"w\xc3\xb6rld\n")
    assert short.text(1000) == "hello\nw\u00f6rld"

    long = _Capture(1000)
    long.feed(b"A" * 20 + b"B" * 50 + b"C" * 30)
    assert long.text(50) == "A" * 20 + "\n… 50 characters omitted …\n" + "C" * 30


def test_terminal_output_loses_its_colours_and_keeps_progress_bars_at_their_last_state() -> None:
    raw = "\x1b[32mPASS\x1b[0m test\r\n 10%\r 50%\r100%\n\x1b]0;title\x07done"
    assert _clean(raw) == "PASS test\n100%\ndone"


def test_the_command_is_shown_as_you_would_type_it() -> None:
    argv = ["npm", "run", "my script", "--flag=a b"]
    expected = 'npm run "my script" "--flag=a b"' if os.name == "nt" else "npm run 'my script' '--flag=a b'"
    assert display_command(argv) == expected


# -- the environment -------------------------------------------------------------------


def test_path_loses_every_entry_a_file_in_the_project_could_hide_in(
    root: Path, scratch_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    venv, tools = scratch_dir / "venv", scratch_dir / "tools"
    (venv / "bin").mkdir(parents=True)
    tools.mkdir()
    (root / "bin").mkdir()
    monkeypatch.setattr(sys, "prefix", str(venv))
    monkeypatch.setattr(sys, "base_prefix", str(scratch_dir / "base"))
    value = os.pathsep.join([
        "", ".", "node_modules/.bin", str(root), str(root / "bin"), str(venv / "bin"),
        f'"{tools}"', str(tools), "  ",
    ])  # fmt: skip
    assert safe_path(value, root) == str(tools)


def test_path_keeps_the_interpreter_folder_outside_a_virtual_environment(
    root: Path, scratch_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = scratch_dir / "python"
    base.mkdir()
    monkeypatch.setattr(sys, "prefix", str(base))
    monkeypatch.setattr(sys, "base_prefix", str(base))
    assert safe_path(str(base), root) == str(base)


def test_the_environment_keeps_what_builds_need_and_drops_the_rest(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("KUMARU_TEST_API_KEY", "s3cret")
    monkeypatch.setenv("JAVA_HOME", "C:\\jdk")
    monkeypatch.setenv("HTTP_PROXY", "http://proxy.example:8080")
    monkeypatch.setenv("HTTPS_PROXY", "http://user:pass@proxy.example:8080")
    env = build_env(root)
    assert "KUMARU_TEST_API_KEY" not in env
    assert "HTTPS_PROXY" not in env
    assert env["JAVA_HOME"] == "C:\\jdk"
    assert env["HTTP_PROXY"] == "http://proxy.example:8080"
    added = {
        "PATH": env["PATH"],
        "PYTHONIOENCODING": "utf-8",
        "PYTHONUNBUFFERED": "1",
        "NO_COLOR": "1",
        "NoDefaultCurrentDirectoryInExePath": "1",
    }
    assert {key: env[key] for key in added} == added
    kept = commands._KEEP_ENV | commands._PROXY_ENV
    assert all(key in added or key.upper() in kept for key in env)


def test_which_finds_only_real_program_files(scratch_dir: Path) -> None:
    first, second = scratch_dir / "first", scratch_dir / "second"
    first.mkdir()
    second.mkdir()
    (first / ("tool.exe" if os.name == "nt" else "tool")).mkdir()  # a folder with a program's name
    (first / "tool.txt").write_text("x", encoding="utf-8")
    target = second / ("tool.cmd" if os.name == "nt" else "tool")
    target.write_text("x", encoding="utf-8")
    target.chmod(0o755)
    path = os.pathsep.join(["", str(first), str(second)])
    assert which("tool", path) == target
    assert which("missing", path) is None


@windows_only
def test_which_uses_a_given_extension_only_if_it_names_a_program(scratch_dir: Path) -> None:
    (scratch_dir / "tool.cmd").write_text("x", encoding="utf-8")
    (scratch_dir / "tool.txt").write_text("x", encoding="utf-8")
    assert which("tool.cmd", str(scratch_dir)) == scratch_dir / "tool.cmd"
    assert which("tool.txt", str(scratch_dir)) is None

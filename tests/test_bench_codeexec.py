"""Protect the executor every code benchmark runs model-written programs through.

What must never regress: a pass needs exit code 0 *and* the marker printed
after the last test; a timeout kills everything the program started; secrets
stay out of its environment; its temporary folder goes away; and the Docker
path keeps its isolation flags and reads its results back exactly. The local
tests run real (tiny) Python programs; Docker itself is faked.
"""

import io
import os
import re
import subprocess
import sys
import tarfile
import threading
import time
from pathlib import Path

import pytest

from kumaru.core.errors import KumaruError
from kumaru.eval.benchmarks import codeexec
from kumaru.eval.benchmarks.codeexec import (
    EARLY_EXIT_NOTE,
    TAIL_BYTES,
    Program,
    Step,
    ToolchainMissing,
)

PYTHON = getattr(sys, "_base_executable", None) or sys.executable


def finish(marker: str) -> str:
    return f"\nimport os\nos.write(1, b'\\n{marker}\\n')\n"


def python(code: str, marker: str = "never-printed", *, timeout_s: float = 30, **fields) -> Program:
    step = Step((PYTHON, "-I", "-X", "utf8", "main.py"), timeout_s)
    return Program({"main.py": code}, (step,), marker, **fields)


# -- local runs -------------------------------------------------------------------


def test_a_pass_needs_exit_code_0_and_the_marker() -> None:
    marker = codeexec.new_marker()
    outcome = codeexec.run(python("print('hi')" + finish(marker), marker))
    assert (outcome.status, outcome.stage, outcome.returncode) == ("passed", "run", 0)
    assert "hi" in outcome.stdout

    # Exiting cleanly before the tests (sys.exit(0) in model code) is not a pass.
    early = codeexec.run(python("print('hi')", marker))
    assert (early.status, early.returncode, early.note) == ("failed", 0, EARLY_EXIT_NOTE)

    crashed = codeexec.run(python("raise SystemExit(3)", marker))
    assert (crashed.status, crashed.returncode, crashed.note) == ("failed", 3, "")


def test_a_failed_compile_step_is_a_compile_error_and_nothing_after_it_runs(scratch_dir: Path) -> None:
    ran = scratch_dir / "ran"
    program = Program(
        {},
        (
            Step((PYTHON, "-c", "import sys; sys.exit(2)"), 30, "compile"),
            Step((PYTHON, "-c", f"open({str(ran)!r}, 'w')"), 30),
        ),
        "marker",
    )
    outcome = codeexec.run(program)
    assert (outcome.status, outcome.stage, outcome.returncode) == ("compile_error", "compile", 2)
    assert not ran.exists()


def test_a_failed_setup_step_means_the_sandbox_is_broken() -> None:
    program = Program({}, (Step((PYTHON, "-c", "raise SystemExit(1)"), 30, "setup"),), "marker")
    with pytest.raises(KumaruError, match="setup step"):
        codeexec.run(program)


def test_a_timeout_kills_every_process_the_program_started(scratch_dir: Path) -> None:
    beat = scratch_dir / "heartbeat"
    grandchild = (
        "import time\nwhile True:\n"
        f"    with open({str(beat)!r}, 'a') as f:\n        f.write('.')\n    time.sleep(0.05)\n"
    )
    code = f"import subprocess, sys, time\nsubprocess.Popen([sys.executable, '-c', {grandchild!r}])\ntime.sleep(60)\n"
    start = time.monotonic()
    outcome = codeexec.run(python(code, timeout_s=3))
    assert (outcome.status, outcome.returncode) == ("timeout", None)
    assert time.monotonic() - start < 20
    assert beat.exists()  # the grandchild was running...
    size = beat.stat().st_size
    time.sleep(0.5)
    assert beat.stat().st_size == size  # ...and died with the program


def test_secrets_are_withheld_and_temp_files_land_in_a_folder_that_is_removed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KUMARU_TEST_SECRET", "s3cret")
    monkeypatch.setenv("KUMARU_TEST_KEPT", "kept")
    marker = codeexec.new_marker()
    code = (
        "import os, tempfile\n"
        "assert 'KUMARU_TEST_SECRET' not in os.environ\n"
        "assert os.environ['KUMARU_TEST_KEPT'] == 'kept'\n"
        "assert os.environ['KUMARU_TEST_SET'] == 'set'\n"
        "assert os.path.samefile(tempfile.gettempdir(), os.getcwd())\n"
        "print(os.getcwd(), flush=True)\n" + finish(marker)
    )
    program = python(
        code, marker, env={"KUMARU_TEST_SET": "set"}, keep_env=frozenset({"KUMARU_TEST_KEPT"})
    )
    outcome = codeexec.run(program)
    assert outcome.passed, outcome.stderr
    workdir = Path(outcome.stdout.splitlines()[0].strip())
    assert workdir.name.startswith("kumaru-code-")
    assert not workdir.exists()


def test_only_the_tail_of_the_output_is_kept_and_the_marker_is_seen_across_reads() -> None:
    tail = codeexec._Tail("MARK", limit=8)
    for data in (b"aaaaMA", b"RKbbbbbbbbb"):
        tail.feed(data)
    assert tail.seen
    assert tail.text() == "bbbbbbbb"

    marker = codeexec.new_marker()
    outcome = codeexec.run(python("import sys\nsys.stdout.write('x' * 200_000)\n" + finish(marker), marker))
    assert outcome.passed
    assert len(outcome.stdout) <= TAIL_BYTES


@pytest.mark.skipif(sys.platform != "win32", reason="the memory cap is a Windows job-object limit")
def test_a_program_cannot_use_more_memory_than_the_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KUMARU_CODE_MEMORY_MB", "256")
    marker = codeexec.new_marker()
    outcome = codeexec.run(python("block = bytearray(1024 * 1024 * 1024)\n" + finish(marker), marker))
    assert outcome.status == "failed"
    assert "MemoryError" in outcome.stderr


@pytest.mark.skipif(sys.platform != "win32", reason="the process cap is a Windows job-object limit")
def test_a_program_cannot_run_more_processes_at_once_than_the_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(codeexec, "MAX_PROCESSES", 3)
    marker = codeexec.new_marker()
    code = (
        "import subprocess, sys\nchildren = []\ntry:\n"
        "    for _ in range(4):\n"
        "        children.append(subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)']))\n"
        "except OSError as exc:\n    print('refused after', len(children), exc.winerror)\n"
        "for child in children:\n    child.kill()\n" + finish(marker)
    )
    outcome = codeexec.run(python(code, marker))
    assert outcome.passed, outcome.stderr
    # The program itself and two children; the third is ERROR_NOT_ENOUGH_QUOTA.
    assert "refused after 2 1816" in outcome.stdout


def test_a_compiler_that_is_not_installed_is_a_missing_toolchain() -> None:
    program = Program({}, (Step(("kumaru-no-such-compiler", "x.c"), 30, "compile"),), "marker")
    with pytest.raises(ToolchainMissing, match="Could not start"):
        codeexec.run(program)


def test_run_refuses_an_unknown_sandbox_and_a_program_without_steps() -> None:
    with pytest.raises(ValueError, match="unknown sandbox"):
        codeexec.run(python("pass"), sandbox="vm")
    with pytest.raises(ValueError, match="at least one step"):
        codeexec.run(Program({}, (), "marker"))


def test_at_most_max_workers_programs_run_at_once(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KUMARU_CODE_WORKERS", "3")
    assert codeexec.max_workers() == 3
    monkeypatch.setenv("KUMARU_CODE_WORKERS", "none")
    assert 1 <= codeexec.max_workers() <= 4

    monkeypatch.setattr(codeexec, "_slots", None)
    codeexec.set_max_workers(1)
    entered = threading.Event()

    def second() -> None:
        with codeexec._slot():
            entered.set()

    with codeexec._slot():
        thread = threading.Thread(target=second)
        thread.start()
        assert not entered.wait(0.2)  # waits for the first to finish
    assert entered.wait(5)
    thread.join()
    with pytest.raises(ValueError):
        codeexec.set_max_workers(0)


# -- finding programs -------------------------------------------------------------

WINDOWS = sys.platform == "win32"


def plant(folder: Path, name: str) -> Path:
    """A runnable stand-in for the program `name` in `folder` (a .cmd on Windows)."""
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / (f"{name}.cmd" if WINDOWS else name)
    path.write_text("@echo off\n" if WINDOWS else "#!/bin/sh\n")
    path.chmod(0o755)
    return path


def test_find_program_never_runs_a_decoy_from_the_current_folder(
    scratch_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # On Windows shutil.which looks in the current folder first, so a node.cmd
    # planted where Kumaru was started from would run as Node.js.
    here = scratch_dir / "here"
    plant(here, "node")
    plant(here / "rel", "node")
    real = plant(scratch_dir / "bin", "node")
    monkeypatch.chdir(here)
    relative = ["", ".", "rel", os.path.join("..", "here")]
    if WINDOWS:
        # Relative too: to the current folder on drive C:, and to the current drive.
        relative += [here.drive + "rel", str(here / "rel").removeprefix(here.drive)]
        monkeypatch.setenv("PATHEXT", ".COM;.EXE;.BAT;.CMD")
    monkeypatch.setenv("PATH", os.pathsep.join(relative))
    assert codeexec.find_program("node") is None

    folder = f'"{real.parent}"' if WINDOWS else str(real.parent)  # cmd accepts a quoted entry
    monkeypatch.setenv("PATH", os.pathsep.join([*relative, folder]))
    found = codeexec.find_program("node")
    assert found is not None and Path(found).is_absolute()
    assert os.path.samefile(found, real)
    assert codeexec.find_program("kumaru-no-such-program") is None
    if WINDOWS:
        assert os.path.samefile(codeexec.find_program("node.cmd") or "", real)


# -- Docker -----------------------------------------------------------------------

DOCKER = "/opt/docker/bin/docker"


def find_docker(monkeypatch: pytest.MonkeyPatch, path: str | None = DOCKER) -> None:
    monkeypatch.setattr(codeexec, "find_program", lambda name: path if name == "docker" else None)


TWO_STEPS = Program(
    {"p.c": "int main() {}"},
    (Step(("cc", "-o", "my prog", "p.c"), 120, "compile"), Step(("./my prog",), 15)),
    "MARK",
)


def record(token: str, index: int, rc: int, millis: int, marks: int, out: str = "", err: str = "") -> str:
    """One step's report, as `docker_script` prints it."""
    return f"\n{token} {index} {rc} {millis} {marks}\n{out}\n{token} stderr\n{err}\n{token} end\n"


def test_docker_runs_each_program_without_network_privileges_or_a_writable_root() -> None:
    argv = codeexec.docker_command(DOCKER, "img", "kumaru-x", {"B": "2", "A": "1"}, "echo hi")
    joined = " ".join(argv)
    for flag in (
        "--network none",
        "--read-only",
        "--cap-drop ALL",
        "--security-opt no-new-privileges",
        "--user 65534:65534",
        "--pull never",
        "--pids-limit 512",
        "--memory 2g --memory-swap 2g",
        "--ulimit core=1",
    ):
        assert flag in joined
    assert joined.index("--env A=1") < joined.index("--env B=2")
    assert argv[:2] == [DOCKER, "run"]
    assert argv[-3:] == ["img", "-c", "echo hi"]


def test_docker_script_unpacks_then_runs_each_step_under_its_time_limit() -> None:
    script = codeexec.docker_script(TWO_STEPS, "tok")
    assert script.startswith('mkdir -p "$HOME"\ntar -xf - || exit 3\n')
    assert "timeout -k 2 120 cc -o 'my prog' p.c </dev/null" in script
    assert "timeout -k 2 15 './my prog' </dev/null" in script
    assert script.count('[ "$rc" -eq 0 ] || exit 0') == 2


@pytest.mark.parametrize(
    ("records", "expected"),
    [
        ([(0, 0, 900, 0), (1, 0, 20, 1, "ok\nMARK")], ("passed", "run", 0, "")),
        ([(0, 0, 900, 0), (1, 0, 20, 0, "ok")], ("failed", "run", 0, EARLY_EXIT_NOTE)),
        ([(0, 1, 900, 0, "", "p.c:1: error")], ("compile_error", "compile", 1, "")),
        # `timeout` ends the step at its limit: 124, or 137 if it needed SIGKILL.
        ([(0, 0, 900, 0), (1, 124, 15_000, 0)], ("timeout", "run", None, "")),
        ([(0, 0, 900, 0), (1, 137, 15_100, 0)], ("timeout", "run", None, "")),
        # Killed long before its limit: the memory limit, not a timeout.
        ([(0, 0, 900, 0), (1, 137, 300, 0)], ("failed", "run", 137, "")),
    ],
)
def test_docker_results_read_back_into_the_outcomes_a_local_run_gives(
    records: list[tuple], expected: tuple
) -> None:
    stdout = "".join(record("tok", *fields) for fields in records)
    outcome = codeexec.parse_docker_output(stdout, "tok", TWO_STEPS)
    assert (outcome.status, outcome.stage, outcome.returncode, outcome.note) == expected


def test_docker_output_without_every_step_is_an_error() -> None:
    with pytest.raises(KumaruError, match="printed no results"):
        codeexec.parse_docker_output("container noise", "tok", TWO_STEPS)
    with pytest.raises(KumaruError, match="stopped after step 1 of 2"):
        codeexec.parse_docker_output(record("tok", 0, 0, 900, 0), "tok", TWO_STEPS)


def fake_docker(monkeypatch: pytest.MonkeyPatch, reply) -> list[list[str]]:
    """Replace `docker` with `reply(argv, stdin, token) -> (exit code, stdout)`."""
    calls: list[list[str]] = []

    def run(argv, input=None, **kwargs):
        calls.append(argv)
        token = re.search(r'echo "(kumaru-[0-9a-f]+) stderr"', argv[-1])
        code, stdout = reply(argv, input, token.group(1) if token else "")
        return subprocess.CompletedProcess(argv, code, stdout.encode(), b"docker said no")

    find_docker(monkeypatch)
    monkeypatch.setattr(codeexec.subprocess, "run", run)
    return calls


def test_a_docker_run_sends_the_files_as_a_tar_and_reads_the_reports(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KUMARU_SANDBOX_IMAGE", "my-image")
    sent: dict[str, str] = {}

    def reply(argv, stdin, token):
        with tarfile.open(fileobj=io.BytesIO(stdin)) as archive:
            for member in archive.getmembers():
                sent[member.name] = archive.extractfile(member).read().decode()
        return 0, record(token, 0, 0, 900, 0) + record(token, 1, 0, 20, 1, "MARK")

    calls = fake_docker(monkeypatch, reply)
    outcome = codeexec.run(TWO_STEPS, sandbox="docker")
    assert outcome.passed
    assert sent == {"p.c": "int main() {}"}
    assert calls[0][0] == DOCKER
    assert "my-image" in calls[0]


def test_a_container_killed_for_memory_fails_the_answer_and_a_broken_docker_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Killed while compiling, before any step reported.
    fake_docker(monkeypatch, lambda argv, stdin, token: (137, ""))
    outcome = codeexec.run(TWO_STEPS, sandbox="docker")
    assert (outcome.status, outcome.stage, outcome.note) == ("compile_error", "compile", "killed: out of memory?")

    fake_docker(monkeypatch, lambda argv, stdin, token: (137, record(token, 0, 0, 900, 0)))
    assert codeexec.run(TWO_STEPS, sandbox="docker").status == "failed"

    fake_docker(monkeypatch, lambda argv, stdin, token: (125, ""))
    with pytest.raises(KumaruError, match="exited with 125") as caught:
        codeexec.run(TWO_STEPS, sandbox="docker")
    assert "docker build" in str(caught.value.hint)


def test_a_container_that_never_finishes_is_removed(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []

    def run(argv, **kwargs):
        calls.append(argv)
        if argv[:2] == [DOCKER, "run"]:
            raise subprocess.TimeoutExpired(argv, kwargs["timeout"])
        return subprocess.CompletedProcess(argv, 0, b"", b"")

    find_docker(monkeypatch)
    monkeypatch.setattr(codeexec.subprocess, "run", run)
    outcome = codeexec.run(TWO_STEPS, sandbox="docker")
    assert outcome.status == "timeout"
    name = calls[0][calls[0].index("--name") + 1]
    assert calls[1] == [DOCKER, "rm", "-f", name]


def test_check_docker_says_how_to_get_docker_and_the_image(monkeypatch: pytest.MonkeyPatch) -> None:
    build = "docker build -t kumaru-sandbox -f docker/sandbox/Dockerfile ."
    monkeypatch.delenv("KUMARU_SANDBOX_IMAGE", raising=False)
    calls: list[list[str]] = []
    answer = {"code": 1, "out": ""}

    def run(argv, **kwargs):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, answer["code"], answer["out"], "no such image")

    monkeypatch.setattr(codeexec.subprocess, "run", run)
    find_docker(monkeypatch, None)
    with pytest.raises(ToolchainMissing, match="not installed") as caught:
        codeexec.check_docker()
    assert build in str(caught.value.hint)
    with pytest.raises(ToolchainMissing, match="not installed"):
        codeexec.run(TWO_STEPS, sandbox="docker")
    assert calls == []  # never a bare "docker", which Windows looks for in the current folder first

    find_docker(monkeypatch)
    with pytest.raises(ToolchainMissing, match="no image 'kumaru-sandbox'") as caught:
        codeexec.check_docker()
    assert build in str(caught.value.hint)

    answer.update(code=0, out="sha256:abc\n")
    assert codeexec.check_docker() == "sha256:abc"
    assert calls[-1][:3] == [DOCKER, "image", "inspect"]

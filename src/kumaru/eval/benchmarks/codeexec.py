"""Run a model-written program: write its files, compile, run, kill everything.

Grading a coding benchmark means running code a model wrote. This module runs
one program as a list of steps (say, compile then run) and reports how it
ended. Every code benchmark goes through it, so all of them get the same
protections:

- a fresh temporary directory, deleted afterwards, which is both the working
  directory and TEMP/TMP/TMPDIR, so compilers' scratch files land in it too;
- a scrubbed environment: only what toolchains need to start, never API keys;
- a time limit per step. When a step ends or times out, everything it started
  is killed, not just its own process. On Windows each step runs in a job
  object, which also caps memory (`KUMARU_CODE_MEMORY_MB`, default 4096) and
  running processes (512, as in Docker) and turns crash dialogs off. On Linux
  and macOS each step gets its own process group;
- bounded output: only the last 64 KB of each stream is kept;
- a random marker the program prints after its last test, so code that exits
  early with status 0 cannot pass untested.

That stops accidents, not adversaries: the program still runs as you, can read
your files and can reach the network. `sandbox="docker"` instead runs each
program in a throwaway container with no network, a read-only root, an
unprivileged user and CPU, memory and process limits. Its image is built from
docker/sandbox/Dockerfile (see `docker_command`).

At most `KUMARU_CODE_WORKERS` programs run at once (default: half the CPU
cores, between 1 and 4), however many answers the runner grades in parallel.
"""

from __future__ import annotations

import ctypes
import functools
import io
import os
import re
import secrets
import shlex
import shutil
import signal
import stat
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Literal

from kumaru.core.errors import KumaruError

Status = Literal["passed", "failed", "timeout", "compile_error"]
Stage = Literal["setup", "compile", "run"]
SANDBOXES = ("none", "docker")

TAIL_BYTES = 64 * 1024
DEFAULT_MEMORY_MB = 4096
DEFAULT_IMAGE = "kumaru-sandbox"
DOCKERFILE = "docker/sandbox/Dockerfile"
EARLY_EXIT_NOTE = "exited with code 0 before the tests finished (an early exit?)"

# Environment variables every program may see. Everything else (API keys,
# tokens) is withheld; a language adds what its toolchain needs (`Program.keep_env`).
ENV_BASE = frozenset({
    "PATH", "PATHEXT", "SYSTEMROOT", "SYSTEMDRIVE", "WINDIR", "COMSPEC", "HOME", "USERPROFILE",
    "LANG", "LC_ALL", "NUMBER_OF_PROCESSORS", "PROCESSOR_ARCHITECTURE", "OS",
})  # fmt: skip
_WINDOWS = sys.platform == "win32"
_CREATE_SUSPENDED = 0x00000004
_CREATE_NO_WINDOW = 0x08000000


class ToolchainMissing(KumaruError):
    """A compiler or runtime a benchmark needs is not installed (or not usable)."""

    http_status = 400

    @property
    def why(self) -> str:
        """The message, then the hint, as two sentences: hints have parentheses of their own."""
        return f"{self.message} {self.hint}" if self.hint else self.message


@dataclass(frozen=True)
class Step:
    """One command, run in the program's directory."""

    # A relative first item ("./prog") names a file an earlier step built.
    argv: tuple[str, ...]
    timeout_s: float
    # A failed "compile" step is a compile_error, a failed "run" step a failure,
    # and a failed "setup" step means the sandbox is broken (KumaruError).
    stage: Stage = "run"


@dataclass(frozen=True)
class Program:
    """Everything needed to run one answer: its files and the steps to run them."""

    # Relative path -> UTF-8 text, written byte for byte (no newline translation).
    files: dict[str, str]
    steps: tuple[Step, ...]
    # Printed by the program after its last test; "passed" requires it.
    marker: str
    # Set for every step, after the scrubbing.
    env: dict[str, str] = field(default_factory=dict)
    # Extra variables passed through from this process's environment.
    keep_env: frozenset[str] = frozenset()


@dataclass(frozen=True)
class Outcome:
    """How a program ended."""

    status: Status
    # The step it ended in.
    stage: Stage
    # None after a timeout.
    returncode: int | None
    # The last 64 KB of the final step's output (4 KB in Docker).
    stdout: str
    stderr: str
    elapsed_s: float
    note: str = ""

    @property
    def passed(self) -> bool:
        return self.status == "passed"


def new_marker() -> str:
    return f"kumaru-finished-{secrets.token_hex(8)}"


def run(program: Program, *, sandbox: str = "none") -> Outcome:
    """Run `program`'s steps in order, stopping at the first that fails."""
    if sandbox not in SANDBOXES:
        raise ValueError(f"unknown sandbox {sandbox!r}; expected one of {', '.join(SANDBOXES)}")
    if not program.steps:
        raise ValueError("a program needs at least one step")
    with _slot():
        return _run_docker(program) if sandbox == "docker" else _run_local(program)


# -- finding programs ---------------------------------------------------------


def find_program(name: str) -> str | None:
    """The absolute path of the program `name` in a PATH folder, or None.

    Unlike `shutil.which`, it never looks in the current folder, which `which`
    searches first on Windows: a node.cmd left in the folder Kumaru was started
    from would run instead of Node.js. Empty and relative PATH entries, which
    also depend on the current folder, are skipped. On Windows a name without
    one of PATHEXT's extensions is tried with each of them.
    """
    names = [name]
    if _WINDOWS:
        pathext = [ext for ext in os.environ.get("PATHEXT", "").split(os.pathsep) if len(ext) > 1 and ext[0] == "."]
        pathext = pathext or [".COM", ".EXE", ".BAT", ".CMD"]
        if not name.upper().endswith(tuple(ext.upper() for ext in pathext)):
            names = [name + ext for ext in pathext]
    for entry in os.environ.get("PATH", "").split(os.pathsep):
        folder = entry.strip('"') if _WINDOWS else entry
        # On Windows pathlib also wants a drive: "\bin" and "C:bin" depend on the current folder.
        if not folder or not Path(folder).is_absolute():
            continue
        for candidate in names:
            path = os.path.join(folder, candidate)
            if os.path.isfile(path) and os.access(path, os.X_OK):
                return path
    return None


# -- how many at once ---------------------------------------------------------

_slots_lock = threading.Lock()
_slots: threading.BoundedSemaphore | None = None


def max_workers() -> int:
    raw = os.environ.get("KUMARU_CODE_WORKERS", "").strip()
    if raw.isdigit() and int(raw) > 0:
        return int(raw)
    return max(1, min(4, (os.cpu_count() or 2) // 2))


def set_max_workers(n: int) -> None:
    """Change how many programs may run at once. Call before grading starts."""
    global _slots
    if n < 1:
        raise ValueError("need at least one worker")
    with _slots_lock:
        _slots = threading.BoundedSemaphore(n)


@contextmanager
def _slot() -> Iterator[None]:
    global _slots
    with _slots_lock:
        if _slots is None:
            _slots = threading.BoundedSemaphore(max_workers())
        slots = _slots
    with slots:
        yield


# -- output -------------------------------------------------------------------


class _Tail:
    """The last `limit` bytes of a stream, and whether `marker` ever appeared in it."""

    def __init__(self, marker: str | None = None, limit: int = TAIL_BYTES) -> None:
        self.marker = marker.encode() if marker else b""
        self.limit = limit
        self.buf = bytearray()
        self.seen = False
        self._carry = b""

    def feed(self, data: bytes) -> None:
        if self.marker and not self.seen:
            window = self._carry + data
            self.seen = self.marker in window
            self._carry = window[-(len(self.marker) - 1) :] if len(self.marker) > 1 else b""
        self.buf += data
        if len(self.buf) > self.limit:
            del self.buf[: len(self.buf) - self.limit]

    def text(self) -> str:
        return self.buf.decode("utf-8", errors="replace")


def _pump(stream: IO[bytes], tail: _Tail) -> None:
    try:
        while chunk := stream.read1(65536):  # type: ignore[attr-defined]
            tail.feed(chunk)
    except (OSError, ValueError):
        pass


# -- Windows job objects ------------------------------------------------------
# A job object groups processes so the kernel can kill them together: every
# process a step starts (compilers' helpers, a program's children) joins it,
# and none can leave. Each step's process is created suspended, put in a new
# job, then resumed, so it cannot start a child before it is contained.

_JOB_OBJECT_LIMIT_ACTIVE_PROCESS = 0x00000008
_JOB_OBJECT_LIMIT_JOB_MEMORY = 0x00000200
_JOB_OBJECT_LIMIT_DIE_ON_UNHANDLED_EXCEPTION = 0x00000400  # no crash dialogs
_JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
_JOB_BASIC_ACCOUNTING = 1
_JOB_EXTENDED_LIMITS = 9
_HANDLE = ctypes.c_void_p
_DWORD = ctypes.c_uint32
# Processes a step may have running at once, as Docker's --pids-limit: a fork bomb stops here.
MAX_PROCESSES = 512


class _BasicLimits(ctypes.Structure):
    _fields_ = [
        ("PerProcessUserTimeLimit", ctypes.c_int64),
        ("PerJobUserTimeLimit", ctypes.c_int64),
        ("LimitFlags", _DWORD),
        ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", _DWORD),
        ("Affinity", ctypes.c_size_t),
        ("PriorityClass", _DWORD),
        ("SchedulingClass", _DWORD),
    ]


class _IoCounters(ctypes.Structure):
    _fields_ = [
        (name, ctypes.c_uint64)
        for name in ("Reads", "Writes", "Others", "ReadBytes", "WriteBytes", "OtherBytes")
    ]


class _ExtendedLimits(ctypes.Structure):
    _fields_ = [
        ("BasicLimitInformation", _BasicLimits),
        ("IoInfo", _IoCounters),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    ]


class _BasicAccounting(ctypes.Structure):
    _fields_ = [
        ("TotalUserTime", ctypes.c_int64),
        ("TotalKernelTime", ctypes.c_int64),
        ("ThisPeriodTotalUserTime", ctypes.c_int64),
        ("ThisPeriodTotalKernelTime", ctypes.c_int64),
        ("TotalPageFaultCount", _DWORD),
        ("TotalProcesses", _DWORD),
        ("ActiveProcesses", _DWORD),
        ("TotalTerminatedProcesses", _DWORD),
    ]


@functools.cache
def _kernel32() -> ctypes.WinDLL:  # type: ignore[name-defined]
    # A private instance, so these signatures cannot clash with other users of ctypes.windll.
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)  # type: ignore[attr-defined]
    signatures = {
        "CreateJobObjectW": ([ctypes.c_void_p, ctypes.c_wchar_p], _HANDLE),
        "SetInformationJobObject": ([_HANDLE, ctypes.c_int, ctypes.c_void_p, _DWORD], ctypes.c_int),
        "QueryInformationJobObject": (
            [_HANDLE, ctypes.c_int, ctypes.c_void_p, _DWORD, ctypes.POINTER(_DWORD)],
            ctypes.c_int,
        ),
        "AssignProcessToJobObject": ([_HANDLE, _HANDLE], ctypes.c_int),
        "TerminateJobObject": ([_HANDLE, ctypes.c_uint], ctypes.c_int),
        "CloseHandle": ([_HANDLE], ctypes.c_int),
    }
    for name, (argtypes, restype) in signatures.items():
        function = getattr(k32, name)
        function.argtypes, function.restype = argtypes, restype
    return k32


@functools.cache
def _resume_process() -> ctypes._CFuncPtr:  # type: ignore[name-defined]
    # Popen closes the main thread's handle, so ResumeThread is out of reach;
    # NtResumeProcess resumes a process by its process handle.
    function = ctypes.WinDLL("ntdll").NtResumeProcess  # type: ignore[attr-defined]
    function.argtypes, function.restype = [_HANDLE], ctypes.c_long
    return function


def memory_limit_mb() -> int:
    raw = os.environ.get("KUMARU_CODE_MEMORY_MB", "").strip()
    return int(raw) if raw.isdigit() else DEFAULT_MEMORY_MB


class _Job:
    def __init__(self) -> None:
        k32 = _kernel32()
        self.handle = k32.CreateJobObjectW(None, None)
        if not self.handle:
            raise ctypes.WinError(ctypes.get_last_error())  # type: ignore[attr-defined]
        info = _ExtendedLimits()
        flags = (
            _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            | _JOB_OBJECT_LIMIT_DIE_ON_UNHANDLED_EXCEPTION
            | _JOB_OBJECT_LIMIT_ACTIVE_PROCESS
        )
        info.BasicLimitInformation.ActiveProcessLimit = MAX_PROCESSES
        limit = memory_limit_mb()
        if limit:
            flags |= _JOB_OBJECT_LIMIT_JOB_MEMORY
            info.JobMemoryLimit = limit * 1024 * 1024
        info.BasicLimitInformation.LimitFlags = flags
        if not k32.SetInformationJobObject(
            self.handle, _JOB_EXTENDED_LIMITS, ctypes.byref(info), ctypes.sizeof(info)
        ):
            error = ctypes.get_last_error()  # type: ignore[attr-defined]
            self.close()
            raise ctypes.WinError(error)  # type: ignore[attr-defined]

    def adopt(self, proc: subprocess.Popen[bytes]) -> None:
        """Put a suspended process in the job, then let it run."""
        handle = int(proc._handle)  # type: ignore[attr-defined]
        if not _kernel32().AssignProcessToJobObject(self.handle, handle):
            error = ctypes.get_last_error()  # type: ignore[attr-defined]
            proc.kill()
            raise ctypes.WinError(error)  # type: ignore[attr-defined]
        status = _resume_process()(handle)
        if status:
            proc.kill()
            raise OSError(f"NtResumeProcess failed with status {status & 0xFFFFFFFF:#x}")

    def active(self) -> int:
        info = _BasicAccounting()
        ok = _kernel32().QueryInformationJobObject(
            self.handle, _JOB_BASIC_ACCOUNTING, ctypes.byref(info), ctypes.sizeof(info), None
        )
        return int(info.ActiveProcesses) if ok else 0

    def terminate(self) -> None:
        """Kill every process in the job and wait (up to 5 s) until they are gone."""
        _kernel32().TerminateJobObject(self.handle, 1)
        deadline = time.monotonic() + 5
        while self.active() and time.monotonic() < deadline:
            time.sleep(0.02)

    def close(self) -> None:
        if self.handle:
            _kernel32().CloseHandle(self.handle)
            self.handle = None


# -- local runs ---------------------------------------------------------------


def child_env(program: Program, workdir: Path) -> dict[str, str]:
    """The environment a step sees: the allowed variables, TEMP in `workdir`, then `program.env`."""
    keep = {name.upper() for name in ENV_BASE | program.keep_env}
    env = {key: value for key, value in os.environ.items() if key.upper() in keep}
    env.update(TEMP=str(workdir), TMP=str(workdir), TMPDIR=str(workdir))
    env.update(program.env)
    return env


def _run_local(program: Program) -> Outcome:
    workdir = Path(tempfile.mkdtemp(prefix="kumaru-code-"))
    try:
        for name, content in program.files.items():
            path = workdir / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content.encode("utf-8"))
        env = child_env(program, workdir)
        elapsed = 0.0
        for index, step in enumerate(program.steps):
            final = index == len(program.steps) - 1
            returncode, out, err, seconds = _run_step(
                step, workdir, env, program.marker if final else None
            )
            elapsed += seconds
            outcome = _step_outcome(step, returncode, out.text(), err.text(), elapsed)
            if outcome is not None:
                return outcome
        if out.seen:
            return Outcome("passed", step.stage, 0, out.text(), err.text(), elapsed)
        return Outcome("failed", step.stage, 0, out.text(), err.text(), elapsed, EARLY_EXIT_NOTE)
    finally:
        _remove(workdir)


def _step_outcome(
    step: Step, returncode: int | None, stdout: str, stderr: str, elapsed: float
) -> Outcome | None:
    """The outcome a step ends the program with, or None to go on to the next step."""
    if returncode == 0:
        return None
    if step.stage == "setup":
        raise KumaruError(
            f"Sandbox setup step {' '.join(step.argv)!r} failed "
            f"({'timed out' if returncode is None else f'exit code {returncode}'}): {stderr[-400:]}"
        )
    if returncode is None:
        return Outcome("timeout", step.stage, None, stdout, stderr, elapsed)
    status: Status = "compile_error" if step.stage == "compile" else "failed"
    return Outcome(status, step.stage, returncode, stdout, stderr, elapsed)


def _run_step(
    step: Step, workdir: Path, env: dict[str, str], marker: str | None
) -> tuple[int | None, _Tail, _Tail, float]:
    """Run one step. Returns (exit code or None on timeout, stdout, stderr, seconds)."""
    argv = list(step.argv)
    built = workdir / argv[0]
    if not os.path.isabs(argv[0]) and built.is_file():
        argv[0] = str(built)
    out, err = _Tail(marker), _Tail()
    job = None
    if _WINDOWS:
        try:
            job = _Job()
        except OSError:
            job = None  # fall back to taskkill /T below
    flags = (_CREATE_NO_WINDOW | (_CREATE_SUSPENDED if job else 0)) if _WINDOWS else 0
    start = time.monotonic()
    try:
        proc = subprocess.Popen(
            argv,
            cwd=workdir,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=flags,
            start_new_session=not _WINDOWS,
        )
    except OSError as exc:
        if job:
            job.close()
        raise ToolchainMissing(f"Could not start {argv[0]}: {exc}") from exc
    try:
        if job:
            try:
                job.adopt(proc)
            except OSError:
                proc.wait()
                job.close()
                raise
        pumps = [
            threading.Thread(target=_pump, args=(proc.stdout, out), daemon=True),
            threading.Thread(target=_pump, args=(proc.stderr, err), daemon=True),
        ]
        for pump in pumps:
            pump.start()
        returncode: int | None
        try:
            returncode = proc.wait(timeout=step.timeout_s)
        except subprocess.TimeoutExpired:
            returncode = None
        # Whether it finished or not, nothing it started may outlive the step.
        _kill_tree(proc, job)
        proc.wait()
        for pump in pumps:
            pump.join(timeout=5)
    finally:
        for stream in (proc.stdout, proc.stderr):
            if stream:
                stream.close()
        if job:
            job.close()
    return returncode, out, err, time.monotonic() - start


def _kill_tree(proc: subprocess.Popen[bytes], job: _Job | None) -> None:
    if job:
        job.terminate()
    elif _WINDOWS:
        if proc.poll() is None:
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                capture_output=True,
                creationflags=_CREATE_NO_WINDOW,
                check=False,
            )
            proc.kill()
    else:
        with suppress(ProcessLookupError, PermissionError):  # the group may be gone already
            os.killpg(proc.pid, signal.SIGKILL)


def _remove(path: Path) -> None:
    for attempt in range(3):
        shutil.rmtree(path, ignore_errors=True)
        if not path.exists():
            return
        # Read-only files (or a virus scanner's brief lock) can stop the first try.
        for root, _dirs, files in os.walk(path):
            for name in files:
                with suppress(OSError):
                    os.chmod(os.path.join(root, name), stat.S_IWRITE | stat.S_IREAD)
        time.sleep(0.2 * (attempt + 1))


# -- Docker -------------------------------------------------------------------


def docker_image() -> str:
    return os.environ.get("KUMARU_SANDBOX_IMAGE", "").strip() or DEFAULT_IMAGE


def _build_command(image: str) -> str:
    return f"docker build -t {image} -f {DOCKERFILE} ."


def _docker_program(image: str) -> str:
    """docker's absolute path, which is what runs (see `find_program`)."""
    docker = find_program("docker")
    if not docker:
        raise ToolchainMissing(
            "--sandbox docker needs Docker, which is not installed.",
            hint=f"Install Docker, then build the image once: {_build_command(image)}",
        )
    return docker


def _ask_docker(argv: list[str], image: str) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=60,
            creationflags=_CREATE_NO_WINDOW if _WINDOWS else 0,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ToolchainMissing(f"Could not ask Docker about image '{image}': {exc}") from exc


def check_docker(image: str | None = None) -> str:
    """The sandbox image's ID. Raises ToolchainMissing unless Docker runs and has the image.

    After 5 minutes without a container, Docker Desktop's Resource Saver
    pauses the engine, and answers commands that start no container without
    waking it. Paused, it finds an image by its ID but not by its name, so the
    lookup fails ("No such image") although the image exists. Starting a
    container wakes it. So when the lookup fails, this starts one from the
    image (`/bin/true`, no network) and looks again; if the image really is
    missing, the container can't start either, and Docker says why.
    """
    image = image or docker_image()
    build = _build_command(image)
    docker = _docker_program(image)
    inspect = [docker, "image", "inspect", "--format", "{{.Id}}", image]
    proc = _ask_docker(inspect, image)
    if proc.returncode != 0:
        wake = [docker, "run", "--rm", "--pull", "never", "--network", "none", "--entrypoint", "/bin/true", image]
        woken = _ask_docker(wake, image)
        proc = _ask_docker(inspect, image) if woken.returncode == 0 else woken
    if proc.returncode != 0:
        raise ToolchainMissing(
            f"Docker has no image '{image}', or the Docker daemon is not running: "
            f"{(proc.stderr or '').strip()[-300:]}",
            hint=f"Start Docker, then build the image once from the Kumaru folder: {build}",
        )
    return proc.stdout.strip()


def docker_command(docker: str, image: str, name: str, env: dict[str, str], script: str) -> list[str]:
    """`docker run` for one program: no network, read-only root, unprivileged, limited.

    `docker` is the docker program's absolute path. The program's files arrive
    on stdin as a tar archive; only /work and /tmp (both in memory) are writable.

    `--ulimit core=1` stops core dumps of crashed programs. Docker Desktop's WSL
    VM pipes them to Windows, into %TEMP%\\wsl-crashes (1.1 GB for one crashed
    Node.js); Linux skips a piped dump only when the limit is exactly 1, not 0.
    """
    argv = [
        docker, "run", "--rm", "-i", "--pull", "never", "--name", name,
        "--network", "none", "--read-only",
        "--tmpfs", "/work:rw,exec,nosuid,nodev,size=512m,mode=1777",
        "--tmpfs", "/tmp:rw,exec,nosuid,nodev,size=1g,mode=1777",
        "--memory", "2g", "--memory-swap", "2g", "--cpus", "2", "--pids-limit", "512",
        "--ulimit", "core=1",
        "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
        "--user", "65534:65534", "--workdir", "/work",
        "--env", "HOME=/tmp/home", "--env", "TMPDIR=/tmp", "--env", "DOTNET_CLI_HOME=/tmp/home",
    ]  # fmt: skip
    for key, value in sorted(env.items()):
        argv += ["--env", f"{key}={value}"]
    return [*argv, "--entrypoint", "/bin/sh", image, "-c", script]


def docker_script(program: Program, token: str) -> str:
    """The shell script a container runs: unpack, then each step under `timeout`.

    After each step it prints a record fenced by `token`: step number, exit
    code, milliseconds, whether the marker was printed, and the last 4 KB of
    each stream. It stops after the first step that fails.
    """
    lines = ['mkdir -p "$HOME"', "tar -xf - || exit 3"]
    marker = shlex.quote(program.marker)
    for index, step in enumerate(program.steps):
        command = " ".join(shlex.quote(part) for part in step.argv)
        lines += [
            f"s=$(date +%s%N); timeout -k 2 {step.timeout_s:g} {command} </dev/null >/tmp/.o 2>/tmp/.e",
            "rc=$?; e=$(date +%s%N)",
            f"m=$(grep -c -F -e {marker} /tmp/.o)",
            f'echo; echo "{token} {index} $rc $(( (e - s) / 1000000 )) $m"',
            "tail -c 4096 /tmp/.o; echo",
            f'echo "{token} stderr"',
            "tail -c 4096 /tmp/.e; echo",
            f'echo "{token} end"',
            '[ "$rc" -eq 0 ] || exit 0',
        ]
    return "\n".join(lines) + "\n"


def parse_docker_output(stdout: str, token: str, program: Program) -> Outcome:
    """Turn `docker_script`'s records back into an Outcome."""
    record = re.compile(
        rf"^{token} (\d+) (\d+) (\d+) (\d+)\n(.*?)\n{token} stderr\n(.*?)\n{token} end$",
        re.MULTILINE | re.DOTALL,
    )
    elapsed = 0.0
    found = list(record.finditer(stdout))
    if not found:
        raise KumaruError(f"The sandbox container printed no results: {stdout[-400:]!r}")
    for match in found:
        index, returncode, millis, marks = (int(match.group(k)) for k in range(1, 5))
        step = program.steps[index]
        out, err = match.group(5).removesuffix("\n"), match.group(6).removesuffix("\n")
        elapsed += millis / 1000
        # GNU timeout exits 124 after SIGTERM and 137 if it needed SIGKILL.
        timed_out = returncode in (124, 137) and millis >= step.timeout_s * 1000 * 0.95
        outcome = _step_outcome(step, None if timed_out else returncode, out, err, elapsed)
        if outcome is not None:
            return outcome
    if index != len(program.steps) - 1:
        raise KumaruError(f"The sandbox container stopped after step {index + 1} of {len(program.steps)}.")
    if marks:
        return Outcome("passed", step.stage, 0, out, err, elapsed)
    return Outcome("failed", step.stage, 0, out, err, elapsed, EARLY_EXIT_NOTE)


def _tar(files: dict[str, str]) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        for name, content in files.items():
            data = content.encode("utf-8")
            info = tarfile.TarInfo(name)
            info.size, info.mode, info.mtime = len(data), 0o644, int(time.time())
            archive.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


def _run_docker(program: Program) -> Outcome:
    image = docker_image()
    docker = _docker_program(image)
    name = f"kumaru-{secrets.token_hex(6)}"
    token = f"kumaru-{secrets.token_hex(12)}"
    argv = docker_command(docker, image, name, program.env, docker_script(program, token))
    # Every step's own limit, plus time to start and remove the container.
    budget = sum(step.timeout_s + 2 for step in program.steps) + 30
    flags = _CREATE_NO_WINDOW if _WINDOWS else 0
    start = time.monotonic()
    try:
        proc = subprocess.run(
            argv, input=_tar(program.files), capture_output=True, timeout=budget, creationflags=flags, check=False
        )
    except subprocess.TimeoutExpired:
        subprocess.run([docker, "rm", "-f", name], capture_output=True, timeout=60, creationflags=flags, check=False)
        return Outcome("timeout", program.steps[-1].stage, None, "", "", budget, "the container did not finish in time")
    elapsed = time.monotonic() - start
    stdout = proc.stdout.decode("utf-8", errors="replace")
    stderr = proc.stderr.decode("utf-8", errors="replace")
    if proc.returncode == 137:
        # The container's memory limit killed the script itself, in the step
        # after the last one that reported.
        step = program.steps[min(stdout.count(f"{token} end"), len(program.steps) - 1)]
        status: Status = "compile_error" if step.stage == "compile" else "failed"
        return Outcome(status, step.stage, 137, "", stderr[-4096:], elapsed, "killed: out of memory?")
    if proc.returncode != 0:
        raise KumaruError(
            f"The sandbox container failed (docker run exited with {proc.returncode}): {stderr.strip()[-400:]}",
            hint=f"Check that Docker is running and the image is built: {_build_command(image)}",
        )
    return parse_docker_output(stdout, token, program)

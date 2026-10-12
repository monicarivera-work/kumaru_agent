"""Managed `llama-server` backend — Kumaru's default engine for GGUF models.

What it does
------------
On `load()`, Kumaru finds (or downloads) a GGUF file, starts llama.cpp's own
HTTP server as a child process, waits until it reports healthy, and then talks
to it over the OpenAI protocol (this class *is* an `OpenAICompatBackend` with a
process attached). On `aclose()` the process is stopped and its VRAM released.

Why a subprocess instead of Python bindings
-------------------------------------------
* **Currency.** New architectures (hybrid attention, MoE routing, MTP heads)
  land in llama.cpp's C++ first; Python bindings trail by weeks or months.
  Upgrading here means downloading a newer binary — no rebuild, no compiler.
* **Throughput.** llama-server does continuous batching across parallel
  "slots", reuses KV cache across turns of a conversation (prompt caching), and
  separates thinking from answer (`reasoning_content`) natively.
* **Isolation.** A CUDA crash kills the child, not your API server: requests
  then fail with the exit code and the end of the server's log, and restarting
  ``kumaru serve`` starts a fresh one.

Model specs
-----------
``backend.model`` accepts three forms:

* ``C:/models/foo.gguf`` — a local file (or set ``backend.model_path``);
* ``org/repo:QUANT`` — e.g. ``unsloth/Qwen3.6-35B-A3B-GGUF:UD-IQ4_XS``. Looked up
  under ``backend.models_dir/org/repo/`` first, downloaded from Hugging Face if
  absent. Multi-file ("-00001-of-00003") GGUFs are fetched completely;
* ``org/repo`` — allowed when the repo contains exactly one GGUF.

Orphan protection
-----------------
A model server that outlives its parent silently holds 20 GB of VRAM. On
Windows the child is placed in a Job Object with KILL_ON_JOB_CLOSE, so the OS
kills it even if Kumaru is force-terminated; on Linux the child receives
SIGTERM when the parent dies (PR_SET_PDEATHSIG). An `atexit` hook covers the
normal exit path everywhere.
"""

from __future__ import annotations

import asyncio
import atexit
import contextlib
import os
import re
import shutil
import socket
import subprocess
import sys
import time
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from kumaru.backends.openai_compat import OpenAICompatBackend
from kumaru.core.config import REPO_ROOT, BackendConfig
from kumaru.core.errors import BackendError
from kumaru.core.logging import get_logger
from kumaru.core.registry import BACKENDS
from kumaru.core.types import BackendInfo, GenerationParams

logger = get_logger(__name__)

EXE_NAME = "llama-server.exe" if os.name == "nt" else "llama-server"
BUNDLED_BINARY = REPO_ROOT / "tools" / "llama.cpp" / EXE_NAME
_SPLIT_SUFFIX = re.compile(r"-\d{5}-of-\d{5}$", re.IGNORECASE)
_SPLIT_PART = re.compile(r"-(\d{5})-of-(\d{5})\.gguf$", re.IGNORECASE)
_LIVE_PROCESSES: set[subprocess.Popen] = set()


# ---------------------------------------------------------------------------
# Model resolution
# ---------------------------------------------------------------------------


def parse_model_spec(spec: str) -> tuple[str, str | None] | None:
    """Split ``"org/repo:QUANT"`` into ``("org/repo", "QUANT")``.

    Returns None when `spec` is not a Hugging Face repo reference (for
    example a filesystem path).
    """
    if spec.lower().endswith(".gguf") or "\\" in spec or os.path.isabs(spec):
        return None
    repo, _, quant = spec.partition(":")
    if repo.count("/") != 1 or not all(repo.split("/")):
        return None
    return repo, (quant or None)


def _stem(filename: str) -> str:
    base = filename.replace("\\", "/").rsplit("/", 1)[-1]
    if base.lower().endswith(".gguf"):
        base = base[:-5]
    return _SPLIT_SUFFIX.sub("", base)


def match_gguf_files(files: Iterable[str], quant: str | None) -> list[str]:
    """Pick the GGUF file set for `quant` from a repo listing.

    Matching is on the filename stem (split suffix removed), which must end in
    the quant name after a ``-``, ``_`` or ``.`` separator, case-insensitively.
    Multimodal projectors (``mmproj-*``) are excluded. When several sets match,
    the shortest stem wins, so ``IQ4_XS`` prefers ``Model-IQ4_XS`` over
    ``Model-UD-IQ4_XS``. Returns every shard of the winner, first shard first.
    """
    groups: dict[tuple[str, str], list[str]] = {}
    for f in files:
        norm = f.replace("\\", "/")
        name = norm.rsplit("/", 1)[-1]
        if not name.lower().endswith(".gguf") or name.lower().startswith("mmproj"):
            continue
        stem = _stem(name).lower()
        if quant is not None:
            q = quant.lower()
            if not stem.endswith(q):
                continue
            prefix = stem[: len(stem) - len(q)]
            if prefix and prefix[-1] not in "-_.":
                continue
        folder = norm.rsplit("/", 1)[0] if "/" in norm else ""
        groups.setdefault((folder, stem), []).append(f)

    if not groups:
        return []
    if quant is None and len(groups) > 1:
        return []  # ambiguous: caller must name a quant
    best = min(groups, key=lambda k: (len(k[1]), k))
    return sorted(groups[best])


def _available_quants(files: Iterable[str]) -> str:
    stems = sorted({_stem(f) for f in files if f.lower().endswith(".gguf") and "mmproj" not in f.lower()})
    return ", ".join(stems[:25]) or "(no .gguf files)"


def shards_complete(files: Sequence[str]) -> bool:
    """True when `files` is one whole GGUF, or a split set with every part present.

    Each part names its own number and the total (``-00002-of-00003.gguf``), so
    a set is complete exactly when parts 1..total are all present. A download
    interrupted between parts leaves an incomplete set that llama-server
    cannot load.
    """
    numbers: set[int] = set()
    totals: set[int] = set()
    for name in files:
        part = _SPLIT_PART.search(name)
        if part is None:
            return len(files) == 1
        numbers.add(int(part.group(1)))
        totals.add(int(part.group(2)))
    return len(totals) == 1 and numbers == set(range(1, totals.pop() + 1))


def resolve_model_file(config: BackendConfig) -> Path:
    """Return a local path to the (first shard of the) GGUF, downloading if needed.

    Blocking (network + disk); call it from a worker thread.
    """
    if config.model_path:
        path = Path(config.model_path).expanduser()
        if not path.is_file():
            raise BackendError(f"backend.model_path does not exist: {path}")
        return path

    spec = config.model
    local = Path(spec).expanduser()
    if local.is_file():
        return local

    parsed = parse_model_spec(spec)
    if parsed is None:
        raise BackendError(
            f"GGUF model not found: {spec}",
            hint="Use a local .gguf path or a Hugging Face spec like "
            "'unsloth/Qwen3.6-35B-A3B-GGUF:UD-IQ4_XS'.",
        )
    repo, quant = parsed
    target = Path(config.models_dir).expanduser() / repo

    if target.is_dir():
        found = [p.relative_to(target).as_posix() for p in target.rglob("*.gguf")]
        chosen = match_gguf_files(found, quant)
        if chosen and shards_complete(chosen):
            logger.info("using cached GGUF %s", target / chosen[0])
            return target / chosen[0]
        if chosen:
            logger.warning("cached GGUF %s is missing parts; fetching them", target / chosen[0])

    try:
        from huggingface_hub import hf_hub_download, list_repo_files
    except ImportError as exc:  # pragma: no cover - a base dependency; only a broken install lacks it
        raise BackendError(
            "huggingface_hub is required to download models.",
            hint='It is a core dependency, so the install is incomplete: run pip install -e "." '
            "in the Kumaru folder, or download the .gguf yourself and set backend.model_path.",
        ) from exc

    try:
        files = list_repo_files(repo)
    except Exception as exc:
        raise BackendError(f"Could not list files of '{repo}' on Hugging Face: {exc}") from exc

    chosen = match_gguf_files(files, quant)
    if not chosen:
        raise BackendError(
            f"No GGUF matching quant '{quant}' in {repo}."
            if quant
            else f"{repo} contains several GGUF variants; name one as '{repo}:QUANT'.",
            hint=f"Available: {_available_quants(files)}",
        )

    target.mkdir(parents=True, exist_ok=True)
    for i, filename in enumerate(chosen, 1):
        logger.info("downloading %s (%d/%d) into %s", filename, i, len(chosen), target)
        hf_hub_download(repo_id=repo, filename=filename, local_dir=str(target))
    return target / chosen[0]


# ---------------------------------------------------------------------------
# Process management
# ---------------------------------------------------------------------------


def find_llama_server(binary: str | None) -> Path:
    """Locate the llama-server executable: explicit path, bundled copy, then PATH."""
    if binary:
        path = Path(binary).expanduser()
        if path.is_file():
            return path
        raise BackendError(f"backend.llama_server.binary does not exist: {path}")
    if BUNDLED_BINARY.is_file():
        return BUNDLED_BINARY
    on_path = shutil.which("llama-server")
    if on_path:
        return Path(on_path)
    raise BackendError(
        "llama-server was not found.",
        hint="Run .\\scripts\\get-llama-server.ps1 (Windows), or install llama.cpp and put "
        "llama-server on PATH, or set backend.llama_server.binary.",
    )


def dial_address(host: str) -> str:
    """The address a client connects to for a server bound to `host`.

    A wildcard bind ("0.0.0.0" or "::") listens on every interface but is not
    itself a destination (Windows refuses to connect to 0.0.0.0), so use the
    loopback address of the same family.
    """
    address = host.strip("[]")
    if address in ("", "0.0.0.0"):
        return "127.0.0.1"
    if address == "::":
        return "::1"
    return address


def url_host(host: str) -> str:
    """`dial_address`, bracketed when it is an IPv6 literal, as URLs require."""
    address = dial_address(host)
    return f"[{address}]" if ":" in address else address


def free_port(host: str) -> int:
    """A TCP port that nothing is bound to on `host` right now."""
    address = host.strip("[]")
    family = socket.AF_INET6 if ":" in address else socket.AF_INET
    with socket.socket(family, socket.SOCK_STREAM) as s:
        s.bind((address, 0))
        return int(s.getsockname()[1])


def port_in_use(host: str, port: int, timeout_s: float = 0.5) -> bool:
    """True when something already accepts connections at `host`:`port`."""
    try:
        with socket.create_connection((dial_address(host), port), timeout=timeout_s):
            return True
    except OSError:
        return False


def build_server_args(binary: Path, model_file: Path, config: BackendConfig, port: int) -> list[str]:
    """The exact command line. Kept pure so tests can assert on it."""
    o = config.llama_server
    args = [
        str(binary),
        "--model", str(model_file),
        "--host", o.host,
        "--port", str(port),
        "--ctx-size", str(config.max_context_length),
        "--parallel", str(o.parallel),
        "--flash-attn", o.flash_attn,
        "--jinja",
        "--alias", config.model,
        "--no-webui",
        "--log-colors", "off",
    ]  # fmt: skip
    if o.kv_unified:
        args.append("--kv-unified")
    # -1 means "as many as fit": leave --n-gpu-layers unset so llama.cpp's
    # --fit logic sizes the offload to the free VRAM it actually measures.
    if config.n_gpu_layers != -1:
        args += ["--n-gpu-layers", str(config.n_gpu_layers)]
    if config.n_threads:
        args += ["--threads", str(config.n_threads)]
    if o.cache_type_k:
        args += ["--cache-type-k", o.cache_type_k]
    if o.cache_type_v:
        args += ["--cache-type-v", o.cache_type_v]
    if o.n_cpu_moe is not None:
        args += ["--n-cpu-moe", str(o.n_cpu_moe)]
    if o.batch_size:
        args += ["--batch-size", str(o.batch_size)]
    if o.ubatch_size:
        args += ["--ubatch-size", str(o.ubatch_size)]
    args += list(o.extra_args)
    return args


def _kill_on_parent_exit_linux() -> None:  # pragma: no cover - runs in the child
    import ctypes
    import signal

    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    libc.prctl(1, signal.SIGTERM)  # PR_SET_PDEATHSIG


class _WindowsJob:  # pragma: no cover - Windows only
    """A Job Object that kills its processes when the last handle closes.

    Kumaru holds the only handle, so if Kumaru dies for any reason — including
    a hard kill from Task Manager — Windows terminates llama-server too.
    """

    def __init__(self) -> None:
        import ctypes
        from ctypes import wintypes

        class IO_COUNTERS(ctypes.Structure):
            _fields_ = [(n, ctypes.c_ulonglong) for n in (
                "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
                "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]  # fmt: skip

        class BASIC_LIMIT(ctypes.Structure):
            _fields_ = [
                ("PerProcessUserTimeLimit", ctypes.c_int64),
                ("PerJobUserTimeLimit", ctypes.c_int64),
                ("LimitFlags", wintypes.DWORD),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.c_size_t),
                ("PriorityClass", wintypes.DWORD),
                ("SchedulingClass", wintypes.DWORD),
            ]

        class EXTENDED_LIMIT(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", BASIC_LIMIT),
                ("IoInfo", IO_COUNTERS),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t),
            ]

        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.CreateJobObjectW.restype = wintypes.HANDLE
        k32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        k32.SetInformationJobObject.restype = wintypes.BOOL
        k32.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
        k32.AssignProcessToJobObject.restype = wintypes.BOOL
        k32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        k32.CloseHandle.argtypes = [wintypes.HANDLE]
        self._k32 = k32

        self.handle = k32.CreateJobObjectW(None, None)
        if not self.handle:
            raise OSError(ctypes.get_last_error(), "CreateJobObjectW failed")
        info = EXTENDED_LIMIT()
        info.BasicLimitInformation.LimitFlags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not k32.SetInformationJobObject(self.handle, 9, ctypes.byref(info), ctypes.sizeof(info)):
            raise OSError(ctypes.get_last_error(), "SetInformationJobObject failed")

    def assign(self, proc: subprocess.Popen) -> None:
        import ctypes

        if not self._k32.AssignProcessToJobObject(self.handle, int(proc._handle)):  # type: ignore[attr-defined]
            raise OSError(ctypes.get_last_error(), "AssignProcessToJobObject failed")

    def close(self) -> None:
        if self.handle:
            self._k32.CloseHandle(self.handle)
            self.handle = None


def _terminate(proc: subprocess.Popen, timeout_s: float = 10.0) -> None:
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
    _LIVE_PROCESSES.discard(proc)


@atexit.register
def _terminate_all() -> None:
    for proc in list(_LIVE_PROCESSES):
        with contextlib.suppress(Exception):
            _terminate(proc, timeout_s=5)


def _log_tail(path: Path | None, lines: int = 25) -> str:
    if path is None or not path.exists():
        return ""
    with path.open("rb") as fh:
        fh.seek(0, os.SEEK_END)
        fh.seek(max(0, fh.tell() - 16_384))
        text = fh.read().decode("utf-8", "replace")
    return "\n".join(text.splitlines()[-lines:])


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------


@BACKENDS.register("llamacpp_server", aliases=("llama-server", "llama_server"))
class LlamaServerBackend(OpenAICompatBackend):
    """Spawn and drive a local llama.cpp `llama-server`.

    Set ``backend.base_url`` to *attach* to a llama-server you started yourself
    instead; Kumaru then never spawns or stops a process.
    """

    name = "llamacpp_server"

    def __init__(self, config: BackendConfig) -> None:
        super().__init__(config)
        self._managed = config.base_url is None
        self._proc: subprocess.Popen | None = None
        self._job: Any = None
        self._log_path: Path | None = None
        self._log_fh: Any = None
        self._model_file: Path | None = None
        self._root_url = self._base_url.removesuffix("/v1")
        self._props: dict[str, Any] = {}

    # -- lifecycle ----------------------------------------------------------

    async def _load(self) -> None:
        # Any failure or cancellation (e.g. the server shutting down mid-load)
        # must not leave a half-started process holding VRAM.
        try:
            if self._managed:
                await self._spawn()
            await super()._load()
            with contextlib.suppress(Exception):
                resp = await self._client.get(f"{self._root_url}/props")
                if resp.status_code == 200:
                    self._props = resp.json()
        except BaseException:
            self._stop_process()
            raise

    async def _spawn(self) -> None:
        opts = self.config.llama_server
        binary = find_llama_server(opts.binary)
        self._model_file = await asyncio.to_thread(resolve_model_file, self.config)
        # With a fixed port, a server that is already there (say, one left over
        # from an earlier run) would answer the health check in place of ours.
        if opts.port and await asyncio.to_thread(port_in_use, opts.host, opts.port):
            raise BackendError(
                f"Port {opts.port} on {opts.host} is already in use.",
                hint="Stop the program listening there (perhaps an earlier llama-server), choose "
                "another backend.llama_server.port, or leave it unset to pick a free one. To use "
                "a llama-server you started yourself, set backend.base_url instead.",
            )
        port = opts.port or free_port(opts.host)
        args = build_server_args(binary, self._model_file, self.config, port)

        self._log_path = Path(opts.log_file) if opts.log_file else REPO_ROOT / "runs" / "logs" / f"llama-server-{port}.log"
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_fh = self._log_path.open("ab")
        logger.info("starting llama-server on port %d (log: %s)", port, self._log_path)
        logger.debug("command: %s", subprocess.list2cmdline(args))

        kwargs: dict[str, Any] = {
            "stdin": subprocess.DEVNULL,
            "stdout": self._log_fh,
            "stderr": subprocess.STDOUT,
        }
        if os.name == "nt":
            kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
        elif sys.platform.startswith("linux"):
            kwargs["preexec_fn"] = _kill_on_parent_exit_linux

        self._proc = subprocess.Popen(args, **kwargs)
        _LIVE_PROCESSES.add(self._proc)
        if os.name == "nt":
            try:
                self._job = _WindowsJob()
                self._job.assign(self._proc)
            except OSError as exc:  # pragma: no cover - nested-job edge cases
                logger.warning("could not attach llama-server to a job object: %s", exc)

        address = url_host(opts.host)
        self._base_url = f"http://{address}:{port}/v1"
        self._root_url = f"http://{address}:{port}"
        await self._wait_until_healthy(opts.startup_timeout_s)

    async def _wait_until_healthy(self, timeout_s: float) -> None:
        """Poll /health: 503 while the model loads, 200 once it can serve."""
        import httpx

        deadline = time.monotonic() + timeout_s
        async with httpx.AsyncClient(timeout=5.0) as client:
            while True:
                assert self._proc is not None
                code = self._proc.poll()
                if code is not None:
                    tail = _log_tail(self._log_path)
                    self._stop_process()
                    raise BackendError(
                        f"llama-server exited during startup (code {code}). Last log lines:\n{tail}",
                        hint="Common causes: not enough VRAM (lower backend.max_context_length, "
                        "set llama_server.n_cpu_moe, or use a smaller quant), port in use, or a "
                        "llama.cpp build too old for this model (re-run get-llama-server.ps1).",
                    )
                with contextlib.suppress(httpx.TransportError):
                    resp = await client.get(f"{self._root_url}/health")
                    if resp.status_code == 200:
                        return
                if time.monotonic() > deadline:
                    self._stop_process()
                    raise BackendError(
                        f"llama-server did not become healthy within {timeout_s:.0f}s.",
                        hint=f"See the log: {self._log_path}",
                    )
                await asyncio.sleep(0.5)

    def _stop_process(self) -> None:
        if self._proc is not None:
            with contextlib.suppress(Exception):
                _terminate(self._proc)
            self._proc = None
        if self._job is not None:
            self._job.close()
            self._job = None
        if self._log_fh is not None:
            with contextlib.suppress(Exception):
                self._log_fh.close()
            self._log_fh = None

    async def aclose(self) -> None:
        await super().aclose()
        if self._managed:
            await asyncio.to_thread(self._stop_process)

    # -- protocol extensions --------------------------------------------------

    def _connection_lost(self, exc: Exception) -> BackendError:
        code = self._proc.poll() if self._proc is not None else None
        if code is None:
            return super()._connection_lost(exc)
        return BackendError(
            f"llama-server exited (code {code}) during the request. Last log lines:\n"
            f"{_log_tail(self._log_path)}",
            hint="Restart `kumaru serve` to start a fresh server. If it happens again, the log "
            "names the cause, most often running out of VRAM: lower backend.max_context_length "
            "or set llama_server.n_cpu_moe.",
        )

    def _extended_fields(self, params: GenerationParams) -> dict[str, Any]:
        fields = super()._extended_fields(params)
        # llama.cpp spells this knob differently from vLLM/transformers.
        if "repetition_penalty" in fields:
            fields["repeat_penalty"] = fields.pop("repetition_penalty")
        if params.thinking_budget is not None and params.thinking is not False:
            fields["reasoning_budget_tokens"] = params.thinking_budget
            fields["reasoning_budget_message"] = self.config.llama_server.reasoning_budget_message
        return fields

    # -- introspection --------------------------------------------------------

    def _device_label(self) -> str:
        if not self._managed:
            return "external server"  # placement is whatever its owner chose
        layers = self.config.n_gpu_layers
        if layers == 0:
            return "cpu"
        return "gpu (auto-fit)" if layers == -1 else f"gpu ({layers} layers)"

    def info(self) -> BackendInfo:
        parsed = parse_model_spec(self.config.model)
        settings = self._props.get("default_generation_settings") or {}
        extra: dict[str, Any] = {
            "engine": "llama.cpp llama-server",
            "base_url": self._base_url,
            "managed": self._managed,
            "build": self._props.get("build_info"),
            "slots": self._props.get("total_slots"),
            "model_file": str(self._model_file) if self._model_file else None,
            "pid": self._proc.pid if self._proc else None,
            "log_file": str(self._log_path) if self._log_path else None,
        }
        return BackendInfo(
            name=self.name,
            model_id=self.config.model,
            device=self._device_label(),
            dtype="gguf",
            context_length=int(settings.get("n_ctx") or self.config.max_context_length),
            supports_streaming=True,
            quantization=(parsed[1] if parsed and parsed[1] else "gguf"),
            extra={k: v for k, v in extra.items() if v is not None},
        )

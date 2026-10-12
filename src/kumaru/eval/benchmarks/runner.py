"""Run benchmarks, keep every answer, and compare runs with real statistics.

What a run does
---------------
1. Resolve the benchmarks, run their preflight checks, then load and sample the
   questions — all *before* the model loads, so a typo or a missing package
   fails in seconds instead of after a two-minute model load.
2. Load the backend once and reuse it for every benchmark.
3. Ask each question with the same decoding settings `kumaru chat` uses
   (`ChatAgent.resolve_params`) plus a per-question seed, `concurrency` at a
   time, and grade each answer the moment it arrives.
4. Append every record to `items.jsonl` as soon as it is graded, so a crash
   or Ctrl+C loses only the questions in flight — and re-running with the
   same `--out` resumes where it stopped. `summary.json` is written when a
   benchmark starts and again when it stops (finished, failed or stopped);
   after a hard kill it is stale until a resume rewrites it from `items.jsonl`.

Layout::

    runs/bench/20250101-120000-qwen36-think8192/
        run.json             # settings, resolved config, backend, hardware, git commit
        gsm8k/items.jsonl    # one line per question: output, reasoning, extracted
        gsm8k/summary.json   #   answer, verdict, tokens, latency, seed
        ...

Why keep records and not just scores: a score says *that* something changed;
the records say *what*. Compare two runs and you can read the exact questions
the new model gets right that the old one missed.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import platform
import re
import statistics
import subprocess
import sys
import time
import zlib
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field
from rich.console import Console
from rich.markup import escape
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

from kumaru import __version__
from kumaru.core.config import REPO_ROOT, KumaruConfig
from kumaru.core.errors import ConfigError, KumaruError
from kumaru.core.types import GenerationParams, Message
from kumaru.eval.benchmarks.base import BenchItem, Benchmark, select_items
from kumaru.eval.benchmarks.stats import (
    holm_adjust,
    mcnemar_exact,
    paired_bootstrap_ci,
    wilson_interval,
)

DEFAULT_OUT = REPO_ROOT / "runs" / "bench"
# Five failures in a row means the server is down or misconfigured (or the
# grader is broken), not that five questions were hard. Stop rather than fill
# the results with errors.
MAX_CONSECUTIVE_ERRORS = 5
# Outcomes a code benchmark records instead of how it found the answer.
TEST_OUTCOMES = frozenset({"passed", "failed", "timeout"})
# Head-room added to the estimated prompt size when sizing concurrency.
PROMPT_MARGIN_TOKENS = 256
MAX_OUTPUT_TOKENS = 131_072


# ---------------------------------------------------------------------------
# Data shapes (all JSON-serializable; these are the files on disk)
# ---------------------------------------------------------------------------


class BenchSettings(BaseModel):
    """How to run. Every field is recorded in run.json."""

    # Questions per benchmark (stratified sample); None = all of them.
    limit: int | None = 100
    # Same seed = same questions and same per-question sampling seeds.
    seed: int = 1234
    # Questions in flight; None = `default_concurrency`.
    concurrency: int | None = Field(default=None, ge=1)
    # None = the profile's default. Other None fields likewise defer to the profile.
    thinking: bool | None = None
    thinking_budget: int | None = Field(default=None, ge=0)
    max_tokens: int | None = Field(default=None, ge=1, le=MAX_OUTPUT_TOKENS)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    # Prepend Kumaru's own system prompt, exactly as chat does. Off by default:
    # public numbers are measured on the raw benchmark prompt.
    use_agent_prompt: bool = False
    allow_code_exec: bool = False
    label: str | None = None
    out_dir: Path | None = None


class ItemRecord(BaseModel):
    """Everything about one question: what was asked, said, extracted and judged."""

    id: str
    # Hash of exactly what was sent. Resuming reuses a record only if it matches.
    key: str = ""
    group: str | None = None
    # The question as sent (the user message), so a record is readable on its own.
    prompt: str = ""
    correct: bool = False
    score: float = 0.0
    extracted: str | None = None
    expected: Any = None
    output: str = ""
    reasoning: str | None = None
    finish_reason: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    seed: int | None = None
    # Set when the request or the grader failed. Errors count as wrong answers.
    error: str | None = None
    detail: dict[str, Any] = Field(default_factory=dict)


class BenchSummary(BaseModel):
    """One benchmark's result. Written to summary.json, read by compare/report."""

    benchmark: str
    label: str
    model: str
    backend: str
    thinking: bool | None = None
    thinking_budget: int | None = None
    params: dict[str, Any] = Field(default_factory=dict)
    n: int = 0
    n_planned: int = 0
    n_correct: int = 0
    accuracy: float = 0.0
    ci95: tuple[float, float] = (0.0, 1.0)
    mean_score: float = 0.0
    metrics: dict[str, float] = Field(default_factory=dict)
    by_group: dict[str, dict[str, float]] = Field(default_factory=dict)
    # How answers were found ("answer_tag", "last_number", "error", ...); a code
    # benchmark records its test outcomes instead ("passed", "failed", "timeout").
    # Graders that judge the whole reply (contains, regex, IFEval) add nothing.
    extraction: dict[str, int] = Field(default_factory=dict)
    n_errors: int = 0
    # Stopped by the token limit (finish_reason "length"): raise the budget.
    n_truncated: int = 0
    mean_completion_tokens: float = 0.0
    median_latency_ms: float = 0.0
    wall_seconds: float = 0.0
    items_per_minute: float = 0.0
    complete: bool = False
    seed: int = 0
    limit: int | None = None
    fingerprint: str = ""
    items_sha256: str = ""
    started_at: str = ""
    kumaru_version: str = __version__
    git_commit: str | None = None
    item_ids: list[str] = Field(default_factory=list)


class Comparison(BaseModel):
    """A paired comparison of two runs on one benchmark (A = baseline, B = candidate)."""

    benchmark: str
    baseline: str
    candidate: str
    n: int
    # Questions present in only one of the two runs (excluded from the pairing).
    unpaired: int = 0
    baseline_accuracy: float
    baseline_ci: tuple[float, float]
    candidate_accuracy: float
    candidate_ci: tuple[float, float]
    delta: float
    delta_ci: tuple[float, float]
    p_value: float
    p_holm: float = 1.0
    # Discordant pairs: only these carry information about which is better.
    baseline_only: int
    candidate_only: int
    baseline_tokens: float = 0.0
    candidate_tokens: float = 0.0
    baseline_latency_ms: float = 0.0
    candidate_latency_ms: float = 0.0


@dataclass
class _Question:
    item: BenchItem
    messages: list[Message]
    params: GenerationParams
    key: str


@dataclass
class _Job:
    """One benchmark of a run, fully resolved before anything is written."""

    bench: Benchmark
    items: list[BenchItem]
    params: GenerationParams
    fingerprint: str
    # Summary left by an earlier session of this same run (we are resuming).
    previous: BenchSummary | None


# ---------------------------------------------------------------------------
# Small pure helpers
# ---------------------------------------------------------------------------


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def safe_name(text: str) -> str:
    """A string usable as a directory name on every OS."""
    return re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-.") or "run"


def item_seed(seed: int, item_id: str) -> int:
    """Per-question sampling seed: different for every question, identical
    across runs and models, so two runs stay paired question by question."""
    return zlib.crc32(f"{seed}:{item_id}".encode()) & 0x7FFFFFFF


def record_key(fingerprint: str, messages: Sequence[Message], params: GenerationParams) -> str:
    payload = {
        "fingerprint": fingerprint,
        "messages": [m.to_openai() for m in messages],
        "max_tokens": params.max_tokens,
        "seed": params.seed,
    }
    return _sha(json.dumps(payload, sort_keys=True, ensure_ascii=False))[:16]


def items_digest(items: Sequence[BenchItem]) -> str:
    """Fingerprint of the question set: equal digests = identical questions."""
    rows = [[item.id, item.system, item.prompt] for item in items]
    return _sha(json.dumps(rows, ensure_ascii=False))[:16]


def model_name(model_id: str) -> str:
    """The short name in a model id: `org/repo:quant` -> `repo`, `C:\\m\\x.gguf` -> `x`."""
    name = re.split(r"[\\/]", model_id.rstrip("\\/"))[-1].split(":")[0]
    # Only weight-file extensions: "Qwen2.5-7B-Instruct" must keep its ".5-7B...".
    return re.sub(r"\.(gguf|safetensors|bin)$", "", name, flags=re.IGNORECASE)


def default_label(model_id: str, profile: str | None, params: GenerationParams) -> str:
    """`<profile or model>-<think mode>`, e.g. `qwen36-think8192`."""
    base = Path(profile).stem if profile else model_name(model_id)
    if params.thinking:
        mode = "think" if params.thinking_budget is None else f"think{params.thinking_budget}"
    else:
        mode = "nothink" if params.thinking is False else "default"
    return safe_name(f"{base}-{mode}")


def default_concurrency(cfg: KumaruConfig, backend: Any, per_request_tokens: int) -> int:
    """Questions in flight when `--concurrency` is not given.

    Only llama-server batches concurrent requests (its `parallel` slots);
    every other backend serves one at a time, so sending more only queues
    them — and a queued request can hit the read timeout. With a unified KV
    cache all slots share one context window, so the number in flight is also
    capped by how many worst-case requests fit in it at once.
    """
    if backend.name != "llamacpp_server":
        return 1
    options = cfg.backend.llama_server
    info = backend.info()
    slots = int(info.extra.get("slots") or options.parallel)
    if not options.kv_unified:
        return slots
    return max(1, min(slots, info.context_length // max(1, per_request_tokens)))


def git_commit() -> str | None:
    """Short commit of the code that produced a result, "+dirty" if modified."""
    def git(*args: str) -> str | None:
        try:
            proc = subprocess.run(
                ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, timeout=5,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return proc.stdout.strip() if proc.returncode == 0 else None

    sha = git("rev-parse", "--short", "HEAD")
    if not sha:
        return None
    return sha + ("+dirty" if git("status", "--porcelain", "--untracked-files=no") else "")


def environment() -> dict[str, Any]:
    """Hardware and software facts that change benchmark speed (never accuracy)."""
    env: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
    }
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.used,driver_version",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )  # fmt: skip
        if proc.returncode == 0:
            env["gpus"] = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    except (OSError, subprocess.SubprocessError):
        pass
    return env


# Credential-like key names ("api_key", "hf_token", "client_secret", "x-api-key",
# "Authorization"), matched as whole name segments so that "max_tokens" or
# "tokenizer" do not count.
_SECRET_KEY = re.compile(
    r"(?:^|[_.-])(?:api[_-]?key|token|secret|password|passwd|authorization|credentials?)(?:$|[_.-])",
    re.IGNORECASE,
)


def _redact(value: Any) -> Any:
    """Copy of a config dump with secrets removed (run.json is meant to be shared).

    Numbers are never hidden: a secret is text, and a count such as a token
    budget is exactly what a reader of the run needs to see.
    """
    if isinstance(value, dict):
        return {
            key: ("<redacted>" if item and not isinstance(item, int | float)
                  and _SECRET_KEY.search(str(key)) else _redact(item))
            for key, item in value.items()
        }  # fmt: skip
    if isinstance(value, list):
        return [_redact(item) for item in value]
    return value


def _write_atomic(path: Path, text: str) -> None:
    # A hard kill mid-write must never leave a half-written summary behind.
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _ends_mid_line(path: Path) -> bool:
    """True when a non-empty file's last line has no newline (a write torn by a kill)."""
    if not path.exists():
        return False
    with path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            return False
        handle.seek(-1, os.SEEK_END)
        return handle.read(1) != b"\n"


# ---------------------------------------------------------------------------
# Reading results
# ---------------------------------------------------------------------------


def read_records(path: Path) -> dict[str, ItemRecord]:
    """All records in an items.jsonl; the last one per id wins (retries supersede)."""
    out: dict[str, ItemRecord] = {}
    if not path.exists():
        return out
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                record = ItemRecord.model_validate_json(line)
            except ValueError:
                continue  # a line torn by a hard kill; that question is simply asked again
            out[record.id] = record
    return out


def read_summary(path: Path) -> BenchSummary:
    try:
        return BenchSummary.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ConfigError(f"Cannot read {path}: {exc}") from exc


def find_results(path: Path) -> dict[str, Path]:
    """Benchmark name -> result directory, for one benchmark's dir or a whole run."""
    path = Path(path)
    if (path / "summary.json").exists():
        return {read_summary(path / "summary.json").benchmark: path}
    found: dict[str, Path] = {}
    if path.is_dir():
        for child in sorted(path.iterdir()):
            if (child / "summary.json").exists():
                found[read_summary(child / "summary.json").benchmark] = child
    if not found:
        raise ConfigError(
            f"No benchmark results in {path}",
            hint="Point at a directory created by `kumaru bench run` (under runs/bench/).",
        )
    return found


def load_results(bench_dir: Path) -> tuple[BenchSummary, dict[str, ItemRecord]]:
    """A benchmark's summary plus its records, restricted to the planned questions."""
    summary = read_summary(bench_dir / "summary.json")
    records = read_records(bench_dir / "items.jsonl")
    if summary.item_ids:
        records = {i: records[i] for i in summary.item_ids if i in records}
    return summary, records


def load_summaries(path: Path) -> list[BenchSummary]:
    return [read_summary(d / "summary.json") for d in find_results(path).values()]


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------


def resolve_benchmark(spec: str) -> Benchmark:
    """A registry name ("gsm8k") or the path of a golden-set JSONL file."""
    from kumaru.core.registry import BENCHMARKS

    path = Path(spec)
    if path.suffix.lower() == ".jsonl" or (spec not in BENCHMARKS and path.is_file()):
        from kumaru.eval.benchmarks.golden import GoldenSet

        return GoldenSet(path)
    return BENCHMARKS.create(spec)


def list_benchmarks() -> list[dict[str, Any]]:
    from kumaru.core.registry import BENCHMARKS

    rows = []
    for name in BENCHMARKS.names():
        cls = BENCHMARKS.get(name)
        rows.append({
            "name": name,
            "description": cls.description,
            "source": cls.source,
            "code_exec": cls.needs_code_execution,
        })  # fmt: skip
    return rows


def summarize(
    bench: Benchmark,
    items: Sequence[BenchItem],
    records: dict[str, ItemRecord],
    header: dict[str, Any],
    *,
    wall_seconds: float,
) -> BenchSummary:
    """Turn the records of the planned questions into a summary."""
    selected = [records[item.id] for item in items if item.id in records]
    n = len(selected)
    n_correct = sum(r.correct for r in selected)
    answered = [r for r in selected if r.error is None]

    groups: dict[str, list[bool]] = defaultdict(list)
    extraction: Counter[str] = Counter()
    for record in selected:
        if record.group:
            groups[record.group].append(record.correct)
        if record.error:
            extraction["error"] += 1
        elif "method" in record.detail:
            extraction[str(record.detail["method"])] += 1
        elif "status" in record.detail:
            extraction[str(record.detail["status"])] += 1

    low, high = wilson_interval(n_correct, n)
    items_by_id = {item.id: item for item in items}
    tokens = [r.completion_tokens for r in answered]
    latencies = [r.latency_ms for r in answered]
    return BenchSummary(
        **header,
        n=n,
        n_planned=len(items),
        n_correct=n_correct,
        accuracy=n_correct / n if n else 0.0,
        ci95=(low, high),
        mean_score=statistics.fmean(r.score for r in selected) if selected else 0.0,
        metrics=bench.aggregate(selected, items_by_id) if selected else {},
        by_group={
            group: {"n": len(flags), "accuracy": sum(flags) / len(flags)}
            for group, flags in sorted(groups.items())
        },
        extraction=dict(extraction.most_common()),
        n_errors=sum(1 for r in selected if r.error),
        n_truncated=sum(1 for r in selected if r.finish_reason == "length"),
        mean_completion_tokens=statistics.fmean(tokens) if tokens else 0.0,
        median_latency_ms=statistics.median(latencies) if latencies else 0.0,
        wall_seconds=round(wall_seconds, 1),
        items_per_minute=n / wall_seconds * 60 if wall_seconds > 0 else 0.0,
        complete=n == len(items),
        item_ids=[item.id for item in items],
    )


class BenchRunner:
    """Runs benchmarks against the backend described by a `KumaruConfig`."""

    def __init__(
        self,
        cfg: KumaruConfig,
        settings: BenchSettings,
        *,
        profile: str | None = None,
        console: Console | None = None,
        show_progress: bool = True,
    ) -> None:
        self.cfg = cfg
        self.settings = settings
        self.profile = profile
        self.console = console or Console()
        self.show_progress = show_progress
        self.out_dir: Path | None = None

    async def run(self, specs: Sequence[str]) -> list[BenchSummary]:
        benches = self._resolve(specs)
        # Questions first: dataset problems should fail before the model loads.
        plans = [
            (bench, select_items(bench.load(), self.settings.limit, self.settings.seed))
            for bench in benches
        ]

        from kumaru.agent.chat import ChatAgent
        from kumaru.core.registry import BACKENDS

        backend = BACKENDS.create(self.cfg.backend.name, self.cfg.backend)
        try:
            with self.console.status(f"[cyan]loading {self.cfg.backend.model}...", spinner="dots"):
                await backend.load()
            agent = ChatAgent(backend, self.cfg)
            base = agent.resolve_params(
                thinking=self.settings.thinking,
                thinking_budget=self.settings.thinking_budget,
                temperature=self.settings.temperature,
            )
            info = backend.info()
            label = safe_name(self.settings.label or default_label(info.model_id, self.profile, base))
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            self.out_dir = Path(self.settings.out_dir or DEFAULT_OUT / f"{stamp}-{label}")
            # Check every benchmark against earlier results *before* writing
            # anything, so a refused resume leaves the old run untouched.
            jobs = [self._job(bench, items, base, backend.name) for bench, items in plans]
            self.out_dir.mkdir(parents=True, exist_ok=True)
            self.console.print(f"[dim]results:[/] {self.out_dir}")
            self._write_run_json(info.model_dump(mode="json"), base, [b.name for b in benches], label)

            return [
                await self._run_benchmark(job, backend, agent, label, info.model_id)
                for job in jobs
            ]
        except (KeyboardInterrupt, asyncio.CancelledError):
            if self.out_dir is not None:
                self.console.print(
                    f"\n[yellow]Stopped.[/] Finished answers are saved; resume with --out {self.out_dir}"
                )
            raise
        finally:
            await backend.aclose()

    # -- internals ------------------------------------------------------------

    def _resolve(self, specs: Sequence[str]) -> list[Benchmark]:
        benches: list[Benchmark] = []
        seen: dict[str, tuple[type, str | None]] = {}
        for spec in specs:
            bench = resolve_benchmark(spec)
            path = getattr(bench, "path", None)
            identity = (type(bench), str(Path(path).resolve()) if path else None)
            if bench.name in seen:
                if seen[bench.name] != identity:
                    raise ConfigError(
                        f"Two benchmarks share the name '{bench.name}' (the second is {spec}), "
                        "so their results would share one folder.",
                        hint="Rename one of the files: a golden set is named after its file.",
                    )
                continue  # the same benchmark twice: run it once
            seen[bench.name] = identity
            if bench.needs_code_execution and not self.settings.allow_code_exec:
                raise ConfigError(
                    f"'{bench.name}' grades answers by running model-written code on this machine.",
                    hint="Add --allow-code-exec to accept that (docs/BENCHMARKS.md explains the "
                    "safeguards and their limits), or leave it out of the list.",
                )
            bench.preflight()
            benches.append(bench)
        if not benches:
            raise ConfigError("No benchmarks given.", hint="See `kumaru bench list`.")
        return benches

    def _job(
        self, bench: Benchmark, items: list[BenchItem], base: GenerationParams, backend_name: str
    ) -> _Job:
        """Resolve one benchmark's settings; refuse to mix them into a different run."""
        assert self.out_dir is not None
        params = base.merged(**bench.generation_overrides())
        fingerprint = self._fingerprint(bench, backend_name, params)
        bench_dir = self.out_dir / safe_name(bench.name)
        summary_path = bench_dir / "summary.json"
        previous = read_summary(summary_path) if summary_path.exists() else None
        if previous is not None and previous.fingerprint != fingerprint:
            raise ConfigError(
                f"{bench_dir} holds results from different settings "
                "(model, decoding parameters, seed or prompt mode).",
                hint="Omit --out (or choose a new directory) to start a fresh run.",
            )
        return _Job(bench, items, params, fingerprint, previous)

    def _max_tokens(self, bench: Benchmark, item: BenchItem, params: GenerationParams) -> int:
        """Output budget for one question.

        Thinking off: the benchmark's answer budget. Thinking on: the thinking
        budget plus the answer budget, or the profile's thinking `max_tokens`
        when thinking is unbudgeted. A truncated answer is graded as it stands
        and usually fails, so these are generous on purpose; `n_truncated`
        shows when they were not.
        """
        if self.settings.max_tokens:
            return self.settings.max_tokens
        tokens = item.max_tokens or bench.answer_tokens
        if params.thinking:
            if params.thinking_budget is not None:
                tokens += params.thinking_budget
            else:
                tokens = max(tokens, params.max_tokens)
        return min(tokens, MAX_OUTPUT_TOKENS)

    def _fingerprint(self, bench: Benchmark, backend_name: str, params: GenerationParams) -> str:
        """Everything that must match for two sessions to be one run.

        Settings only, not code: after changing a grader or a prompt template,
        start a new run directory rather than resuming an old one.
        """
        b, a = self.cfg.backend, self.cfg.agent
        # What shapes the prepended prompt. The record keys hash the bare
        # question instead, because that prompt also states today's date.
        agent_prompt = (
            {
                "system_prompt": a.system_prompt,
                "enabled_tools": a.enabled_tools,
                "max_history_tokens": a.max_history_tokens,
                "memory_policy": a.memory_policy,
            }
            if self.settings.use_agent_prompt
            else None
        )
        payload = {
            "benchmark": bench.name,
            "backend": backend_name,
            "model": b.model,
            "model_path": b.model_path,
            "adapter": b.adapter_path,
            "params": params.model_dump(mode="json", exclude={"seed", "max_tokens"}),
            "max_tokens": self.settings.max_tokens,
            "seed": self.settings.seed,
            "agent_prompt": agent_prompt,
        }
        return _sha(json.dumps(payload, sort_keys=True))[:16]

    def _question(
        self, bench: Benchmark, item: BenchItem, params: GenerationParams, fingerprint: str, agent: Any
    ) -> _Question:
        max_tokens = self._max_tokens(bench, item, params)
        messages = bench.messages(item)
        item_params = params.merged(max_tokens=max_tokens, seed=item_seed(self.settings.seed, item.id))
        # Keyed before the agent prompt is added: it states today's date, so a
        # run resumed the next day would otherwise ask every question again.
        key = record_key(fingerprint, messages, item_params)
        if self.settings.use_agent_prompt:
            messages = agent.prepare(messages, reserve_tokens=max_tokens)
        return _Question(item, messages, item_params, key)

    def _write_run_json(self, info: dict[str, Any], base: GenerationParams, names: list[str], label: str) -> None:
        assert self.out_dir is not None
        payload = {
            "label": label,
            "benchmarks": names,
            "started_at": _now_iso(),
            "kumaru_version": __version__,
            "git_commit": git_commit(),
            "command": sys.argv,
            "settings": self.settings.model_dump(mode="json"),
            "params": base.model_dump(mode="json"),
            "backend": info,
            "environment": environment(),
            "config": _redact(self.cfg.model_dump(mode="json")),
        }
        path = self.out_dir / "run.json"
        if path.exists():  # a resumed run: keep the original, record this session too
            path = self.out_dir / f"run.resumed-{datetime.now():%Y%m%d-%H%M%S}.json"
        _write_atomic(path, json.dumps(payload, indent=2, ensure_ascii=False))

    async def _run_benchmark(
        self, job: _Job, backend: Any, agent: Any, label: str, model_id: str
    ) -> BenchSummary:
        assert self.out_dir is not None
        bench, items, params, fingerprint, previous = (
            job.bench, job.items, job.params, job.fingerprint, job.previous
        )  # fmt: skip
        bench_dir = self.out_dir / safe_name(bench.name)
        bench_dir.mkdir(parents=True, exist_ok=True)
        items_path = bench_dir / "items.jsonl"
        summary_path = bench_dir / "summary.json"

        questions = [self._question(bench, item, params, fingerprint, agent) for item in items]
        prior = read_records(items_path)
        torn = _ends_mid_line(items_path)
        records: dict[str, ItemRecord] = {}
        for question in questions:
            old = prior.get(question.item.id)
            if old is not None and old.error is None and old.key == question.key:
                records[old.id] = old
        todo = [q for q in questions if q.item.id not in records]

        worst_case = max(
            (backend.count_tokens(q.messages) + q.params.max_tokens for q in todo), default=0
        )
        concurrency = self.settings.concurrency or default_concurrency(
            self.cfg, backend, worst_case + PROMPT_MARGIN_TOKENS
        )

        recorded_params = params.model_dump(mode="json", exclude={"seed"})
        if questions:
            recorded_params["max_tokens"] = max(q.params.max_tokens for q in questions)
        header = {
            "benchmark": bench.name,
            "label": label,
            "model": model_id,
            "backend": backend.name,
            "thinking": params.thinking,
            "thinking_budget": params.thinking_budget,
            "params": recorded_params,
            "seed": self.settings.seed,
            "limit": self.settings.limit,
            "fingerprint": fingerprint,
            "items_sha256": items_digest(items),
            "started_at": previous.started_at if previous else _now_iso(),
            "git_commit": git_commit(),
        }
        earlier_wall = previous.wall_seconds if previous else 0.0
        started = time.perf_counter()

        def write_summary() -> BenchSummary:
            summary = summarize(
                bench, items, records, header,
                wall_seconds=earlier_wall + time.perf_counter() - started,
            )  # fmt: skip
            _write_atomic(summary_path, summary.model_dump_json(indent=2))
            return summary

        # Written before the first question, so even a hard kill leaves a
        # fingerprint that a resume can check.
        write_summary()

        tally = Counter(
            answered=len(records), correct=sum(r.correct for r in records.values())
        )
        progress = Progress(
            TextColumn("[bold cyan]{task.description:<12}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("{task.fields[status]}"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=self.console,
            disable=not self.show_progress,
        )

        def status() -> str:
            answered = tally["answered"]
            accuracy = tally["correct"] / answered if answered else 0.0
            errors = f"  [red]{tally['errors']} errors[/]" if tally["errors"] else ""
            return f"acc {accuracy:5.1%}{errors}  j={concurrency}"

        task = progress.add_task(
            bench.name, total=len(questions), completed=len(records), status=status()
        )
        semaphore = asyncio.Semaphore(concurrency)
        consecutive_errors = 0

        with items_path.open("a", encoding="utf-8") as sink, progress:
            if torn:
                # Finish the torn line, or the first new record would be glued
                # onto it and lost with it.
                sink.write("\n")

            async def worker(question: _Question) -> None:
                nonlocal consecutive_errors
                async with semaphore:
                    record = await self._ask(bench, backend, question)
                sink.write(record.model_dump_json() + "\n")
                sink.flush()
                records[record.id] = record
                tally["answered"] += 1
                tally["correct"] += record.correct
                tally["errors"] += record.error is not None
                consecutive_errors = consecutive_errors + 1 if record.error else 0
                progress.update(task, advance=1, status=status())
                if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                    raise KumaruError(
                        f"Stopped {bench.name}: {consecutive_errors} questions failed in a row. "
                        f"Last error: {record.error}",
                        hint="An error starting 'grader failed' is the benchmark's own bug; otherwise "
                        "check the backend with `kumaru chat -q hi` and the same flags. "
                        f"Re-run with --out {self.out_dir} to resume; failed questions are retried.",
                    )

            tasks = [asyncio.create_task(worker(q)) for q in todo]
            try:
                await asyncio.gather(*tasks)
            finally:
                for pending in tasks:
                    pending.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                summary = write_summary()
        return summary

    async def _ask(self, bench: Benchmark, backend: Any, question: _Question) -> ItemRecord:
        """Ask one question and grade the reply. Never raises for model failures."""
        item = question.item
        record = ItemRecord(
            id=item.id,
            key=question.key,
            group=item.group,
            prompt=item.prompt,
            expected=item.answer,
            seed=question.params.seed,
        )
        try:
            result = await backend.generate(question.messages, question.params)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # one failed request must not end the run
            record.error = f"{type(exc).__name__}: {exc}"[:500]
            return record

        record.output = result.message.content
        record.reasoning = result.message.reasoning
        record.finish_reason = result.finish_reason
        record.prompt_tokens = result.usage.prompt_tokens
        record.completion_tokens = result.usage.completion_tokens
        record.latency_ms = round(result.latency_ms, 1)
        if result.finish_reason == "error":
            record.error = "the backend reported an error during generation"
            return record

        try:
            if bench.blocking_grader:
                grade = await asyncio.to_thread(bench.grade, item, record.output)
            else:
                grade = bench.grade(item, record.output)
        except Exception as exc:
            record.error = f"grader failed: {type(exc).__name__}: {exc}"[:500]
            return record

        record.correct = grade.correct
        record.score = grade.score if grade.score is not None else float(grade.correct)
        record.extracted = grade.extracted
        record.detail = grade.detail
        return record


async def run_benchmarks(
    cfg: KumaruConfig,
    specs: Sequence[str],
    settings: BenchSettings,
    *,
    profile: str | None = None,
    console: Console | None = None,
    show_progress: bool = True,
) -> tuple[Path, list[BenchSummary]]:
    """Run `specs` and return `(results_dir, summaries)`."""
    runner = BenchRunner(cfg, settings, profile=profile, console=console, show_progress=show_progress)
    summaries = await runner.run(specs)
    assert runner.out_dir is not None
    return runner.out_dir, summaries


# ---------------------------------------------------------------------------
# Comparing
# ---------------------------------------------------------------------------


def _cost(records: dict[str, ItemRecord], ids: Sequence[str]) -> tuple[float, float]:
    """Mean completion tokens and median latency over the answered questions."""
    ok = [records[i] for i in ids if records[i].error is None]
    if not ok:
        return 0.0, 0.0
    return (
        statistics.fmean(r.completion_tokens for r in ok),
        statistics.median(r.latency_ms for r in ok),
    )


def compare_runs(baseline: Path, candidate: Path, *, resamples: int = 10_000) -> list[Comparison]:
    """Paired comparison on every benchmark the two runs share.

    Pairs on question ids present in both runs, so a partial run can still be
    compared on what it finished. p-values are Holm-adjusted across the
    benchmarks compared, because testing five benchmarks at once makes a
    chance "win" on one of them likely.
    """
    a_dirs, b_dirs = find_results(baseline), find_results(candidate)
    common = [name for name in a_dirs if name in b_dirs]
    if not common:
        raise ConfigError(
            "The two runs share no benchmark.",
            hint=f"Baseline has {', '.join(a_dirs)}; candidate has {', '.join(b_dirs)}.",
        )

    out: list[Comparison] = []
    for name in common:
        a_summary, a_records = load_results(a_dirs[name])
        b_summary, b_records = load_results(b_dirs[name])
        ids = [i for i in a_records if i in b_records]
        unpaired = len(set(a_records) ^ set(b_records))
        x = [a_records[i].correct for i in ids]
        y = [b_records[i].correct for i in ids]
        baseline_only = sum(1 for p, q in zip(x, y, strict=True) if p and not q)
        candidate_only = sum(1 for p, q in zip(x, y, strict=True) if q and not p)
        delta, low, high = paired_bootstrap_ci(
            [float(v) for v in x], [float(v) for v in y], resamples=resamples
        )
        a_tokens, a_latency = _cost(a_records, ids)
        b_tokens, b_latency = _cost(b_records, ids)
        n = len(ids)
        out.append(
            Comparison(
                benchmark=name,
                baseline=a_summary.label,
                candidate=b_summary.label,
                n=n,
                unpaired=unpaired,
                baseline_accuracy=sum(x) / n if n else 0.0,
                baseline_ci=wilson_interval(sum(x), n),
                candidate_accuracy=sum(y) / n if n else 0.0,
                candidate_ci=wilson_interval(sum(y), n),
                delta=delta,
                delta_ci=(low, high),
                p_value=mcnemar_exact(baseline_only, candidate_only),
                baseline_only=baseline_only,
                candidate_only=candidate_only,
                baseline_tokens=a_tokens,
                candidate_tokens=b_tokens,
                baseline_latency_ms=a_latency,
                candidate_latency_ms=b_latency,
            )
        )

    for comparison, adjusted in zip(out, holm_adjust([c.p_value for c in out]), strict=True):
        comparison.p_holm = adjusted
    return out


def verdict(c: Comparison) -> str:
    """Plain-language reading of a comparison, at the 5% level after correction."""
    if c.n == 0:
        return "no paired questions"
    winner = "B better" if c.delta > 0 else "A better"
    if c.p_holm < 0.05:
        return winner
    if c.p_value < 0.05:
        return f"{winner}? (not after correction)"
    return "no clear difference"


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _pct(value: float) -> str:
    return f"{value * 100:.1f}"


def render_summaries(console: Console, summaries: Sequence[BenchSummary], *, details: bool = False) -> None:
    if not summaries:
        console.print("[dim]no results[/]")
        return
    first = summaries[0]
    mode = "think" if first.thinking else "no-think" if first.thinking is False else "default"
    if first.thinking and first.thinking_budget is not None:
        mode += f" (budget {first.thinking_budget})"
    table = Table(
        title=f"{escape(first.label)}  [dim]{escape(first.model)} · {mode}[/]",
        border_style="dim",
        title_justify="left",
    )
    for column in ("benchmark", "n", "accuracy %", "95% CI", "errors", "cut off",
                   "tokens/answer", "answers/min"):  # fmt: skip
        table.add_column(column, justify="left" if column == "benchmark" else "right")
    for s in summaries:
        n_text = f"{s.n}" if s.complete else f"{s.n}/{s.n_planned} [yellow](partial)[/]"
        table.add_row(
            escape(s.benchmark),
            n_text,
            f"[bold]{_pct(s.accuracy)}[/]",
            f"{_pct(s.ci95[0])}-{_pct(s.ci95[1])}",
            f"[red]{s.n_errors}[/]" if s.n_errors else "0",
            f"[yellow]{s.n_truncated}[/]" if s.n_truncated else "0",
            f"{s.mean_completion_tokens:,.0f}",
            f"{s.items_per_minute:.1f}",
        )
    console.print(table)

    for s in summaries:
        lines = []
        if s.metrics:
            shown = {k: v for k, v in s.metrics.items() if details or "/" not in k}
            lines.append("  ".join(f"{k} {_pct(v)}" for k, v in shown.items()))
        if s.extraction:
            lines.append(_extraction_line(s))
        if details and s.by_group:
            lines.append("by group: " + ", ".join(
                f"{g} {_pct(v['accuracy'])} (n={int(v['n'])})" for g, v in s.by_group.items()
            ))  # fmt: skip
        if lines:
            body = escape("\n    ".join(lines))
            console.print(f"[cyan]{escape(s.benchmark)}[/]  {body}", highlight=False)
    if any(s.n_truncated for s in summaries):
        console.print(
            "[yellow]Some answers hit the token limit (cut off) and were graded as they stood, "
            "usually wrong; raise --max-tokens or --thinking-budget.[/]"
        )


def _extraction_line(s: BenchSummary) -> str:
    """`answers found by: answer_tag 14, whole reply 13`, or `tests: passed 80, failed 20`."""
    counts = dict(s.extraction)
    if TEST_OUTCOMES & counts.keys():
        label = "tests"
    else:
        label = "answers found by"
        # contains/regex-style graders judge the whole reply and record no
        # method; count them, or a mixed golden set seems to lose answers.
        if (unread := s.n - sum(counts.values())) > 0:
            counts["whole reply"] = unread
    return f"{label}: " + ", ".join(f"{k} {v}" for k, v in counts.items())


def render_comparisons(console: Console, comparisons: Sequence[Comparison]) -> None:
    if not comparisons:
        console.print("[dim]nothing to compare[/]")
        return
    first = comparisons[0]
    table = Table(
        title=f"A = {escape(first.baseline)}   B = {escape(first.candidate)}",
        border_style="dim",
        title_justify="left",
    )
    for column in ("benchmark", "n", "A %", "B %", "B - A (95% CI)", "only A / only B",
                   "p (Holm)", "verdict", "tokens A -> B"):  # fmt: skip
        table.add_column(column, justify="left" if column in ("benchmark", "verdict") else "right")
    for c in comparisons:
        colour = "green" if c.delta > 0 else "red" if c.delta < 0 else "white"
        table.add_row(
            escape(c.benchmark),
            f"{c.n}" + (f" [dim](+{c.unpaired} unpaired)[/]" if c.unpaired else ""),
            _pct(c.baseline_accuracy),
            _pct(c.candidate_accuracy),
            f"[{colour}]{c.delta * 100:+.1f}[/] ({c.delta_ci[0] * 100:+.1f} to {c.delta_ci[1] * 100:+.1f})",
            f"{c.baseline_only} / {c.candidate_only}",
            f"{c.p_holm:.3g}",
            verdict(c),
            f"{c.baseline_tokens:,.0f} -> {c.candidate_tokens:,.0f}",
        )
    console.print(table)
    console.print(
        "[dim]Only questions the two runs answered differently decide the test (only A / only B). "
        "p is an exact McNemar test, Holm-corrected across benchmarks.[/]"
    )


def _short(value: Any, width: int = 100) -> str:
    text = " ".join(str(value).split()) if value is not None else "-"
    return text if len(text) <= width else text[: width - 1] + "…"


def render_flips(console: Console, baseline: Path, candidate: Path, limit: int) -> None:
    """Print up to `limit` questions per direction that one run got right and the
    other got wrong — the fastest way to see *what* a change actually changed."""
    a_dirs, b_dirs = find_results(baseline), find_results(candidate)
    for name in (n for n in a_dirs if n in b_dirs):
        _, a = load_results(a_dirs[name])
        _, b = load_results(b_dirs[name])
        ids = [i for i in a if i in b]
        for title, chosen in (
            ("only B right", [i for i in ids if b[i].correct and not a[i].correct]),
            ("only A right", [i for i in ids if a[i].correct and not b[i].correct]),
        ):
            if not chosen:
                continue
            console.print(f"\n[bold cyan]{escape(name)}[/]: {title} ({len(chosen)})")
            for i in chosen[:limit]:
                console.print(f"  [bold]{escape(i)}[/]  {escape(_short(a[i].prompt))}")
                console.print(
                    f"    expected {escape(_short(a[i].expected, 40))}"
                    f"  ·  A: {escape(_short(a[i].extracted or a[i].error, 40))}"
                    f"  ·  B: {escape(_short(b[i].extracted or b[i].error, 40))}",
                    style="dim",
                )

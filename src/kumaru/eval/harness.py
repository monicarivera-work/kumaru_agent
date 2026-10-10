"""Evaluation: turning "it feels better" into a number.

Why this exists at all
----------------------
Every change you make — a new system prompt, a quantization setting, a
fine-tune — will *feel* like an improvement, because you chose it. Without
measurement you will ship regressions with total confidence. A harness that
takes thirty seconds to run is the difference between engineering and vibes.

Two dimensions, both required
-----------------------------
**Speed** (`speed` suite) — time to first token, tokens/second, total latency.
Fully objective, and the numbers you need to decide whether a scaling change was
worth it.

**Quality** (`smoke` suite, or your own JSONL) — does the answer contain what a
correct answer must contain? Substring checking is crude, and that is a
deliberate trade: it is fast, deterministic, needs no judge model, and reliably
catches the failures that matter (the model stopped answering, started looping,
forgot its format, lost a capability after fine-tuning). For nuanced grading,
add an LLM-judge scorer — the `Case`/`EvalResult` structures already support it.
"""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

from pydantic import BaseModel, Field

from kumaru.core.config import KumaruConfig
from kumaru.core.errors import ConfigError
from kumaru.core.logging import get_logger
from kumaru.core.types import Message

logger = get_logger(__name__)


class Case(BaseModel):
    """One test: a prompt plus what a correct answer must and must not contain."""

    prompt: str
    # All of these must appear (case-insensitive) for the case to pass.
    must_include: list[str] = Field(default_factory=list)
    # Any of these appearing fails the case — catches refusals and loops.
    must_not_include: list[str] = Field(default_factory=list)
    max_tokens: int = 256
    tag: str = "general"


class CaseResult(BaseModel):
    prompt: str
    tag: str
    output: str
    passed: bool
    missing: list[str] = Field(default_factory=list)
    forbidden_hits: list[str] = Field(default_factory=list)
    latency_ms: float = 0.0
    ttft_ms: float | None = None
    completion_tokens: int = 0

    @property
    def tokens_per_second(self) -> float:
        return self.completion_tokens / max(0.001, self.latency_ms / 1000)


class EvalResult(BaseModel):
    suite: str
    model: str
    backend: str
    n_cases: int
    n_passed: int
    pass_rate: float
    median_ttft_ms: float | None = None
    median_tokens_per_second: float = 0.0
    total_seconds: float = 0.0
    cases: list[CaseResult] = Field(default_factory=list)


# A tiny built-in suite. These are not a benchmark — they are a *regression
# tripwire*. Each one fails loudly if the model has broken in a specific way:
# instruction-following, arithmetic, format compliance, code generation, and
# calibrated refusal (which fine-tuning damages more often than anything else).
SMOKE_SUITE: list[Case] = [
    Case(
        prompt="What is the capital of Japan? Answer in one word.",
        must_include=["tokyo"],
        max_tokens=24,
        tag="factual",
    ),
    Case(
        prompt="Compute 17 * 23. Reply with only the number.",
        must_include=["391"],
        max_tokens=24,
        tag="arithmetic",
    ),
    Case(
        prompt="List exactly three primary colors as a JSON array of strings. Output only the JSON.",
        must_include=["[", "]", "red"],
        max_tokens=64,
        tag="format",
    ),
    Case(
        prompt="Write a Python function named `add` that returns the sum of two arguments. Code only.",
        must_include=["def add", "return"],
        max_tokens=96,
        tag="code",
    ),
    Case(
        prompt="In one sentence, what does the acronym GPU stand for?",
        must_include=["graphics"],
        max_tokens=48,
        tag="factual",
    ),
    Case(
        prompt=(
            "What did I have for breakfast this morning? You have no way to know. "
            "Say so directly."
        ),
        must_include=["know"],
        max_tokens=64,
        tag="calibration",
    ),
]

SPEED_SUITE: list[Case] = [
    Case(prompt="Count from 1 to 50, separated by spaces.", max_tokens=256, tag="speed"),
    Case(prompt="Write a 200-word explanation of how a CPU cache works.", max_tokens=512, tag="speed"),
    Case(prompt="Say 'ok'.", max_tokens=8, tag="latency"),
]


def load_suite(suite: str) -> tuple[str, list[Case]]:
    """Resolve a suite name or a path to a JSONL file of `Case` objects."""
    if suite == "smoke":
        return "smoke", SMOKE_SUITE
    if suite == "speed":
        return "speed", SPEED_SUITE

    path = Path(suite)
    if not path.exists():
        raise ConfigError(
            f"Unknown eval suite: {suite}",
            hint="Use 'smoke', 'speed', or a path to a JSONL file of eval cases.",
        )
    cases = [
        Case(**json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return path.stem, cases


async def run_eval(config: KumaruConfig, *, suite: str = "smoke", output: str | None = None) -> EvalResult:
    """Run a suite against a configured backend and report the results."""
    from rich.console import Console
    from rich.markup import escape
    from rich.table import Table

    from kumaru.agent.chat import ChatAgent
    from kumaru.core.registry import BACKENDS

    console = Console()
    suite_name, cases = load_suite(suite)

    backend = BACKENDS.create(config.backend.name, config.backend)
    with console.status(f"[cyan]loading {config.backend.model}…", spinner="dots"):
        await backend.load()
    agent = ChatAgent(backend, config)

    results: list[CaseResult] = []
    started = time.perf_counter()

    try:
        for i, case in enumerate(cases, 1):
            # Prompts are user data: rich would read "[/INST]" as a closing tag and raise.
            console.print(f"[dim]{i}/{len(cases)}[/] {escape(case.prompt[:64])}…")
            # temperature=0 makes eval deterministic. Evaluating with sampling on
            # means re-running gives a different score, and you can no longer
            # attribute a change to your edit rather than to luck.
            result = await agent.complete(
                [Message.user(case.prompt)], temperature=0.0, max_tokens=case.max_tokens
            )
            text = result.message.content
            low = text.lower()

            missing = [s for s in case.must_include if s.lower() not in low]
            forbidden = [s for s in case.must_not_include if s.lower() in low]

            results.append(
                CaseResult(
                    prompt=case.prompt,
                    tag=case.tag,
                    output=text,
                    passed=not missing and not forbidden,
                    missing=missing,
                    forbidden_hits=forbidden,
                    latency_ms=result.latency_ms,
                    ttft_ms=result.time_to_first_token_ms,
                    completion_tokens=result.usage.completion_tokens,
                )
            )
    finally:
        await backend.aclose()

    n_passed = sum(1 for r in results if r.passed)
    ttfts = [r.ttft_ms for r in results if r.ttft_ms]
    speeds = [r.tokens_per_second for r in results if r.completion_tokens > 4]

    report = EvalResult(
        suite=suite_name,
        model=config.backend.model,
        backend=config.backend.name,
        n_cases=len(results),
        n_passed=n_passed,
        pass_rate=round(n_passed / max(1, len(results)), 3),
        median_ttft_ms=round(statistics.median(ttfts), 1) if ttfts else None,
        median_tokens_per_second=round(statistics.median(speeds), 2) if speeds else 0.0,
        total_seconds=round(time.perf_counter() - started, 1),
        cases=results,
    )

    table = Table(title=f"eval: {suite_name} · {config.backend.model}", border_style="dim")
    table.add_column("tag", style="cyan")
    table.add_column("result")
    table.add_column("tok/s", justify="right")
    table.add_column("ttft", justify="right")
    table.add_column("note", overflow="fold")

    for r in results:
        note = ""
        if r.missing:
            note = f"missing: {', '.join(r.missing)}"
        elif r.forbidden_hits:
            note = f"forbidden: {', '.join(r.forbidden_hits)}"
        table.add_row(
            escape(r.tag),
            "[green]pass[/]" if r.passed else "[red]FAIL[/]",
            f"{r.tokens_per_second:.1f}",
            f"{r.ttft_ms:.0f}ms" if r.ttft_ms else "—",
            escape(note),
        )

    console.print(table)
    console.print(
        f"\n[bold]{n_passed}/{len(results)} passed[/] "
        f"({report.pass_rate:.0%}) · median {report.median_tokens_per_second} tok/s"
        + (f" · median ttft {report.median_ttft_ms:.0f}ms" if report.median_ttft_ms else "")
        + f" · {report.total_seconds}s total"
    )

    if output:
        Path(output).write_text(report.model_dump_json(indent=2), encoding="utf-8")
        console.print(f"[dim]results written to {output}[/]")

    return report

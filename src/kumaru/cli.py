"""Command-line entry point: `kumaru <command>`.

One binary, eight commands::

    kumaru serve     # API + web UI
    kumaru chat      # terminal REPL, no server needed
    kumaru info      # what is installed, what hardware is visible
    kumaru config    # print the fully resolved configuration
    kumaru train     # fine-tune or pretrain
    kumaru eval      # quick check: speed and a few known answers
    kumaru bench     # public benchmarks with confidence intervals; compare runs
    kumaru distill   # turn a teacher model's graded answers into training data
    kumaru learn     # build a reviewed local reference pack

Two rules keep the CLI fast and debuggable:

* **Heavy imports happen inside command bodies**, never at module scope. `kumaru
  --help` must not import torch; if it did, every invocation would cost several
  seconds.
* **Every command that loads a model accepts `--config`** plus flags for the
  settings you change most, for one-off experiments without editing YAML.
  Backend flags are written into the config; `chat`'s generation flags travel
  with each message, as the UI's settings do, so they beat the profile's presets.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table

from kumaru import __version__
from kumaru.core.errors import KumaruError

app = typer.Typer(
    name="kumaru",
    help="Local-first, scalable LLM agent.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()

CONTEXT_HELP = (
    "Context window in tokens. With llama-server it sizes the KV cache, the main "
    "VRAM cost after the weights: lower it (e.g. 65536) to free GPU memory."
)


def _build_overrides(**kwargs: Any) -> dict[str, Any]:
    """Turn flat CLI flags into the nested dict shape `load_config` expects."""
    mapping = {
        "backend": ("backend", "name"),
        "model": ("backend", "model"),
        "device": ("backend", "device"),
        "dtype": ("backend", "dtype"),
        "adapter": ("backend", "adapter_path"),
        "base_url": ("backend", "base_url"),
        "load_in_4bit": ("backend", "load_in_4bit"),
        "context": ("backend", "max_context_length"),
        "host": ("server", "host"),
        "port": ("server", "port"),
        "temperature": ("generation", "temperature"),
        "max_tokens": ("generation", "max_tokens"),
        "thinking": ("generation", "thinking"),
        "thinking_budget": ("generation", "thinking_budget"),
        "log_level": ("telemetry", "log_level"),
    }
    out: dict[str, Any] = {}
    for key, value in kwargs.items():
        if value is None or key not in mapping:
            continue
        section, field = mapping[key]
        out.setdefault(section, {})[field] = value
    return out


def _load(config: str | None, **flags: Any):
    from kumaru.core.config import load_config

    return load_config(config, overrides=_build_overrides(**flags))


# ---------------------------------------------------------------------------


@app.command()
def serve(
    config: str = typer.Option(None, "--config", "-c", help="YAML file or profile name."),
    backend: str = typer.Option(None, "--backend", "-b", help="echo | llamacpp_server | transformers | llamacpp | openai_compat"),
    model: str = typer.Option(None, "--model", "-m", help="Model id or path."),
    device: str = typer.Option(None, "--device", help="auto | cuda | cpu | mps"),
    adapter: str = typer.Option(None, "--adapter", help="Path to a trained LoRA adapter."),
    context: int = typer.Option(None, "--context", min=512, help=CONTEXT_HELP),
    host: str = typer.Option(None, "--host"),
    port: int = typer.Option(None, "--port", "-p"),
    log_level: str = typer.Option(None, "--log-level"),
) -> None:
    """Start the HTTP API and the web UI."""
    cfg = _load(config, backend=backend, model=model, device=device, adapter=adapter,
                context=context, host=host, port=port, log_level=log_level)

    console.print(
        Panel.fit(
            f"[bold cyan]Kumaru[/] v{__version__}\n\n"
            f"[dim]backend[/]  {cfg.backend.name}\n"
            f"[dim]model[/]    {cfg.backend.model}\n"
            f"[dim]device[/]   {cfg.backend.device}\n\n"
            f"[bold]UI[/]   http://{cfg.server.host}:{cfg.server.port}\n"
            f"[bold]API[/]  http://{cfg.server.host}:{cfg.server.port}/v1\n"
            f"[bold]Docs[/] http://{cfg.server.host}:{cfg.server.port}/docs",
            border_style="cyan",
        )
    )
    console.print("[dim]Model loads in the background; the UI shows progress.[/]\n")

    from kumaru.server.app import run

    run(cfg)


@app.command()
def chat(
    config: str = typer.Option(None, "--config", "-c"),
    backend: str = typer.Option(None, "--backend", "-b"),
    model: str = typer.Option(None, "--model", "-m"),
    device: str = typer.Option(None, "--device"),
    adapter: str = typer.Option(None, "--adapter"),
    context: int = typer.Option(None, "--context", min=512, help=CONTEXT_HELP),
    temperature: float = typer.Option(None, "--temperature", "-t", min=0.0, max=2.0),
    max_tokens: int = typer.Option(None, "--max-tokens", min=1, max=131072),
    think: bool = typer.Option(None, "--think/--no-think", help="Force reasoning on or off (default: profile)."),
    thinking_budget: int = typer.Option(None, "--thinking-budget", min=0, help="Max thinking tokens before answering."),
    prompt: str = typer.Option(None, "--prompt", "-q", help="Answer one question and exit."),
    workspace: str = typer.Option(
        None, "--workspace", "-w",
        help="Project folder the agent may work in: list, read, create and edit files there, "
        "and run allowed programs (agent.workspace.allowed_commands). It shows you each change "
        "and command first, and does it only if you answer y.",
    ),
) -> None:
    """Chat in the terminal, with no server in between."""
    import asyncio

    cfg = _load(config, backend=backend, model=model, device=device, adapter=adapter, context=context)
    if workspace:
        from kumaru.agent.tools import WORKSPACE_TOOLS

        cfg.agent.workspace.root = workspace
        cfg.agent.enabled_tools = list(dict.fromkeys([*cfg.agent.enabled_tools, *WORKSPACE_TOOLS]))
    # Sent with every message, as the UI sends its settings, so a flag beats both
    # of the profile's presets: `generation` and, while thinking, `thinking_mode`.
    overrides = {"temperature": temperature, "max_tokens": max_tokens,
                 "thinking": think, "thinking_budget": thinking_budget}
    with _exit_cleanly_on_signals():
        asyncio.run(_chat_loop(cfg, prompt, overrides))


def _visible(text: str) -> str:
    """`text` with control and invisible formatting characters spelled out.

    What the model sends is untrusted: an escape sequence or a right-to-left
    override in a file's content could hide or disguise lines of the approval
    prompt itself. Tabs and line breaks stay as they are.
    """
    import unicodedata

    return "".join(
        ch if ch in "\t\n" or unicodedata.category(ch) not in ("Cc", "Cf")
        else f"\\x{ord(ch):02x}" if ord(ch) < 0x100 else f"\\u{ord(ch):04x}"
        for ch in text
    )


def _printable(text: str) -> str:
    """Streamed model text with the control characters that steer a terminal spelled out.

    The model can repeat what a file or a command's output told it to, and an
    escape sequence printed raw could clear the screen, hide the text after it
    or redraw what came before, including the next approval prompt. A carriage
    return, which would let a line overwrite itself, is dropped. Unlike
    `_visible`, this keeps formatting characters such as the joiners inside
    emoji, since a reply is read rather than approved.
    """
    import unicodedata

    return "".join(
        "" if ch == "\r"
        else ch if ch in "\t\n" or unicodedata.category(ch) != "Cc"
        else f"\\x{ord(ch):02x}"
        for ch in text
    )


@contextmanager
def _exit_cleanly_on_signals() -> Iterator[None]:
    """On POSIX, turn a hangup or a `kill` into an exit that runs the cleanups.

    The programs `run_command` starts run in a session of their own, out of
    reach of the terminal, and are stopped by the cleanup that ends the turn.
    By default these signals end Python on the spot, skipping that cleanup, so
    closing the terminal would leave a dev server running. Windows needs none
    of this: a Job Object ends them however Kumaru exits. A signal someone
    chose to ignore (``nohup``) or handle stays as it was.
    """
    if sys.platform == "win32":
        yield
        return
    import signal

    def leave(signum: int, frame: Any) -> None:
        raise SystemExit(128 + signum)

    previous = {}
    for signum in (signal.SIGHUP, signal.SIGTERM):
        if signal.getsignal(signum) == signal.SIG_DFL:
            previous[signum] = signal.signal(signum, leave)
    try:
        yield
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


async def _approve(request: Any) -> bool:
    """Show exactly what a tool call will do, and ask. Anything but yes is no."""
    from rich.text import Text

    body = Text()
    for line in _visible(request.summary).splitlines():
        style = ""
        if line.startswith("+") and not line.startswith("+++"):
            style = "green"
        elif line.startswith("-") and not line.startswith("---"):
            style = "red"
        elif line.startswith("@@"):
            style = "cyan"
        body.append(line + "\n", style=style)
    body.rstrip()
    console.print(Panel(body, title=f"[bold]{escape(request.tool)}[/]", title_align="left",
                        border_style="yellow"))
    try:
        # Blocking on purpose: nothing else may run while you decide.
        answer = console.input(escape("Allow this? [y/N] "))
    except (EOFError, KeyboardInterrupt):  # no one to answer: the safe answer is no
        console.print()
        return False
    return answer.strip().lower() in ("y", "yes")


async def _chat_loop(cfg: Any, single_prompt: str | None, overrides: dict[str, Any]) -> None:
    from rich.text import Text

    from kumaru.agent.chat import ChatAgent
    from kumaru.core.registry import BACKENDS
    from kumaru.core.types import Message

    with console.status(f"[cyan]loading {cfg.backend.model}…", spinner="dots"):
        backend = BACKENDS.create(cfg.backend.name, cfg.backend)
        # Before load: a bad knowledge pack, tool name or project folder fails
        # with nothing to release. The terminal can ask, so it is the approver.
        agent = ChatAgent(backend, cfg, approver=_approve)
        await backend.load()

    info = backend.info()
    console.print(f"[dim]{info.model_id} on {info.device} ({info.dtype}), "
                  f"{info.context_length:,} token context[/]")
    if agent.tools:
        console.print(f"[dim]tools: {escape(', '.join(agent.tools.tools))}[/]")
    if agent.tools.workspace is not None:
        console.print(f"[dim]project folder:[/] {escape(str(agent.tools.workspace.root))} "
                      "[dim](every change and command asks you first)[/]")

    history: list[Message] = []

    async def ask(question: str) -> None:
        history.append(Message.user(question))
        console.print()
        turn: list[Message] = []
        stats = None
        thinking = False
        mid_line = False
        # Print tokens as they arrive: the point of streaming is that you start
        # reading before generation finishes. Thoughts print dimmed; they are
        # never added to the history sent back to the model. Both are untrusted
        # text, so control characters are spelled out.
        async for chunk in agent.stream(history, transcript=turn, **overrides):
            if chunk.reasoning:
                if not thinking:
                    console.print("[dim italic]thinking…[/]")
                    thinking = True
                console.print(_printable(chunk.reasoning), end="", style="dim", markup=False, highlight=False)
            if chunk.delta:
                if thinking:
                    console.print("\n")
                    thinking = False
                printed = _printable(chunk.delta)
                if printed:
                    console.print(printed, end="", markup=False, highlight=False)
                    mid_line = not printed.endswith("\n")
            if chunk.tool_calls:
                if thinking or mid_line:
                    console.print()
                    thinking = mid_line = False
                for call in chunk.tool_calls:
                    shown = call["function"]["arguments"]
                    shown = shown if len(shown) <= 120 else shown[:120] + "…"
                    console.print(Text(_visible(f"» {call['function']['name']}({shown})"), style="cyan"))
            if chunk.tool_result:
                lines = chunk.tool_result.content.strip().splitlines()
                first = lines[0] if lines else "(empty)"
                style = ("red" if first.startswith("ERROR")
                         else "yellow" if first.startswith(("DENIED", "NOT RUN")) else "dim")
                console.print(Text(_visible(f"« {first[:200]}"), style=style))
            if chunk.usage:
                stats = chunk.usage
        console.print("\n")
        if stats:
            console.print(f"[dim]{stats.prompt_tokens} prompt + {stats.completion_tokens} completion tokens[/]")
        # The calls, their results and the answer, so the next turn knows what was done.
        history.extend(turn)

    try:
        if single_prompt:
            await ask(single_prompt)
            return

        console.print("[dim]Type your message. /reset clears history, /exit quits.[/]\n")
        while True:
            try:
                question = console.input("[bold cyan]you ›[/] ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not question:
                continue
            if question in ("/exit", "/quit"):
                break
            if question == "/reset":
                history.clear()
                console.print("[dim]history cleared[/]")
                continue
            await ask(question)
    finally:
        await backend.aclose()


@app.command()
def info() -> None:
    """Show versions, hardware, and which optional components are installed."""
    import kumaru.agent.tools  # noqa: F401 - registers the built-in tools
    from kumaru.core.config import list_profiles
    from kumaru.core.registry import BACKENDS, BENCHMARKS, TOOLS

    table = Table(title=f"Kumaru v{__version__}", show_header=False, border_style="dim")
    table.add_column(style="cyan", no_wrap=True)
    table.add_column()

    table.add_row("python", sys.version.split()[0])
    table.add_row("backends", ", ".join(BACKENDS.names()))
    table.add_row("tools", ", ".join(TOOLS.names()) or "(none)")
    table.add_row("benchmarks", ", ".join(BENCHMARKS.names()))
    table.add_row("profiles", ", ".join(list_profiles()) or "(none)")

    def probe(module: str) -> str:
        try:
            mod = __import__(module)
            return getattr(mod, "__version__", "installed")
        except ImportError:
            return "[dim]not installed[/]"

    for mod in ("torch", "transformers", "peft", "trl", "datasets", "bitsandbytes", "llama_cpp",
                "langdetect", "nltk"):
        table.add_row(mod, probe(mod))

    try:
        import torch

        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                table.add_row(
                    f"cuda:{i}",
                    f"{props.name} · {props.total_memory / 1024**3:.1f} GB · "
                    f"sm_{props.major}{props.minor} · bf16={'yes' if torch.cuda.is_bf16_supported() else 'no'}",
                )
        else:
            table.add_row("cuda", "[dim]unavailable (CPU only)[/]")
    except ImportError:
        table.add_row("cuda", "[dim]torch not installed[/]")

    console.print(table)


@app.command("config")
def show_config(
    config: str = typer.Option(None, "--config", "-c"),
    backend: str = typer.Option(None, "--backend", "-b"),
    model: str = typer.Option(None, "--model", "-m"),
) -> None:
    """Print the fully merged configuration and exit.

    Run this whenever Kumaru is not doing what you expect. It shows the result
    of all four config layers, so 'which setting actually won' stops being a
    guess.
    """
    cfg = _load(config, backend=backend, model=model)
    # Escaped: system prompts and stop sequences can hold brackets, such as "[/INST]".
    console.print(Panel(escape(cfg.dump_yaml()), title="resolved configuration", border_style="cyan"))


@app.command()
def learn(
    topic: str = typer.Argument(..., help="Language or platform to learn."),
    source: Path = typer.Option(..., "--source", help="Reviewed UTF-8 reference file."),
    version: str = typer.Option(..., "--version", help="Language/platform version."),
    output: Path = typer.Option(..., "--output", "-o", help="New knowledge pack JSON file."),
) -> None:
    """Learn from local references without downloads, execution, or fine-tuning."""
    from pydantic import ValidationError

    from kumaru.agent.knowledge import KnowledgePack, describe_errors, read_local_text, save_pack
    from kumaru.core.errors import ConfigError

    try:
        pack = KnowledgePack(
            topic=topic, version=version, source=source.name, content=read_local_text(source)
        )
    except ValidationError as exc:
        raise ConfigError(
            f"Cannot create knowledge pack {output}: {describe_errors(exc)}",
            hint="Topic, version and the --source text must be nonempty and within "
            "their length limits.",
        ) from exc
    save_pack(pack, output)
    console.print(f"Saved reference pack: {output}", markup=False)
    console.print("Add its absolute path to agent.knowledge_files and restart chat/server.")


@app.command()
def train(
    config: str = typer.Argument(..., help="Training config YAML (see configs/train/)."),
    resume: str = typer.Option(None, "--resume", help="Checkpoint directory to resume from."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Validate config and data, then stop."),
) -> None:
    """Fine-tune (LoRA) or pretrain a model. See docs/TRAINING.md."""
    from kumaru.train.runner import run_training

    run_training(Path(config), resume=resume, dry_run=dry_run)


@app.command("eval")
def evaluate(
    config: str = typer.Option(None, "--config", "-c"),
    backend: str = typer.Option(None, "--backend", "-b"),
    model: str = typer.Option(None, "--model", "-m"),
    adapter: str = typer.Option(None, "--adapter"),
    suite: str = typer.Option("smoke", "--suite", "-s", help="smoke | speed | jsonl file path"),
    output: str = typer.Option(None, "--output", "-o", help="Write results to this JSON file."),
) -> None:
    """Measure latency, throughput, and answer quality."""
    import asyncio

    from kumaru.eval.harness import run_eval

    cfg = _load(config, backend=backend, model=model, adapter=adapter)
    asyncio.run(run_eval(cfg, suite=suite, output=output))


# ---------------------------------------------------------------------------
# kumaru bench ...
# ---------------------------------------------------------------------------

bench_app = typer.Typer(
    help="Score models on public benchmarks; compare runs with significance tests.",
    no_args_is_help=True,
)
app.add_typer(bench_app, name="bench")


@bench_app.command("list")
def bench_list() -> None:
    """Show the available benchmarks."""
    from kumaru.eval.benchmarks.runner import list_benchmarks

    table = Table(border_style="dim")
    table.add_column("name", style="cyan", no_wrap=True)
    table.add_column("what it measures")
    table.add_column("source", style="dim")
    table.add_column("runs code", justify="center")
    for row in list_benchmarks():
        table.add_row(row["name"], row["description"], row["source"],
                      "[yellow]yes[/]" if row["code_exec"] else "")
    console.print(table)
    console.print("[dim]Any .jsonl file in the golden-set format also works as a benchmark "
                  "(see data/bench/golden.jsonl).[/]")


@bench_app.command("run")
def bench_run(
    benchmarks: list[str] = typer.Argument(..., help="Names from `kumaru bench list`, or golden-set .jsonl paths."),
    config: str = typer.Option(None, "--config", "-c", help="YAML file or profile name."),
    backend: str = typer.Option(None, "--backend", "-b"),
    model: str = typer.Option(None, "--model", "-m"),
    adapter: str = typer.Option(None, "--adapter"),
    context: int = typer.Option(None, "--context", min=512, help=CONTEXT_HELP),
    limit: int = typer.Option(100, "--limit", "-n", min=1, help="Questions per benchmark (a fixed, stratified sample)."),
    full: bool = typer.Option(False, "--full", help="Every question; overrides --limit."),
    seed: int = typer.Option(1234, "--seed", help="Picks the questions and each question's sampling seed."),
    think: bool = typer.Option(None, "--think/--no-think", help="Force reasoning on or off (default: profile)."),
    thinking_budget: int = typer.Option(None, "--thinking-budget", min=0, help="Max thinking tokens per question."),
    max_tokens: int = typer.Option(None, "--max-tokens", min=1, max=131072, help="Output cap per question (default: 8192; thinking adds --thinking-budget, or uses the profile's max_tokens without one)."),
    temperature: float = typer.Option(None, "--temperature", "-t", min=0.0, max=2.0, help="0 = greedy (default: the profile's preset)."),
    concurrency: int = typer.Option(None, "--concurrency", "-j", min=1, help="Questions in flight (default: what the backend can batch)."),
    label: str = typer.Option(None, "--label", help="Run name (default: <profile or model>-<thinking mode>, e.g. qwen36-think4096)."),
    out: str = typer.Option(None, "--out", "-o", help="This run's own folder (default runs/bench/<timestamp>-<label>). Point at an earlier run to resume it."),
    allow_code_exec: bool = typer.Option(False, "--allow-code-exec", help="Allow benchmarks that execute model-written code (humaneval)."),
    use_agent_prompt: bool = typer.Option(False, "--use-agent-prompt", help="Prepend Kumaru's system prompt, as chat does."),
) -> None:
    """Run benchmarks; every answer is saved under runs/bench/."""
    import asyncio

    from kumaru.eval.benchmarks.runner import BenchSettings, render_summaries, run_benchmarks

    cfg = _load(config, backend=backend, model=model, adapter=adapter, context=context)
    settings = BenchSettings(
        limit=None if full else limit,
        seed=seed,
        concurrency=concurrency,
        thinking=think,
        thinking_budget=thinking_budget,
        max_tokens=max_tokens,
        temperature=temperature,
        use_agent_prompt=use_agent_prompt,
        allow_code_exec=allow_code_exec,
        label=label,
        out_dir=Path(out) if out else None,
    )
    out_dir, summaries = asyncio.run(
        run_benchmarks(cfg, benchmarks, settings, profile=config, console=console)
    )
    console.print()
    render_summaries(console, summaries)
    console.print(f"\n[dim]saved to[/] {out_dir}")
    console.print(f"[dim]details:[/] kumaru bench report {out_dir}")
    console.print(f"[dim]compare:[/] kumaru bench compare <baseline-run> {out_dir}")


@bench_app.command("compare")
def bench_compare(
    baseline: str = typer.Argument(..., help="Results directory of the baseline run (A)."),
    candidate: str = typer.Argument(..., help="Results directory of the candidate run (B)."),
    examples: int = typer.Option(0, "--examples", "-e", min=0, help="Also show up to N questions per benchmark that the runs answered differently."),
) -> None:
    """Did B really beat A? Paired differences with confidence intervals and p-values."""
    from kumaru.eval.benchmarks.runner import compare_runs, render_comparisons, render_flips

    render_comparisons(console, compare_runs(Path(baseline), Path(candidate)))
    if examples:
        render_flips(console, Path(baseline), Path(candidate), examples)


@bench_app.command("report")
def bench_report(
    run: str = typer.Argument(..., help="Results directory from `kumaru bench run`."),
) -> None:
    """Show a saved run in detail: per-group accuracy, answer extraction, extra metrics."""
    from kumaru.eval.benchmarks.runner import load_summaries, render_summaries

    render_summaries(console, load_summaries(Path(run)), details=True)


# ---------------------------------------------------------------------------
# kumaru distill ...
# ---------------------------------------------------------------------------

distill_app = typer.Typer(
    help="Build training data from a teacher model's graded answers (knowledge distillation).",
    no_args_is_help=True,
)
app.add_typer(distill_app, name="distill")

DISTILL_OUT_HELP = "New folder for train.jsonl, val.jsonl and manifest.json."
VAL_FRACTION_HELP = "Share of questions held out for validation loss."
SPLIT_SEED_HELP = "Picks the held-out questions; keep it equal across datasets you compare."
ARITHMETIC_HELP = 'Drop answers with a wrong step such as "3 x 4 = 13".'
EVAL_SET_HELP = "A benchmark the student will be scored on (repeat for several); its questions never become training data."
DECONTAMINATE_HELP = "Accept only training-split sources (e.g. gsm8k_train) and drop near-copies of evaluation questions. Turn off only for smoke tests."
FORCE_HELP = "Replace a dataset already in --out."


@distill_app.command("build")
def distill_build(
    runs: list[str] = typer.Argument(..., help="Teacher runs from `kumaru bench run` (a run folder or one benchmark's folder)."),
    out: str = typer.Option(..., "--out", "-o", help=DISTILL_OUT_HELP),
    max_tokens: int = typer.Option(1024, "--max-tokens", min=1, help="Drop answers longer than this many tokens; keep it below the training max_seq_length."),
    per_question: int = typer.Option(4, "--per-question", min=1, help="Distinct answers kept per question."),
    val_fraction: float = typer.Option(0.05, "--val-fraction", min=0.0, max=0.5, help=VAL_FRACTION_HELP),
    seed: int = typer.Option(42, "--seed", help=SPLIT_SEED_HELP),
    check_arithmetic: bool = typer.Option(True, "--check-arithmetic/--no-check-arithmetic", help=ARITHMETIC_HELP),
    eval_set: list[str] = typer.Option(["gsm8k"], "--eval-set", help=EVAL_SET_HELP),
    decontaminate: bool = typer.Option(True, "--decontaminate/--no-decontaminate", help=DECONTAMINATE_HELP),
    allow_thinking: bool = typer.Option(False, "--allow-thinking", help="Accept thinking runs; the student sees the answers, not the reasoning."),
    force: bool = typer.Option(False, "--force", help=FORCE_HELP),
) -> None:
    """Teacher answers -> SFT data, keeping only correct, complete, clean answers."""
    from kumaru.train.distill import DistillSettings, build_from_runs, render

    settings = DistillSettings(
        max_tokens=max_tokens,
        per_question=per_question,
        val_fraction=val_fraction,
        seed=seed,
        check_arithmetic=check_arithmetic,
        eval_sets=eval_set,
        decontaminate=decontaminate,
        allow_thinking=allow_thinking,
    )
    render(console, build_from_runs(runs, out, settings, force=force), Path(out))


@distill_app.command("gold")
def distill_gold(
    out: str = typer.Option(..., "--out", "-o", help=DISTILL_OUT_HELP),
    match: str = typer.Option(None, "--match", help="A teacher dataset folder: keep only its questions, each on the same side of the split (then --val-fraction and --seed are unused)."),
    val_fraction: float = typer.Option(0.05, "--val-fraction", min=0.0, max=0.5, help=VAL_FRACTION_HELP),
    seed: int = typer.Option(42, "--seed", help=SPLIT_SEED_HELP),
    check_arithmetic: bool = typer.Option(True, "--check-arithmetic/--no-check-arithmetic", help=ARITHMETIC_HELP),
    eval_set: list[str] = typer.Option(["gsm8k"], "--eval-set", help=EVAL_SET_HELP),
    decontaminate: bool = typer.Option(True, "--decontaminate/--no-decontaminate", help=DECONTAMINATE_HELP),
    force: bool = typer.Option(False, "--force", help=FORCE_HELP),
) -> None:
    """The control arm: GSM8K's human-written solutions, in the teacher's format."""
    from kumaru.train.distill import DistillSettings, build_gold, render

    settings = DistillSettings(
        val_fraction=val_fraction,
        seed=seed,
        check_arithmetic=check_arithmetic,
        eval_sets=eval_set,
        decontaminate=decontaminate,
    )
    render(console, build_gold(out, settings, match=match, force=force), Path(out))


def main() -> None:
    """Console-script entry point with uniform error handling."""
    from kumaru.core.logging import ensure_utf8_streams

    ensure_utf8_streams()
    try:
        app()
    except KumaruError as exc:
        # Messages carry literal brackets, such as pip extras (".[gguf]") and
        # pydantic details ("[type=missing]"); rich would read them as markup.
        console.print(f"[red]error:[/] {escape(exc.message)}")
        if exc.hint:
            console.print(f"[yellow]hint:[/] {escape(exc.hint)}")
        raise SystemExit(1) from exc
    except KeyboardInterrupt:
        console.print("\n[dim]interrupted[/]")
        raise SystemExit(130) from None


if __name__ == "__main__":
    main()

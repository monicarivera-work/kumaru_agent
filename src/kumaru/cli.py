"""Command-line entry point: `kumaru <command>`.

One binary, five verbs::

    kumaru serve     # API + web UI
    kumaru chat      # terminal REPL, no server needed
    kumaru info      # what is installed, what hardware is visible
    kumaru config    # print the fully resolved configuration
    kumaru train     # fine-tune or pretrain
    kumaru eval      # measure speed and quality

Two rules keep the CLI fast and debuggable:

* **Heavy imports happen inside command bodies**, never at module scope. `kumaru
  --help` must not import torch; if it did, every invocation would cost several
  seconds.
* **Every command accepts `--config`** and the same override flags, so anything
  you can express in YAML you can also express on the command line for a
  one-off experiment.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
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
        "host": ("server", "host"),
        "port": ("server", "port"),
        "temperature": ("generation", "temperature"),
        "max_tokens": ("generation", "max_tokens"),
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
    backend: str = typer.Option(None, "--backend", "-b", help="echo | transformers | llamacpp | openai_compat"),
    model: str = typer.Option(None, "--model", "-m", help="Model id or path."),
    device: str = typer.Option(None, "--device", help="auto | cuda | cpu | mps"),
    adapter: str = typer.Option(None, "--adapter", help="Path to a trained LoRA adapter."),
    host: str = typer.Option(None, "--host"),
    port: int = typer.Option(None, "--port", "-p"),
    log_level: str = typer.Option(None, "--log-level"),
) -> None:
    """Start the HTTP API and the web UI."""
    cfg = _load(config, backend=backend, model=model, device=device, adapter=adapter,
                host=host, port=port, log_level=log_level)

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
    temperature: float = typer.Option(None, "--temperature", "-t"),
    max_tokens: int = typer.Option(None, "--max-tokens"),
    prompt: str = typer.Option(None, "--prompt", "-q", help="Answer one question and exit."),
) -> None:
    """Chat in the terminal, with no server in between."""
    import asyncio

    cfg = _load(config, backend=backend, model=model, device=device, adapter=adapter,
                temperature=temperature, max_tokens=max_tokens)
    asyncio.run(_chat_loop(cfg, prompt))


async def _chat_loop(cfg: Any, single_prompt: str | None) -> None:
    from kumaru.agent.chat import ChatAgent
    from kumaru.core.registry import BACKENDS
    from kumaru.core.types import Message

    with console.status(f"[cyan]loading {cfg.backend.model}…", spinner="dots"):
        backend = BACKENDS.create(cfg.backend.name, cfg.backend)
        await backend.load()

    agent = ChatAgent(backend, cfg)
    info = backend.info()
    console.print(f"[dim]{info.model_id} on {info.device} ({info.dtype}), "
                  f"{info.context_length:,} token context[/]")

    history: list[Message] = []

    async def ask(question: str) -> None:
        history.append(Message.user(question))
        console.print()
        parts: list[str] = []
        stats = None
        # Print tokens as they arrive: the point of streaming is that you start
        # reading before generation finishes.
        async for chunk in agent.stream(history):
            if chunk.delta:
                parts.append(chunk.delta)
                console.print(chunk.delta, end="", markup=False, highlight=False)
            if chunk.usage:
                stats = chunk.usage
        console.print("\n")
        if stats:
            console.print(f"[dim]{stats.prompt_tokens} prompt + {stats.completion_tokens} completion tokens[/]")
        history.append(Message.assistant("".join(parts)))

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
    from kumaru.core.registry import BACKENDS, TOOLS

    table = Table(title=f"Kumaru v{__version__}", show_header=False, border_style="dim")
    table.add_column(style="cyan", no_wrap=True)
    table.add_column()

    table.add_row("python", sys.version.split()[0])
    table.add_row("backends", ", ".join(BACKENDS.names()))
    table.add_row("tools", ", ".join(TOOLS.names()) or "(none)")
    table.add_row("profiles", ", ".join(list_profiles()) or "(none)")

    def probe(module: str) -> str:
        try:
            mod = __import__(module)
            return getattr(mod, "__version__", "installed")
        except ImportError:
            return "[dim]not installed[/]"

    for mod in ("torch", "transformers", "peft", "trl", "datasets", "bitsandbytes", "llama_cpp"):
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
    console.print(Panel(cfg.dump_yaml(), title="resolved configuration", border_style="cyan"))


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


def main() -> None:
    """Console-script entry point with uniform error handling."""
    from kumaru.core.logging import ensure_utf8_streams

    ensure_utf8_streams()
    try:
        app()
    except KumaruError as exc:
        console.print(f"[red]error:[/] {exc.message}")
        if exc.hint:
            console.print(f"[yellow]hint:[/] {exc.hint}")
        raise SystemExit(1) from exc
    except KeyboardInterrupt:
        console.print("\n[dim]interrupted[/]")
        raise SystemExit(130) from None


if __name__ == "__main__":
    main()

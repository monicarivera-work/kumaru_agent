# `src\kumaru\cli.py`

> Typer-based command-line entry point for serving, chatting, inspecting config, training, and evaluating Kumaru.

**Read this when:** you add a CLI flag, diagnose a failed `kumaru` command, need to keep `kumaru --help` fast, or map a command-line override to configuration.

---

## What it does
This file defines the `kumaru` command and its subcommands. It keeps heavyweight imports inside command bodies so help and lightweight commands do not import model libraries. It also centralises CLI error handling so deliberate Kumaru errors print a concise message and optional hint.

## Why it exists
Operators need one consistent binary for local chat, server startup, config inspection, training, and evaluation. Without this layer, each subsystem would parse its own flags, configuration overrides would drift, and startup diagnostics would become inconsistent.

## Mental model
The CLI is a thin adapter from command-line flags to subsystem calls. Configuration always flows through `_load()`, which calls `load_config()` with nested overrides built from flat flags. Heavy modules are imported only after a command has been selected.

## Public API
| Name | Signature | What it's for |
|---|---|---|
| `app` | `typer.Typer` | The Typer application registered as the `kumaru` CLI. |
| `console` | `rich.console.Console` | Shared Rich console used for CLI output. |
| `serve` | `serve(config=None, backend=None, model=None, device=None, adapter=None, host=None, port=None, log_level=None) -> None` | Start HTTP API and web UI. |
| `chat` | `chat(config=None, backend=None, model=None, device=None, adapter=None, temperature=None, max_tokens=None, prompt=None) -> None` | Run terminal chat, optionally answering one prompt and exiting. |
| `info` | `info() -> None` | Print Python version, registered components, profiles, optional packages, and CUDA visibility. |
| `show_config` | `show_config(config=None, backend=None, model=None) -> None` | Print the fully resolved configuration. Registered as command `config`. |
| `train` | `train(config: str, resume=None, dry_run=False) -> None` | Run training from a training YAML file. |
| `evaluate` | `evaluate(config=None, backend=None, model=None, adapter=None, suite="smoke", output=None) -> None` | Run evaluation. Registered as command `eval`. |
| `main` | `main() -> None` | Console-script entry point with `KumaruError` and interrupt handling. |

Private helpers `_build_overrides()`, `_load()`, and `_chat_loop()` are implementation details but are important when changing CLI flags.

## How to use it

Start the server with a profile and one-off overrides:

```shell
kumaru serve --config local-gpu --host 127.0.0.1 --port 8080 --backend transformers
```

Ask one terminal question without starting the HTTP server:

```shell
kumaru chat --backend echo --prompt "Say hello in one sentence."
```

Inspect the exact configuration that won after defaults, YAML, environment, and flags:

```shell
kumaru config --config local-gpu --model Qwen/Qwen2.5-1.5B-Instruct
```

Run the built-in smoke evaluation and write JSON results:

```shell
kumaru eval --config local-gpu --suite smoke --output results.json
```

Call the entry point from Python:

```python
from kumaru.cli import main

main()
```

## How to extend it

Add new flags in two places when they override configuration:

1. Add the Typer option to the command function.
2. Add the flat flag name to `_build_overrides()` with the target config path.

For example, to expose `backend.load_in_4bit` on `serve`, add an option and pass it through:

```python
@app.command()
def serve(
    config: str = typer.Option(None, "--config", "-c"),
    load_in_4bit: bool = typer.Option(None, "--load-in-4bit"),
) -> None:
    cfg = _load(config, load_in_4bit=load_in_4bit)
```

`_build_overrides()` already maps `load_in_4bit` to `("backend", "load_in_4bit")`, so `load_config()` receives:

```python
{"backend": {"load_in_4bit": True}}
```

When adding a new command, keep imports for its subsystem inside the command body. Do not import training, server, torch, or backend modules at file import time.

## Failure modes
| Symptom | Likely cause | Fix |
|---|---|---|
| `kumaru --help` takes several seconds | A heavy dependency was imported at module scope. | Move the import into the relevant command function. |
| A CLI flag appears to be ignored | The flag is not passed to `_load()` or is missing from `_build_overrides()`. | Add the parameter to both the command call and the mapping. |
| `error: Invalid configuration: ...` | Pydantic rejected the merged config, often due to a typo or wrong type. | Run `kumaru config ...` and correct the YAML, environment variable, or flag. |
| `Unknown backend: 'name'` | The selected backend is not registered in `BACKENDS`. | Use `kumaru info` to list names, install/register the backend, or fix the config. |
| Terminal chat exits with code `130` | The user interrupted with Ctrl+C. | No fix required; this is intentional interrupt handling. |
| Training command cannot find config | `train` takes a required path argument, not `--config`. | Run `kumaru train configs\train\your-file.yaml`. |

## Related files
- `src\kumaru\core\config.py` — loads profiles, environment variables, and CLI overrides.
- `src\kumaru\core\errors.py` — defines `KumaruError`, caught by `main()`.
- `src\kumaru\core\registry.py` — resolves backend names used by `chat`.
- `src\kumaru\agent\chat.py` — powers the terminal chat loop.
- `src\kumaru\server\app.py` — started by `serve()`.
- `src\kumaru\train\runner.py` — called by `train()`.
- `src\kumaru\eval\harness.py` — called by `evaluate()`.

# `src/kumaru/cli.py`

> Command-line interface for serving, chatting, asking one question, printing config, and listing backends.

**Read this when:** operating Kumaru from a shell or adding a new CLI command.

---

## What it does
It builds the `kumaru` argument parser, loads configuration, applies CLI overrides, sets up logging, and dispatches subcommands. It is the only layer that exits with numeric status codes or configures process logging.

## Why it exists
Library code should raise exceptions so notebooks, tests, and servers can handle failures. The CLI converts those exceptions into human-readable stderr and exit codes.

## Mental model
Common flags can appear before or after the subcommand. CLI flags override env vars and YAML because they are the most explicit operator input.

## Public API
| Name | Signature | What it's for |
|---|---|---|
| `EXIT_OK` | `int` | Successful command status. |
| `EXIT_ERROR` | `int` | Failed command status. |
| `build_parser` | `build_parser() -> argparse.ArgumentParser` | Construct the command parser. |
| `main` | `main(argv: Sequence[str] | None = None) -> int` | Parse args and run a command. |

## How to use it
```bash
kumaru serve -c configs/default.yaml --host 127.0.0.1 --port 8000
```

```bash
kumaru ask --backend echo "What backend is active?"
```

```python
from kumaru.cli import main
exit_code = main(["backends"])
print(exit_code)
```

```python
from kumaru.cli import build_parser
args = build_parser().parse_args(["ask", "hello"])
print(args.command)
```

## How to extend it
Add subcommands in `build_parser()`, then dispatch them in `main()`. Keep `setup_logging` and exception-to-exit-code handling in this file; deeper code should raise `KumaruError` subclasses instead.

## Gotchas
| Symptom | Likely cause | Fix |
|---|---|---|
| `kumaru ask` exits 1 | Config validation or backend loading raised `KumaruError`. | Read stderr hint; run `kumaru config -c <file>` to inspect effective settings. |
| Flag ignored | The flag is not copied in `_config_from_args`. | Add an override assignment before `config.validate()`. |
| Backend list omits a custom backend | Runtime registration has not happened in this process. | Register it before calling `available_backends()` or add it to `BACKEND_PATHS`. |

## Related files
- [Package facade](./__init__.md)
- [Agent](./agent/chat.md)
- [Config](./core/config.md)
- [Backends](./backends/__init__.md)

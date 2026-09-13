"""Command line entry point: ``kumaru <command>``.

Commands
--------
``kumaru serve``    start the HTTP server + browser UI (the normal way to use it)
``kumaru chat``     a terminal REPL, useful when debugging a backend
``kumaru ask``      one-shot question, for scripts and pipes
``kumaru config``   print the effective configuration after env overrides
``kumaru backends`` list selectable backend names

This module is the *only* place allowed to call ``setup_logging`` or
``sys.exit`` - everything below it raises exceptions instead, which is what
makes the library usable from a notebook or another app.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

from kumaru import __version__
from kumaru.agent.chat import ChatAgent
from kumaru.backends import available_backends
from kumaru.core.config import KumaruConfig, load_config
from kumaru.core.errors import KumaruError
from kumaru.core.logging import get_logger, setup_logging

log = get_logger("cli")

EXIT_OK = 0
EXIT_ERROR = 1


def build_parser() -> argparse.ArgumentParser:
    # Common flags live on a parent parser so they are accepted both before
    # and after the subcommand: `kumaru -c x.yaml serve` and
    # `kumaru serve -c x.yaml` both work, which is what people actually type.
    #
    # argparse.SUPPRESS is essential here: without it the subparser would write
    # its own `None` default over a value already parsed before the subcommand,
    # so `kumaru --model m config` would silently lose the model.
    common = argparse.ArgumentParser(add_help=False, argument_default=argparse.SUPPRESS)
    common.add_argument(
        "-c", "--config", help="path to a YAML config file (see configs/)"
    )
    common.add_argument("--backend", help="override backend.name")
    common.add_argument("--model", help="override backend.model")
    common.add_argument("--log-level", help="DEBUG, INFO, WARNING, ERROR")

    parser = argparse.ArgumentParser(
        prog="kumaru",
        description="Kumaru - a local-first LLM chat agent",
        parents=[common],
    )
    parser.add_argument("--version", action="version", version=f"kumaru {__version__}")

    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser(
        "serve", help="run the web UI and API", parents=[common]
    )
    serve.add_argument("--host", help="override server.host")
    serve.add_argument("--port", type=int, help="override server.port")

    sub.add_parser("chat", help="interactive terminal chat", parents=[common])

    ask = sub.add_parser(
        "ask", help="ask a single question and print the answer", parents=[common]
    )
    ask.add_argument("question", nargs="+")
    ask.add_argument(
        "--no-stream", action="store_true", help="wait for the full answer"
    )

    sub.add_parser(
        "config", help="print the effective configuration as JSON", parents=[common]
    )
    sub.add_parser("backends", help="list available backends", parents=[common])
    return parser


def _config_from_args(args: argparse.Namespace) -> KumaruConfig:
    # Flags use argparse.SUPPRESS, so an unset flag is an *absent* attribute.
    config = load_config(getattr(args, "config", None))
    # CLI flags sit above env vars: they are the most explicit signal.
    if getattr(args, "backend", None):
        config.backend.name = args.backend
    if getattr(args, "model", None):
        config.backend.model = args.model
    if getattr(args, "log_level", None):
        config.log_level = args.log_level
    if getattr(args, "host", None):
        config.server.host = args.host
    if getattr(args, "port", None):
        config.server.port = args.port
    config.validate()
    return config


def _cmd_serve(config: KumaruConfig) -> int:
    import uvicorn

    from kumaru.server.app import create_app

    url = f"http://{config.server.host}:{config.server.port}"
    print(f"Kumaru UI -> {url}   (backend: {config.backend.name})", file=sys.stderr)
    uvicorn.run(
        create_app(config),
        host=config.server.host,
        port=config.server.port,
        log_level=config.log_level.lower(),
    )
    return EXIT_OK


def _stream_to_stdout(agent: ChatAgent, question: str, session_id: str) -> None:
    for chunk in agent.stream(question, session_id=session_id):
        if chunk.delta:
            print(chunk.delta, end="", flush=True)
    print()


def _cmd_chat(config: KumaruConfig) -> int:
    print(
        f"Kumaru {__version__} ({config.backend.name}). "
        "Type /exit to quit, /reset to clear history.",
        file=sys.stderr,
    )
    with ChatAgent(config) as agent:
        while True:
            try:
                question = input("\nyou> ").strip()
            except (EOFError, KeyboardInterrupt):
                print(file=sys.stderr)
                return EXIT_OK
            if not question:
                continue
            if question in {"/exit", "/quit"}:
                return EXIT_OK
            if question == "/reset":
                agent.reset("cli")
                print("history cleared", file=sys.stderr)
                continue
            print("\nkumaru> ", end="", flush=True)
            try:
                _stream_to_stdout(agent, question, "cli")
            except KumaruError as exc:
                print(f"\n[{type(exc).__name__}] {exc.message}", file=sys.stderr)
                if exc.hint:
                    print(f"hint: {exc.hint}", file=sys.stderr)


def _cmd_ask(config: KumaruConfig, args: argparse.Namespace) -> int:
    question = " ".join(args.question)
    with ChatAgent(config) as agent:
        if args.no_stream:
            print(agent.ask(question, session_id="cli"))
        else:
            _stream_to_stdout(agent, question, "cli")
    return EXIT_OK


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "backends":
        print("\n".join(available_backends()))
        return EXIT_OK

    try:
        config = _config_from_args(args)
    except KumaruError as exc:
        print(f"[{type(exc).__name__}] {exc.message}", file=sys.stderr)
        if exc.hint:
            print(f"hint: {exc.hint}", file=sys.stderr)
        return EXIT_ERROR

    setup_logging(config.log_level, json_format=config.log_json)

    try:
        if args.command == "config":
            print(json.dumps(config.to_dict(), indent=2, default=str))
            return EXIT_OK
        if args.command == "serve":
            return _cmd_serve(config)
        if args.command == "chat":
            return _cmd_chat(config)
        if args.command == "ask":
            return _cmd_ask(config, args)
    except KumaruError as exc:
        log.debug("command failed", exc_info=True)
        print(f"[{type(exc).__name__}] {exc.message}", file=sys.stderr)
        if exc.hint:
            print(f"hint: {exc.hint}", file=sys.stderr)
        return EXIT_ERROR
    except KeyboardInterrupt:  # pragma: no cover - interactive only
        return EXIT_OK

    return EXIT_ERROR  # pragma: no cover - argparse rejects unknown commands


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

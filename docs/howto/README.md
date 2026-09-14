# Kumaru how-to index

Operator-oriented how-to docs for source files and shipped configs.

## Package and CLI
- [src/kumaru/__init__.py](./src/kumaru/__init__.md) - Stable package-level imports and version.
- [src/kumaru/cli.py](./src/kumaru/cli.md) - Command-line entry point and subcommand dispatch.

## Core layer
- [src/kumaru/core/__init__.py](./src/kumaru/core/__init__.md) - Core facade and dependency boundary.
- [src/kumaru/core/types.py](./src/kumaru/core/types.md) - Shared chat value types.
- [src/kumaru/core/errors.py](./src/kumaru/core/errors.md) - Deliberate exception hierarchy.
- [src/kumaru/core/config.py](./src/kumaru/core/config.md) - Typed YAML/env configuration.
- [src/kumaru/core/registry.py](./src/kumaru/core/registry.md) - Generic plugin registry.
- [src/kumaru/core/logging.py](./src/kumaru/core/logging.md) - Logging setup and logger naming.

## Backend layer
- [src/kumaru/backends/__init__.py](./src/kumaru/backends/__init__.md) - Lazy backend registry and loader.
- [src/kumaru/backends/base.py](./src/kumaru/backends/base.md) - Backend interface contract.
- [src/kumaru/backends/echo.py](./src/kumaru/backends/echo.md) - Deterministic echo backend.
- [src/kumaru/backends/openai_compat.py](./src/kumaru/backends/openai_compat.md) - OpenAI-compatible HTTP backend.
- [src/kumaru/backends/transformers_backend.py](./src/kumaru/backends/transformers_backend.md) - In-process Hugging Face backend.
- [src/kumaru/backends/llamacpp_backend.py](./src/kumaru/backends/llamacpp_backend.md) - In-process GGUF llama.cpp backend.

## Agent layer
- [src/kumaru/agent/__init__.py](./src/kumaru/agent/__init__.md) - Agent facade.
- [src/kumaru/agent/prompt.py](./src/kumaru/agent/prompt.md) - Prompt rendering and message assembly.
- [src/kumaru/agent/memory.py](./src/kumaru/agent/memory.md) - Bounded session memory.
- [src/kumaru/agent/chat.py](./src/kumaru/agent/chat.md) - Chat orchestration loop.

## Server layer
- [src/kumaru/server/__init__.py](./src/kumaru/server/__init__.md) - Server facade.
- [src/kumaru/server/schemas.py](./src/kumaru/server/schemas.md) - HTTP wire models.
- [src/kumaru/server/app.py](./src/kumaru/server/app.md) - FastAPI app factory.
- [src/kumaru/server/routes_chat.py](./src/kumaru/server/routes_chat.md) - Chat, stream, history routes.

## UI layer
- [ui/index.html](./ui/index.html.md) - Static browser UI shell.
- [ui/styles.css](./ui/styles.css.md) - Theme, layout, and message styles.
- [ui/app.js](./ui/app.js.md) - Browser chat client and SSE handling.

## Config profiles
- [configs/default.yaml](./configs/default.yaml.md) - Documented built-in defaults.
- [configs/cpu-only.yaml](./configs/cpu-only.yaml.md) - CPU-only GGUF inference.
- [configs/low-vram.yaml](./configs/low-vram.yaml.md) - Low-VRAM llama.cpp offload profile.
- [configs/local-gpu.yaml](./configs/local-gpu.yaml.md) - Transformers GPU profile.
- [configs/openai-compat.yaml](./configs/openai-compat.yaml.md) - OpenAI-compatible server profile.
- [configs/test.yaml](./configs/test.yaml.md) - Deterministic test/CI profile.

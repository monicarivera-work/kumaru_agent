# `src\kumaru\backends\__init__.py`

> Exports the two lightweight backend primitives, `Backend` and `iterate_in_thread`. Concrete engines are never imported here; you reach them by name through `kumaru.core.registry.BACKENDS`.

**Read this when:**

- you are writing a backend and want the supported import path;
- `from kumaru.backends import LlamaServerBackend` (or any other concrete class) fails;
- you need the backend names and aliases, or must decide where a new backend gets registered;
- a light command such as `kumaru config` or `kumaru chat -c test` suddenly loads torch or starts slowly.

---

## What it does

The file is one import plus `__all__`:

```python
from kumaru.backends.base import Backend, iterate_in_thread

__all__ = ["Backend", "iterate_in_thread"]
```

Its docstring states the goal of the whole package: every engine sits behind the same `Backend` contract, so the agent, server and UI never learn which one is loaded, and moving to bigger hardware is a YAML edit. In v0.2 there are five engines (see [Backend names](#backend-names)).

Importing `kumaru.backends` loads `kumaru.backends.base` and the `kumaru.core` modules it uses, nothing else.

## Why it exists

Python runs a package's `__init__.py` before any of its submodules. Every concrete backend lives in this package, so resolving `echo` (`import kumaru.backends.echo`) runs this file first, and whatever it imports is paid by every backend, every CLI command that creates one, and every test.

Kumaru keeps optional engines off that path in two layers, and this file must not undo them:

1. **Lazy registration.** `src\kumaru\core\registry.py` maps each name to a `"module:Class"` string with `register_lazy`. The module is imported on the first `BACKENDS.get()`/`create()` for that name; listing names imports nothing.
2. **Deferred engine imports.** No backend module imports its engine at module level. `torch`, `transformers`, `peft`, `llama_cpp`, `huggingface_hub` and even `httpx` are imported inside `_load()` or the method that uses them. A missing optional dependency therefore surfaces as `DependencyMissingError` (with an install hint) when you `load()` that backend, not as an `ImportError` when something merely lists, configures or imports it.

Re-exporting a concrete class here would not import torch today (layer 2 holds), but it would make every backend import every other backend module, and the first engine that does import something heavy at module level would slow down all of them.

## Mental model

1. **`kumaru.backends` is for backend authors.** It gives you `Backend` to subclass and `iterate_in_thread` to drive a blocking engine.
2. **`BACKENDS` is for everyone else.** The CLI (`src\kumaru\cli.py`), the server (`src\kumaru\server\app.py`), the eval harness and the benchmark runner all do `BACKENDS.create(cfg.backend.name, cfg.backend)`; none imports a concrete class.
3. **A backend name is a string, validated late.** `BackendConfig.name` is a plain `str`. Nothing checks it until `BACKENDS.create()` resolves it, so `kumaru config -b nonexistent` prints happily, `kumaru chat -b nonexistent` exits 1 with `Unknown backend`, and the server stores the error and answers `/healthz` with 503.
4. **Aliases are pure synonyms.** `ollama`, `vllm` and `remote` all create the same `OpenAICompatBackend`, whose `name` is `"openai_compat"`; no alias carries its own defaults (you still set `base_url`). Aliases resolve everywhere but are hidden from `BACKENDS.names()`, `kumaru info` and `/v1/info`'s `available_backends`.
5. **`iterate_in_thread` moves work off the event loop, and stops it when the reader leaves.** It is how `transformers` and `llamacpp` stream. If the consumer stops early, the worker stops at the producer's next item and closes the producer on its own thread. The producer's `finally` therefore runs there, and that is where a backend halts the engine or releases its lock ([base.md](base.md#wrap-a-blocking-producer)).

## Public API

| Name | Signature | What it's for |
|---|---|---|
| `Backend` | `class Backend(config: BackendConfig)` (abstract) | The inference contract, re-exported from `kumaru.backends.base`. Subclasses set the class attributes `name` and, only if `_load()` attaches `adapter_path`, `supports_adapters = True`; they implement the abstract `_load`, `info` and `stream`; `load`, `generate`, `aclose`, `is_loaded`, `count_tokens`, `context_length` and `apply_stop_sequences` are inherited. Full contract: [base.md](base.md). |
| `iterate_in_thread` | `async def iterate_in_thread(make_iterator: Any, *, queue_size: int = 256) -> AsyncGenerator[Any, None]` | Calls the zero-argument factory `make_iterator` on a daemon thread named `kumaru-generate` and yields its items to the event loop through a bounded queue (backpressure). If the consumer stops early, the worker stops at the next item (within 0.1 s when the queue is full) and closes the producer on the worker thread. An exception raised by the producer is re-raised in the consumer while it is still reading. |
| `__all__` | `["Backend", "iterate_in_thread"]` | The supported surface, and what `from kumaru.backends import *` binds. |

The other shared helpers are deliberately not re-exported. Import them from where they are defined:

| Name | Import from | Signature |
|---|---|---|
| `ReasoningSplitter` | `kumaru.backends.base` | `class ReasoningSplitter(*, start_in_reasoning: bool = False, open_tag: str = "<think>", close_tag: str = "</think>")` |
| `StopSequenceTracker` | `kumaru.backends.base` | `class StopSequenceTracker(stop: Sequence[str])` |
| `chunks_from_text` | `kumaru.backends.base` | `def chunks_from_text(text: str, size: int = 4) -> Iterator[str]` |
| `normalize_finish_reason` | `kumaru.backends.openai_compat` | `def normalize_finish_reason(value: Any) -> FinishReason` |

What they do: [base.md](base.md) and [openai_compat.md](openai_compat.md).

### Backend names

`BACKENDS.names()` (what `kumaru info`, `/v1/info` and the UI list) returns the five primary names; `BACKENDS.all_names()` adds the aliases. The registrations are in `src\kumaru\core\registry.py`.

| `backend.name` | Aliases | Lazy target | Engine | Needs | Page |
|---|---|---|---|---|---|
| `echo` | — | `kumaru.backends.echo:EchoBackend` | none: deterministic test double | nothing | [echo.md](echo.md) |
| `transformers` | `hf` | `kumaru.backends.transformers_backend:TransformersBackend` | PyTorch, in process | `[torch]` extra; `[quant]` for 4/8-bit; `[train]` (peft) for `adapter_path` | [transformers_backend.md](transformers_backend.md) |
| `llamacpp` | `gguf` | `kumaru.backends.llamacpp_backend:LlamaCppBackend` | llama-cpp-python, in process | `[gguf]` extra | [llamacpp_backend.md](llamacpp_backend.md) |
| `openai_compat` | `ollama`, `vllm`, `remote` | `kumaru.backends.openai_compat:OpenAICompatBackend` | any OpenAI-compatible HTTP server | nothing (`httpx` is a core dependency) | [openai_compat.md](openai_compat.md) |
| `llamacpp_server` | `llama-server`, `llama_server` | `kumaru.backends.llamacpp_server:LlamaServerBackend` | managed `llama-server` child process, spoken to over the OpenAI protocol | the binary from `scripts\get-llama-server.ps1` (`huggingface_hub`, a core dependency, downloads the GGUFs) | [llamacpp_server.md](llamacpp_server.md) |

## How to use it

### Write a backend against the public import

```python
import asyncio
from collections.abc import AsyncIterator, Sequence

from kumaru.backends import Backend
from kumaru.core.config import BackendConfig
from kumaru.core.types import BackendInfo, GenerationParams, Message, StreamChunk


class ShoutBackend(Backend):
    name = "shout"

    async def _load(self) -> None:
        pass  # open clients, load weights, spawn processes; runs once per load()

    def info(self) -> BackendInfo:
        return BackendInfo(name=self.name, model_id=self.config.model, context_length=self.context_length)

    async def stream(self, messages: Sequence[Message], params: GenerationParams) -> AsyncIterator[StreamChunk]:
        yield StreamChunk(delta=messages[-1].content.upper())
        yield StreamChunk(finish_reason="stop")


async def main() -> None:
    backend = ShoutBackend(BackendConfig(name="shout", model="none"))
    await backend.load()
    result = await backend.generate([Message.user("hello")], GenerationParams())
    print(result.message.content, result.finish_reason, result.usage.completion_tokens)


asyncio.run(main())
```

This prints `HELLO stop 1`: the stream reported no usage, so `generate()` estimated it. A thinking-capable backend also yields `StreamChunk(reasoning=...)`; the contract is in [base.md](base.md).

Two arguments are enough for a backend that cannot call tools: `ChatAgent` passes `tools=` only when it offers tools, and it refuses to build when tools are enabled on a backend whose `supports_tools` is `False`. To let the model call tools, set `supports_tools = True` and accept `*, tools=None`, as [base.md](base.md) describes.

### Drive a blocking engine with `iterate_in_thread`

```python
import asyncio
import time

from kumaru.backends import iterate_in_thread


def blocking_tokens():
    for token in ["Hel", "lo", "!"]:
        time.sleep(0.1)  # stands in for one blocking engine step
        yield token


async def main() -> None:
    async for token in iterate_in_thread(blocking_tokens):
        print(token, end="", flush=True)
    print()


asyncio.run(main())
```

Pass the factory, not a generator object (`iterate_in_thread(blocking_tokens)` or `iterate_in_thread(lambda: engine.generate(prompt))`). The factory runs on the worker thread, so expensive setup inside it (tokenizing a long prompt, moving tensors to the GPU) never blocks the event loop. A backend that may stop reading early should wrap the call in `contextlib.aclosing(...)`, as the shipped backends do, so that the producer is closed as soon as the backend's own stream is.

### Get a concrete backend by name

```python
import asyncio

from kumaru.core.config import load_config
from kumaru.core.registry import BACKENDS
from kumaru.core.types import Message


async def main() -> None:
    cfg = load_config("test")
    backend = BACKENDS.create(cfg.backend.name, cfg.backend)
    await backend.load()
    try:
        result = await backend.generate([Message.user("hello")], cfg.generation)
        print(type(backend).__name__, repr(result.message.content))
    finally:
        await backend.aclose()


asyncio.run(main())
```

### List names and check the import budget

```powershell
.\.venv\Scripts\python.exe -c "from kumaru.core.registry import BACKENDS; print(BACKENDS.names()); print(BACKENDS.all_names())"
.\.venv\Scripts\python.exe -c "import sys, kumaru.backends; print([m for m in ('torch', 'transformers', 'llama_cpp', 'httpx') if m in sys.modules])"
.\.venv\Scripts\kumaru.exe chat -c test -q "smoke test"
```

The second command must print `[]`. `kumaru info` also lists the primary names, but it imports torch and the other optional packages to report their versions and CUDA devices, so it is not a cheap check.

## How to extend it

### Add a backend to Kumaru itself

1. Create `src\kumaru\backends\<name>.py` with a `Backend` subclass. Import the engine inside `_load()` and raise `DependencyMissingError("<package>", extra="<extra>")` from the `ImportError`.
2. Decorate the class like the built-ins: `@BACKENDS.register("<name>", aliases=(...))`. That registers it whenever the module is imported directly.
3. Add the line that makes the name resolvable without importing anything, next to the five built-ins in `src\kumaru\core\registry.py`:

   ```python
   BACKENDS.register_lazy("toy", "kumaru.backends.toy:ToyBackend")
   ```

   Without it nothing imports your module, the decorator never runs, and `create("toy", ...)` raises `Unknown backend`.
4. Add the name to the `backend.name` comment in `configs\default.yaml` and the `--backend` help of `kumaru serve` in `src\kumaru\cli.py`, then add a profile and tests.

### Ship a backend from another package

Declare an entry point in the plugin's `pyproject.toml`, install it into the same venv, and select it with `backend.name: toy`:

```toml
[project.entry-points."kumaru.backends"]
toy = "kumaru_toy.backend:ToyBackend"
```

`Registry.get()` looks a name up in three places: names already registered or resolved, then the `register_lazy` table, then (only on a miss, and only once per process) every installed `kumaru.backends` entry point, all imported at that moment. So plugins do not appear in `BACKENDS.names()`, `kumaru info` or `/v1/info` until some lookup has missed, a plugin installed while the server runs is not seen until restart, and a plugin that fails to import is logged (`failed to load plugin backend '<name>': ...`) and skipped rather than crashing the app.

### Register in-process (tests, notebooks, scripts)

```python
from kumaru.backends import Backend
from kumaru.core.registry import BACKENDS


@BACKENDS.register("toy")
class ToyBackend(Backend):
    ...
```

This must run before `BACKENDS.create("toy", ...)`. The name is listed by `BACKENDS.names()` immediately.

### Rules for this file

- Add a name to `__all__` only if it is lightweight (standard library plus `kumaru.core`), stable, and needed by backend authors.
- Never import a concrete backend module here; it would run on every import of any `kumaru.backends.*` module.
- Never reuse a built-in name or alias (`echo`, `transformers`, `hf`, `llamacpp`, `gguf`, `openai_compat`, `ollama`, `vllm`, `remote`, `llamacpp_server`, `llama-server`, `llama_server`). `register()` refuses names that are already registered, so an earlier registration of, say, `vllm` makes `openai_compat.py`'s own decorator raise `RegistryError: backend 'vllm' is already registered` on first import. That breaks `openai_compat` and `llamacpp_server`, which subclasses it.

## Failure modes

| Symptom | Likely cause | Fix |
|---|---|---|
| `ImportError: cannot import name 'LlamaServerBackend' from 'kumaru.backends'` | Concrete classes are not re-exported. | `BACKENDS.create("llamacpp_server", cfg.backend)`, or import from `kumaru.backends.llamacpp_server`. |
| `error: Unknown backend: 'x'.` and `hint: Available backends: ...` (CLI exits 1; the server reports it through `/healthz` 503) | Typo, a plugin not installed in this venv, or a decorator-only module that nothing imports. | Compare with `BACKENDS.all_names()`; add the `register_lazy` line; install the plugin. |
| `kumaru config -b x` accepted a wrong name | Expected: names are only resolved by `BACKENDS.create()`. | Validate with `kumaru chat -c <profile> -q hi`, or check `/healthz` after `kumaru serve`. |
| `RegistryError: backend 'toy' is already registered` | The decorator ran twice (a re-run notebook cell, a module imported under two names), or the name or alias belongs to a built-in. | Register once; guard with `if "toy" not in BACKENDS:`; choose an unused name. |
| `Optional dependency '<package>' is not installed.` on `load()` | Expected deferral: the engine is imported inside `_load()`. | Run the command in the hint, which `install_command(extra)` builds. On Windows it names the `setup.ps1` switch where there is one (`-Gpu` for `torch` and `train`, `-Quant`, `-Gguf`); otherwise it is `pip install -e ".[<extra>]" in the Kumaru folder`. See [errors.md](../core/errors.md). |
| A plugin works with `-b toy` but is missing from `kumaru info` or the UI | `names()` never scans entry points; only a lookup miss does. | Expected; select it by name. |
| `TypeError: 'generator' object is not callable`, raised from `iterate_in_thread` | A generator object was passed instead of a zero-argument factory. | `iterate_in_thread(make_gen)` or `iterate_in_thread(lambda: engine.generate(...))`. |
| An in-process engine keeps computing for a while after a client disconnects or a stop sequence hits | Abandonment is seen only when the producer hands over its next item, so one long blocking step (a long prefill, a slow first token) finishes first. Or the engine runs on its own thread (transformers' `generate()`) and the producer's `finally` does not tell it to stop. | In your own backend, stop the engine in the producer's `finally`: close its iterator if it is pull-based (as `llamacpp` does), or set a `threading.Event` that a stopping-criteria hook reads (as `transformers` does). HTTP backends are unaffected: closing their stream closes the request. |
| `kumaru chat -c test` loads torch or starts seconds slower | A heavyweight import was added here, to `base.py`, or at module level in a backend. | Move it into `_load()`; recheck with the `sys.modules` one-liner above. |

## Related files

- `src\kumaru\backends\base.py` — defines `Backend`, `iterate_in_thread` and the helpers that are not re-exported ([base.md](base.md)).
- `src\kumaru\core\registry.py` — `Registry`, `BACKENDS` and the five lazy registrations with their aliases.
- `src\kumaru\core\errors.py` — `RegistryError` and `DependencyMissingError`.
- `src\kumaru\cli.py`, `src\kumaru\server\app.py`, `src\kumaru\eval\harness.py`, `src\kumaru\eval\benchmarks\runner.py` — the four places that call `BACKENDS.create()`.
- `configs\default.yaml` — documents `backend.name` and every backend setting ([default.yaml.md](../../../configs/default.yaml.md)).

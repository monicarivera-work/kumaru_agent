# `src\kumaru\__init__.py`

> Package-level public surface for Kumaru's stable chat data types and version.

**Read this when:** you need to import Kumaru types from application code, check the installed package version, or decide whether a symbol belongs in the top-level API.

---

## What it does
This file makes the package importable as `kumaru` and exposes a deliberately small set of stable types. It re-exports the chat request, response, streaming, role, usage, and generation parameter models from `kumaru.core.types`. It also defines the package version string.

## Why it exists
Kumaru keeps its top-level API small so callers do not couple themselves to internal package layout. Without this file, users would import from deeper modules for common types, making future refactors harder and encouraging accidental dependencies on non-public code.

## Mental model
Treat `kumaru` as the stable front door. The top-level package is for shared data contracts and version checks, not for constructing backends, agents, or servers. New capabilities should usually register themselves in the registry rather than being added here.

## Public API
| Name | Signature | What it's for |
|---|---|---|
| `ChatResult` | `class ChatResult(BaseModel)` | Completed non-streaming generation result. Re-exported from `kumaru.core.types`. |
| `GenerationParams` | `class GenerationParams(BaseModel)` | Normalised decoding parameters used by every backend. Since v0.2 it also has `min_p` and `presence_penalty` (`None` means "not sent"), `thinking` (`None` = model default, `True`, `False`) and `thinking_budget`. |
| `Message` | `class Message(BaseModel)` | One conversation turn using Kumaru's canonical message schema. Since v0.2, `reasoning` holds the model's thinking text; input also accepts it as `reasoning_content`. `to_openai()` leaves it out unless you pass `include_reasoning=True`. |
| `Role` | `class Role(str, Enum)` | Message role enum: `system`, `user`, `assistant`, or `tool`. |
| `StreamChunk` | `class StreamChunk(BaseModel)` | One incremental piece of a streaming reply. `delta` carries new answer text and `reasoning` new thinking text (since v0.2); a chunk can carry either or both. `finish_reason` and `usage` default to `None`. |
| `Usage` | `class Usage(BaseModel)` | Prompt and completion token accounting. |
| `__version__` | `str` | Package version string, `"0.2.0"`. It must match `version` in `pyproject.toml`; nothing checks that it does. |
| `__all__` | `list[str]` | Exact list exported by `from kumaru import *`. |

## How to use it

Import common types from the package root:

```python
from kumaru import GenerationParams, Message

messages = [
    Message.system("You are concise."),
    Message.user("Summarise the deployment risk."),
]

params = GenerationParams(max_tokens=256, temperature=0.2)
```

Check the installed package version:

```python
import kumaru

print(kumaru.__version__)
```

Use `Role` when building messages directly:

```python
from kumaru import Message, Role

message = Message(role=Role.USER, content="What models are loaded?")
print(message.to_openai())
```

Accumulate streamed output. Chunks hold only new text, so you append. Keep the thinking separate from the answer; one chunk can carry both.

```python
from kumaru import StreamChunk, Usage

chunks = [
    StreamChunk(reasoning="Greet "),
    StreamChunk(reasoning="briefly."),
    StreamChunk(delta="Hello"),
    StreamChunk(delta=" world"),
    StreamChunk(finish_reason="stop", usage=Usage(prompt_tokens=4, completion_tokens=2)),
]

thought = "".join(chunk.reasoning for chunk in chunks)  # "Greet briefly."
text = "".join(chunk.delta for chunk in chunks)         # "Hello world"
```

Ask for thinking per call with `GenerationParams`:

```python
from kumaru import GenerationParams

params = GenerationParams(max_tokens=256, temperature=0.2, thinking=True, thinking_budget=512)
```

## How to extend it

Only add symbols here when they are stable enough for external callers. Most new features should be discoverable through registries or imported from their owning module.

For example, if `BackendInfo` becomes part of the supported top-level API, update both the import list and `__all__`:

```python
from kumaru.core.types import BackendInfo

__all__ = [
    "BackendInfo",
    "ChatResult",
    "GenerationParams",
    "Message",
    "Role",
    "StreamChunk",
    "Usage",
    "__version__",
]
```

Do not add backend classes here just because a backend exists. Backends are resolved by name through `kumaru.core.registry.BACKENDS`.

## Failure modes
| Symptom | Likely cause | Fix |
|---|---|---|
| `ImportError: cannot import name 'BackendInfo' from 'kumaru'` | The symbol exists in `kumaru.core.types` but is not exported at package root. | Import it from `kumaru.core.types`, or deliberately add it to this file and `__all__`. |
| `from kumaru import *` misses a new symbol | `__all__` was not updated. | Add the symbol to `__all__` if it is meant to be public. |
| External code imports deep internal modules for common types | The top-level API does not expose a stable type callers need. | Consider adding only that data contract here, not implementation classes. |
| `pip show kumaru` and `kumaru info` report different versions | The version lives in two places and only one was bumped, or the editable install still has the old metadata (pip reads it at install time; `kumaru info` reads `__version__` live). | Bump both, then re-run `pip install -e ".[dev]"`. See [`pyproject.toml`](../../pyproject.toml.md). |

## Related files
- `src\kumaru\core\types.py` — defines every type re-exported here.
- `src\kumaru\cli.py` — reads `__version__` for `kumaru serve` and `kumaru info` output.
- `src\kumaru\server\app.py`, `src\kumaru\server\routes_chat.py` — report `__version__` as the FastAPI app version and in `/healthz` and `/v1/info`.
- `src\kumaru\core\registry.py` — provides the extension mechanism that keeps this top-level surface small.
- `src\kumaru\agent\chat.py` — consumes the same `Message`, `GenerationParams`, `ChatResult`, and `StreamChunk` contracts.

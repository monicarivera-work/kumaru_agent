# `src\kumaru\backends\echo.py`

> Implements a deterministic, dependency-free backend for tests, CI and install checks, including a fake reasoning channel.

**Read this when:**
- you need to verify Kumaru without model weights;
- you want to isolate app-layer failures from model failures;
- you are writing tests for streaming behaviour;
- you are testing the thinking path (SSE `reasoning_content`, the UI's thinking panel, `thinking_ms`, budgets) without a thinking model;
- you are testing what happens when `max_tokens` cuts a reply short (`finish_reason` `length`);
- you are testing the tool loop, or the approval prompt of `kumaru chat --workspace`, without a model.

---

## What it does

`EchoBackend` returns a structured reply based on the last user message. It never loads a model, never uses a GPU, and streams small chunks with an optional artificial delay. The reply includes a stable SHA-256 prefix of the last user message, the message count, approximate prompt tokens, `max_tokens` and `temperature`:

```text
[echo:2cf24dba] You said: hello
Context: 1 message(s), ~11 prompt tokens, max_tokens=1024, temperature=0.7.
```

Since v0.2 it also emulates thinking. When `params.thinking` is truthy, it first streams a fixed "thought" on the reasoning channel, then the reply on the answer channel:

```text
The user asked: hello. Echo cannot reason; this text only exercises the reasoning channel.
```

Like a real model, it stops at `max_tokens`, which counts the thought and the reply together, and then reports `finish_reason` `length` instead of `stop`.

With tools offered (`kumaru chat --workspace`, or `agent.enabled_tools`), it also scripts tool calls. Each line of the last user message that reads `/tool <name> [json arguments]` becomes one call, one per round. When every line has been called, echo answers as usual, plus one `Tool <name>: <first line of the result>` line per result:

```text
[echo:96acd169] You said: /tool current_time
What time is it?
Context: 3 message(s), ~63 prompt tokens, max_tokens=1024, temperature=0.7.
Tool current_time: 2025-01-01T12:00:00+00:00 (UTC)
```

That exercises the whole agent loop, the approval prompt included, without a model.

## Why it exists

Most bugs above inference are in:
- HTTP routing;
- SSE framing;
- memory trimming;
- terminal output;
- UI rendering;
- error handling.

Those layers do not need a 15 GB model or a CUDA device. Echo gives operators a fast bisection: if `kumaru chat --backend echo` works, the problem is probably in the model backend or its dependencies.

The thinking emulation exists for the same reason. The reasoning path spans:
- `StreamChunk.reasoning`;
- SSE `delta.reasoning_content`;
- the non-streaming `message.reasoning_content`;
- the UI panel;
- the `thinking_ms` stat;
- budget plumbing from CLI, API and config.

All of it can be tested in milliseconds instead of loading the 17.73 GB Qwen3.6 GGUF.

## Mental model

1. Echo is fake inference but a real `Backend` implementation.
2. Output is deterministic for a given last user message. The hash is SHA-256 of the raw last user content, and everything else is derived from the inputs.
3. Text is streamed in fixed-size pieces, and **one piece counts as one token**:
   - **Reasoning:** the thought, in 8-character pieces. It is emitted only when `params.thinking` is truthy; `None` means off, because echo's default is no thinking. `thinking_budget` truncates the thought to that many pieces, and `0` means no reasoning at all.
   - **Answer:** the reply, in 6-character pieces.
   - **Usage:** `completion_tokens` = reasoning pieces + answer pieces actually emitted. `prompt_tokens` = the base heuristic `count_tokens()`.
4. `max_tokens` caps reasoning and answer pieces together, as on a real model. `thinking_budget` is applied to the thought first; the cap then covers what is left. When the cap cuts the reply short, the final chunk says `finish_reason="length"`; otherwise it says `"stop"`, including when the reply fits exactly. A thought that uses up the whole cap leaves no answer at all, which is what the UI's "No answer: the max-tokens cap ran out while the model was still thinking" note is for. Echo still ignores `stop`: a test of stop sequences must use a different backend or a stub.
5. `backend.extra.delay_s` (default `0.01`) is slept before *every* piece, reasoning and answer alike. Use `0` for tests and at least `0.05` to watch streaming by eye.
6. Tool calls are scripted, never chosen, and only when `stream()` receives `tools`; otherwise `/tool` lines are ordinary text.
   - **Script:** every line of the last user message that matches `^/tool\s+(\S+)(?:\s+(.*))?$` once stripped, in order. No arguments means `{}`. Arguments pass through as written, so malformed JSON and unknown tool names test the agent's error paths. Echo does not check names against the offered tools; the agent does.
   - **Round:** echo counts the assistant messages with `tool_calls` after the last user message. If a line is left, it returns that line as the only chunk: `StreamChunk(finish_reason="tool_calls", tool_calls=[{"id": "call_<n>_<hash>", "type": "function", "function": {...}}], usage=Usage(prompt_tokens=count_tokens(messages), completion_tokens=1))`. `delay_s` is slept once before it.
   - **Answer:** after the last line, the normal reply, with one `Tool <name>: <first line>` line per tool message after the last user message.
   - **Cap:** `agent.max_tool_iterations` (10) still applies. One line is one round, so the line after the cap gets the agent's `NOT RUN` result and the later lines are never requested.

## Public API

| Name | Signature | What it's for |
|---|---|---|
| `EchoBackend` | `class EchoBackend(config: BackendConfig)` | Dependency-free backend registered as `echo`, both by decorator and lazily in `src\kumaru\core\registry.py`. |
| `EchoBackend.name` | `name = "echo"` | Registry and introspection name. |
| `EchoBackend.supports_tools` | `supports_tools = True` | Lets `ChatAgent` offer tools; the calls come from `/tool` lines. |
| `EchoBackend.delay_s` | `delay_s: float = float(config.extra.get("delay_s", 0.01))` | Seconds slept before each streamed piece. Set in `__init__`; mutable on the instance. |
| `EchoBackend._load` | `async def _load(self) -> None` | No-op load hook. |
| `EchoBackend.info` | `def info(self) -> BackendInfo` | `model_id="echo"`, `device="cpu"`, `dtype="none"`, `context_length=backend.max_context_length`, `supports_streaming=True`, `extra={"purpose": "testing and install verification"}`. |
| `EchoBackend.stream` | `async def stream(self, messages: Sequence[Message], params: GenerationParams, *, tools: Sequence[dict[str, Any]] \| None = None) -> AsyncIterator[StreamChunk]` | With `tools` and a `/tool` line not yet called: that call, alone, in one `finish_reason="tool_calls"` chunk. Otherwise optional reasoning pieces, then answer pieces, at most `max_tokens` of them in total, then `StreamChunk(finish_reason="length" or "stop", usage=...)`. |
| `scripted_calls` | `def scripted_calls(text: str) -> list[tuple[str, str]]` | The `(name, raw arguments)` of every `/tool` line in `text`, in order; arguments default to `"{}"`. `scripted_calls('/tool current_time\nWhat time is it?')` is `[('current_time', '{}')]`. |

Exact piece counts for `"hello"` with `GenerationParams()` defaults, measured: 18 answer pieces. With thinking on, 12 reasoning pieces + 18 = 30, and `completion_tokens` is 30. With `max_tokens=17` the reply stops one piece short (`length`, 17); with `max_tokens=18` it fits exactly (`stop`, 18). With thinking on and `max_tokens=5`, the cap runs out mid-thought: 5 reasoning pieces (40 characters), no answer, `length`, 5.

## How to use it

### Run one prompt with no model dependencies

```powershell
kumaru chat --backend echo -q "hello"
kumaru chat -c test -q "hello"
```

### Exercise the reasoning channel

```powershell
kumaru chat -c test --think -q "hello"                       # dim thought, then the reply
kumaru chat -c test --think --thinking-budget 2 -q "hello"   # thought cut to 2 pieces (16 characters)
kumaru chat -c test --no-think -q "hello"                    # reply only
```

Over HTTP, `kumaru serve -c test` serves the same thing on port 8080. Send `"thinking": true` (and optionally `"thinking_budget": N`) in a chat-completions request:
- streaming: thinking arrives in `delta.reasoning_content`, and the final frame has a non-null `thinking_ms`;
- non-streaming: it is in `choices[0].message.reasoning_content`.

### Script tool calls

In `kumaru chat`, type one `/tool` line per message. Changes and commands ask first, exactly as with a real model; read-only tools run at once. The folder must exist:

```powershell
New-Item -ItemType Directory .\scratch -Force | Out-Null
kumaru chat -c test --workspace .\scratch
# you › /tool write_file {"path": "hello.py", "content": "print(\"hello\")\n"}
# you › /tool run_command {"command": ["python", "hello.py"]}
```

[`cli.md`](../cli.md#let-it-work-in-a-project-folder-kumaru-chat---workspace) shows the approval prompts this produces. To run several calls in one message, put the lines in a here-string; single quotes keep PowerShell away from the JSON:

```powershell
$script = @'
/tool list_files
/tool read_file {"path": "hello.py"}
'@
kumaru chat -c test --workspace .\scratch -q $script
```

```text
» list_files({})
« Folder '.' (1 entry, 2 levels deep):
» read_file({"path": "hello.py"})
« File 'hello.py' (lines 1-1 of 1):
[echo:5d1dc02f] You said: /tool list_files
/tool read_file {"path": "hello.py"}
Context: 6 message(s), ~336 prompt tokens, max_tokens=256, temperature=0.0.
Tool list_files: Folder '.' (1 entry, 2 levels deep):
Tool read_file: File 'hello.py' (lines 1-1 of 1):
```

The backend alone, round by round, as the agent drives it:

```python
import asyncio

from kumaru.agent.tools import ToolBelt
from kumaru.backends.echo import EchoBackend
from kumaru.core.config import BackendConfig
from kumaru.core.types import GenerationParams, Message, Role


async def main() -> None:
    backend = EchoBackend(BackendConfig(name="echo", model="echo", extra={"delay_s": 0}))
    tools = ToolBelt(["current_time"]).specs()
    messages = [Message.user("/tool current_time\nWhat time is it?")]

    chunks = [c async for c in backend.stream(messages, GenerationParams(), tools=tools)]
    call = chunks[-1].tool_calls[0]
    print(len(chunks), chunks[-1].finish_reason, call["id"], call["function"])

    messages += [
        Message(role=Role.ASSISTANT, tool_calls=[call]),
        Message(role=Role.TOOL, content="2025-01-01T12:00:00+00:00 (UTC)",
                name="current_time", tool_call_id=call["id"]),
    ]
    chunks = [c async for c in backend.stream(messages, GenerationParams(), tools=tools)]
    print(chunks[-1].finish_reason, "".join(c.delta for c in chunks).splitlines()[-1])


asyncio.run(main())
```

```text
1 tool_calls call_0_96acd169 {'name': 'current_time', 'arguments': '{}'}
stop Tool current_time: 2025-01-01T12:00:00+00:00 (UTC)
```

With `ChatAgent` in between, the agent runs the call itself; see "Let the model use tools, with no model" in [`chat.md`](../agent/chat.md).

### Start the HTTP server with the test profile

```powershell
kumaru serve -c test
```

`configs\test.yaml` uses:

```yaml
backend:
  name: echo
  model: echo
  max_context_length: 8192
  extra:
    delay_s: 0.01

generation:
  max_tokens: 256
  temperature: 0.0
```

### Instantiate directly in a test

```python
import asyncio

from kumaru.backends.echo import EchoBackend
from kumaru.core.config import BackendConfig
from kumaru.core.types import GenerationParams, Message


async def main() -> None:
    backend = EchoBackend(BackendConfig(name="echo", model="echo", extra={"delay_s": 0}))
    await backend.load()
    result = await backend.generate(
        [Message.user("2+2?")], GenerationParams(thinking=True, thinking_budget=3)
    )
    print(result.message.reasoning)  # 'The user asked: 2+2?. Ec'  (3 pieces x 8 chars)
    print(result.message.content)    # '[echo:30ad2052] You said: 2+2?\nContext: ...'
    print(result.usage)              # prompt_tokens=11 completion_tokens=21


asyncio.run(main())
```

### Choose `delay_s`

| Value | Use | Measured on Windows (`"hello"`, 18 answer pieces) |
|---|---|---|
| `0` | Unit tests, CI, benchmarks of the app layer. | 0 ms |
| `0.01` (default, `configs\test.yaml`) | Realistic-but-fast streaming for smoke tests. | ~280 ms, ~16 ms per piece |
| `0.05` or more | Watching tokens arrive in the web UI, or debugging rendering, cancellation and the Stop button. | ~1.1 s, ~62 ms per piece |

Windows rounds short `asyncio.sleep` calls up to its ~15.6 ms timer tick. On this machine, `0.01` behaves like ~16 ms and `0.05` like ~62 ms, so a sub-tick value is not a precise rate.

### Bisect an operator issue

```powershell
kumaru chat --backend echo -q "smoke test"
kumaru chat -c qwen36 -q "smoke test"
```

If echo works and the real profile fails, inspect:
- optional dependencies;
- the `llama-server` binary;
- CUDA and VRAM;
- model ids, and adapter paths (transformers backend only);
- upstream servers.

## How to extend it

Use echo as the template for other test-only backends:
- keep construction cheap;
- leave `_load()` a no-op if there are no resources;
- return accurate `BackendInfo`;
- stream `StreamChunk(delta=...)`, and `StreamChunk(reasoning=...)` if you test thinking;
- stop at `params.max_tokens` pieces across both channels;
- emit a final chunk with `finish_reason` (`length` if the cap cut the reply short, else `stop`) and `Usage`.

```python
from kumaru.backends.base import Backend, chunks_from_text
from kumaru.core.registry import BACKENDS
from kumaru.core.types import BackendInfo, StreamChunk, Usage


@BACKENDS.register("fixed")
class FixedBackend(Backend):
    name = "fixed"

    async def _load(self) -> None:
        return None

    def info(self) -> BackendInfo:
        return BackendInfo(name=self.name, model_id="fixed", device="cpu", dtype="none")

    async def stream(self, messages, params):
        pieces = [("reasoning", "fixed thought")] if params.thinking else []
        pieces += [("delta", piece) for piece in chunks_from_text("fixed response", size=6)]
        for channel, piece in pieces[: params.max_tokens]:  # max_tokens covers reasoning too
            yield StreamChunk(**{channel: piece})
        yield StreamChunk(
            finish_reason="length" if len(pieces) > params.max_tokens else "stop",
            usage=Usage(prompt_tokens=0, completion_tokens=min(len(pieces), params.max_tokens)),
        )
```

With thinking on and `max_tokens=2`, this yields the thought and `'fixed '`, then `length` with 2 completion tokens. If a test needs `stop` semantics too, base it on the `ReverseBackend` example in [`base.md`](base.md) instead.

## Failure modes

| Symptom | Likely cause | Fix |
|---|---|---|
| Tests are slower than expected | `backend.extra.delay_s` defaults to `0.01`, slept before every piece (≈16 ms each on Windows). | Set `extra.delay_s: 0`. |
| Streaming looks instant in the UI | `delay_s` is `0` or `0.01`; a short reply finishes in a fraction of a second. | Use `0.05` or more while watching the UI. |
| Timing assertions flaky on Windows | Sub-tick sleeps round up to the ~15.6 ms timer tick. | Assert on ordering and counts, not durations; use `delay_s: 0`. |
| No reasoning / thinking panel in a test | `thinking` was `None` or `false`. Echo's default is off. | Send `"thinking": true`, `--think`, or `GenerationParams(thinking=True)`. |
| `thinking_ms` is null and no thought appears despite `--think` | `thinking_budget` is `0`, which truncates the thought to zero pieces. | Use a budget ≥ 1, or none. |
| Reply ignores `stop` | Echo does not implement stop sequences. | Expected. Use a stub backend, such as the `ReverseBackend` example in [`base.md`](base.md), for stop tests. |
| Reply ends mid-sentence with `finish_reason` `length`, or a thinking reply has no answer | `max_tokens` is smaller than the pieces echo needs (18 answer pieces for `hello`, 30 with thinking), and the cap counts the thought too. | Raise `max_tokens` (`configs\test.yaml` uses 256) or set a `thinking_budget` that leaves room for the answer; or keep the cap small on purpose to test truncation. |
| Reply says `You said: (nothing)` | No user message, or empty user content. | Send `Message.user(...)` or CLI `-q "..."`. |
| `/tool` lines come back as plain text, and nothing is called | No tools were offered: no `--workspace`, and `agent.enabled_tools` is empty. Echo scripts calls only when the request carries tools. | Add `--workspace DIR`, or list the tools in `agent.enabled_tools`. Over HTTP the change-making tools are never offered ([`tools.md`](../agent/tools.md)). |
| `« ERROR: the arguments for <tool> are not valid JSON (...)`, or `« ERROR: <tool> has no argument ...` | The `/tool` line's arguments are not one JSON object with the tool's parameter names. Echo passes them on as written. | Fix the JSON. In PowerShell, wrap a `-q` script in single quotes or a single-quoted here-string. |
| Later `/tool` lines never run, and the reply ends with `[Stopped after N tool round(s) ...]` | More lines than `agent.max_tool_iterations` (10): one line is one round. | Raise `agent.max_tool_iterations`, or split the script across messages. |
| Token counts are approximate | Echo uses the base heuristic for prompts, and pieces for completions. | Use a tokenizer-backed backend for exact counts. |
| `Unknown backend: 'echo'.` | Registry lazy registration is missing. | Restore `BACKENDS.register_lazy("echo", "kumaru.backends.echo:EchoBackend")`. |
| Echo works but a real profile fails | Echo needs no optional dependencies, binaries or GPU. | For `llamacpp_server` profiles (`qwen36`, `qwen25-7b-gguf`), run `.\scripts\get-llama-server.ps1`. For transformers profiles, run `.\scripts\setup.ps1 -Gpu`. Then check CUDA and VRAM. |
| `error: The echo backend cannot attach a LoRA adapter (backend.adapter_path is '…').` | `--adapter`, `adapter_path` or `$env:KUMARU_BACKEND__ADAPTER_PATH` is set. Only `transformers` attaches adapters; every other backend refuses one at `load()`. | Drop the adapter for echo runs (`Remove-Item Env:KUMARU_BACKEND__ADAPTER_PATH`), or test the adapter with a transformers profile. |

## Related files

- `src\kumaru\backends\base.py` — provides `Backend`, `chunks_from_text` and default token counting. [`base.md`](base.md) documents the reasoning contract echo emulates.
- `src\kumaru\core\registry.py` — lazily registers `echo`.
- `src\kumaru\core\types.py` — defines the message, params, chunk, usage and info models.
- `configs\test.yaml` — ready-to-run echo profile; see [`test.yaml.md`](../../../configs/test.yaml.md).
- `src\kumaru\cli.py` — `chat` and `serve` instantiate this backend through the registry.
- `tests\test_backends.py` — echo determinism, thinking-channel, budget and `max_tokens`-cap tests (`length` in both channels, and `stop` on an exact fit).
- `tests\test_server.py` — `test_a_thought_that_fills_max_tokens_ends_with_length_and_no_answer` checks the same cap through the HTTP API, streaming and not.
- `src\kumaru\agent\chat.py` — the tool loop that echo's scripted calls drive; see [`chat.md`](../agent/chat.md).
- `tests\test_tool_loop.py` — scripted calls, and the `kumaru chat --workspace` approval tests built on them.

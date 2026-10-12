# `src\kumaru\agent\chat.py`

> `ChatAgent` is the stateless per-turn orchestrator. It resolves the decoding parameters (`generation` < `thinking_mode` < per-call overrides), fits the history into the context budget, calls the backend, and logs timing.

**Read this when:** you change how a turn is assembled or decoded, you debug "why did the model see this prompt" or "why did it decode with these settings", you wire a new caller to a backend, or you reason about history, reasoning traces or horizontal scaling.

---

## What it does

`ChatAgent(backend, config)` wraps a `Backend` and a `KumaruConfig`. Each `complete()` or `stream()` call runs four steps:

1. **Resolve parameters.** `resolve_params(**overrides)` layers `config.generation`, then `config.thinking_mode` (only when thinking resolves on), then the per-call overrides.
2. **Prepare the prompt.** `prepare(messages, extra_context=..., reserve_tokens=params.max_tokens)` does three things:
   - builds the system prompt (`agent\prompt.py`);
   - forces exactly one system message, at index 0;
   - trims the history with the configured memory policy (`agent\memory.py`) to fit the budget.

   It then trims `max_tokens` to the room the prepared prompt leaves in the context window.
3. **Generate.** `backend.generate()` for `complete()`, `backend.stream()` for `stream()`.
4. **Log.** It writes one INFO line on the `agent.chat` logger with token counts and timing.

The agent stores no conversation. Its only state is four attributes, all set in `__init__`:

- `backend`;
- `config`;
- `memory`: `get_policy(config.agent.memory_policy)`;
- `tools`: `ToolBelt(config.agent.enabled_tools)`.

Every caller constructs it after `backend.load()`:

| Caller | Uses | Per-call overrides passed |
|---|---|---|
| `src\kumaru\server\app.py` / `routes_chat.py` | `complete()` and `stream()`; `describe()` for `GET /v1/info` | The request's sampling fields (`ChatCompletionRequest.generation_overrides()`). |
| `src\kumaru\cli.py` (`kumaru chat`) | `stream(history, **overrides)` | `temperature`, `max_tokens`, `thinking`, `thinking_budget` from the flags, with every message. A flag left out is `None`, so the presets apply; see "CLI flags" below. |
| `src\kumaru\eval\harness.py` (`kumaru eval`) | `complete()` | `temperature=0.0`, plus `max_tokens` from the case. |
| `src\kumaru\eval\benchmarks\runner.py` (`kumaru bench run`) | `resolve_params()`; `prepare()` only with `--use-agent-prompt` | `thinking`, `thinking_budget`, `temperature`. It calls the backend directly. |

## Why it exists

- **Policy here, mechanism in the backend.**
  - Backends own tokenization, chat templates and transport.
  - The agent owns the system prompt, the history budget, the decoding presets and the telemetry.
  - Otherwise five backends would each re-implement the same policy and drift apart.
- **Stateless on purpose.**
  - Like the OpenAI API, the caller sends the full history on every turn. Any replica can serve any request, so you scale by adding processes behind a load balancer, with no session affinity.
  - Persistence belongs to the caller. The UI keeps history in `localStorage`; the CLI keeps it in a local list.
- **Two sampling presets, not one.**
  - Reasoning models are tuned to decode differently when they think. `configs\qwen36.yaml` uses Qwen's recommended presets (its comments cite the model card): temperature 0.7 and `top_p` 0.8 without thinking, 1.0 and 0.95 with it.
  - A single `generation` block cannot serve both modes. So `thinking_mode` holds the thinking preset, and it applies only when thinking is on.
- **`resolve_params` is public** so that the benchmark runner decodes exactly the way chat does. A benchmark score is only meaningful for the configuration that users actually run.
- **`prepare` is public and pure** so that tests can assert on the exact prompt without running a model. That is the fastest way to catch the bug class "the model is fine, but the prompt is wrong".

## Mental model

### Parameter precedence

`resolve_params(**overrides)` layers three sources, lowest to highest:

1. **`config.generation`.** It is already the merge of field defaults < YAML < `KUMARU_GENERATION__*` env vars (`core\config.md`). No CLI flag writes it.
2. **`config.thinking_mode`.** Only its non-`None` fields, and only if thinking resolves on.
3. **`overrides`.** Only non-`None` values. `GenerationParams.merged()` drops `None`, so callers can pass optional request fields straight through.

Thinking resolves to `overrides["thinking"]` when that is not `None`, and to `generation.thinking` otherwise:

| `overrides["thinking"]` | `generation.thinking` | `thinking_mode` applied | Resulting `params.thinking` |
|---|---|---|---|
| `True` | any | yes | `True` |
| `False` | any | no | `False` |
| absent or `None` | `True` | yes | `True` |
| absent or `None` | `False` | no | `False` |
| absent or `None` | `None` | **no** | `None`: nothing is sent, and the model's chat template decides. |

The last row is a trap for models whose chat template thinks by default, as Qwen3's does. Such a model can think while it decodes with the non-thinking preset and `max_tokens`.

`configs\qwen36.yaml` avoids this:

- It pins `generation.thinking: false` (its comment says "fast by default").
- Thinking is opt-in: `--think`, the UI toggle, or a request field.

Every other shipped profile leaves `generation.thinking` as `None` and `thinking_mode` empty, so there the question does not arise.

### CLI flags

`kumaru chat` sends `-t`, `--max-tokens`, `--think/--no-think` and `--thinking-budget` with every message, as `agent.stream(history, **overrides)`, the same way the UI sends its settings. So a flag beats both presets: `generation` and, on a thinking turn, `thinking_mode`. Verified with the echo backend, which reports what it received (all with `kumaru chat -c qwen36 -b echo -q "hi"`):

| Flags | Decodes with |
|---|---|
| `--think -t 0.3 --max-tokens 100` | `max_tokens=100, temperature=0.3` |
| `--no-think -t 0.3 --max-tokens 100` | `max_tokens=100, temperature=0.3` |
| `--think` | `max_tokens=32768, temperature=1.0`: the `thinking_mode` preset |

The other flags (`-c`, `-b`, `-m`, `--device`, `--adapter`, `--context`) are config overrides: they choose and load the model once, at start-up.

### The history budget

```text
reserved   = reserve_tokens if reserve_tokens is not None else config.generation.max_tokens
reserved   = min(reserved, backend.context_length // 2)
budget     = min(config.agent.max_history_tokens, max(512, backend.context_length - reserved - 128))
# after prepare(), in _fit():
max_tokens = min(max_tokens, max(1, backend.context_length - count_tokens(prepared)))
```

- **Reserved room.**
  - `complete()` and `stream()` pass `reserve_tokens=params.max_tokens`, the *resolved* value. A thinking turn therefore reserves `thinking_mode.max_tokens`, not `generation.max_tokens`.
  - The room for the reply has to be reserved up front: a prompt that fills the window leaves nothing to generate into.
  - **At most half the window is reserved.** `max_tokens` is a ceiling, not a forecast; most replies stop far below it. Reserving qwen36's 32,768-token cap in full in a 32,768-token window would leave history only the 512-token floor.
- **The reply cap is trimmed to the room actually left.** Strict servers (vLLM, OpenAI) reject a request whose prompt plus `max_tokens` exceeds the window, instead of stopping at its end. So the reply gets whatever the history did not use.
- **Margin and floor.**
  - 128 tokens is a fixed margin, and 512 is a floor. With the half-window cap, the floor only matters for windows under 1,280 tokens.
  - Example, measured with the echo backend: `qwen36` and a 10-exchange chat of ~1,000-token messages. With `--context 32768` the budget is 16,256 tokens: the system prompt and the last 16 of 21 messages are kept (~15,300 tokens), and the reply cap drops from 32,768 to 17,490. At the default 131,072 window the budget is 65,536 (`max_history_tokens`): all 21 messages are kept and the cap stays 32,768.
- **The system prompt counts.**
  - The budget covers the whole list, including the system prompt. For the `test` profile, the system prompt alone is about 215 tokens by the heuristic.
  - The policy always keeps the system message. What else survives depends on the policy (`agent\memory.md`).
  - With `memory_policy: none`, an over-budget conversation raises `ContextLengthExceededError`.
- **Token counts come from `backend.count_tokens()`.**
  - Exact for `transformers` (it renders the chat template) and for `llamacpp` (the model tokenizer).
  - `echo`, `openai_compat` and `llamacpp_server` use a heuristic of about 3.6 characters per token.

### Reasoning and history

- **Where thoughts arrive.**
  - `complete()` returns a `ChatResult` whose `message.reasoning` holds the thought text. It is `None` when there was none.
  - `stream()` yields `StreamChunk`s with thought text in `chunk.reasoning` and answer text in `chunk.delta`. A chunk can carry both.
- **The agent keeps neither. Callers keep answers only:**
  - the CLI appends `Message.assistant("".join(parts))`, built from the deltas;
  - the UI sends `{role, content}` and nothing else;
  - the server drops inbound `reasoning_content` (`ChatMessageIn` ignores unknown fields).
- **A kept thought is inert.**
  - `count_tokens()` and `Message.to_openai()` (default `include_reasoning=False`) read `content` only. So a stored thought is neither budgeted nor sent.
  - What is counted is what is sent.
- **Why answers only.**
  - Model authors train multi-turn behaviour on histories whose earlier assistant turns hold final answers, not thoughts. Qwen3's chat template strips old `<think>` blocks itself, and its model card says history should keep only the final output.
  - Thoughts are usually much longer than answers. Resending them would push real conversation out of the budget and add prefill time to every turn.

### Streaming lifecycle

- **Nothing runs until the first iteration.**
  - `stream()` is an async generator. Neither `resolve_params` nor `prepare` runs until the first `__anext__()`.
  - So a `ContextLengthExceededError` surfaces on the first iteration, not at the call.
  - In the server that moment is after the 200 headers have gone out, so the error arrives as an SSE error frame instead of a 413 (`server\routes_chat.md`).
- **TTFT.**
  - It runs from the end of `prepare()` to the first chunk whose `delta` *or* `reasoning` is non-empty. With thinking on, that is the time to the first thought, which is what the user sees as "it started".
  - The server measures its own `stats` separately, and they include queueing.
- **The `streamed:` log line.**
  - Format: `streamed: <n> chunks in <ms>ms (ttft <ms>ms)`.
  - It counts chunks that carry text, not tokens.
  - It is written only when the backend's stream runs to the end. A consumer that stops early (`break`, client disconnect) produces no line.
  - `ttft -1` means no text chunk arrived.
- **The `completed:` log line.** `complete()` logs `completed: <p> prompt + <c> completion tokens in <ms>ms (<x> tok/s)` from the `ChatResult`. `Backend.generate()` estimates the usage when the backend reported none.

## Public API

| Name | Signature | What it's for |
|---|---|---|
| `ChatAgent` | `ChatAgent(backend: Backend, config: KumaruConfig)` | Builds `memory = get_policy(config.agent.memory_policy)` (an unknown name logs a warning and falls back to `sliding_window`) and `tools = ToolBelt(config.agent.enabled_tools)` (an unknown tool name raises `RegistryError`). Does not load the backend. |
| `ChatAgent.backend`, `.config`, `.memory`, `.tools` | attributes | The wrapped `Backend`, the `KumaruConfig`, the `MemoryPolicy` and the `ToolBelt`. |
| `ChatAgent.prepare` | `prepare(messages: Sequence[Message], *, extra_context: str \| None = None, reserve_tokens: int \| None = None) -> list[Message]` | Pure. Returns `memory.apply(ensure_system_message(messages, build_system_prompt(...)), budget, backend.count_tokens)`. `extra_context` becomes the last system-prompt section ("Relevant context:"). `reserve_tokens` defaults to `config.generation.max_tokens`; at most half the window is reserved. |
| `ChatAgent.resolve_params` | `resolve_params(**overrides) -> GenerationParams` | Applies the precedence above. `overrides` are `GenerationParams` field names. `None` values are ignored. Values are not re-validated (`model_copy`). An unknown name is attached to the copy but never sent. |
| `ChatAgent.complete` | `async complete(messages: Sequence[Message], *, extra_context: str \| None = None, **overrides) -> ChatResult` | Resolves, prepares, trims `max_tokens` to the room left in the window, and calls `backend.generate()`. Logs `completed: ...`. Errors propagate. |
| `ChatAgent.stream` | `async stream(messages: Sequence[Message], *, extra_context: str \| None = None, **overrides) -> AsyncIterator[StreamChunk]` | Async generator. Resolves, prepares and trims `max_tokens` like `complete()`, then re-yields `backend.stream()` chunks unchanged. Logs `streamed: ...` on exhaustion. |
| `ChatAgent.describe` | `describe() -> dict` | Keys: `backend` (`backend.info().model_dump()`), `memory_policy` (the resolved policy's name), `max_history_tokens`, `tools` (enabled tool names), `generation` (full dump) and `thinking_mode` (non-`None` fields only). Served as `agent` in `GET /v1/info` once the backend has loaded. |

## How to use it

All the examples below were run against the echo backend, so they need no model.

### Inspect the exact prompt without a model

```python
from kumaru.agent.chat import ChatAgent
from kumaru.core.config import load_config
from kumaru.core.registry import BACKENDS
from kumaru.core.types import Message

config = load_config("test")
backend = BACKENDS.create(config.backend.name, config.backend)
agent = ChatAgent(backend, config)  # prepare() needs no load()

prepared = agent.prepare([Message.user("What can you do?")], extra_context="Kumaru runs locally.")
print([m.role.value for m in prepared])         # ['system', 'user']
print(prepared[0].content.splitlines()[-2:])    # ['Relevant context:', 'Kumaru runs locally.']
```

`extra_context` applies to one call only: put retrieved documents there, and they sit last in the system prompt, nearest the question.

### See which decoding parameters a turn will use

This uses the `qwen36` presets with echo swapped in, so nothing loads:

```python
cfg = load_config("qwen36", overrides={"backend": {"name": "echo"}})
qwen = ChatAgent(BACKENDS.create("echo", cfg.backend), cfg)
for overrides in ({}, {"thinking": True}, {"thinking": True, "temperature": 0.6}, {"temperature": None}):
    p = qwen.resolve_params(**overrides)
    print(overrides, p.thinking, p.temperature, p.top_p, p.max_tokens)
```

```text
{} False 0.7 0.8 32768
{'thinking': True} True 1.0 0.95 32768
{'thinking': True, 'temperature': 0.6} True 0.6 0.95 32768
{'temperature': None} False 0.7 0.8 32768
```

The same precedence over HTTP is captured in `server\routes_chat.md` ("Thinking resolution").

### Multi-turn with thinking, keeping answers only

```python
import asyncio

async def main() -> None:
    await backend.load()
    history: list[Message] = []
    for question in ["hi", "and again"]:
        history.append(Message.user(question))
        result = await agent.complete(history, thinking=True, max_tokens=128)
        print(result.message.reasoning[:40], result.finish_reason)  # show it...
        history.append(Message.assistant(result.message.content))   # ...but keep only the answer
    await backend.aclose()

asyncio.run(main())
```

```text
The user asked: hi. Echo cannot reason;  stop
The user asked: and again. Echo cannot r stop
```

### Stream, separating thoughts from the answer

```python
async def stream_once() -> None:
    thoughts, answer = [], []
    async for chunk in agent.stream([Message.user("Count to five.")], thinking=True, max_tokens=64):
        thoughts.append(chunk.reasoning)
        answer.append(chunk.delta)
        if chunk.finish_reason:
            print(chunk.finish_reason, chunk.usage)  # stop prompt_tokens=221 completion_tokens=33
    print("".join(thoughts))  # The user asked: Count to five.. Echo cannot reason; ...
    print("".join(answer))    # [echo:f562228d] You said: Count to five. ...
```

Every shipped backend ends the stream with a chunk that carries `finish_reason` and `usage`. For `openai_compat` and `llamacpp_server`, the usage is zeros when the server sent none. Render `reasoning` dimmed or collapsed, as the CLI and UI do.

### Decode like chat outside chat (the bench pattern)

Resolve once, then derive per-item parameters with `merged()`:

```python
base = agent.resolve_params(thinking=True, thinking_budget=2048)
item = base.merged(max_tokens=64, seed=7)
messages = agent.prepare([Message.user("What is 2+2?")], reserve_tokens=item.max_tokens)
result = await agent.backend.generate(messages, item)
```

## How to extend it

### Add a decoding knob

1. Add the field to `GenerationParams` (`src\kumaru\core\types.py`), defaulting to `None`, which means "not sent".
2. If the value should differ when thinking, add the same name to `ThinkingModeConfig` (`src\kumaru\core\config.py`).
   - `resolve_params` needs no change, because it merges `thinking_mode.model_dump(exclude_none=True)`.
   - A `ThinkingModeConfig` field with no matching `GenerationParams` field would be attached by `model_copy` and silently never sent.
3. Forward the field in each backend's payload, and accept it in `src\kumaru\server\schemas.py`. `server\schemas.md` walks through this for `frequency_penalty`.

### Add context before generation (RAG)

Retrieve in the caller and pass `extra_context=`. To retrieve over the history itself, write a `MemoryPolicy` (`agent\memory.md`). Either way, backends stay unaware of it.

### Add a tool-call loop

Some of the parts exist already:

- `ToolBelt.descriptions()` is already in the system prompt (the "You can call these tools:" section).
- `ToolBelt.specs()` returns JSON specs.
- `ToolBelt.invoke(name, arguments)` runs a tool with its timeout and turns failures into text.
- `config.agent.max_tool_iterations` (default 4) exists but nothing reads it yet.

Still missing: a way to pass the specs to the backend, and parsing tool calls out of the model's output. A sketch:

```python
from kumaru.core.types import Message, Role

async def complete_with_tools(self, messages, **overrides):
    params = self.resolve_params(**overrides)
    convo = list(messages)
    for _ in range(self.config.agent.max_tool_iterations):
        prepared = self.prepare(convo, reserve_tokens=params.max_tokens)
        result = await self.backend.generate(prepared, params)  # backend must accept tool specs
        call = parse_tool_call(result.message)                   # yours to write
        if call is None:
            return result
        output = await self.tools.invoke(call.name, call.arguments)
        convo += [result.message, Message(role=Role.TOOL, content=output)]
    return result
```

## Failure modes

| Symptom | Likely cause | Fix |
|---|---|---|
| `ContextLengthExceededError`. Non-streaming HTTP gets a 413; streaming gets an SSE error frame; the CLI prints `error:` and exits 1. | `agent.memory_policy: none` and the conversation exceeds the budget. | Use `sliding_window`, lower the resolved `max_tokens`, or raise `max_history_tokens` or `--context`. |
| Long chats forget their early messages. | The history budget is `min(max_history_tokens, window - min(max_tokens, window/2) - 128)`, and the memory policy drops what does not fit. With `--context 32768`, `qwen36` keeps about 16,000 tokens of history. | Keep the window larger (the default 131072 keeps 65,536 tokens), or raise `agent.max_history_tokens`. |
| Thinking turns trim more history than non-thinking turns. | The budget reserves the resolved `max_tokens` (up to half the window), and `thinking_mode.max_tokens` is usually the larger value. | Expected. Lower `thinking_mode.max_tokens` (bounding thoughts with `thinking_budget`) if history matters more. |
| The model thinks but uses the non-thinking preset, and its thoughts get cut off. | `thinking` resolved to `None` and the template thinks by default. `thinking_mode` applies only when thinking resolves to `True`. | Pin `generation.thinking` to `true` or `false` in the profile, or send `thinking` per request. |
| On a long chat, a reply stops with `finish_reason: "length"` well below `max_tokens`. | `max_tokens` was trimmed to the room the prompt left in the window. | Start a new chat, or raise `--context`. |
| An out-of-range value (for example `temperature=5`) reaches the backend from Python. | `merged()` uses `model_copy`, which does not validate. | Validate in the caller. HTTP requests are validated by `schemas.py`. |
| A misspelled override (`temprature=0.1`) has no effect. | Unknown keys are attached to the copy, and no backend reads them. | Use exact `GenerationParams` field names. |
| The server's `/healthz` returns 503 `Backend failed to load: Unknown tool: 'x'. (Available tools: ...)`, but only after the model has loaded. | `ChatAgent.__init__` builds the `ToolBelt`, and the server constructs the agent after `backend.load()`. | Fix the name in `agent.enabled_tools`. `kumaru info` lists the tools. |
| A stored thought seems to change nothing. | By design, `reasoning` is never counted or sent. | Keep answers only, and render thoughts separately. |
| The `streamed:` log line is missing. | The consumer stopped iterating early (`break` or a client disconnect). | Expected. |
| The logged TTFT is far below the time to the first answer word. | TTFT counts the first thought. | Expected with thinking on. The server's final `stats.thinking_ms` measures the thinking phase. |
| Tool names appear in the system prompt, but no tool ever runs. | There is no tool-call loop yet. | See "Add a tool-call loop". |
| Replicas "lose" the conversation. | The agent is stateless by design. | Send the full `messages` list on every turn. |

## Related files

- `src\kumaru\agent\prompt.py`: `build_system_prompt()` and `ensure_system_message()`.
- `src\kumaru\agent\memory.py`: `MemoryPolicy` and `get_policy()`.
- `src\kumaru\agent\tools.py`: `ToolBelt`.
- `src\kumaru\backends\base.py`: `Backend.generate()`, `stream()`, `count_tokens()`, `context_length`.
- `src\kumaru\core\types.py`: `GenerationParams.merged()`, `Message`, `StreamChunk`, `ChatResult`.
- `src\kumaru\core\config.py`: `GenerationParams` as `generation`, `ThinkingModeConfig`, `AgentConfig`.
- `src\kumaru\server\routes_chat.py`: the HTTP caller; it turns errors into status codes or SSE error frames.
- `src\kumaru\cli.py`: the `chat` REPL.
- `src\kumaru\eval\benchmarks\runner.py`: calls `resolve_params()` so that benchmarks decode like chat.

# `src\kumaru\agent\memory.py`

> Context-window memory policies: given a conversation, a token budget and a token counter, decide which messages the model gets to see.

**Read this when:** you hit a context-length error, a long chat "forgets" something, you change history trimming, or you add summarising or retrieval (RAG) memory.

---

## What it does
This file defines the `MemoryPolicy` interface and three policies:

| Config name | Class | Behaviour |
|---|---|---|
| `none` | `NoTrimPolicy` | Sends everything. Raises `ContextLengthExceededError` if it does not fit. |
| `drop_oldest` | `DropOldestPolicy` | Drops body messages from the front until the rest fits. |
| `sliding_window` (default) | `SlidingWindowPolicy` | Keeps the system message plus the longest run of newest messages that fits, and never fewer than `min_recent_turns` (2) messages. |

Each policy takes a message list, a token budget and the backend's token counter, and returns the messages to send. `get_policy()` maps a config name to a fresh policy instance. An unknown name falls back to `sliding_window` and logs a warning.

## Why it exists
A model has a fixed context window. Room for the reply must also be reserved before generation starts. Long conversations therefore have to lose something, and *what* to lose is a policy decision.

Centralising that decision gives every backend the same trimming behaviour. It also gives the standard upgrades one place to plug in, without touching `ChatAgent`: summarising the dropped middle, or retrieving relevant history from a store.

## Mental model
- **Who calls it.** Only `ChatAgent.prepare()` (`src\kumaru\agent\chat.py`) calls a policy. `ChatAgent.__init__` resolves the policy once, with `get_policy(config.agent.memory_policy)`.
- **One system message.** `prepare()` first normalises the conversation to exactly one system message at index 0, built by `ensure_system_message`. Policies keep only the *first* system message (`[:1]`) and treat everything else as "body".
- **The budget, computed in `prepare()`:**

  ```
  budget = min(agent.max_history_tokens, max(512, context_length - reserved - 128 - tool_tokens))
  ```

  - `reserved` is the request's *resolved* `max_tokens`. With thinking on, that is `thinking_mode.max_tokens`, which is often much larger than `generation.max_tokens`.
  - `128` is a safety margin.
  - `tool_tokens` is the estimated size of the tool declarations sent with the request, 0 when no tools are offered.
  - `512` is a floor, so the budget never reaches zero. When the reply reservation eats the window, the floor lets prompt plus reply exceed the window. See `docs\howto\src\kumaru\agent\chat.md`.
- **After the policy.** In a tool conversation the cut can fall between a call and its results, or drop the user's request. `prepare()` then repairs the policy's output: it drops tool results that answer no call, and puts the latest user message back (`chat.md`, "The history budget").
- **The counter comes from the backend**, because tokenizer accuracy is backend-specific:
  - `transformers` renders the real chat template, so its count is exact.
  - `llamacpp` (in-process) tokenizes a `role: content` blob, then adds 8.
  - `echo`, `openai_compat` and `llamacpp_server` use the base heuristic `int(sum(len(content) + len(role) + 4 + len(json.dumps(tool_calls))) / 3.6) + 8`, where `tool_calls` counts only on assistant messages that call tools. This can be wrong in either direction, so leave headroom.
- **Reasoning is invisible to memory.** A `Message.reasoning` field survives `apply()`, because policies return the same `Message` objects. But:
  - No counter counts it. The base heuristic reads `content`, `role` and `tool_calls` only, and `transformers` counts `to_openai()` output, which omits reasoning by default.
  - No backend sends it.

  Old thoughts therefore cost nothing here. The design reasons for dropping them, and how chat histories avoid storing them, are in `docs\howto\src\kumaru\agent\chat.md`.
- **`min_recent_turns` counts messages, not exchanges.** The default of 2 normally means "the last reply and the new question". The guarantee wins over the budget: `sliding_window` can return *more* tokens than `budget`.
- **The window is contiguous from the newest message.** Once a message does not fit, everything older is dropped, even smaller messages that would have fitted.

## Public API

| Name | Signature | What it's for |
|---|---|---|
| `TokenCounter` | `TokenCounter = Callable[[Sequence[Message]], int]` | Type of the counter passed to policies. In practice it is `backend.count_tokens`. |
| `MemoryPolicy` | `class MemoryPolicy(ABC)` | Base interface. |
| `MemoryPolicy.name` | `name: str = "base"` | Config name. `describe()` and `/v1/info` report it as `agent.memory_policy`. |
| `MemoryPolicy.apply` | `apply(self, messages: Sequence[Message], budget: int, count_tokens: TokenCounter) -> list[Message]` | Abstract. Returns the sub-list of `messages` to send. |
| `NoTrimPolicy` | `class NoTrimPolicy(MemoryPolicy)`, `name = "none"` | Returns `list(messages)`, or raises `ContextLengthExceededError` (hint: `"Switch agent.memory_policy to 'sliding_window', or raise the budget."`). |
| `DropOldestPolicy` | `class DropOldestPolicy(MemoryPolicy)`, `name = "drop_oldest"` | Keeps the first system message. Pops body messages from the front while the total exceeds `budget`. |
| `SlidingWindowPolicy` | `class SlidingWindowPolicy(MemoryPolicy)`, `name = "sliding_window"` | Keeps the first system message and grows a window backwards from the newest message. Logs `memory: dropped N older message(s) to fit B-token budget` at INFO. |
| `SlidingWindowPolicy.__init__` | `__init__(self, min_recent_turns: int = 2) -> None` | The number of newest body messages kept even when over budget. |
| `POLICIES` | `POLICIES: dict[str, Callable[[], MemoryPolicy]]` | Name-to-factory map: `none`, `drop_oldest`, `sliding_window`. It is a plain dict, not a `Registry`, so it has no plugin discovery. |
| `get_policy` | `get_policy(name: str) -> MemoryPolicy` | Builds a policy by name. Unknown names log `unknown memory_policy '<name>'; using sliding_window` at WARNING. |

The config fields, in `src\kumaru\core\config.py`, `AgentConfig`:
- `max_history_tokens: int = 6144`
- `memory_policy: str = "sliding_window"`. This is a free string, not validated against `POLICIES`.

## How to use it

### Pick a policy in config

```yaml
agent:
  max_history_tokens: 6144
  memory_policy: sliding_window     # sliding_window | drop_oldest | none
```

For a one-off run, use the environment override: `$env:KUMARU_AGENT__MEMORY_POLICY='none'`.

### See what a conversation will look like (no model needed)

`ChatAgent.prepare()` is pure: it runs the policy without generating anything. The echo backend needs no `load()` for this.

```python
from kumaru.agent.chat import ChatAgent
from kumaru.core.config import load_config
from kumaru.core.registry import BACKENDS
from kumaru.core.types import Message

cfg = load_config("test", overrides={"agent": {"memory_policy": "drop_oldest", "max_history_tokens": 600}})
backend = BACKENDS.create(cfg.backend.name, cfg.backend)
agent = ChatAgent(backend, cfg)

history = [Message.user("x" * 400) if i % 2 == 0 else Message.assistant("y" * 400) for i in range(9)]
prepared = agent.prepare(history)
print(len(history), "->", len(prepared), backend.count_tokens(prepared))
# 9 -> 4 556          (system prompt + the newest 3 messages, under the 600 budget)
```

### Apply a policy directly

```python
from kumaru.agent.memory import SlidingWindowPolicy
from kumaru.core.types import Message

messages = [Message.system("You are Kumaru."), *[Message.user("x" * 2000) for _ in range(5)]]
count = lambda ms: sum(len(m.content) for m in ms) // 4
trimmed = SlidingWindowPolicy(min_recent_turns=2).apply(messages, budget=1000, count_tokens=count)

print([m.role.value for m in trimmed], count(trimmed))
# ['system', 'user', 'user'] 1003     <- over the 1000 budget: min_recent_turns wins
```

### Make overflow fail loudly

```python
from kumaru.agent.memory import NoTrimPolicy
from kumaru.core.errors import ContextLengthExceededError
from kumaru.core.types import Message

try:
    NoTrimPolicy().apply([Message.user("too long")], budget=1, count_tokens=lambda ms: 99)
except ContextLengthExceededError as exc:
    print(exc.message)  # Conversation needs 99 tokens but the budget is 1.
    print(exc.hint)     # Switch agent.memory_policy to 'sliding_window', or raise the budget.
```

Where that error surfaces:
- A non-streaming HTTP request gets **413**.
- A streaming request gets a `data: {"error": ...}` frame after the 200.
- `kumaru chat` prints `error:` / `hint:` and exits 1.

### Confirm reasoning costs no budget

```python
# continues the prepare() example above: `backend` is the echo backend
from kumaru.core.types import Message

plain = Message.assistant("The answer is 4.")
thought = Message.assistant("The answer is 4.", reasoning="Let me think step by step " * 50)
print(backend.count_tokens([plain]), backend.count_tokens([thought]))   # 16 16
print(sorted(thought.to_openai()))                                       # ['content', 'role']
```

## How to extend it
Subclass `MemoryPolicy`, implement `apply`, and add a factory to `POLICIES`.

`POLICIES` has no plugin discovery and `ChatAgent` resolves the name in its constructor. So the entry must exist before any agent is built. In practice that means you either:
- add it in `memory.py` itself, or
- register it in code that runs before `ChatAgent(...)` (your own entry point, or a test fixture).

Otherwise the name silently, apart from a warning, becomes `sliding_window`.

Worked example: compress the dropped middle into one note.

```python
from collections.abc import Sequence
from kumaru.agent.memory import MemoryPolicy, POLICIES, TokenCounter
from kumaru.core.types import Message, Role

class SummarizingPolicy(MemoryPolicy):
    name = "summarize_middle"

    def apply(self, messages: Sequence[Message], budget: int, count_tokens: TokenCounter) -> list[Message]:
        system = [m for m in messages if m.role == Role.SYSTEM][:1]
        body = [m for m in messages if m.role != Role.SYSTEM]
        if count_tokens([*system, *body]) <= budget:
            return [*system, *body]

        recent = body[-4:]
        dropped = body[:-4]
        summary = Message.assistant(
            "Earlier conversation summary: "
            + " ".join(m.content[:120].replace("\n", " ") for m in dropped)
        )
        candidate = [*system, summary, *recent]

        while recent and count_tokens(candidate) > budget:
            recent.pop(0)
            candidate = [*system, summary, *recent]
        return candidate

POLICIES["summarize_middle"] = SummarizingPolicy
```

```yaml
agent:
  memory_policy: summarize_middle
```

This example is a placeholder:
- Its "summary" is truncation.
- The summary itself can outgrow the budget.
- With the history used above, it returned `[system, summary, newest user]`.

A production version should call a deterministic summariser or a retrieval store, cap the summary's own size, and never drop the newest user message.

Rules every policy should keep:
- Return `Message` objects in conversation order.
- Keep the system message, which carries the guardrails.
- Keep the newest user message.
- Never call the model inside `apply`: `prepare()` is meant to stay pure and fast.

## Failure modes

| Symptom | Likely cause | Fix |
|---|---|---|
| HTTP 413 `ContextLengthExceededError`, an SSE `error` frame, or `error: Conversation needs N tokens but the budget is B.` | `memory_policy: none` and the history is over budget. | Switch to `sliding_window`, start a new conversation, or raise `agent.max_history_tokens` (and the context, if the formula's middle term is the limit). |
| The model answers as if it never saw the question | `drop_oldest` with a budget smaller than system prompt plus question pops *every* body message and returns only the system message. Verified: `[system, user(400 chars)]` at budget 10 gives `['system']`. | Use `sliding_window`, which always keeps the newest 2 messages. Raise the budget. |
| The prompt is larger than `max_history_tokens` | `sliding_window` keeps `min_recent_turns` messages even over budget, by design. | Expected. If the backend rejects it, the budget is too small for single messages; raise the context or shorten the input. |
| Long chats keep only 1–3 recent messages, and early context is "forgotten" | The budget collapsed. `context_length - reserved - 128` is tiny because the resolved `max_tokens` is close to the window. Thinking presets (`thinking_mode.max_tokens`) are the usual cause, e.g. qwen36 with `--context 32768`. | Raise the context, or lower the resolved `max_tokens` (`thinking_mode.max_tokens` when thinking). `kumaru config` shows both. |
| The model forgets early requirements in a long chat | `sliding_window` and `drop_oldest` both discard old turns. | Put durable instructions in `agent.system_prompt`, or add a summarising policy. |
| The backend still rejects the prompt as too long | The heuristic counter (`openai_compat`, `llamacpp_server`, `echo`) underestimated, e.g. for code or non-English text. Or the 512 floor pushed prompt plus reply past the window. | Lower `agent.max_history_tokens` to leave headroom, or override `count_tokens` with a real tokenizer. |
| `TemplateError` saying roles must alternate (some templates, e.g. Gemma's) | The trimmed window can start with an assistant message, because the window is contiguous from the end, not aligned to user turns. | Write a policy that trims whole user/assistant pairs (see "How to extend it"). |
| The configured policy is not used | A typo, or a custom policy registered after `ChatAgent` was built. `get_policy` falls back to `sliding_window` with a WARNING. | Check the log line, or `agent.memory_policy` in `/v1/info` (the policy actually in use). Register custom policies before building the agent. |
| "Thinking" text seems to vanish from long conversations | By design: reasoning is never counted or sent back. | Nothing to fix. See `docs\howto\src\kumaru\agent\chat.md` for why. |

## Related files
- `src\kumaru\agent\chat.py` — `ChatAgent.prepare()` computes the budget and calls `MemoryPolicy.apply()`. `describe()` reports `memory_policy`.
- `src\kumaru\agent\prompt.py` — `ensure_system_message()` normalises the system message before the policy runs.
- `src\kumaru\core\config.py` — `AgentConfig.max_history_tokens` and `memory_policy`.
- `src\kumaru\core\errors.py` — `ContextLengthExceededError` (HTTP 413).
- `src\kumaru\core\types.py` — `Message` (including `reasoning` and `to_openai()`) and `Role`.
- `src\kumaru\backends\base.py` — the heuristic `count_tokens()` and `context_length`. `transformers_backend.py` and `llamacpp_backend.py` override the counter.
- `docs\SCALABILITY.md` — where summarisation and RAG memory fit.

# `src\kumaru\agent\memory.py`

> Context-window memory policies that decide which messages fit into a model prompt.

**Read this when:** you are handling context length errors, changing history trimming, adding summarization/RAG memory, or tuning long-chat behavior.

---

## What it does
This file defines the `MemoryPolicy` interface and three policies: no trimming, drop oldest, and sliding window. Each policy receives a message list, a token budget, and a backend token counter, then returns the messages to send. `get_policy()` resolves the configured policy name and safely falls back to `sliding_window`.

## Why it exists
Every model has a finite context window and completion tokens must be reserved before generation starts. Without a memory policy, long chats would either fail unpredictably or force each backend to invent its own trimming behavior. Centralizing the policy keeps prompt budgeting consistent across local and remote backends.

## Mental model
- The system message is special and should survive trimming.
- Token counting comes from the backend because tokenizer accuracy is backend-specific.
- `sliding_window` preserves instructions plus the newest messages, accepting loss of older middle context.

## Public API

| Name | Signature | What it's for |
|---|---|---|
| `TokenCounter` | `Callable[[Sequence[Message]], int]` | Function type used by policies to count prompt tokens. |
| `MemoryPolicy` | `abstract class MemoryPolicy` | Base interface for all memory policies. |
| `MemoryPolicy.name` | `str` | Config name for the policy. |
| `MemoryPolicy.apply` | `apply(messages: Sequence[Message], budget: int, count_tokens: TokenCounter) -> list[Message]` | Returns a message subset that should fit in `budget`. |
| `NoTrimPolicy` | `class NoTrimPolicy(MemoryPolicy)` | Sends everything and raises if it exceeds budget. |
| `DropOldestPolicy` | `class DropOldestPolicy(MemoryPolicy)` | Keeps the first system message and drops oldest non-system messages until the budget fits. |
| `SlidingWindowPolicy` | `class SlidingWindowPolicy(MemoryPolicy)` | Keeps the first system message and newest body messages, with a minimum recent-turn guarantee. |
| `SlidingWindowPolicy.__init__` | `__init__(min_recent_turns: int = 2) -> None` | Configures how many recent body messages survive even under small budgets. |
| `POLICIES` | `dict[str, Callable[[], MemoryPolicy]]` | Registry from config names to policy factories. |
| `get_policy` | `get_policy(name: str) -> MemoryPolicy` | Creates a policy by name; unknown names log a warning and return `SlidingWindowPolicy`. |

## How to use it

### Resolve the configured policy

```python
from kumaru.agent.memory import get_policy
from kumaru.core.config import AgentConfig

agent_config = AgentConfig(memory_policy="sliding_window")
policy = get_policy(agent_config.memory_policy)
print(policy.name)
```

### Apply a policy with a backend counter

```python
from kumaru.agent.memory import SlidingWindowPolicy
from kumaru.core.types import Message

messages = [Message.system("You are Kumaru."), *[Message.user("x" * 2000) for _ in range(5)]]
policy = SlidingWindowPolicy(min_recent_turns=2)
trimmed = policy.apply(messages, budget=1000, count_tokens=lambda ms: sum(len(m.content) for m in ms) // 4)

assert trimmed[0].role.value == "system"
assert len(trimmed) >= 3
```

### Make over-budget prompts fail loudly

```python
from kumaru.agent.memory import NoTrimPolicy
from kumaru.core.errors import ContextLengthExceededError
from kumaru.core.types import Message

try:
    NoTrimPolicy().apply([Message.user("too long")], budget=1, count_tokens=lambda ms: 99)
except ContextLengthExceededError as exc:
    print(exc.hint)
```

### Use with `ChatAgent.prepare()`

```python
config = config.model_copy(update={"agent": config.agent.model_copy(update={"memory_policy": "drop_oldest"})})
agent = ChatAgent(backend, config)
prepared = agent.prepare(history)
```

## How to extend it
Create a `MemoryPolicy` subclass and register a factory in `POLICIES`. The policy must return `Message` objects and should preserve the system message unless you intentionally want an unguided model.

Worked example: a summarizing policy placeholder that compresses dropped messages into one assistant note.

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

Then configure it:

```yaml
agent:
  memory_policy: summarize_middle
```

For production summarization, call a deterministic summarizer or retrieve from a store; do not silently discard the latest user request.

## Failure modes

| Symptom | Likely cause | Fix |
|---|---|---|
| HTTP 413 context error | `NoTrimPolicy` sent an over-budget conversation. | Switch to `sliding_window`, reduce history, or increase backend context length. |
| Model forgets early requirements | `drop_oldest` or `sliding_window` removed old user messages. | Put durable instructions in the system prompt or add a summarizing policy. |
| Prompt still exceeds real context | Backend `count_tokens()` underestimates tokens. | Override `Backend.count_tokens()` with the model tokenizer. |
| Empty or incoherent answer | Budget is so small that only a tiny recent window survives. | Lower `generation.max_tokens` or raise `agent.max_history_tokens`/model context. |
| Unknown policy silently not used | `get_policy()` falls back to `sliding_window` after logging a warning. | Check config spelling and add the policy to `POLICIES`. |

## Related files

- `src\kumaru\agent\chat.py` — computes the budget and calls `MemoryPolicy.apply()`.
- `src\kumaru\core\config.py` — defines `agent.memory_policy` and `agent.max_history_tokens`.
- `src\kumaru\core\errors.py` — defines `ContextLengthExceededError`.
- `src\kumaru\core\types.py` — defines `Message` and `Role`.
- `src\kumaru\backends\base.py` — provides default `count_tokens()` and `context_length`.


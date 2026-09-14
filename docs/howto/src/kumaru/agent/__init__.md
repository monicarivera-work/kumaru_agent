# `src/kumaru/agent/__init__.py`

> Agent-layer facade for chat orchestration, memory, and prompt assembly.

**Read this when:** importing agent utilities without depending on individual module paths.

---

## What it does
It re-exports `ChatAgent`, `ConversationMemory`, `build_messages`, and `render_system_prompt`. This is the public import surface for conversation orchestration.

## Why it exists
Callers should not have to know whether orchestration code lives in `chat.py`, `memory.py`, or `prompt.py`. The facade keeps agent-level imports stable.

## Mental model
The agent layer sits between backend generation and user-facing interfaces. It owns conversation state and prompt construction, not HTTP or CLI concerns.

## Public API
| Name | Signature | What it's for |
|---|---|---|
| `ChatAgent` | `class ChatAgent` | Stateful chat orchestration over a backend. |
| `ConversationMemory` | `class ConversationMemory` | Bounded per-session history. |
| `build_messages` | `build_messages(system_prompt: str, history: Iterable[Message], user_input: str, *, now: datetime | None = None) -> list[Message]` | Assemble backend message list. |
| `render_system_prompt` | `render_system_prompt(template: str, *, now: datetime | None = None) -> str` | Fill date/time prompt placeholders. |

## How to use it
```python
from kumaru.agent import ChatAgent
from kumaru.core.config import load_config
agent = ChatAgent(load_config())
```

```python
from kumaru.agent import build_messages
messages = build_messages("Be brief", [], "hello")
```

```python
from kumaru.agent import ConversationMemory
memory = ConversationMemory(max_turns=4, max_chars=1000)
```

## How to extend it
Re-export only stable agent APIs. If a new planner becomes public, expose it here after its module-level interface is settled.

## Gotchas
| Symptom | Likely cause | Fix |
|---|---|---|
| Importing private helper fails | The facade exports only public orchestration APIs. | Import private helpers from their module only in tests, or make a public API intentionally. |
| Backend dependency imported unexpectedly | Agent facade imported a concrete backend. | Keep concrete backend selection in `ChatAgent`/`load_backend`. |

## Related files
- [Chat agent](./chat.md)
- [Memory](./memory.md)
- [Prompt](./prompt.md)

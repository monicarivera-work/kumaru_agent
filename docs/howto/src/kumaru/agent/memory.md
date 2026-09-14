# `src/kumaru/agent/memory.py`

> Bounded in-process per-session conversation memory.

**Read this when:** tuning history retention, resetting sessions, or replacing memory storage.

---

## What it does
It stores non-system messages in per-session deques, enforces a message count limit and a character budget, and can clear one or all sessions. Oldest messages are removed first without splitting messages.

## Why it exists
Unbounded chat history eventually overflows model context or produces backend errors. A small in-process store is enough for v1 and has a storage-shaped interface for later SQLite or Redis replacement.

## Mental model
Each session id maps to oldest-first message history. System prompts are never stored because they are rebuilt for every turn.

## Public API
| Name | Signature | What it's for |
|---|---|---|
| `ConversationMemory` | `ConversationMemory(max_turns: int = 12, max_chars: int = 12000)` | Create bounded memory store. |
| `ConversationMemory.add` | `add(self, session_id: str, message: Message) -> None` | Append one non-system message. |
| `ConversationMemory.extend` | `extend(self, session_id: str, messages: Iterable[Message]) -> None` | Append many messages. |
| `ConversationMemory.history` | `history(self, session_id: str) -> list[Message]` | Return oldest-first copy. |
| `ConversationMemory.clear` | `clear(self, session_id: str) -> None` | Forget one session. |
| `ConversationMemory.clear_all` | `clear_all(self) -> None` | Forget all sessions. |
| `ConversationMemory.sessions` | `sessions(self) -> list[str]` | Sorted active session ids. |

## How to use it
```python
from kumaru.agent.memory import ConversationMemory
from kumaru.core.types import Message
m = ConversationMemory(max_turns=2, max_chars=100)
m.add("s1", Message.user("hello"))
print(m.history("s1"))
```

```python
m.extend("s1", [Message.assistant("hi"), Message.user("next")])
print([msg.content for msg in m.history("s1")])
```

```python
m.clear("s1")
assert m.history("s1") == []
```

```python
m.add("s2", Message.user("x"))
print(m.sessions())
```

## How to extend it
Replace it by implementing the same methods (`add`, `extend`, `history`, `clear`, `clear_all`, `sessions`, `__len__`). Example: a Redis-backed memory can keep the `ChatAgent` constructor unchanged.

## Gotchas
| Symptom | Likely cause | Fix |
|---|---|---|
| System prompt not in history | `add()` drops `Role.SYSTEM`. | Use `build_messages()` to add the current system prompt each turn. |
| Old messages disappear | `max_turns` or `max_chars` trimming ran. | Increase `agent.max_history_turns` or `agent.max_history_chars`. |
| `ValueError` at construction | Limit was less than 1. | Pass positive `max_turns` and `max_chars`. |

## Related files
- [Chat agent](./chat.md)
- [Prompt](./prompt.md)
- [Config](../core/config.md)

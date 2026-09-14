# `src/kumaru/agent/prompt.py`

> Prompt assembly helpers that decide the exact message list sent to a backend.

**Read this when:** changing prompts or debugging why a model saw particular context.

---

## What it does
It renders date/time placeholders in the system prompt and builds a message sequence from system prompt, history, and new user input. Stray system messages in history are dropped.

## Why it exists
Prompt changes have high impact and high risk. Keeping prompt assembly in one file makes answer-quality regressions easier to review and revert.

## Mental model
The backend sees at most one system message, first. Non-system history stays ordered. The current user message is appended last if it is not blank.

## Public API
| Name | Signature | What it's for |
|---|---|---|
| `render_system_prompt` | `render_system_prompt(template: str, *, now: datetime | None = None) -> str` | Replace `{date}` and `{time}` placeholders. |
| `build_messages` | `build_messages(system_prompt: str, history: Iterable[Message], user_input: str, *, now: datetime | None = None) -> list[Message]` | Create backend-ready message list. |

## How to use it
```python
from kumaru.agent.prompt import render_system_prompt
print(render_system_prompt("Today is {date}"))
```

```python
from datetime import datetime
from kumaru.agent.prompt import render_system_prompt
print(render_system_prompt("{date} {time}", now=datetime(2026, 9, 13, 21, 12)))
```

```python
from kumaru.agent.prompt import build_messages
from kumaru.core.types import Message
history = [Message.system("old"), Message.user("hi"), Message.assistant("hello")]
messages = build_messages("new system", history, "next")
print([m.to_dict() for m in messages])
```

## How to extend it
Add new prompt variables by extending `render_system_prompt()` with safe string replacement. Example: replace `{app}` with a configured app name while leaving unknown placeholders untouched.

## Gotchas
| Symptom | Likely cause | Fix |
|---|---|---|
| Two system prompts reach backend | History was assembled outside `build_messages`. | Use `build_messages`; it filters history system messages. |
| Literal JSON braces fail expectation | Only `{date}` and `{time}` are replaced. | Leave other braces as literals or add explicit supported placeholders. |
| Blank user turn missing | `user_input.strip()` was empty. | Validate input before calling, or expect no final user message. |

## Related files
- [Chat agent](./chat.md)
- [Memory](./memory.md)
- [Core types](../core/types.md)

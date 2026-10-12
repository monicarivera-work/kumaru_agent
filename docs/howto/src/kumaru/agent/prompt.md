# `src\kumaru\agent\prompt.py`

> System-prompt assembly and normalization helpers for Kumaru chat turns.

**Read this when:** you are editing assistant behavior, adding retrieved context, enabling tools, or debugging why a client system prompt did not replace server guardrails.

---

## What it does
This file composes the full system prompt from configured persona text, quality directives, the current date, enabled tool descriptions, and optional retrieved context. It also ensures every message list has exactly one system message at index 0. Client-supplied system text augments, rather than replaces, server-side instructions.

## Why it exists
The system prompt is the fastest way to change answer behavior without fine-tuning. Keeping assembly in one module makes prompt order deliberate and testable. Without normalization, clients could send no system prompt, several system prompts, or a replacement prompt that accidentally drops Kumaru's guardrails.

## Mental model
- Stable identity goes first; volatile context goes last near the user's question.
- Server guardrails always apply, even when a client sends `role="system"`.
- `ensure_system_message()` removes all original system messages from the body.

## Public API

| Name | Signature | What it's for |
|---|---|---|
| `QUALITY_DIRECTIVES` | `str` | Shared behavioral guardrails appended to every system prompt. |
| `build_system_prompt` | `build_system_prompt(config: AgentConfig, *, tool_descriptions: Sequence[str] = (), extra_context: str \| None = None, include_date: bool = True) -> str` | Builds the complete system prompt string for a request. |
| `ensure_system_message` | `ensure_system_message(messages: Sequence[Message], system_prompt: str) -> list[Message]` | Returns a new list with exactly one merged system message at index 0. |

## How to use it

### Build the default prompt

```python
from kumaru.agent.prompt import build_system_prompt
from kumaru.core.config import AgentConfig

prompt = build_system_prompt(AgentConfig(), include_date=False)
print(prompt)
```

### Include enabled tool descriptions

```python
from kumaru.agent.prompt import build_system_prompt
from kumaru.core.config import AgentConfig

prompt = build_system_prompt(
    AgentConfig(),
    tool_descriptions=["current_time: Return the current local date and time."],
    include_date=False,
)
assert "You can call these tools:" in prompt
```

### Add retrieved context for one request

```python
prompt = build_system_prompt(
    AgentConfig(system_prompt="You are Kumaru."),
    extra_context="Runbook: /healthz returns loading while the model warms up.",
    include_date=False,
)
assert prompt.endswith("Runbook: /healthz returns loading while the model warms up.")
```

### Normalize a client message list

```python
from kumaru.agent.prompt import ensure_system_message
from kumaru.core.types import Message, Role

messages = [
    Message(role=Role.SYSTEM, content="Use terse answers."),
    Message.user("What is Kumaru?"),
]
normalized = ensure_system_message(messages, "You are Kumaru.")

assert normalized[0].role == Role.SYSTEM
assert normalized[0].content == "You are Kumaru.\n\nUse terse answers."
assert normalized[1].content == "What is Kumaru?"
```

## How to extend it
Add a new prompt section in `build_system_prompt()` when it applies to every caller. Preserve the section-order rule: identity and guardrails first, request-specific context last.

Worked example: add deployment metadata from config.

```python
def build_system_prompt(config, *, tool_descriptions=(), extra_context=None, include_date=True):
    sections = [config.system_prompt.strip(), QUALITY_DIRECTIVES]
    if include_date:
        sections.append(f"Today's date is {_dt.datetime.now().strftime('%Y-%m-%d')}.")
    if getattr(config, "deployment_name", None):
        sections.append(f"Deployment: {config.deployment_name}")
    if tool_descriptions:
        sections.append("You can call these tools:\n" + "\n".join(f"- {d}" for d in tool_descriptions))
    if extra_context:
        sections.append(f"Relevant context:\n{extra_context.strip()}")
    return "\n\n".join(s for s in sections if s)
```

If the new section is optional and request-specific, prefer the existing `extra_context` parameter instead of adding global state.

## Failure modes

| Symptom | Likely cause | Fix |
|---|---|---|
| Client prompt does not override Kumaru behavior | `ensure_system_message()` merges client text after server text by design. | Change `config.agent.system_prompt` for server-wide behavior. |
| Prompt contains a stale date in tests | `include_date=True` uses wall-clock time. | Pass `include_date=False` in deterministic tests. |
| Tool use is never mentioned | `ToolBelt.descriptions()` is empty because `agent.enabled_tools` is empty. | Register the tool and add its name to config. |
| Retrieved context seems ignored | Context was placed in an earlier user message instead of `extra_context`. | Pass `extra_context` to `ChatAgent.prepare`, `complete`, or `stream`. |
| Multiple system messages disappear | Normalization intentionally keeps only the first client system message. | Combine required client instructions into one system message before sending. |

## Related files

- `src\kumaru\agent\chat.py` — calls `build_system_prompt()` and `ensure_system_message()`.
- `src\kumaru\agent\tools.py` — provides tool descriptions inserted into the prompt.
- `src\kumaru\core\config.py` — defines `AgentConfig.system_prompt`.
- `src\kumaru\core\types.py` — defines `Message` and `Role.SYSTEM`.
- `ui\app.js` — lets the browser prepend one client system message from the UI.


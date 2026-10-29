# `src\kumaru\agent\prompt.py`

> System-prompt assembly and normalization helpers for Kumaru chat turns.

**Read this when:** you are editing assistant behavior, adding retrieved context, or debugging why a client system prompt did not replace server guardrails.

---

## What it does
This file composes the full system prompt from configured persona text, quality directives, engineering guidance (`agent.expertise`), the documentation-excerpts rule, the current date, and optional retrieved context. It can also list tool descriptions, but `ChatAgent` passes none: tool declarations travel in the request's `tools` field, and the model's chat template renders them (`agent\chat.md`, "The tool loop"). It also ensures every message list has exactly one system message at index 0. Client-supplied system text augments, rather than replaces, server-side instructions: every nonblank client system message is appended after Kumaru's prompt, in the order sent.

## Why it exists
The system prompt is the fastest way to change answer behavior without fine-tuning. Keeping assembly in one module makes prompt order deliberate and testable. Without normalization, clients could send no system prompt, several system prompts, or a replacement prompt that accidentally drops Kumaru's guardrails.

## Mental model
- Stable identity goes first; volatile context goes last near the user's question.
- Documentation excerpts are not part of the system prompt. With `agent.knowledge_indexes` set, `ChatAgent` puts them at the start of the latest user message, and `excerpts=True` adds only `EXCERPTS_RULE`, a fixed sentence saying the user did not write that block and it is untrusted data. The system prompt then stays the same whatever is retrieved, so servers such as llama.cpp can reuse their prompt cache ([doc_index.md](doc_index.md)).
- Server guardrails always apply, even when a client sends `role="system"`.
- `ensure_system_message()` removes all original system messages from the body and merges them into the one at index 0: Kumaru's prompt first, then each nonblank client system message, stripped, in the order sent, separated by blank lines. Blank ones add nothing, and none is ever dropped.

## Public API

| Name | Signature | What it's for |
|---|---|---|
| `QUALITY_DIRECTIVES` | `str` | Shared behavioral guardrails appended to every system prompt. |
| `build_system_prompt` | `build_system_prompt(config: AgentConfig, *, tool_descriptions: Sequence[str] = (), extra_context: str \| None = None, include_date: bool = True, excerpts: bool = False) -> str` | Builds the complete system prompt string for a request. `excerpts=True` adds `EXCERPTS_RULE` (from `agent\doc_index.py`) after the engineering guidance; `ChatAgent` passes it when documentation indexes are configured. |
| `ensure_system_message` | `ensure_system_message(messages: Sequence[Message], system_prompt: str) -> list[Message]` | Returns a new list with exactly one merged system message at index 0. |

## How to use it

### Build the default prompt

```python
from kumaru.agent.prompt import build_system_prompt
from kumaru.core.config import AgentConfig

prompt = build_system_prompt(AgentConfig(), include_date=False)
print(prompt)
```

### List tools in the prompt

For a model reached without native tool calling, the prompt can list tools as text. `ChatAgent` does not do this: listing them again beside the request's `tools` would only spend history budget.

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
    Message(role=Role.SYSTEM, content="Answer in English."),
]
normalized = ensure_system_message(messages, "You are Kumaru.")

assert [m.role for m in normalized] == [Role.SYSTEM, Role.USER]
assert normalized[0].content == "You are Kumaru.\n\nUse terse answers.\n\nAnswer in English."
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

If the new section is optional and request-specific, prefer the existing `extra_context` parameter instead of adding global state. The system prompt comes before the whole history, so text there that changes from turn to turn stops servers such as llama.cpp from reusing their cache of that history. That is why documentation excerpts go in the latest user message instead.

## Failure modes

| Symptom | Likely cause | Fix |
|---|---|---|
| Client prompt does not override Kumaru behavior | `ensure_system_message()` merges client text after server text by design. | Change `config.agent.system_prompt` for server-wide behavior. |
| Prompt contains a stale date in tests | `include_date=True` uses wall-clock time. | Pass `include_date=False` in deterministic tests. |
| The system prompt does not list the tools | By design: `ChatAgent` sends the declarations in the request (`tools`), and the chat template renders them into the model's prompt. | See what is offered in the `tools:` line of `kumaru chat`, or `agent.tools` in `GET /v1/info`. |
| Retrieved context seems ignored | Context was placed in an earlier user message instead of `extra_context`. | Pass `extra_context` to `ChatAgent.prepare`, `complete`, or `stream`. For documentation, use `agent.knowledge_indexes`, which adds the matching passages to every question ([doc_index.md](doc_index.md)). |
| A client system message seems ignored | It was blank or whitespace-only, which adds nothing; otherwise it is in the merged message, after Kumaru's prompt. | Send nonblank text. To see the merged prompt, call `ChatAgent.prepare()` on the same messages. |

## Related files

- `src\kumaru\agent\chat.py` — calls `build_system_prompt()` and `ensure_system_message()`.
- `src\kumaru\agent\tools.py` — `ToolBelt.descriptions()` produces lines for `tool_descriptions`; `ChatAgent` sends `ToolBelt.specs()` in the request instead.
- `src\kumaru\agent\doc_index.py` — `EXCERPTS_RULE`, and the excerpts `ChatAgent` adds to the latest user message.
- `src\kumaru\core\config.py` — defines `AgentConfig.system_prompt`.
- `src\kumaru\core\types.py` — defines `Message` and `Role.SYSTEM`.
- `ui\app.js` — lets the browser prepend one client system message from the UI.
- `tests\test_agent.py` — pins the merge: order, stripping, blank and missing client system messages.
- `tests\test_server.py` — `test_every_client_system_message_reaches_the_model` checks the HTTP path end to end.


# `src\kumaru\agent\tools.py`

> Tool abstraction, registry integration, and safe invocation seam for future agent capabilities.

**Read this when:** you are adding a tool, enabling configured tools, designing the tool-call loop, or debugging tool timeouts and model-visible errors.

---

## What it does
This file defines a `Tool` base class, a registered `current_time` reference tool, and `ToolBelt`, which resolves enabled tool names into runnable instances. `Tool.safe_run()` wraps execution with a wall-clock timeout and converts failures into strings the model can read. In v0.1, the chat path is intentionally inert unless tools are enabled, and `ChatAgent` only inserts tool descriptions into the prompt.

## Why it exists
Tool calling turns a chat model into an agent, but it also creates an unbounded model-driven loop. This seam lets Kumaru add tools without changing every layer at once, while preserving load-bearing safety properties: iteration caps in config and per-tool timeouts. Without the seam, adding tools later would require simultaneous changes to message schemas, streaming, backend calls, and orchestration.

## Mental model
- A tool is a typed async function with JSON Schema parameters.
- Tools are registered by name in `kumaru.core.registry.TOOLS`.
- Tool errors return as text so the model can recover in the next iteration.

## Public API

| Name | Signature | What it's for |
|---|---|---|
| `Tool` | `abstract class Tool` | Base class for capabilities the model may invoke. |
| `Tool.name` | `str` | Stable registry/model identifier. |
| `Tool.description` | `str` | Model-facing documentation inserted into prompts/specs. |
| `Tool.parameters` | `dict[str, Any]` | JSON Schema for the tool arguments object. |
| `Tool.timeout_s` | `float` | Hard timeout for one invocation. |
| `Tool.read_only` | `bool` | Indicates whether the tool mutates state and should require confirmation. |
| `Tool.run` | `async run(**kwargs: Any) -> str` | Subclass implementation hook. |
| `Tool.spec` | `spec() -> dict[str, Any]` | Returns an OpenAI-style function declaration. |
| `Tool.safe_run` | `async safe_run(**kwargs: Any) -> str` | Runs with timeout and converts exceptions to `ERROR:` text. |
| `CurrentTimeTool` | `class CurrentTimeTool(Tool)` | Built-in read-only example tool registered as `current_time`. |
| `CurrentTimeTool.run` | `async run(**kwargs: Any) -> str` | Returns `datetime.now().strftime(format)`. |
| `ToolBelt` | `class ToolBelt(names: list[str])` | Instantiates the enabled tools for one agent. |
| `ToolBelt.__bool__` | `__bool__() -> bool` | True when any tools are enabled. |
| `ToolBelt.specs` | `specs() -> list[dict[str, Any]]` | Returns OpenAI-style tool specs. |
| `ToolBelt.descriptions` | `descriptions() -> list[str]` | Returns compact prompt descriptions. |
| `ToolBelt.invoke` | `async invoke(name: str, arguments: dict[str, Any]) -> str` | Runs a named enabled tool safely, or returns an unknown-tool error. |

## How to use it

### Enable and describe the built-in tool

```python
from kumaru.agent.tools import ToolBelt

belt = ToolBelt(["current_time"])
print(belt.descriptions())
print(belt.specs()[0]["function"]["name"])
```

### Invoke a tool directly

```python
from kumaru.agent.tools import ToolBelt

belt = ToolBelt(["current_time"])
value = await belt.invoke("current_time", {"format": "%Y-%m-%d"})
print(value)
```

### Handle unknown tools as model-readable text

```python
belt = ToolBelt([])
result = await belt.invoke("current_time", {})
assert result.startswith("ERROR: unknown tool")
```

### Enable a tool through config

```yaml
agent:
  enabled_tools:
    - current_time
  max_tool_iterations: 4
```

Then `ChatAgent` includes `current_time: ...` in the system prompt. A full execution loop must still be implemented before model-emitted tool calls run automatically.

## How to extend it
Subclass `Tool`, register it, and add the name to `agent.enabled_tools`. Keep `description` clear because the model reads it as API documentation.

Worked example: complete read-only calculator tool.

```python
from typing import Any
from kumaru.agent.tools import Tool
from kumaru.core.errors import ToolError
from kumaru.core.registry import TOOLS

@TOOLS.register("add_numbers")
class AddNumbersTool(Tool):
    name = "add_numbers"
    description = "Add two numbers and return the sum. Use for exact arithmetic with two numeric inputs."
    parameters = {
        "type": "object",
        "properties": {
            "a": {"type": "number", "description": "First addend."},
            "b": {"type": "number", "description": "Second addend."},
        },
        "required": ["a", "b"],
    }
    timeout_s = 2.0
    read_only = True

    async def run(self, **kwargs: Any) -> str:
        try:
            return str(float(kwargs["a"]) + float(kwargs["b"]))
        except KeyError as exc:
            raise ToolError(f"Missing argument: {exc.args[0]}") from exc
```

Configure it:

```yaml
agent:
  enabled_tools:
    - current_time
    - add_numbers
```

If the tool lives in another package, expose it via the `kumaru.tools` entry-point group so registry discovery can import it.

## Failure modes

| Symptom | Likely cause | Fix |
|---|---|---|
| `RegistryError` when constructing `ToolBelt` | Config names a tool that was not registered. | Import the module that registers it or fix `agent.enabled_tools`. |
| Tool returns `ERROR: tool 'x' timed out...` | `run()` exceeded `timeout_s`. | Shorten the operation, raise the timeout carefully, or make it cancellable. |
| Tool exception ends up in model text | `safe_run()` intentionally converts exceptions. | Raise `ToolError` for expected failures with actionable messages. |
| Mutating tool ran without confirmation | Future loop ignored `Tool.read_only`. | Gate `read_only=False` tools before calling `invoke()`. |
| Model loops forever asking for tools | Missing hard iteration cap in orchestration. | Enforce `config.agent.max_tool_iterations` in the tool loop. |
| Enabled tools do not run automatically | v0.1 only describes tools; it has no model tool-call loop. | Implement the loop in `ChatAgent` and backend schema support. |

## Related files

- `src\kumaru\agent\chat.py` — creates `ToolBelt` and inserts descriptions into prompts.
- `src\kumaru\agent\prompt.py` — formats tool descriptions in the system message.
- `src\kumaru\core\config.py` — defines `enabled_tools` and `max_tool_iterations`.
- `src\kumaru\core\errors.py` — defines `ToolError`.
- `src\kumaru\core\registry.py` — owns the `TOOLS` registry and entry-point discovery.
- `src\kumaru\core\types.py` — already includes tool-related message fields for future use.


# `src\kumaru\agent\tools.py`

> Tools: what the model can do besides talk, and the approval step that guards every tool that would change something.

**Read this when:** you enable or add a tool, write an approver for a new interface, or debug a tool call that was refused, declined, withheld or timed out.

---

## What it does
This file defines:
- **`Tool`**, the base class every tool subclasses. It has a name, a description and a JSON Schema the model reads; `run()` for tools that only read; `prepare()` for tools that change things; and `safe_*` wrappers that turn every failure into text.
- **`Action`**, a change that is ready to make: the `summary` shown to the user, and the `apply` coroutine that makes it.
- **`ApprovalRequest`** and **`Approver`**: the question put to whoever approves (the tool, its arguments, and the summary), and the async callback that answers it.
- **`ToolBelt`**, the tools one agent offers. It is built from `agent.enabled_tools` and withholds every tool that needs approval when there is no approver. Its `invoke()` runs one call: straight away for a read-only tool, otherwise prepare, ask, then apply or decline.
- **`current_time`**, the reference read-only tool. **`WORKSPACE_TOOLS`** lists the five project-folder tools (`list_files`, `read_file`, `write_file`, `edit_file`, `run_command`), which are registered lazily from `workspace.py` and `commands.py`.

`ChatAgent` (`src\kumaru\agent\chat.py`) runs the loop. It sends `ToolBelt.specs()` with each request, runs each call the model makes through `ToolBelt.invoke()`, appends the results as `TOOL` messages, and asks the model again. It stops when the model answers in prose, or after `agent.max_tool_iterations` rounds.

## Why it exists
A chat model can only print code, and you then copy, save, run and fix it. Tools let it do that work itself: write the files, run the build, read the error, fix the file. That turns Kumaru into an agent.

The risk is that the model's next action is a sample from a probability distribution, steered by everything in its context, including text in files and command output that someone else wrote. So this file makes one property hold everywhere: **nothing changes unless a person saw exactly what will happen and said yes.** Read-only tools run freely. Every other tool runs in two steps with a person in between. An interface with no person to ask is not offered those tools at all.

## Mental model

```text
model asks for a tool call
  │
  ToolBelt.invoke(name, arguments)
  ├─ unknown name ───────────────► "ERROR: unknown tool ..."
  ├─ read_only tool ─────────────► safe_run() ─► result text
  └─ changing tool
       safe_prepare() ── invalid ─► "ERROR: ..."   (the user is not asked)
          │ Action(summary, apply)
       approver(ApprovalRequest) ── no ──► "DENIED: the user declined ..."
          │ yes
       safe_apply(action) ─────────► result text, e.g. "Created src/app.js (12 lines, 340 B)."
```

- **Withheld, not refused.** `ToolBelt` leaves out a tool that changes anything (`read_only = False`) or works in the project folder (`needs_workspace = True`) when there is no approver. It lists such tools in `withheld`, and `ChatAgent` logs a warning. The model never sees their declarations, so it cannot ask. The folder tools are withheld even though two of them only read: without someone at the keyboard, nothing should show the folder to the model either.
- **Who approves.** Only `kumaru chat` passes an approver (`_approve` in `src\kumaru\cli.py`), which shows the summary and asks `Allow this? [y/N]`. The HTTP server, the web UI, `kumaru eval` and `kumaru bench` pass none, so they offer only read-only tools such as `current_time`.
- **The summary is the contract.** `prepare()` must describe what `apply` will really do: the diff, or the exact program and folder. It must not repeat the model's own description. The user approves the summary, so a tool whose summary hides an effect breaks the safety property.
- **Errors are text.** Timeouts, `ToolError`s and unexpected exceptions all come back as `ERROR: ...` strings the model reads and can correct, rather than exceptions that end the conversation.
- **The caps are safety, not polish.** The loop is driven by a sampler. `agent.max_tool_iterations` bounds the rounds per message (default 10), at most 16 calls of one reply run (the rest get `NOT RUN`), each tool has a `timeout_s`, and each result is cut to `agent.workspace.max_output_chars`.

## Public API

| Name | Signature | What it's for |
|---|---|---|
| `Tool` | `abstract class Tool` | Base class for capabilities the model may invoke. |
| `Tool.name` | `str` | Stable identifier the model emits. |
| `Tool.description` | `str` | What the model reads to decide when and how to call it. |
| `Tool.parameters` | `dict[str, Any]` | JSON Schema for the arguments object. |
| `Tool.timeout_s` | `float` | Hard limit for one `run`, `prepare` or `apply`. Default 30. |
| `Tool.read_only` | `bool` | `True` (default): runs as soon as the model asks. `False`: runs only through `prepare()` and approval. |
| `Tool.needs_workspace` | `bool` | `True` for project-folder tools: `ToolBelt` builds them with the shared `Workspace`, and offers them only with an approver. |
| `Tool.run` | `async run(**kwargs) -> str` | Do the work and return text for the model. Changing tools may refuse here. |
| `Tool.prepare` | `async prepare(**kwargs) -> Action` | Check the arguments and describe the effect without causing it. The default shows `name({json arguments})` and applies `run`. |
| `Tool.spec` | `spec() -> dict[str, Any]` | The OpenAI-style function declaration sent to the model. |
| `Tool.safe_run` | `async safe_run(**kwargs) -> str` | `run` with the timeout, every failure returned as text: `ERROR: tool '<name>' timed out after <timeout_s>s.`, `ERROR: <message>` for a `ToolError` (with its hint in parentheses), or `ERROR: tool '<name>' failed: <Type>: <message>` (also logged with its traceback). |
| `Tool.safe_prepare` | `async safe_prepare(**kwargs) -> Action \| str` | `prepare` behind the same net: an `Action`, or the `ERROR:` text. |
| `Tool.safe_apply` | `async safe_apply(action: Action) -> str` | `action.apply()` behind the same net. |
| `Action` | `Action(summary: str, apply: Callable[[], Awaitable[str]])` | A prepared change. |
| `ApprovalRequest` | `ApprovalRequest(tool: str, arguments: dict, summary: str)` (frozen) | One decision for the approver. |
| `Approver` | `Callable[[ApprovalRequest], Awaitable[bool]]` | `True` applies the action; `False` declines it. |
| `declined` | `declined(name: str) -> str` | The text the model gets on a no: `DENIED: the user declined this <name> call. Nothing was done. Do not retry it unchanged; ask the user what they want instead.` |
| `WORKSPACE_TOOLS` | `tuple[str, ...]` | `("list_files", "read_file", "write_file", "edit_file", "run_command")`, which `kumaru chat --workspace` adds to `agent.enabled_tools`. |
| `CurrentTimeTool` | `class CurrentTimeTool(Tool)` | `current_time`: the local date and time, in an optional strftime `format`. |
| `ToolBelt` | `ToolBelt(names: Sequence[str], *, workspace: WorkspaceConfig \| None = None, approver: Approver \| None = None)` | Build the tools for one agent. Duplicate names count once. Raises `RegistryError` for an unknown name, and `ConfigError` when a folder tool is offered but `workspace.root` is not set. |
| `ToolBelt.tools` | `dict[str, Tool]` | The offered tools, by name. |
| `ToolBelt.withheld` | `list[str]` | Enabled tools left out because there is no approver. |
| `ToolBelt.workspace` | `Workspace \| None` | The project folder shared by the folder tools, if any is offered. |
| `ToolBelt.__bool__` | `__bool__() -> bool` | True when any tool is offered. |
| `ToolBelt.specs` | `specs() -> list[dict[str, Any]]` | The declarations sent with each request. |
| `ToolBelt.descriptions` | `descriptions() -> list[str]` | `"name: description"` lines (not used by `ChatAgent`, which sends `specs()` instead). |
| `ToolBelt.invoke` | `async invoke(name: str, arguments: dict[str, Any]) -> str` | Run one call as in the diagram above. Never raises for tool failures. |

## How to use it

### Turn tools on
In `kumaru chat`, `--workspace DIR` turns on the five project-folder tools. See [workspace.md](workspace.md) and [commands.md](commands.md). Other tools are listed in config:

```yaml
agent:
  enabled_tools: [current_time]
  max_tool_iterations: 10     # rounds of tool calls per message
```

Tools need a backend that can call them: `llamacpp_server` (for example `kumaru chat -c qwen36`), `openai_compat` with a server that supports tool calls, or `echo` (scripted calls for tests). The `transformers` and `llamacpp` backends refuse at startup with a `ConfigError`, rather than leave the model writing tool calls as text that nothing runs.

### Invoke tools directly

```python
import asyncio
from kumaru.agent.tools import ToolBelt

belt = ToolBelt(["current_time"])
print(belt.specs()[0]["function"]["name"])                          # current_time
print(asyncio.run(belt.invoke("current_time", {"format": "%Y-%m-%d"})))
print(asyncio.run(belt.invoke("nope", {})))   # ERROR: unknown tool 'nope'. Available: current_time
```

### Write an approver
An interface that can ask a person passes an `Approver` to `ChatAgent(backend, config, approver=...)`, which hands it to its `ToolBelt`:

```python
from kumaru.agent.tools import ApprovalRequest

async def approve(request: ApprovalRequest) -> bool:
    print(f"{request.tool} wants to:\n{request.summary}")
    return input("Allow this? [y/N] ").strip().lower() in ("y", "yes")
```

Rules for an approver:
- Show `request.summary` in full. It is the diff or the command line, not the model's description.
- Treat everything but an explicit yes as no, including end-of-input and errors.
- Ask once per call. Never remember a yes for later calls.
- Display control characters visibly, since the summary contains text the model wrote. `kumaru chat` uses `_visible` in `cli.py` so an escape sequence or a right-to-left override cannot disguise the prompt.

## How to extend it

### A read-only tool
Subclass `Tool`, register it, and add its name to `agent.enabled_tools`. Write `description` carefully: it is the API documentation the model reads.

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

    async def run(self, **kwargs: Any) -> str:
        try:
            return str(float(kwargs["a"]) + float(kwargs["b"]))
        except KeyError as exc:
            raise ToolError(f"Missing argument: {exc.args[0]}") from exc
```

### A tool that changes something
Set `read_only = False` and implement `prepare()`. Validate everything there, raise `ToolError` for anything that cannot happen (the user is then not asked), and return an `Action` whose summary shows the real effect. Check again inside `apply`, since the world can change while the user reads. A tool that touches files should work in the project folder: subclass `_WorkspaceTool` from `workspace.py` and resolve every path with `self.workspace.resolve`.

```python
import asyncio
from typing import Any
from kumaru.agent.tools import Action
from kumaru.agent.workspace import _WorkspaceTool
from kumaru.core.errors import ToolError

class MakeFolderTool(_WorkspaceTool):
    name = "make_folder"
    read_only = False
    description = "Create an empty folder in the project folder. The user approves it first."
    parameters = {
        "type": "object",
        "properties": {"path": {"type": "string", "description": "Folder to create, relative to the project folder."}},
        "required": ["path"],
    }

    async def run(self, **kwargs: Any) -> str:
        raise ToolError("make_folder runs only through approval.")

    async def prepare(self, **kwargs: Any) -> Action:
        args = self._arguments(kwargs)
        path, shown = self.workspace.resolve(args["path"])
        if path.exists():
            raise ToolError(f"'{shown}' already exists.")

        async def apply() -> str:
            again, _ = self.workspace.resolve(args["path"])  # re-check: links may have appeared
            await asyncio.to_thread(again.mkdir, parents=True)
            return f"Created the folder {shown}/."

        return Action(f"Create the folder {shown}/", apply)
```

Register it lazily, so it is imported only when enabled, and turn it on:

```python
TOOLS.register_lazy("make_folder", "your_package.tools:MakeFolderTool")
```

```yaml
agent:
  enabled_tools: [make_folder]     # and run `kumaru chat --workspace DIR`
```

Tools in another pip package are found through the `kumaru.tools` entry-point group.

## Failure modes

| Symptom | Likely cause | Fix |
|---|---|---|
| `RegistryError` when the agent starts | `agent.enabled_tools` names a tool that is not registered. | Fix the name (`kumaru info` lists tools), or install the package that provides it. |
| `error: The <name> backend cannot call tools, and tools are enabled` | Tools enabled on `transformers` or `llamacpp`. | Use `llamacpp_server` (`-c qwen36`) or `openai_compat`, or empty `enabled_tools`. |
| `error: The project-folder tools are enabled, but no project folder is set.` | A folder tool in `enabled_tools` in `kumaru chat`, without `--workspace`. | Pass `--workspace DIR`, or set `agent.workspace.root`. |
| Warning `not offering write_file, ...: they need approval, and nobody here can give it` | Changing or folder tools enabled in the server, UI, eval or bench. | Expected: only `kumaru chat` can ask. Use it for project work. |
| `DENIED: the user declined this <name> call...` | You answered anything but `y`. | Expected. The model is told not to retry unchanged and to ask what you want. |
| `ERROR: tool '<name>' timed out after Ns.` | `run`, `prepare` or `apply` exceeded `timeout_s`. | Make it faster, raise `timeout_s` carefully, or make it cancellable. |
| `ERROR: <name> has no argument 'x'; it takes ...` or `ERROR: tool '<name>' failed: TypeError: ... unexpected keyword argument` | The model misnamed an argument. | Expected; the model retries. In a new tool, check arguments and raise `ToolError` naming the right ones. |
| `ERROR: the arguments for <name> are not valid JSON ...` | The model emitted broken JSON (raised by `ChatAgent`, not here). | Expected; the model resends. |
| `[Stopped after 10 tool rounds (agent.max_tool_iterations). Reply 'continue' to let it keep going.]` | The model used its rounds for this message. | Reply `continue`, or raise `agent.max_tool_iterations` (up to 100). |
| A changing tool ran without anyone seeing it | Its `read_only` is `True`, or its `prepare()` summary hides an effect. | Set `read_only = False`, and make the summary show exactly what `apply` does. |

## Related files
- `src\kumaru\agent\chat.py` — the tool loop: sends `specs()`, runs calls through `invoke()`, enforces `max_tool_iterations`, clips results.
- `src\kumaru\agent\workspace.py` — `Workspace` and the four file tools.
- `src\kumaru\agent\commands.py` — `run_command`.
- `src\kumaru\cli.py` — `kumaru chat --workspace`, and `_approve`, the terminal approver.
- `src\kumaru\backends\base.py` — `supports_tools` and `check_tools`.
- `src\kumaru\core\config.py` — `enabled_tools`, `max_tool_iterations`, `WorkspaceConfig`.
- `src\kumaru\core\registry.py` — the `TOOLS` registry, lazy registration and entry-point discovery.
- `src\kumaru\core\errors.py` — `ToolError`, `ConfigError`.
- `tests\test_tools.py` — `safe_run` and `invoke`, plus approval: a change only after yes, an invalid change reported without asking, read-only tools never asking, apply failures as text, tools withheld without an approver, and folder tools needing (and sharing) one project folder.
- `tests\test_tool_loop.py` — the loop end to end with the echo backend.

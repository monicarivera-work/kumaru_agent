# `src\kumaru\agent\workspace.py`

> The project folder: the one place the agent may list, read, create and edit files, and the checks that keep every path inside it.

**Read this when:** you use `kumaru chat --workspace DIR`, a file tool refuses a path, an approved write says the file "changed after the change was shown", or you add a tool that works on files.

---

## What it does
This file defines `Workspace`, one project folder, and the four file tools that work in it:

| Tool | Class | Changes anything? | What the model gets |
|---|---|---|---|
| `list_files` | `ListFilesTool` | No | The folder's entries, depth-first, as paths relative to the folder, with file sizes. |
| `read_file` | `ReadFileTool` | No | A text file, with line numbers, a page at a time if it is long. |
| `write_file` | `WriteFileTool` | Yes, after approval | Create a file, or replace a whole file. Missing folders are created. |
| `edit_file` | `EditFileTool` | Yes, after approval | Replace one exact piece of a file, which must occur exactly once. |

`kumaru chat --workspace DIR` turns them on, together with `run_command` ([commands.md](commands.md)). They register lazily in `src\kumaru\agent\tools.py`, so they are imported only when enabled.

The two tools that change files do nothing when the model calls them. `prepare()` checks the arguments and returns an `Action` whose summary is a unified diff. `ToolBelt` shows that summary to the approver, and runs `Action.apply` only if the answer is yes.

## Why it exists
To build an app, a website or a game, the model has to write more than one file and come back to them. Printing code in the chat leaves the copying, saving and fixing to you.

Giving a model a file system is also the most direct way for it to do damage. A path such as `..\..\.ssh\config`, a junction to `C:\Windows`, a hook in `.git\hooks`, or text in a file that says "now overwrite this other file" could all lead out of the project. So this file confines the tools to one folder you name, and refuses any path it cannot prove is inside it. Every change waits for your yes.

## Mental model
- **One root, chosen by you.** `WorkspaceConfig.root` (set by `--workspace`) must be an existing folder. It is resolved to its real path once (links and 8.3 short names expanded). It may not be a whole drive, your home folder or a folder that contains it, or anything inside `.git`.
- **Every path is relative to the root.** `Workspace.resolve(raw)` refuses, before touching the disk:
  - absolute paths (`/x`, `\\server\share`), and drive letters (`C:`) through the `:` rule below;
  - `..` anywhere;
  - `:` (also alternate data streams such as `notes.txt:hidden`) and the other characters Windows forbids in names, `< > " | ? *`;
  - control characters, and paths over 1,024 characters;
  - reserved device names (`CON`, `NUL`, `COM1`, `LPT1`, with or without an extension);
  - names that start or end with a space, or end with a dot, which Windows silently changes.
- **Links are never followed.** Every part of the path that exists is checked with `lstat`: symlinks, junctions and other reparse points that redirect (name surrogates) are refused, not only at the last part. Then the real path must still be inside the root.
- **`.git` is off limits.** A hook written there would run on your next commit, so nothing in `.git` is listed, read or written.
- **Changes are shown, then made.** `plan_write` and `plan_edit` return an `Action`. Applying it:
  1. resolves the path again, and checks that the file is still byte-for-byte what the diff was made from (or still absent);
  2. creates missing folders, then resolves and checks once more;
  3. writes a temporary file beside the target and renames it over the target, so a failure never leaves half a file.
- **Files keep their format.** Existing files keep their encoding (UTF-8 with or without BOM, UTF-16 LE/BE) and their line endings (CRLF or LF). In a file that mixes the two, the lines a change leaves alone keep their own ending, and new and changed lines get the more common one; the approval prompt says so. New files are UTF-8 with LF, except `.bat` and `.cmd`, which get CRLF. On POSIX a replaced file keeps its permissions, such as the executable bit.
- **Bounded output.** A file larger than `max_file_bytes` (256 KB) is neither read nor written. `read_file` stops before `max_output_chars` (16,000) and says which `offset` continues. `list_files` stops after `max_list_entries` (500), and lists generated folders (`node_modules`, `.venv`, `bin`, `obj`, `dist`, `build`, `target`, ...) without expanding them unless asked by path.
- **File contents are data.** A file can contain text written to steer the model. The tool descriptions tell the model so, but the real backstop is the approval: you see the actual diff, not the model's description of it.

## Public API

| Name | Signature | What it's for |
|---|---|---|
| `Workspace` | `Workspace(config: WorkspaceConfig)` | One project folder. Raises `ConfigError` when `config.root` is empty, missing or not a folder, a whole drive, your home folder (or contains it), or inside `.git`. |
| `Workspace.root` | `Path` | The folder's real path. |
| `Workspace.config` | `WorkspaceConfig` | The limits: `max_file_bytes`, `max_output_chars`, `max_list_entries`. |
| `Workspace.resolve` | `resolve(raw: Any) -> tuple[Path, str]` | The real path for a relative path, and the path to show (`"src/app.js"`, or `"."` for the root). Raises `ToolError` for every refused path. |
| `Workspace.list_files` | `list_files(raw: Any = ".", depth: Any = 2) -> str` | The folder listing the model reads. `depth` is 1–8. |
| `Workspace.read_file` | `read_file(raw: Any, offset: Any = 1, limit: Any = None) -> str` | Numbered lines (`{n:6}\t{line}`), from `offset`, at most `limit` lines and `max_output_chars - 300` characters. Binary files are refused; bytes that are not valid UTF-8 show as U+FFFD, with a note saying so. |
| `Workspace.plan_write` | `plan_write(raw: Any, content: Any) -> Action` | Prepare creating or replacing a whole file. Refuses binary targets and content identical to the file. |
| `Workspace.plan_edit` | `plan_edit(raw: Any, old_text: Any, new_text: Any) -> Action` | Prepare replacing the one occurrence of `old_text`. Refuses zero or several matches, and files that are not valid UTF-8 (UTF-16 files with a BOM are fine). |
| `ListFilesTool`, `ReadFileTool` | `Tool` subclasses, `read_only = True`, `needs_workspace = True` | `run()` calls the method above in a worker thread. |
| `WriteFileTool`, `EditFileTool` | `Tool` subclasses, `read_only = False`, `needs_workspace = True` | `prepare()` calls `plan_write` / `plan_edit`. `run()` refuses: these tools act only through approval. |

Every tool refuses argument names it does not take (`write_file has no argument 'text'; it takes path, content.`) and missing required ones, so a model that misspells an argument learns the right one.

The config, in `src\kumaru\core\config.py`, `WorkspaceConfig` (under `agent.workspace`):

| Field | Default | Meaning |
|---|---|---|
| `root` | `None` | The project folder. `None` means the folder tools cannot be built. |
| `max_file_bytes` | `262144` | Largest file read or written (1 KB – 8 MB). |
| `max_output_chars` | `16000` | Most characters one tool call returns to the model (1,000 – 200,000). |
| `max_list_entries` | `500` | Most entries per `list_files` call (10 – 10,000). |

`allowed_commands` and `command_timeout_s` belong to `run_command`; see [commands.md](commands.md).

## How to use it

### From the terminal

```powershell
mkdir C:\projects\hello-site
kumaru chat -c qwen36 --workspace C:\projects\hello-site
```

Then ask for something that needs files, for example "Create a one-page website with an index.html and a style.css". Each `write_file` or `edit_file` call shows its diff in a yellow panel and waits for `Allow this? [y/N]`. Anything but `y` or `yes` declines, and the model is told: `DENIED: the user declined this write_file call. Nothing was done. ...`.

### Use a `Workspace` directly (no model needed)

```python
import asyncio
from kumaru.agent.workspace import Workspace
from kumaru.core.config import WorkspaceConfig

ws = Workspace(WorkspaceConfig(root=r"C:\projects\demo"))   # must exist
action = ws.plan_write("src/hello.py", 'print("hello")\n')
print(action.summary)
# Create src/hello.py (1 line), and the folder src/
# --- /dev/null
# +++ b/src/hello.py
# @@ -0,0 +1 @@
# +print("hello")
print(asyncio.run(action.apply()))          # Created src/hello.py (1 line, 15 B).

edit = ws.plan_edit("src/hello.py", '"hello"', '"hello, world"')
print(asyncio.run(edit.apply()))            # Updated src/hello.py (1 line, 22 B).
print(ws.read_file("src/hello.py"))
# File 'src/hello.py' (lines 1-1 of 1):
#      1	print("hello, world")
print(ws.list_files())
# Folder '.' (2 entries, 2 levels deep):
# src/
# src/hello.py  (22 B)
```

### See what is refused

```python
from kumaru.core.errors import ToolError

for path in ["../secrets.txt", "notes.txt:hidden", "CON.txt", ".git/hooks/pre-commit"]:
    try:
        ws.resolve(path)
    except ToolError as exc:
        print(exc)
# path '../secrets.txt' uses '..'; paths must stay inside the project folder.
# path 'notes.txt:hidden' contains :, which file names cannot contain (':' would name a drive or a hidden data stream).
# 'CON.txt' is a reserved device name on Windows, not a file name.
# '.git/hooks/pre-commit' is in .git, which the agent may not read or change: a hook written there would run on your next commit.
```

### Ask before writing, through a `ToolBelt`

```python
import asyncio
from kumaru.agent.tools import WORKSPACE_TOOLS, ToolBelt
from kumaru.core.config import WorkspaceConfig

async def approve(request):          # an ApprovalRequest: tool, arguments, summary
    print(request.summary)
    return input("Allow this? [y/N] ").strip().lower() in ("y", "yes")

belt = ToolBelt(WORKSPACE_TOOLS, workspace=WorkspaceConfig(root=r"C:\projects\demo"), approver=approve)
print(asyncio.run(belt.invoke("write_file", {"path": "notes.md", "content": "# Notes\n"})))
# Created notes.md (1 line, 8 B).   or, if you answer n:
# DENIED: the user declined this write_file call. Nothing was done. Do not retry it unchanged; ...
```

Without an `approver`, `ToolBelt` withholds all five project-folder tools (`belt.withheld`), so nothing that cannot ask can see or change the folder.

## How to extend it
A new file tool subclasses `_WorkspaceTool` (it gets `self.workspace`, `needs_workspace = True` and argument checking through `self._arguments(kwargs)`), then registers in `src\kumaru\agent\tools.py` with `TOOLS.register_lazy(...)`. Add it to `WORKSPACE_TOOLS` if `--workspace` should turn it on.

Rules every file tool should keep:
- Resolve every path with `Workspace.resolve`. Never join the root with a raw path yourself.
- A tool that changes anything sets `read_only = False`, does its checks in `prepare()`, and returns an `Action` whose summary shows the real effect (a diff, the exact files). Check again inside `apply`, since the folder can change while the user reads.
- Write through a temporary file and `os.replace`, never in place.
- Return text, raise `ToolError` with a message the model can act on, and keep output under `max_output_chars`.

Worked example: a read-only tool that counts a file's lines.

```python
import asyncio
from typing import Any
from kumaru.agent.workspace import _WorkspaceTool

class CountLinesTool(_WorkspaceTool):
    name = "count_lines"
    description = "Count the lines in a text file in the project folder."
    parameters = {
        "type": "object",
        "properties": {"path": {"type": "string", "description": "Path relative to the project folder."}},
        "required": ["path"],
    }

    async def run(self, **kwargs: Any) -> str:
        args = self._arguments(kwargs)
        def count() -> str:
            path, shown = self.workspace.resolve(args["path"])
            with open(path, "rb") as handle:
                return f"{shown}: {sum(1 for _ in handle)} lines"
        return await asyncio.to_thread(count)

# in src\kumaru\agent\tools.py:
# TOOLS.register_lazy("count_lines", "your.module:CountLinesTool")
```

## Failure modes

| Symptom | Likely cause | Fix |
|---|---|---|
| `error: The project folder X does not exist, or is not a folder.` | `--workspace` names a missing folder. Kumaru never creates the root for you. | Create it first. |
| `error: X is, or contains, your home folder` / `is a whole drive` | The root must hold only the project. | Make a folder for the project, e.g. `%USERPROFILE%\projects\my-app`. |
| `ERROR: 'x' is a link or junction. Links are not followed...` | The path goes through a symlink or junction, even one that points inside the folder. | Use the real path, or copy the files into the folder. |
| `ERROR: 'x' changed after the change was shown, so nothing was written.` | The file changed (you saved it in an editor, or a build rewrote it) between the diff and your yes. | Expected: the model reads it again and makes the change from the current content. |
| `ERROR: old_text was not found in 'x'. Leave out the line numbers that read_file adds.` | The model copied `read_file` output with its line numbers. | Nothing to do; the hint tells the model. |
| `ERROR: old_text occurs N times in 'x'` | The text to replace is not unique. | Nothing to do; the model includes more surrounding lines. |
| `ERROR: 'x' is 300.0 KB, over the 256.0 KB limit...` | A large or generated file. | Raise `agent.workspace.max_file_bytes` (up to 8 MB), or leave that file out of the task. |
| `ERROR: cannot write 'x': Access is denied. It may be open in another program, or read-only.` | Windows locks files that are open in some programs (a running `.exe`, Excel). | Close the program, or stop the build that holds the file. |
| A hard-linked file elsewhere did not get the change | Writes rename a new file over the name, which replaces the link rather than writing through it. | Expected; the other names keep the old content. |
| A tool you enabled in `agent.enabled_tools` is missing in the server or web UI | They cannot ask for approval, so `ToolBelt` withholds the folder tools there (a warning is logged). | Use `kumaru chat --workspace DIR`. |

## What it does not protect against
- **Concurrent changes.** Another program that swaps a link into the folder between the last check and the rename. The window is small, but the checks are not atomic.
- **Code that runs later.** Files the agent writes are code. Approving a change to `package.json` scripts, a `Makefile` or a test file approves what it will run when you, or an approved `run_command`, run it.
- **What a command does.** Confinement covers the file tools only. An approved command runs with your permissions; see [commands.md](commands.md).
- **Remote models.** With `openai_compat` pointed at a hosted service, the files the agent reads are sent to that service.

## Related files
- `src\kumaru\agent\tools.py` — `Tool`, `Action`, `ApprovalRequest`, `ToolBelt` (which builds one `Workspace` for all folder tools), and the lazy registrations.
- `src\kumaru\agent\commands.py` — `run_command`, which uses `Workspace.resolve` for its `cwd` and for programs run by path.
- `src\kumaru\agent\chat.py` — the tool loop that calls these tools.
- `src\kumaru\cli.py` — `kumaru chat --workspace`, and the approval prompt.
- `src\kumaru\core\config.py` — `WorkspaceConfig`.
- `tests\test_workspace.py` — every refused path spelling (including junctions, symlinks and 8.3 names), encodings and line endings kept (mixed ones too), stale writes refused, atomic writes, and the size and listing limits.

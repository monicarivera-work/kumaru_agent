# `src\kumaru\agent\commands.py`

> `run_command`: how the agent builds, tests and installs in the project folder. Every command is shown to you first, has no shell, and is stopped at its time limit with every program it started.

**Read this when:** the agent should run `npm test`, `dotnet build` or `python -m pytest`, a command is refused, you want to allow another program, or a command's output looks cut.

---

## What it does
This file defines `RunCommandTool`, the `run_command` tool that `kumaru chat --workspace DIR` turns on beside the four file tools ([workspace.md](workspace.md)). The model sends a command as a list, the program and then each argument. Kumaru then:

1. checks the list, and finds the program: an allowed name on a cleaned-up PATH, or an allowed program file inside the project folder;
2. shows you the exact command line, the folder it runs in, the full path of the program, and its time limit, and waits for `Allow this? [y/N]`;
3. on yes, runs it without a shell, with a reduced environment and no input, inside the project folder;
4. at exit, timeout or cancel, ends the program **and everything it started**;
5. returns the exit code and the output (stdout and stderr together), cut to `max_output_chars`.

## Why it exists
A model that writes code without running it cannot know whether the code works. With `run_command` it can run the tests, read the failure, fix the file and run them again, the loop a developer uses for any language. Builds and package managers (`npm`, `dotnet`, `pip`) also need a way to run.

Running programs is also the riskiest thing an agent can do, so the tool is shaped to make each command easy to judge and hard to smuggle into:
- **No shell.** Nothing a model writes, and nothing it read in a file, can add `&& del ...` to an approved command.
- **Allowed programs only**, found where a file the agent wrote cannot pose as one.
- **No secrets** from your environment.
- **Nothing left running** after the time limit.

## Mental model
- **Prepare, then apply.** As with the file tools, calling the tool does nothing. `prepare()` validates and resolves everything; the summary you approve is exactly what `apply` runs.
- **A list, never a string.** `["npm", "test"]`. A single string is refused with a message telling the model so. The program is started directly (`subprocess.Popen` with an argv list), never through cmd.exe, PowerShell or sh: no pipes, redirection, `&&`, wildcards or `%VARIABLES%`.
- **Allowed by name.** `agent.workspace.allowed_commands` (default `python`, `node`, `npm`, `npx`, `dotnet`) holds bare names; `.exe`/`.cmd` suffixes and case are ignored. Shells and launchers are refused when the config loads, because one approved argv to a shell could run anything: `cmd`, `powershell`, `pwsh`, `bash`, `wsl`, `start`, `runas`, `rundll32`, `mshta`, and programs that run a command given to them, such as `xargs`, `nohup`, `timeout`, `su`, `schtasks`, `wmic` and `wt`. That list is best-effort: no list knows every program that can start another, so what you allow, and what you approve, decide.
- **A PATH that cannot be planted.** A bare name is looked up on your PATH without:
  - empty and relative entries (they mean "the current folder");
  - entries inside the project folder;
  - Kumaru's own virtual environment, so `python -m pip install` cannot change Kumaru.

  On Windows only `.com`, `.exe`, `.bat` and `.cmd` files count, and cmd.exe is told not to search the current folder first (`NoDefaultCurrentDirectoryInExePath`).
- **Programs in the folder, by path.** `[".venv/Scripts/python.exe", "-m", "pytest"]` works if the file's name is allowed. The path goes through `Workspace.resolve`, so the same confinement applies: no links, no `..`.
- **Batch files are special.** `npm` and `npx` are `npm.cmd` and `npx.cmd` on Windows, and Windows runs `.cmd` files through cmd.exe, which re-reads their arguments. So for a `.bat` or `.cmd` program, arguments containing `" % ! ^ & | < > ( )` or line breaks are refused, as are command lines over 8,000 characters. On Windows the `program:` line of the approval prompt ends with `(batch script, runs in cmd.exe)` for them, since a batch file in the project folder, such as `tools/python.cmd`, also matches its allowed name.
- **One command, one line.** The approval prompt shows the command on its `$` line (`display_command`). Every line break inside an argument is shown as its escape there: `\n`, `\r`, `\u2028` and the other characters `str.splitlines()` breaks at (`_LINE_BREAKS`). When there are any, a `note:` line says so, for example `note:    \n in the command is a line break inside an argument`. So an argument cannot draw lines of its own, such as a fake `program:` line, or push the real ones off the screen with thousands of blank lines.
- **A reduced environment.** Programs get only:
  - system, user and temp folders;
  - locale settings;
  - certificate settings;
  - toolchain homes (`JAVA_HOME`, `DOTNET_ROOT`, `GOPATH`, `CARGO_HOME`, ...);
  - proxy settings, unless they contain a `user:password@` part.

  Nothing else from your environment is passed, so API keys and tokens stay out. Kumaru adds `PYTHONIOENCODING=utf-8`, `PYTHONUNBUFFERED=1` and `NO_COLOR=1`.
- **No input, no window.** stdin is empty, so an interactive prompt gets end-of-file. On Windows the program gets no console window.
- **Everything stops.** Each run has a limit: `command_timeout_s` (300 s), or less if the model passes `timeout_s` (a number of seconds, at least 1; `NaN` and `Infinity` are refused). After the program exits, times out, or the chat is cancelled, everything it started is ended:
  - on Windows by closing a Job Object (with `taskkill /T` as the fallback);
  - elsewhere by killing its process group.

  So a dev server or a file watcher runs only until the limit, and a command cannot leave a background process behind. The same holds when Kumaru exits mid-command:
  - on Windows the Job Object ends the programs however Kumaru ends;
  - on Linux and macOS, `kumaru chat` turns a closed terminal (SIGHUP) or a `kill` (SIGTERM) into an orderly exit that stops the command first.
- **Output is bounded and untrusted.** stdout and stderr are merged.
  - Kumaru keeps at most `max_output_chars` (16,000) of it: two fifths from the start and three fifths from the end, where errors usually are.
  - Colours are removed, and progress bars keep only their last state.
  - The tool description tells the model that output is data, not instructions.

## Public API

| Name | Signature | What it's for |
|---|---|---|
| `RunCommandTool` | `RunCommandTool(workspace: Workspace)` | The `run_command` tool. `read_only = False` and `needs_workspace = True`, so `ToolBelt` offers it only with an approver. Its description and schema are built per instance, naming the allowed programs and the time limit. |
| `RunCommandTool.prepare` | `async prepare(**kwargs) -> Action` | Arguments: `command` (a list, required), `cwd` (a folder in the project, default `"."`) and `timeout_s` (a finite number, at least 1; at most the configured limit is used). Validates them, resolves the program and folder, and returns the `Action`. Raises `ToolError` for every refusal. |
| `RunCommandTool.run` | `async run(**kwargs) -> str` | Refuses: the tool acts only through `prepare()` and approval. |
| `RunCommandTool.timeout_s` | `int` | `command_timeout_s + 30`: the outer safety net for `Tool.safe_apply`. The process itself stops at the inner limit. |
| `build_env` | `build_env(root: Path) -> dict[str, str]` | The reduced environment, with the cleaned-up PATH. |
| `safe_path` | `safe_path(value: str, root: Path) -> str` | A PATH string without empty, relative, in-project and Kumaru-venv entries. |
| `which` | `which(name: str, path: str) -> Path \| None` | Find a program on `path` as the OS would, accepting only real program files. |
| `display_command` | `display_command(argv: list[str]) -> str` | The command as you would type it here (`list2cmdline` on Windows, `shlex.join` elsewhere), with each line break inside an argument shown as its escape (`\n`, `\u2028`, ...), so it stays on one line. |

The config is in `src\kumaru\core\config.py`, `WorkspaceConfig` (under `agent.workspace`):

| Field | Default | Meaning |
|---|---|---|
| `allowed_commands` | `[python, node, npm, npx, dotnet]` | Programs `run_command` may start, by bare name. Shells and launchers are refused. |
| `command_timeout_s` | `300` | Time limit per command, 1–3600 s. The model can ask for less, never more. |
| `max_output_chars` | `16000` | Most characters of output returned to the model (shared with the file tools). |

## How to use it

### From the terminal

```powershell
mkdir C:\projects\hello-node
kumaru chat -c qwen36 --workspace C:\projects\hello-node
```

Ask "Create a hello-world Node script and run it". After approving the `write_file`, you see:

```text
╭─ run_command ───────────────────────────────────╮
│ $ node hello.js                                 │
│ folder:  .  (C:\projects\hello-node)            │
│ program: C:\Program Files\nodejs\node.exe       │
│ stops after 300s, with every program it starts  │
╰─────────────────────────────────────────────────╯
Allow this? [y/N]
```

Check that `program:` points where you expect. A path inside the project folder means the agent is running a file in the project.

### Use it directly (no model needed)

```python
import asyncio
from kumaru.agent.commands import RunCommandTool
from kumaru.agent.workspace import Workspace
from kumaru.core.config import WorkspaceConfig

tool = RunCommandTool(Workspace(WorkspaceConfig(root=r"C:\projects\demo")))
action = asyncio.run(tool.prepare(command=["node", "-e", "console.log(1+1)"]))
print(action.summary)
# $ node -e console.log(1+1)
# folder:  .  (C:\projects\demo)
# program: C:\Program Files\nodejs\node.exe
# stops after 300s, with every program it starts
print(asyncio.run(action.apply()))
# Command finished with exit code 0 in 0.1s.
# $ node -e console.log(1+1)
# (in: .)
# 2
```

What is refused, and what the model reads:

```text
"npm test && del *"          -> command must be a list, not one string: ... There is no shell, so && | > and quotes do not work; run one command per call.
["powershell", "-c", "dir"]  -> 'powershell' is not an allowed program. Allowed: python, node, npm, npx, dotnet (the user sets this list in agent.workspace.allowed_commands).
["npm", "run", "a&b"]        -> npm is run by cmd.exe, which would treat '&' in the argument 'a&b' as a command character. Leave it out.
```

### Allow another program
Add it in your config. The list replaces the default, so repeat the programs you want to keep:

```yaml
agent:
  workspace:
    allowed_commands: [python, node, npm, npx, dotnet, cargo, go]
    command_timeout_s: 600      # a first `cargo build` downloads and compiles a lot
```

On Linux or macOS, where Python is often only `python3`, add `python3`.

### Python packages for the project
Kumaru's own virtual environment is not on the PATH programs see, so `pip install` cannot change Kumaru. Give the project its own:

```text
["python", "-m", "venv", ".venv"]
[".venv/Scripts/python.exe", "-m", "pip", "install", "pytest"]     (Windows)
[".venv/Scripts/python.exe", "-m", "pytest", "-q"]
```

On Linux and macOS a venv's `bin/python` is a symlink, which the folder rules refuse. Create it with `python -m venv --copies .venv` to get real files.

## How to extend it
- **Another program:** add its name to `allowed_commands` (above). Never add a shell or a launcher; the config refuses the known ones because they defeat the allowlist, but that list is best-effort, so check what a program can start before you allow it.
- **A narrower tool**, such as one that only runs `npm test`: write a `Tool` with `read_only = False` and `needs_workspace = True`, and register it in `src\kumaru\agent\tools.py` with `TOOLS.register_lazy`. Reuse this file's pieces rather than calling `subprocess` yourself:
  - `build_env` for the environment;
  - `which` with `build_env(...)["PATH"]` to find the program;
  - `display_command` for the summary;
  - `_run` to run with the time limit and the process-tree cleanup.

  Show the resolved program and folder in the `Action` summary, never pass `shell=True`, and keep the output under `max_output_chars`.

## Failure modes

| Symptom | Likely cause | Fix |
|---|---|---|
| `ERROR: 'cargo' is not an allowed program. Allowed: ...` | Not in `agent.workspace.allowed_commands`. | Add it to your config, if you want the agent to run it. |
| `ERROR: 'python' is allowed, but it was not found on PATH. Is it installed?` | Not installed, or found only in places removed from the PATH (the project folder, Kumaru's venv, relative entries). | Install it system-wide, or run the project's own copy by path. |
| `error: ... 'bash' is a shell or launcher, which could run any command` when Kumaru starts | A shell in `allowed_commands`. | Remove it; ask the agent for the program the script would run instead. |
| `Command timed out after 300s and was stopped, with every program it started.` | A long build, or something that never exits (a dev server, `--watch`, a prompt waiting for input). | Raise `command_timeout_s` (up to 3600) for long builds. Start servers and watchers yourself in another terminal. |
| `... is run by cmd.exe, which would treat '&' in the argument ...` | An argument to `npm`/`npx` or another `.cmd` program holds a character cmd.exe would act on. | Expected. The model rephrases it, or you run that command yourself. |
| `program: ...\tools\python.cmd  (batch script, runs in cmd.exe)` in the approval prompt | The program is a batch file. `npm` and `npx` always are on Windows; a batch file in the project folder matches its allowed name too. | Check that it is the program you expect before you answer `y`. |
| `note:    \n in the command is a line break inside an argument` in the approval prompt | An argument holds a line break, such as a multi-line `python -c` script. It is shown as `\n` on the `$` line. | Expected. Read each `\n` there as a line break. |
| `ERROR: timeout_s must be a number of seconds, at least 1.` | The model passed `0`, a negative number, a string, `NaN` or `Infinity`. Nothing was run. | Nothing to do; the model sends the call again. |
| `npm init` or `create-vite` fails or gives up at its first question | stdin is empty, so prompts get end-of-file. | Pass non-interactive flags (`npm init -y`, `npx --yes create-vite my-app --template react`). The tool description tells the model to. |
| A build misses a variable (`NODE_OPTIONS`, a registry token) | Only the variables listed in `_KEEP_ENV` are passed, on purpose. | Put the setting in a project file (`.npmrc`, `nuget.config`), or run that build yourself. |
| A proxy setting does not reach the program | Proxy URLs with `user:password@` are dropped as secrets. | Configure the proxy credentials in the tool's own config, or run it yourself. |
| `ERROR: '.venv/bin/python' is a link or junction...` (Linux/macOS) | A venv's python is a symlink. | `python -m venv --copies .venv`. |
| The output starts with `… N characters omitted …` in the middle | The command printed more than `max_output_chars`. | Expected. Raise `max_output_chars` (up to 200,000) if the model needs more, or have it run a narrower command. |
| A test run leaves nothing running, even an intended server | Every process the command started is ended when it ends. | Expected; run long-lived processes yourself. |

## What it does not protect against
- **What an approved command does.** It runs with your permissions: it can read and change any file you can, use the network, and start programs. The allowlist limits which program starts, not what it does: `python -c ...`, `node -e ...` and `npm run <script>` run any code, and `npm install` runs package install scripts. Read each command before you answer `y`.
- **Code the agent wrote earlier.** `npm test` runs whatever the agent put in `package.json` and the test files. Approving the run approves that code.
- **Programs in the folder.** An allowed name inside the project folder (a `node.cmd` the agent wrote) can be run by path. The summary shows the full path, so you can see it is a project file, and on Windows labels a batch file `(batch script, runs in cmd.exe)`.
- **Every launcher.** The refused-shell list is best-effort. A program you allow that starts others (a task runner, a test runner, `npm run`) runs whatever it is given.
- **A small race on Windows.** A program is put in its Job Object just after it starts. A child started in that instant could escape the job. The `taskkill` fallback covers only the case where no job exists.
- **`kill -9` on Linux and macOS.** It ends Kumaru with no cleanup, so a running command carries on until it ends by itself. It runs in its own session, out of reach of the terminal. On Windows the Job Object still ends it.

## Related files
- `src\kumaru\agent\workspace.py` — `Workspace.resolve`, used for `cwd` and for programs run by path.
- `src\kumaru\agent\tools.py` — `Tool`, `Action`, `ToolBelt` and the approval flow; registers `run_command` lazily.
- `src\kumaru\backends\llamacpp_server.py` — `_WindowsJob`, the Job Object helper reused here.
- `src\kumaru\core\config.py` — `WorkspaceConfig`, `program_name` and the list of refused shells.
- `src\kumaru\cli.py` — the approval prompt.
- `tests\test_commands.py` — argv validation, the allowlist and shell refusals, PATH cleaning, batch-file arguments and their label, line breaks kept off the approval prompt, the environment, time limits that end grandchildren, commands stopped when the turn is cancelled or Kumaru exits, and output capping.

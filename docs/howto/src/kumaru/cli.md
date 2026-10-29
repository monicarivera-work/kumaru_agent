# `src\kumaru\cli.py`

> The `kumaru` binary. Its Typer commands turn flags into config overrides and hand off to the server, the agent, the trainer, the eval harness, the benchmark runner or the distillation data builder. Every deliberate failure prints as `error:` plus `hint:` and exits 1.

**Read this when:**
- you run or script `kumaru`;
- you let the agent create, edit and run things in a project folder (`chat --workspace`);
- you add a flag or a command;
- you need to know which setting won (flag, profile or `thinking_mode`);
- you need to decode an exit code or an error line.

---

## What it does

- **Commands.** `app` (a `typer.Typer`) has six commands: `serve`, `chat`, `info`, `config`, `train` and `eval`. The `bench` sub-app adds `list`, `run`, `report` and `compare`; the `distill` sub-app adds `build` and `gold`.
- **Config loading.** Every command that needs a configuration builds it the same way: `_load(config, **flags)` calls `load_config(config, overrides=_build_overrides(**flags))`. `chat`'s generation flags are the exception: they travel with each message instead (invariant 4).
- **The REPL.** `chat` runs `_chat_loop`. It streams reasoning dimmed under `thinking…`, then the answer, then a token count. The answer goes back into the history, along with any tool calls and results from that turn; thoughts never do. Both print through `_printable`, because the model can repeat what a file or a command's output told it to. A control character such as an escape sequence shows as `\x1b` instead of steering the terminal.
- **The project folder.** `chat --workspace DIR` lets the model list, read, create and edit files in `DIR` and run allowed programs there. Before each change and each command, `_approve` shows exactly what will happen and waits for `y`. Listing and reading run without asking.
- **The entry point.** `main()` is the installed console script (`pyproject.toml`: `kumaru = "kumaru.cli:main"`).
  - It forces UTF-8 on stdout and stderr (`ensure_utf8_streams()`), then runs `app()`.
  - It turns any `KumaruError` into `error:`/`hint:` lines and exit code 1.

## Why it exists

- **One flag vocabulary over the config model the server also uses.** A flag is just the top config layer. A one-off experiment on the command line is therefore the same thing as a YAML edit, and `kumaru config` can show the result.
- **Heavy imports live in command bodies.** `import kumaru.cli` takes about 0.6 s and does not import torch. That keeps `--help`, `config`, `bench list|report|compare` and `distill` fast; `distill` never imports torch, and loads the `datasets` library only to read GSM8K. Only commands that need the model stack pay the import cost: `serve`, `chat`, `train`, `eval`, `bench run`, and `info`, which probes torch.
- **`main()` exists because `app()` handles only Click's own exceptions** (usage errors, `Abort`, `Exit`). Typer's `__call__` re-raises everything else. Without `main()`, a deliberate `ConfigError` carrying a hint would print as a traceback.
- **UTF-8 is forced first.** Windows consoles default to a legacy code page (cp1252 or cp437). The output uses `›`, `…`, `·` and box drawing, which would otherwise come out as replacement glyphs.

## Mental model

```text
argv ──► Typer parses argv and checks option types and bounds
           (bad usage: usage panel, exit 2)
     ──► command body: _load(config, **flags)
           field defaults < YAML (-c) < env KUMARU_<SECTION>__<FIELD> < flags
           (a flag that is None was not given)
     ──► subsystem: server.app.run | ChatAgent | run_training | run_eval | bench runner
     ──► a KumaruError anywhere ──► main(): "error: <message>" [+ "hint: <hint>"], exit 1
```

**Invariants:**

1. **Only `None` means "not given".** Every override flag defaults to `None`, and only `None` values are dropped: by `_build_overrides` for config flags, and by `GenerationParams.merged` for `chat`'s per-message flags. So `--no-think`, `--thinking-budget 0` and `-t 0` are real values that override the profile.
2. **Without `-c`, no YAML is read.** You get the field defaults: backend `transformers` and model `Qwen/Qwen2.5-7B-Instruct`, a multi-gigabyte download on first use. `configs\default.yaml` is an annotated reference, not an implicit layer. With `-c NAME`, a bare name (no folder and no `.yaml` or `.yml` suffix) is always the profile `configs\NAME.yaml`, even when the current folder holds a file called `NAME`. Running Kumaru inside a cloned project is normal with `--workspace`, and such a file could otherwise set `base_url` or the llama-server binary before anything asks you. A path, such as `.\my.yaml`, is read as given.
3. **One mapping turns flags into config paths for all commands** (`_build_overrides`):

| Flag (parameter) | Config path | Offered by |
|---|---|---|
| `-b, --backend` (`backend`) | `backend.name` | `serve`, `chat`, `config`, `eval`, `bench run` |
| `-m, --model` (`model`) | `backend.model` | `serve`, `chat`, `config`, `eval`, `bench run` |
| `--device` (`device`) | `backend.device` | `serve`, `chat` |
| `--adapter` (`adapter`) | `backend.adapter_path` | `serve`, `chat`, `eval`, `bench run` |
| `--context` (`context`) | `backend.max_context_length` | `serve`, `chat`, `bench run` |
| `--host`, `-p, --port` | `server.host`, `server.port` | `serve` |
| `--log-level` (`log_level`) | `telemetry.log_level` | `serve` |
| `temperature`, `max_tokens`, `thinking`, `thinking_budget` | `generation.*` | Mapped, but unused: `chat` sends these flags with each message instead. |
| `dtype`, `base_url`, `load_in_4bit` | `backend.*` | Mapped, but no command has the flag yet. |

   `chat` and `bench run` both have `-t`, `--max-tokens`, `--think/--no-think` and `--thinking-budget`, but neither writes them into the config. `chat` sends them with each message (invariant 4), and `bench run` puts them into `BenchSettings` (see [Benchmarks](#benchmarks-kumaru-bench)).

   `chat --workspace` is the one config flag outside this mapping. It adds the project-folder tools to the `agent.enabled_tools` that the profile already resolved, which a nested override would replace. So `chat` sets `agent.workspace.root` and extends `agent.enabled_tools` on the loaded config.

4. **On `chat`, flags beat both presets.**
   - `ChatAgent.resolve_params` merges `generation` < `thinking_mode` < per-request overrides. `thinking_mode` applies only when thinking resolves on.
   - `chat` passes its four generation flags as per-request overrides, `agent.stream(history, **overrides)`, the way the UI sends its settings with each message. So a flag wins even while thinking is on. Verified with the qwen36 profile on the echo backend, which reports what it received:

   ```powershell
   kumaru chat -c qwen36 -b echo --think -q "hi"
   # Context: 2 message(s), ~218 prompt tokens, max_tokens=32768, temperature=1.0.
   kumaru chat -c qwen36 -b echo --think -t 0.3 --max-tokens 100 -q "hi"
   # Context: 2 message(s), ~218 prompt tokens, max_tokens=100, temperature=0.3.
   ```

   - **Only the flags you give override.** The rest of the preset still applies, e.g. `thinking_mode`'s `top_p` and `presence_penalty`.
   - **`bench run` passes its flags the same way,** through `BenchSettings`, so they win there too.

5. **Only `chat` can approve.** It is the one command that passes an approver (`_approve`) to `ChatAgent`. `serve`, `eval` and `bench run` build theirs without one. There, `ToolBelt` withholds every tool that changes something or works in the project folder, and logs `not offering <names>: they need approval, and nobody here can give it`. Changes and commands are never made unasked.

## Public API

The signatures show the CLI defaults. In Python, the real defaults are `typer.Option(...)` and `typer.Argument(...)` objects, so drive commands through `app` (or `CliRunner`), not by calling the functions directly.

| Name | Signature | What it's for |
|---|---|---|
| `app` | `typer.Typer(name="kumaru", help="Local-first, scalable LLM agent.", no_args_is_help=True, add_completion=False)` | The root command. It has no `--version` and no shell-completion options. |
| `bench_app` | `typer.Typer(help="Score models on public benchmarks; compare runs with significance tests.", no_args_is_help=True)` | Mounted as `kumaru bench` by `app.add_typer(bench_app, name="bench")`. |
| `distill_app` | `typer.Typer(help="Build training data from a teacher model's graded answers (knowledge distillation).", no_args_is_help=True)` | Mounted as `kumaru distill`. |
| `console` | `Console()` | Module-level Rich console on stdout. All command output goes here, including the `error:` and `hint:` lines. Log records go to stderr. |
| `CONTEXT_HELP` | `str` | The help text shared by every `--context`. |
| `serve` | `serve(config: str = None, backend: str = None, model: str = None, device: str = None, adapter: str = None, context: int = None, host: str = None, port: int = None, log_level: str = None) -> None` | `kumaru serve`: prints the banner, then calls `kumaru.server.app.run(cfg)`, which blocks. |
| `chat` | `chat(config: str = None, backend: str = None, model: str = None, device: str = None, adapter: str = None, context: int = None, temperature: float = None, max_tokens: int = None, think: bool = None, thinking_budget: int = None, prompt: str = None, workspace: str = None) -> None` | `kumaru chat`: builds the config from the backend flags, then runs `asyncio.run(_chat_loop(cfg, prompt, overrides))` inside `_exit_cleanly_on_signals()`. `overrides` holds the four generation flags. `workspace` sets `agent.workspace.root` and adds `WORKSPACE_TOOLS` to `agent.enabled_tools`. |
| `info` | `info() -> None` | `kumaru info`: versions, registered components, profiles, optional packages and CUDA devices. |
| `show_config` | `show_config(config: str = None, backend: str = None, model: str = None) -> None` | `kumaru config`: prints `Panel(cfg.dump_yaml(), title="resolved configuration")`. |
| `learn` | `learn(topic: str, source: Path, version: str, output: Path) -> None` | `kumaru learn TOPIC`. When `source` is a folder, it calls `_learn_folder`. Otherwise it reads `source` (`read_local_text`), builds a `KnowledgePack` whose `source` is the file's name, and writes it with `save_pack`, which never overwrites. Invalid input raises `ConfigError` `Cannot create knowledge pack <output>: <field>: <reason>`. See [knowledge.md](agent/knowledge.md). |
| `_learn_folder` | `_learn_folder(topic: str, folder: Path, version: str, output: Path) -> None` | Private. Runs `build_index` under a status line that counts files (`Indexing 120/492 files...`), then prints the index's path, its file and passage counts and size, the files with no new text, up to three skipped files per reason and the ignored file types. See [doc_index.md](agent/doc_index.md). |
| `train` | `train(config: str, resume: str = None, dry_run: bool = False) -> None` | `kumaru train CONFIG`: calls `run_training(Path(config), resume=resume, dry_run=dry_run)`. |
| `evaluate` | `evaluate(config: str = None, backend: str = None, model: str = None, adapter: str = None, suite: str = "smoke", output: str = None) -> None` | `kumaru eval`: `asyncio.run(run_eval(cfg, suite=suite, output=output))`. |
| `bench_list` | `bench_list() -> None` | `kumaru bench list`. |
| `bench_run` | `bench_run(benchmarks: list[str], config: str = None, backend: str = None, model: str = None, adapter: str = None, context: int = None, limit: int = 100, full: bool = False, seed: int = 1234, think: bool = None, thinking_budget: int = None, max_tokens: int = None, temperature: float = None, concurrency: int = None, label: str = None, out: str = None, allow_code_exec: bool = False, sandbox: str = "none", use_agent_prompt: bool = False) -> None` | `kumaru bench run`: refuses a `--sandbox` outside `codeexec.SANDBOXES` before loading the profile, builds `BenchSettings` (`limit=None` when `--full` is given), then runs `BenchRunner(cfg, settings, profile=config, console=console).run(benchmarks)`. After the summaries it prints `not run: <name>: <why>` for each suite member the runner skipped. |
| `bench_compare` | `bench_compare(baseline: str, candidate: str, examples: int = 0) -> None` | `kumaru bench compare A B`: `render_comparisons(...)`, plus `render_flips(...)` when `-e` is above 0. |
| `bench_report` | `bench_report(run: str) -> None` | `kumaru bench report RUN`: `render_summaries(console, load_summaries(Path(run)), details=True)`. |
| `distill_build` | `distill_build(runs: list[str], out: str, max_tokens: int = 1024, per_question: int = 4, val_fraction: float = 0.05, seed: int = 42, check_arithmetic: bool = True, eval_set: list[str] = ["gsm8k"], decontaminate: bool = True, allow_thinking: bool = False, force: bool = False) -> None` | `kumaru distill build RUNS... -o DIR`: `render(console, build_from_runs(runs, out, DistillSettings(...), force=force), Path(out))`. |
| `distill_gold` | `distill_gold(out: str, match: str = None, val_fraction: float = 0.05, seed: int = 42, check_arithmetic: bool = True, eval_set: list[str] = ["gsm8k"], decontaminate: bool = True, force: bool = False) -> None` | `kumaru distill gold -o DIR`: the same with `build_gold(out, settings, match=match, force=force)`. |
| `DISTILL_OUT_HELP`, `VAL_FRACTION_HELP`, `SPLIT_SEED_HELP`, `ARITHMETIC_HELP`, `EVAL_SET_HELP`, `DECONTAMINATE_HELP`, `FORCE_HELP` | `str` | Help texts shared by `distill build` and `distill gold`, so the two commands describe a shared flag identically. |
| `main` | `main() -> None` | Console-script entry point. Calls `ensure_utf8_streams()`, then `app()`; a `KumaruError` prints `error:`/`hint:` and raises `SystemExit(1)`. Its `KeyboardInterrupt` branch (`interrupted`, exit 130) does not run in practice; see [Exit codes](#errors-and-exit-codes). |
| `_build_overrides` | `_build_overrides(**kwargs: Any) -> dict[str, Any]` | Private, and the one place flags become config paths. Drops `None` values and unknown keys. `tests\test_cli.py` tests it directly. |
| `_load` | `_load(config: str \| None, **flags: Any)` | Private. Returns the `KumaruConfig` from `load_config(config, overrides=_build_overrides(**flags))`. |
| `_chat_loop` | `async _chat_loop(cfg: Any, single_prompt: str \| None, overrides: dict[str, Any]) -> None` | Private. The terminal REPL behind `chat`. Builds `ChatAgent(backend, cfg, approver=_approve)` before `backend.load()`, so a bad knowledge pack, documentation index, tool name or project folder, or a backend that cannot call tools, fails before the model loads. Passes `overrides` to every `agent.stream` call, prints tool calls and results as they happen, and adds the turn's whole transcript to the history. |
| `_approve` | `async _approve(request: Any) -> bool` | Private. The approver; `request` is an `ApprovalRequest`. Prints `request.summary` in a yellow panel titled with the tool name, with diff lines coloured and the `$` line of a command in bold, then discards type-ahead (`_discard_type_ahead`) and asks `Allow this? [y/N]`. Only `y` or `yes` (any case) approves; anything else, end of input or Ctrl+C is no. |
| `_discard_type_ahead` | `_discard_type_ahead() -> None` | Private. Drops the keys typed before the approval prompt appeared, so that a `y` and Enter typed in reply to the model's prose cannot approve a change you have not seen. Only an interactive terminal is flushed (`msvcrt` on Windows, `termios.tcflush` elsewhere); piped answers are kept for the prompts. |
| `_visible` | `_visible(text: str) -> str` | Private. Spells out control and invisible formatting characters, and every line break except `\n` (Unicode categories Cc, Cf, Zl and Zp, except tab and newline), as `\xNN` or `\uNNNN`, so text from the model cannot hide or disguise lines of the approval panel. |
| `_printable` | `_printable(text: str) -> str` | Private. Streamed thoughts and answers go through it before printing. It spells out control characters (category Cc, except tab and newline) as `\xNN` and drops carriage returns. Then an escape sequence the model repeats from a file or from a command's output cannot clear the screen or redraw earlier lines, including the next approval panel. Unlike `_visible`, it keeps formatting characters such as the joiners inside emoji, since a reply is read, not approved. |
| `_exit_cleanly_on_signals` | `_exit_cleanly_on_signals() -> Iterator[None]` (a context manager) | Private. On Linux and macOS it makes SIGHUP (the terminal closed) and SIGTERM (a `kill`) raise `SystemExit(128 + signal)` while `chat` runs, instead of ending Python on the spot. The cleanup that stops a running command then runs first. It changes only signals still at their default, so one ignored with `nohup` or handled by someone else stays as it was. The old handlers come back when it ends. On Windows it does nothing: the Job Object ends commands however Kumaru exits. |

## How to use it

The examples are PowerShell, run from the repo root with the venv active; otherwise call `.\.venv\Scripts\kumaru.exe`. Rich wraps output at the terminal width, so set `$env:COLUMNS = '120'` when you capture output.

### Commands

| Command | Help text | Loads a model? |
|---|---|---|
| `kumaru serve` | Start the HTTP API and the web UI. | Yes, in the background after the server is up. |
| `kumaru chat` | Chat in the terminal, with no server in between. | Yes, before the first prompt. |
| `kumaru info` | Show versions, hardware, and which optional components are installed. | No. It imports torch and queries CUDA, so it is slow, especially on a cold start. |
| `kumaru config` | Print the fully merged configuration and exit. | No. |
| `kumaru learn TOPIC` | Learn from local references without downloads, execution, or fine-tuning. A file becomes a knowledge pack; a folder becomes a documentation index. | No. |
| `kumaru train CONFIG` | Fine-tune (LoRA) or pretrain a model. See docs/TRAINING.md. | Yes. |
| `kumaru eval` | Measure latency, throughput, and answer quality. | Yes. |
| `kumaru bench list\|run\|report\|compare` | Score models on public benchmarks; compare runs with significance tests. | `run` only. |
| `kumaru distill build\|gold` | Build training data from a teacher model's graded answers (knowledge distillation). | No. It reads GSM8K from the Hugging Face cache, downloading it on first use. |

### Options

Quoted text is the option's `--help` string. A dash means the option has none.

| Option | Commands | Sets | Bounds and notes |
|---|---|---|---|
| `-c, --config` | `serve`, `chat`, `config`, `eval`, `bench run` | YAML layer | `serve` and `bench run`: "YAML file or profile name." A bare name means `configs\<name>.yaml`, never a file in the current folder; a value with a folder or a `.yaml`/`.yml` suffix is a path. |
| `-b, --backend` | same five | `backend.name` | `serve`: "echo \| llamacpp_server \| transformers \| llamacpp \| openai_compat". |
| `-m, --model` | same five | `backend.model` | `serve`: "Model id or path." |
| `--device` | `serve`, `chat` | `backend.device` | `serve`: "auto \| cuda \| cpu \| mps". |
| `--adapter` | `serve`, `chat`, `eval`, `bench run` | `backend.adapter_path` | `serve`: "Path to a trained LoRA adapter." Only the `transformers` backend can attach one. The others refuse it when the model loads: `error: The <name> backend cannot attach a LoRA adapter ...`. `serve` reports that in `/healthz` and the UI instead of exiting. |
| `--context` | `serve`, `chat`, `bench run` | `backend.max_context_length` | `x>=512`. Help is `CONTEXT_HELP`: with llama-server the context sizes the KV cache, so lowering it frees VRAM. |
| `--host` | `serve` | `server.host` | Default `127.0.0.1`. |
| `-p, --port` | `serve` | `server.port` | Default `8080`. |
| `--log-level` | `serve` | `telemetry.log_level` | `DEBUG`, `INFO`, `WARNING`, `ERROR` or `CRITICAL`, in any case; `WARN` and `FATAL` are aliases. Validated with the config, so a typo exits 1 before the banner. |
| `-t, --temperature` | `chat` | per message | `0.0<=x<=2.0`. |
| `--max-tokens` | `chat` | per message | `1<=x<=131072`. Trimmed to the room the prompt leaves in the window. |
| `--think/--no-think` | `chat` | per message | "Force reasoning on or off (default: profile)." |
| `--thinking-budget` | `chat` | per message | "Max thinking tokens before answering." `x>=0`. |
| `-q, --prompt` | `chat` | — | "Answer one question and exit." |
| `-w, --workspace` | `chat` | `agent.workspace.root`, `agent.enabled_tools` | "Project folder the agent may work in: list, read, create and edit files there, and run allowed programs (agent.workspace.allowed_commands). It shows you each change and command first, and does it only if you answer y." The folder must exist. It may not be a whole drive, contain your home folder, or lie inside a `.git` folder. |
| `-s, --suite` | `eval` | — | Default `smoke`. "smoke \| speed \| jsonl file path". |
| `-o, --output` | `eval`, `learn` | — | `eval`: "Write results to this JSON file." `learn`: "New knowledge pack JSON file; for a folder, a new index file such as godot.sqlite." Required by `learn`; its folder must exist, and an existing file is never replaced. |
| `TOPIC` (argument) | `learn` | — | Required. "Language, framework or platform to learn." |
| `--source` | `learn` | — | Required. "Reviewed UTF-8 reference file, or a documentation folder to index (Markdown, MDX, reStructuredText, HTML or plain text)." A file may be at most 128,000 bytes. |
| `--version` | `learn` | — | Required. "Language/platform version." |
| `CONFIG` (argument) | `train` | — | Required. "Training config YAML (see configs/train/)." |
| `--resume` | `train` | — | "Checkpoint directory to resume from." |
| `--dry-run` | `train` | — | "Validate config and data, then stop." |

### See what will run: `kumaru config`

```powershell
kumaru config -c qwen36 -m my-model
$env:KUMARU_BACKEND__MAX_CONTEXT_LENGTH = '32768'; kumaru config -c test; Remove-Item Env:KUMARU_BACKEND__MAX_CONTEXT_LENGTH
```

- **Output.** A Rich panel titled `resolved configuration` that holds the YAML of every section after all four layers.
- **Flags.** `config` takes only `-c`, `-b` and `-m`. To preview any other override, set the matching `KUMARU_<SECTION>__<FIELD>` variable, which sits one layer below flags. The second line above shows `max_context_length: 32768`.
- **Secrets.** The dump is not redacted: `api_key` values print in plain text.

### Chat in the terminal: `kumaru chat`

**Ask one question and exit.** This example uses the echo backend from `configs\test.yaml` and shows stdout only:

```text
PS> kumaru chat -c test -q "hi"
echo on cpu (none), 8,192 token context

[echo:8f434346] You said: hi
Context: 2 message(s), ~218 prompt tokens, max_tokens=256, temperature=0.0.

218 prompt + 18 completion tokens
```

- **Log lines.** Records such as `backend 'echo' ready: echo (0.0s)` and `streamed: 18 chunks in 282ms (ttft 21ms)` go to stderr at INFO. `$env:KUMARU_LOG_LEVEL = 'WARNING'` silences them.
- **Why the environment variable.** `chat` configures logging from `KUMARU_LOG_*`, not from `telemetry.*` (`docs\howto\src\kumaru\core\logging.md`).

**Reasoning.** `--think` prints the thought dimmed under `thinking…`, then the answer:

```text
PS> kumaru chat -c test --think -q "hi"
echo on cpu (none), 8,192 token context

thinking…
The user asked: hi. Echo cannot reason; this text only exercises the reasoning channel.

[echo:8f434346] You said: hi
Context: 2 message(s), ~218 prompt tokens, max_tokens=256, temperature=0.0.

218 prompt + 29 completion tokens
```

- `--think --thinking-budget 3` cuts the echo thought to `The user asked: hi. Echo` (21 completion tokens).
- **Only the answer is appended to the history** (`Message.assistant("".join(parts))`), the same rule the server and the UI follow.
  - Chat templates are trained on histories whose earlier assistant turns hold only final answers.
  - Old thoughts would also spend the context budget on text the model no longer needs.

**REPL.** Lines go through one by one:
- an empty line is skipped;
- `/reset` empties the history;
- `/exit` or `/quit` leaves with exit 0;
- so does end of input: the end of piped input, or Ctrl+Z then Enter on Windows.

Piped input is not echoed, so the prompts run together:

```text
PS> "hi", "", "/reset", "/exit" | kumaru chat -c test
echo on cpu (none), 8,192 token context
Type your message. /reset clears history, /exit quits.

you › 
[echo:8f434346] You said: hi
Context: 2 message(s), ~218 prompt tokens, max_tokens=256, temperature=0.0.

218 prompt + 18 completion tokens
you › you › history cleared
you › 
```

**Ending a session:**
- **Ctrl+C, at the prompt or during an answer, exits 130 with no message**, and the conversation is gone.
  - `asyncio.run` installs its own SIGINT handler, which cancels the chat task.
  - The cancellation reaches Typer as a `KeyboardInterrupt`, and Typer turns it into a silent `Exit(130)`.
  - Use `/exit` for a clean 0.
- **On Linux and macOS, closing the terminal or a `kill` exits 129 or 143** with no message, after the same cleanup as Ctrl+C. A running command is stopped first, with everything it started (on macOS, except programs that detached themselves). `kill -9` cannot be caught: it leaves a running command to finish on its own.
- **A `KumaruError` during a turn ends the whole session** with `error:`/`hint:` and exit 1. For example, `agent.memory_policy: none` refuses a history over the budget. The loop has no per-turn recovery, and the history is not saved.

### Let it work in a project folder: `kumaru chat --workspace`

`--workspace DIR` (`-w`) offers the model five tools that work only inside `DIR`: `list_files`, `read_file`, `write_file`, `edit_file` and `run_command`. Listing and reading run at once. Every change and every command first shows you exactly what will happen, and runs only if you answer `y`:

- **a new file:** its path, line count and whole content, as a diff against nothing;
- **an edit:** a unified diff;
- **a command:** the arguments as they will run, the folder, the program the name resolved to, and the time limit.

With a real model you ask in plain words, and the model decides which calls to make:

```powershell
New-Item -ItemType Directory C:\projects\hello-node -Force | Out-Null
kumaru chat -c qwen36 --workspace C:\projects\hello-node
# you › Create a hello-world Node script and run it.
```

The echo backend in `configs\test.yaml` takes the calls as `/tool` lines instead ([`echo.md`](backends/echo.md)), so you can try the whole flow offline. This session approves a new file, refuses an edit and approves a command. It was captured with `$env:COLUMNS = '100'` and piped answers, and is shown as a terminal shows it, with each typed line after its prompt:

```text
echo on cpu (none), 8,192 token context
tools: list_files, read_file, write_file, edit_file, run_command
project folder: C:\Users\you\projects\scratch (every change and command asks you first)
Type your message. /reset clears history, /exit quits.

you › /tool write_file {"path": "hello.py", "content": "print(\"hello\")\n"}

» write_file({"path": "hello.py", "content": "print(\"hello\")\n"})
┌─ write_file ────────────────────────────────────────────────────────────────────────────────────┐
│ Create hello.py (1 line)                                                                        │
│ --- /dev/null                                                                                   │
│ +++ b/hello.py                                                                                  │
│ @@ -0,0 +1 @@                                                                                   │
│ +print("hello")                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────────────────────┘
Allow this? [y/N] y
« Created hello.py (1 line, 15 B).
[echo:33a26426] You said: /tool write_file {"path": "hello.py", "content": "print(\"hello\")\n"}
Context: 4 message(s), ~298 prompt tokens, max_tokens=256, temperature=0.0.
Tool write_file: Created hello.py (1 line, 15 B).

535 prompt + 38 completion tokens
you › /tool edit_file {"path": "hello.py", "old_text": "hello", "new_text": "hello, world"}

» edit_file({"path": "hello.py", "old_text": "hello", "new_text": "hello, world"})
┌─ edit_file ─────────────────────────────────────────────────────────────────────────────────────┐
│ Edit hello.py                                                                                   │
│ --- a/hello.py                                                                                  │
│ +++ b/hello.py                                                                                  │
│ @@ -1 +1 @@                                                                                     │
│ -print("hello")                                                                                 │
│ +print("hello, world")                                                                          │
└─────────────────────────────────────────────────────────────────────────────────────────────────┘
Allow this? [y/N] n
« DENIED: the user declined this edit_file call. Nothing was done. Do not retry it unchanged; ask
the user what they want instead.
[echo:e5a9e87a] You said: /tool edit_file {"path": "hello.py", "old_text": "hello", "new_text": "hello, world"}
Context: 8 message(s), ~480 prompt tokens, max_tokens=256, temperature=0.0.
Tool edit_file: DENIED: the user declined this edit_file call. Nothing was done. Do not retry it unchanged; ask the user what they want instead.

869 prompt + 57 completion tokens
you › /tool run_command {"command": ["python", "hello.py"]}

» run_command({"command": ["python", "hello.py"]})
┌─ run_command ───────────────────────────────────────────────────────────────────────────────────┐
│ $ python hello.py                                                                               │
│ folder:  .  (C:\Users\you\projects\scratch)                                                     │
│ program: C:\Users\you\AppData\Local\Programs\Python\Python312\python.exe                        │
│ stops after 300s, with every program it starts                                                  │
└─────────────────────────────────────────────────────────────────────────────────────────────────┘
Allow this? [y/N] y
« Command finished with exit code 0 in 0.1s.
[echo:256a4c1c] You said: /tool run_command {"command": ["python", "hello.py"]}
Context: 12 message(s), ~659 prompt tokens, max_tokens=256, temperature=0.0.
Tool run_command: Command finished with exit code 0 in 0.1s.

1252 prompt + 38 completion tokens
you › /exit
```

- **Reading the screen.** Each call prints in cyan as `» name(arguments)`, cut at 120 characters. Each result prints its first line as `« ...`, cut at 200: red for `ERROR`, yellow for `DENIED` and `NOT RUN`, dimmed otherwise. The model gets the whole result, up to `agent.workspace.max_output_chars` (the program's output follows the first line).
- **Answering.** Only `y` or `yes` approves, in any case. Enter alone, anything else, end of input and Ctrl+C all mean no. A refusal reaches the model as `DENIED: ...`, which tells it not to retry the call unchanged. At a terminal, keys typed before the panel appeared are discarded, so a `y` you typed in reply to the model's "Shall I go ahead?" cannot approve a change you have not seen; type your answer after the panel. Piped answers are not discarded.
- **The panel cannot be disguised.** Text from the model goes through `_visible` before it is printed, so a control character such as an escape sequence or a right-to-left override shows as `\x1b` or `\u202e` instead of rewriting the panel, and so does every line break but `\n`, such as `\r` or U+2028. On the `$` line, which is printed in bold, a line break inside an argument shows as `\n` too, and a `note:` line says so ([`commands.md`](agent/commands.md)), so a command cannot draw a fake `program:` line. Streamed thoughts and answers go through `_printable`, which spells out control characters the same way, so text printed before the panel cannot redraw or hide it either.
- **The token line adds up every round of the turn**: above, the round that made the call plus the round that answered.
- **History.** Calls and results stay in the history until `/reset`, so a later message can refer to what was done. They count against `agent.max_history_tokens` like any other message.
- **Rounds.** One message may make up to `agent.max_tool_iterations` rounds of calls (10). Calls beyond that are not run, and the reply ends with `[Stopped after 10 tool rounds (agent.max_tool_iterations). Reply 'continue' to let it keep going.]`.
- **`-q` asks too.** At a terminal it asks like the REPL. Piped, it reads the answers from the following input lines, and at the end of input every change and command is refused.
- **The backend must call tools.** `llamacpp_server` (`-c qwen36`), `openai_compat` and `echo` can. `transformers` and `llamacpp` stop before the model loads, with `error: The <name> backend cannot call tools, ...`.
- **Which programs.** `agent.workspace.allowed_commands`: `python`, `node`, `npm`, `npx` and `dotnet` by default. They run without a shell, with a time limit, an output cap and a scrubbed environment ([`commands.md`](agent/commands.md)). For which paths the file tools refuse, see [`workspace.md`](agent/workspace.md).
- **Approval is the safety boundary for commands.** The folder confines the file tools, but an approved program runs with your permissions and can reach anything you can: `python -c ...`, an npm script, or a file the model wrote earlier. Read each command, and the files it will run, before you answer `y`.

### Serve the API and UI: `kumaru serve`

```powershell
kumaru serve -c qwen36 --context 65536
kumaru serve -c test -p 9000      # echo backend: checks the server and the UI with no model
```

The banner below was captured with `run()` stubbed out. The real command then blocks in uvicorn:

```text
┌─────────────────────────────────┐
│ Kumaru v0.2.0                   │
│                                 │
│ backend  echo                   │
│ model    echo                   │
│ device   auto                   │
│                                 │
│ UI   http://127.0.0.1:8080      │
│ API  http://127.0.0.1:8080/v1   │
│ Docs http://127.0.0.1:8080/docs │
└─────────────────────────────────┘
Model loads in the background; the UI shows progress.
```

- **Version.** The banner shows `kumaru.__version__` (`0.2.0`).
- **URLs.** They come from `server.host`, so `--host 0.0.0.0` prints `http://0.0.0.0:<port>`. That is a bind address, not one you can browse to.
- **No thinking or sampling flags.** Thinking is chosen per request, by the UI toggle or `"thinking": true` in the body. Server-wide defaults come from YAML or env, e.g. `$env:KUMARU_GENERATION__THINKING = 'true'`.
- **Logging.** `serve` follows `telemetry.log_level` (and `--log-level`), not `KUMARU_LOG_LEVEL`.

### Inspect the install: `kumaru info`

One table:
- `python`;
- the registered `backends`, `tools` and `benchmarks`, and the `benchmark suites` (`multipl_e`);
- the `profiles`;
- the version of each optional package (`torch`, `transformers`, `peft`, `trl`, `datasets`, `bitsandbytes`, `llama_cpp`, `langdetect`, `nltk`), or `not installed`;
- the CUDA state, as one of:
  - one `cuda:<i>` row per GPU: `<name> · <GB> GB · sm_<major><minor> · bf16=yes|no`;
  - `cuda  unavailable (CPU only)`;
  - `cuda  torch not installed`.

This is the first thing to paste into a bug report. Unlike `kumaru config`, it contains no secrets.

### Train: `kumaru train`

```powershell
kumaru train configs\train\lora-small.yaml --dry-run
kumaru train configs\train\lora-small.yaml
kumaru train configs\train\lora-small.yaml --resume <checkpoint-dir>
```

- **`CONFIG` is a positional path to a training YAML,** not a `-c` profile: training configs use their own schema (`src\kumaru\train\config.py`). The examples are `lora-qwen.yaml`, `lora-small.yaml` and `pretrain-nano.yaml` in `configs\train\`.
- **Trying the result.** A LoRA run ends by logging `try it:  kumaru chat --model <base> --adapter <dir>`. That works because no `-c` means the `transformers` backend, the only one that attaches adapters.
- Details: `docs\TRAINING.md`.

### Quick check: `kumaru eval`

```powershell
kumaru eval -c test
kumaru eval -c qwen36 -s speed -o eval-speed.json
kumaru eval -c qwen36 -s my_cases.jsonl
```

- **Output.** One table row per case, with columns tag, result, tok/s, ttft and note, then a summary line. On echo (6 smoke cases): `1/6 passed (17%) · median 58.84 tok/s · median ttft 19ms · 3.0s total`.
- **`eval` exits 0 even when cases fail.** To gate CI, read the JSON written by `-o`.
- **An unknown suite** gives the hint `Use 'smoke', 'speed', or a path to a JSONL file of eval cases.`
- Details: `docs\howto\src\kumaru\eval\harness.md`.

### Benchmarks: `kumaru bench`

- **Subcommands:**
  - `bench list` prints the benchmarks: `golden`, `gsm8k`, `gsm8k_train`, `humaneval`, `ifeval`, `mmlu_pro` and the eight `multipl_e_<lang>`, plus the `multipl_e` suite, which runs all eight. Any golden-set `.jsonl` file also works as a benchmark. `gsm8k_train` is GSM8K's training split: teacher data for `kumaru distill`, not a score.
  - `bench report RUN` shows one run in detail.
  - `bench compare BASELINE CANDIDATE [-e N]` answers "did B really beat A?" with paired differences, confidence intervals and p-values.
- **Records.** `bench run` saves every answer.
- **Further reading:**
  - methodology and results: `docs\BENCHMARKS.md`;
  - sampling, fingerprints, concurrency and the record format: `docs\howto\src\kumaru\eval\benchmarks\runner.md`.

`bench run BENCHMARKS...`. The argument help reads "Names from `kumaru bench list`, or golden-set .jsonl paths." A suite name (`multipl_e`) stands for its members.

| Option | Default | Help text and notes |
|---|---|---|
| `-c`, `-b`, `-m`, `--adapter`, `--context` | — | Config overrides, as above. `--context` is `x>=512`. |
| `-n, --limit` | `100` (`x>=1`) | "Questions per benchmark (a fixed, stratified sample)." |
| `--full` | off | "Every question; overrides --limit." |
| `--seed` | `1234` | "Picks the questions and each question's sampling seed." |
| `--think/--no-think` | profile | "Force reasoning on or off (default: profile)." A per-request override, so it beats `thinking_mode`. |
| `--thinking-budget` | none (`x>=0`) | "Max thinking tokens per question." Also a per-request override. |
| `--max-tokens` | `8192`; more with thinking (`1<=x<=131072`) | "Output cap per question (default: 8192; thinking adds --thinking-budget, or uses the profile's max_tokens without one)." |
| `-t, --temperature` | profile (`0.0<=x<=2.0`) | "0 = greedy (default: the profile's preset)." Also a per-request override. |
| `-j, --concurrency` | backend (`x>=1`) | "Questions in flight (default: what the backend can batch)." That is 1, except on `llamacpp_server` (its slots) and on `transformers` at temperature 0 (`backend.max_batch_size`). |
| `--label` | `<profile or model>-<thinking mode>` | "Run name (default: <profile or model>-<thinking mode>, e.g. qwen36-think4096)." The mode is `think`, `think<budget>`, `nothink` or `default`. Without a profile, the model's file or repo name is used: `-m D:\kumaru\models\Qwen3-8B-Q4_K_M.gguf --no-think` gives `Qwen3-8B-Q4_K_M-nothink`. |
| `-o, --out` | `runs\bench\<YYYYMMDD-HHMMSS>-<label>` | "This run's own folder (default runs/bench/<timestamp>-<label>). Point at an earlier run to resume it." |
| `--allow-code-exec` | off | "Allow benchmarks that execute model-written code (humaneval, multipl_e)." |
| `--sandbox` | `none` | "Where code benchmarks run model-written programs: none (this machine: a temp folder, time and memory limits) or docker (a throwaway container with no network; build its image first, see docs/BENCHMARKS.md)." Any other value: `error: Unknown sandbox '<value>'.` with `hint: Use one of: none, docker.`, before the profile loads. |
| `--use-agent-prompt` | off | "Prepend Kumaru's system prompt, as chat does." |

**Recipes:**

```powershell
# Same questions and seeds, thinking off vs on
kumaru bench run gsm8k ifeval -c qwen36 --no-think -n 50
kumaru bench run gsm8k ifeval -c qwen36 --think --thinking-budget 4096 -n 50

# Is the difference real? Show up to 5 questions per benchmark that flipped
kumaru bench compare runs\bench\<stamp>-qwen36-nothink runs\bench\<stamp>-qwen36-think4096 -e 5

# Inspect one run; resume an interrupted run (same settings, same folder)
kumaru bench report runs\bench\<stamp>-qwen36-think4096
kumaru bench run gsm8k ifeval -c qwen36 --think --thinking-budget 4096 -n 50 -o runs\bench\<stamp>-qwen36-think4096

# HumanEval in eight languages; languages without a compiler here are skipped, not scored 0
kumaru bench run multipl_e -c qwen36 --no-think --allow-code-exec
```

- **Resuming.**
  - A rerun with the same `--out` reuses the finished answers.
  - A run whose fingerprint differs (model, decoding parameters, seed, prompt mode, documentation indexes, sandbox, compiler version or memory cap) is refused before anything is written.
  - On Ctrl+C the runner prints `Stopped. Finished answers are saved; resume with --out <dir>`.
- **`humaneval` and `multipl_e*`** refuse to run without `--allow-code-exec`. Read `docs\howto\src\kumaru\eval\benchmarks\multipl_e.md` for the toolchains each language needs.

### Distillation data: `kumaru distill`

```powershell
kumaru distill build runs\distill\teacher-qwen36 -o runs\distill\data-teacher
kumaru distill gold -o runs\distill\data-gold --match runs\distill\data-teacher
```

- **`build RUNS...`** turns teacher runs of `kumaru bench run gsm8k_train` into `train.jsonl`, `val.jsonl` and `manifest.json`. It keeps only answers that are correct, complete and clean, then prints what each check dropped.
- **`gold`** builds the control arm: GSM8K's human-written solutions to the same questions.
- **Both refuse evaluation data.** A run of `gsm8k` (the test split) or of a golden set is refused whole, and near-copies of test questions are dropped. `--no-decontaminate` is for smoke tests only.
- Every flag, the report and the whole experiment: `docs\howto\src\kumaru\train\distill.md`. The training recipes: `docs\howto\configs\distill-gsm8k.yaml.md`.

### Errors and exit codes

```text
PS> kumaru chat -c test -b nope -q hi
error: Unknown backend: 'nope'.
hint: Available backends: echo, llamacpp, llamacpp_server, openai_compat, transformers
```

| Exit | When | What prints |
|---|---|---|
| 0 | Success; `--help`; REPL `/exit`, `/quit` or end of input; `eval` with failing cases. | Normal output. |
| 1 | Any `KumaruError`: config, registry, backend, a missing training file, a bench refusal. | `error: <message>`, then `hint: <hint>` when the error has one. Both print literally: brackets such as `".[gguf]"` or Pydantic's `[type=…]` are escaped, not read as Rich markup. |
| 1 | Any other exception. | A traceback; such an exception is a bug and should have been a `KumaruError`. An `EOFError` that escapes a command prints `Aborted.` to stderr instead. |
| 2 | Usage errors: an unknown command or option, a value of the wrong type or out of bounds, a missing argument. A bare `kumaru` or `kumaru bench` prints the help and also exits 2. | `Usage: ...` and a Rich `Error` panel, e.g. `Invalid value for '--context': 100 is not in the range x>=512.` |
| 130 | Ctrl+C inside any command. | Nothing. Typer converts the `KeyboardInterrupt` to `Exit(130)` before `main()` sees it. |
| 129, 143 | `kumaru chat` on Linux and macOS, when its terminal closes (SIGHUP) or it gets a `kill` (SIGTERM). | Nothing. A running command is stopped first. |

### Script it or test it

```python
from typer.testing import CliRunner
from kumaru.cli import app

result = CliRunner().invoke(app, ["chat", "-c", "test", "-q", "hello"], env={"COLUMNS": "200"})
assert result.exit_code == 0 and "You said: hello" in result.output
```

- **`CliRunner` calls `app`, not `main()`.** A `KumaruError` therefore comes back as `result.exception`, with exit code 1 and no `error:` line.
- **To test the message format,** set `sys.argv` and call `main()`, as `test_errors_become_a_message_a_hint_and_exit_code_1` in `tests\test_cli.py` does.

## How to extend it

### Add a config-override flag (two places)

1. Add the option to the command and pass it to `_load(...)`.
2. Map its parameter name in `_build_overrides`' `mapping`, unless it is already there.

`load_in_4bit` is already mapped to `backend.load_in_4bit`, so exposing it on `serve` takes one edit:

```python
@app.command()
def serve(
    config: str = typer.Option(None, "--config", "-c", help="YAML file or profile name."),
    # ... existing options ...
    load_in_4bit: bool = typer.Option(None, "--load-in-4bit/--no-load-in-4bit",
                                      help="NF4 weights via bitsandbytes (transformers backend)."),
) -> None:
    cfg = _load(config, ..., load_in_4bit=load_in_4bit)
```

- **Default the option to `None`, not `False`.** Otherwise "not given" can't be told apart from "off", and the flag would always overwrite the profile.
- **A name missing from `mapping` is dropped silently.** The flag parses and does nothing.
- **A new sampling flag on `chat`** (say `--top-p`) belongs in `overrides`, not in `_load`. A value written into `generation` ranks below `thinking_mode`; a per-message override beats both presets. Give it Typer bounds (`min=`, `max=`): `GenerationParams.merged` skips validation, so nothing else checks a per-message value.

### Add a command

- **Registration.** Decorate it with `@app.command()`, `@bench_app.command("<name>")` or `@distill_app.command("<name>")`. Accept `-c/--config` plus the overrides it needs, and build the config with `_load`.
- **Imports.** Import the subsystem inside the body. Check with `python -c "import sys, kumaru.cli; print('torch' in sys.modules)"`, which must print `False`.
- **Errors.** For anything the user can fix, raise a `KumaruError` subclass with a `hint=`, and `main()` formats it. Don't print and call `sys.exit` yourself.
- **Async work.** Run it with `asyncio.run(...)` inside the body, as `chat`, `eval` and `bench run` do. Close backends in `finally:` with `await backend.aclose()`.
- **Tests.** Use `CliRunner` against an echo profile, like the `echo_profile` fixture in `tests\test_cli.py` (`delay_s: 0` keeps it fast).

## Failure modes

| Symptom | Likely cause | Fix |
|---|---|---|
| `kumaru --help` takes seconds. | A heavy import moved to module scope, here or in a module `cli.py` imports at the top. | Move it into the command body and re-run the `'torch' in sys.modules` check. |
| `kumaru chat` or `serve` without `-c` starts downloading a 7B model. | No `-c` means field defaults (`transformers`, `Qwen/Qwen2.5-7B-Instruct`). `configs\default.yaml` is never loaded implicitly. | Always pass `-c <profile>`; use `-c test` for an offline check. |
| A flag seems ignored. | It isn't passed to `_load`, it's missing from `mapping`, or the command doesn't offer it (`config` takes only `-c`, `-b` and `-m`). | Add it in both places. Preview with `kumaru config`, using an env var for flags `config` lacks. |
| A thinking answer stops short late in a long chat at a small `--context`. | The reply gets only the room the prompt leaves: `max_tokens` is trimmed to the window minus the prompt, and history may fill up to half the window. | Raise `--context`, `/reset` the chat, or lower `agent.max_history_tokens`. See `docs\howto\src\kumaru\agent\chat.md`. |
| `error: The llamacpp_server backend cannot attach a LoRA adapter (...)`. | Only `transformers` attaches LoRA adapters; every other backend refuses `adapter_path` when it loads, rather than answering with the untuned model. | Drop `-c` (the built-in defaults use `transformers`) and pass the adapter's base model with `-m`. For GGUF, see `docs\TRAINING.md`. |
| A usage panel such as `Invalid value for '--context': 100 is not in the range x>=512.`, exit 2. | Typer checks types and bounds before any config is built. | Fix the value; `kumaru <command> --help` lists the bounds. |
| `error: Invalid configuration: ...`, exit 1. | Pydantic rejected the merged config. Example: `$env:KUMARU_GENERATION__TEMPERATURE = '5'` (the maximum is 2). | Fix the env var or YAML. `kumaru config` shows the merged result. |
| `error: Unknown backend: 'nope'.` or `error: Config file not found: nope`. | A typo in `-b` or `-c`. | The `hint:` line lists the valid backends or profiles. |
| `error: Config file not found: my-profile`, though `.\my-profile` exists. | A bare name always means `configs\my-profile.yaml`, never a file in the current folder. | `-c .\my-profile`, or name the file `my-profile.yaml`. |
| The REPL ends with `error: ...` in the middle of a session. | A `KumaruError` during any turn ends `kumaru chat` with exit 1. | Fix the cause and restart. The history is lost. |
| `kumaru chat -q ""` opens the REPL. | An empty prompt is falsy. | Pass a non-empty question. |
| Ctrl+C ends `kumaru chat` with exit 130 and no output, at the prompt as well as during an answer. | `asyncio.run` turns Ctrl+C into a task cancellation, which Typer maps to a silent `Exit(130)`. `main()`'s `interrupted` branch never runs. | Leave the REPL with `/exit`. Scripts should treat 130 as "interrupted". |
| `error: The project folder <dir> does not exist, or is not a folder.` | `--workspace` never creates the folder. | `New-Item -ItemType Directory <dir>`, then start again. |
| `error: <dir> is, or contains, your home folder; ...`, `... is a whole drive; ...` or `... is inside a .git folder; ...` | The folder would expose far more than one project. | Make a folder for the project, e.g. `C:\projects\my-app`. |
| `error: The transformers backend cannot call tools, and tools are enabled (agent.enabled_tools, or kumaru chat --workspace).` | `--workspace` with a backend that has no tool calling (`transformers` or `llamacpp`). | `-c qwen36` (`llamacpp_server`), or `openai_compat` with a server that parses tool calls. |
| `« ERROR: 'git' is not an allowed program. Allowed: python, node, npm, npx, dotnet (...)` | The program is not in `agent.workspace.allowed_commands`. You are not asked about a call that cannot run. | Add it to the list in your profile YAML, if you want the model to run it. |
| The model prints `<tool_call>{...}` as text instead of making the call. | The server is not parsing tool calls: llama-server started without `--jinja` (attach mode), or vLLM without `--enable-auto-tool-choice`. | See "Let the model work in a project folder" in `docs\howto\src\kumaru\backends\openai_compat.md`. |
| A reply ends with `[Stopped after 10 tool rounds (agent.max_tool_iterations). ...]`. | The model used every round it had for this message. | Reply `continue`, or raise `agent.max_tool_iterations` (up to 100). |
| `Missing argument 'config'.` from `kumaru train`, exit 2. | `train` takes a positional path, not `-c`. | `kumaru train configs\train\lora-small.yaml`. |
| `error: Training config not found: <path>`. | A wrong path. | Start from a file in `configs\train\`. |
| `error: 'humaneval' grades answers by running model-written code on this machine.` (or `'multipl_e' ...`) | Benchmarks that execute code are opt-in. | Add `--allow-code-exec` after reading the safeguards in `docs\BENCHMARKS.md`, or drop the benchmark. |
| `not run: multipl_e_go: Go needs the go command, which is not on PATH. ...` after a suite run | That language's compiler or runtime is missing, so the suite skipped it instead of scoring it 0. | Install it and resume with `-o <run dir>`, or ignore it. |
| `error: <dir> holds results from different settings (model, decoding parameters, seed, prompt mode, documentation indexes, sandbox, compiler version or memory cap).` | `--out` points at a run made with other settings, and resuming would mix them. | Omit `--out`, or choose a new directory. |
| A bench run is not in `runs\bench\<label>`. | The default folder is `runs\bench\<YYYYMMDD-HHMMSS>-<label>`. | Use the `results:` or `saved to` path that the run prints. |
| `error: No benchmark results in <dir>`. | `bench report` was pointed at a folder that `bench run` didn't create. | Point it at a run folder under `runs\bench\`. |
| `error: <dir> holds answers to gsm8k, an evaluation set: training on them would inflate the score it measures.` | `distill build` was given a run of a test split or a golden set. | Run the teacher on `gsm8k_train` and pass that run. See `docs\howto\src\kumaru\train\distill.md`. |
| `error: <dir> already holds a dataset (...)`. | `distill` would overwrite an earlier dataset. | Choose another `-o`, or pass `--force`. |
| `kumaru eval` exits 0 although cases FAIL. | `eval` reports; it does not gate. | Gate on the JSON written by `-o`, or use `kumaru bench` for statistics. |
| The serve banner shows `http://0.0.0.0:...`, which the browser can't open. | The banner prints the bind address. | Browse to `http://127.0.0.1:<port>` or to the machine's address. |
| `error: Invalid configuration: ... telemetry.log_level ...` from `serve --log-level verbose`, exit 1. | The level is validated with the rest of the config. | Use `debug`, `info`, `warning`, `error` or `critical`, in any case; `warn` and `fatal` also work. |
| An API key appears in pasted `kumaru config` output. | `dump_yaml` prints every field verbatim. | Redact `api_key:` lines before sharing, or share `kumaru info` instead. |

## Known limitations

- **Seven mapped overrides are dead.** `_build_overrides` maps `temperature`, `max_tokens`, `thinking` and `thinking_budget` to `generation.*`, and `dtype`, `base_url` and `load_in_4bit` to `backend.*`, but no command passes any of them to `_load`. There is no `--dtype`, `--base-url` or `--load-in-4bit` flag. Workaround: YAML or `$env:KUMARU_BACKEND__DTYPE` and the like; for sampling, the per-request flags of `chat` and `bench run` (invariant 4).

## Related files

- `pyproject.toml`: `[project.scripts] kumaru = "kumaru.cli:main"`.
- `src\kumaru\core\config.py`: `load_config` and the layers `_build_overrides` feeds (`docs\howto\src\kumaru\core\config.md`).
- `src\kumaru\core\errors.py`: `KumaruError` and its `hint`, formatted by `main()` (`docs\howto\src\kumaru\core\errors.md`).
- `src\kumaru\core\logging.py`: `ensure_utf8_streams()`, plus which log settings each command honours (`docs\howto\src\kumaru\core\logging.md`).
- `src\kumaru\agent\chat.py`: `ChatAgent.stream` and `resolve_params`, behind `chat` and `bench run`; the tool loop (`docs\howto\src\kumaru\agent\chat.md`).
- `src\kumaru\agent\tools.py`: `ToolBelt`, `ApprovalRequest` and `WORKSPACE_TOOLS`, behind `--workspace` and `_approve` (`docs\howto\src\kumaru\agent\tools.md`).
- `src\kumaru\agent\workspace.py` and `src\kumaru\agent\commands.py`: the project-folder tools and what they refuse (`docs\howto\src\kumaru\agent\workspace.md`, `docs\howto\src\kumaru\agent\commands.md`).
- `src\kumaru\agent\knowledge.py`: `KnowledgePack`, `read_local_text` and `save_pack`, behind `learn` (`docs\howto\src\kumaru\agent\knowledge.md`).
- `src\kumaru\agent\doc_index.py`: `build_index` and `format_types`, behind `learn` for a folder (`docs\howto\src\kumaru\agent\doc_index.md`).
- `src\kumaru\server\app.py`: `run(cfg)`, called by `serve` (`docs\howto\src\kumaru\server\app.md`).
- `src\kumaru\train\runner.py`: `run_training`, called by `train` (`docs\TRAINING.md`).
- `src\kumaru\eval\harness.py`: `run_eval`, called by `eval` (`docs\howto\src\kumaru\eval\harness.md`).
- `src\kumaru\eval\benchmarks\runner.py`: everything behind `bench` (`docs\howto\src\kumaru\eval\benchmarks\runner.md`, `docs\BENCHMARKS.md`).
- `src\kumaru\eval\benchmarks\codeexec.py`: `SANDBOXES`, the values `--sandbox` accepts (`docs\howto\src\kumaru\eval\benchmarks\codeexec.md`).
- `src\kumaru\train\distill.py`: everything behind `distill` (`docs\howto\src\kumaru\train\distill.md`).
- `tests\test_cli.py`: CliRunner tests and the `main()` error-format test.

# `src\kumaru\cli.py`

> The `kumaru` binary. Its Typer commands turn flags into config overrides and hand off to the server, the agent, the trainer, the eval harness, the benchmark runner or the distillation data builder. Every deliberate failure prints as `error:` plus `hint:` and exits 1.

**Read this when:**
- you run or script `kumaru`;
- you add a flag or a command;
- you need to know which setting won (flag, profile or `thinking_mode`);
- you need to decode an exit code or an error line.

---

## What it does

- **Commands.** `app` (a `typer.Typer`) has six commands: `serve`, `chat`, `info`, `config`, `train` and `eval`. The `bench` sub-app adds `list`, `run`, `report` and `compare`; the `distill` sub-app adds `build` and `gold`.
- **Config loading.** Every command that needs a configuration builds it the same way: `_load(config, **flags)` calls `load_config(config, overrides=_build_overrides(**flags))`. `chat`'s generation flags are the exception: they travel with each message instead (invariant 4).
- **The REPL.** `chat` runs `_chat_loop`. It streams reasoning dimmed under `thinking…`, then the answer, then a token count. Only the answer goes back into the history.
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
2. **Without `-c`, no YAML is read.** You get the field defaults: backend `transformers` and model `Qwen/Qwen2.5-7B-Instruct`, a multi-gigabyte download on first use. `configs\default.yaml` is an annotated reference, not an implicit layer.
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
| `chat` | `chat(config: str = None, backend: str = None, model: str = None, device: str = None, adapter: str = None, context: int = None, temperature: float = None, max_tokens: int = None, think: bool = None, thinking_budget: int = None, prompt: str = None) -> None` | `kumaru chat`: builds the config from the backend flags, then runs `asyncio.run(_chat_loop(cfg, prompt, overrides))`, where `overrides` holds the four generation flags. |
| `info` | `info() -> None` | `kumaru info`: versions, registered components, profiles, optional packages and CUDA devices. |
| `show_config` | `show_config(config: str = None, backend: str = None, model: str = None) -> None` | `kumaru config`: prints `Panel(cfg.dump_yaml(), title="resolved configuration")`. |
| `learn` | `learn(topic: str, source: Path, version: str, output: Path) -> None` | `kumaru learn TOPIC`. When `source` is a folder, it calls `_learn_folder`. Otherwise it reads `source` (`read_local_text`), builds a `KnowledgePack` whose `source` is the file's name, and writes it with `save_pack`, which never overwrites. Invalid input raises `ConfigError` `Cannot create knowledge pack <output>: <field>: <reason>`. See [knowledge.md](agent/knowledge.md). |
| `_learn_folder` | `_learn_folder(topic: str, folder: Path, version: str, output: Path) -> None` | Private. Runs `build_index` under a status line that counts files (`Indexing 120/492 files...`), then prints the index's path, its file and passage counts and size, the files with no new text, up to three skipped files per reason and the ignored file types. See [doc_index.md](agent/doc_index.md). |
| `train` | `train(config: str, resume: str = None, dry_run: bool = False) -> None` | `kumaru train CONFIG`: calls `run_training(Path(config), resume=resume, dry_run=dry_run)`. |
| `evaluate` | `evaluate(config: str = None, backend: str = None, model: str = None, adapter: str = None, suite: str = "smoke", output: str = None) -> None` | `kumaru eval`: `asyncio.run(run_eval(cfg, suite=suite, output=output))`. |
| `bench_list` | `bench_list() -> None` | `kumaru bench list`. |
| `bench_run` | `bench_run(benchmarks: list[str], config: str = None, backend: str = None, model: str = None, adapter: str = None, context: int = None, limit: int = 100, full: bool = False, seed: int = 1234, think: bool = None, thinking_budget: int = None, max_tokens: int = None, temperature: float = None, concurrency: int = None, label: str = None, out: str = None, allow_code_exec: bool = False, use_agent_prompt: bool = False) -> None` | `kumaru bench run`: builds `BenchSettings` (`limit=None` when `--full` is given), then calls `run_benchmarks(cfg, benchmarks, settings, profile=config, console=console)`. |
| `bench_compare` | `bench_compare(baseline: str, candidate: str, examples: int = 0) -> None` | `kumaru bench compare A B`: `render_comparisons(...)`, plus `render_flips(...)` when `-e` is above 0. |
| `bench_report` | `bench_report(run: str) -> None` | `kumaru bench report RUN`: `render_summaries(console, load_summaries(Path(run)), details=True)`. |
| `distill_build` | `distill_build(runs: list[str], out: str, max_tokens: int = 1024, per_question: int = 4, val_fraction: float = 0.05, seed: int = 42, check_arithmetic: bool = True, eval_set: list[str] = ["gsm8k"], decontaminate: bool = True, allow_thinking: bool = False, force: bool = False) -> None` | `kumaru distill build RUNS... -o DIR`: `render(console, build_from_runs(runs, out, DistillSettings(...), force=force), Path(out))`. |
| `distill_gold` | `distill_gold(out: str, match: str = None, val_fraction: float = 0.05, seed: int = 42, check_arithmetic: bool = True, eval_set: list[str] = ["gsm8k"], decontaminate: bool = True, force: bool = False) -> None` | `kumaru distill gold -o DIR`: the same with `build_gold(out, settings, match=match, force=force)`. |
| `DISTILL_OUT_HELP`, `VAL_FRACTION_HELP`, `SPLIT_SEED_HELP`, `ARITHMETIC_HELP`, `EVAL_SET_HELP`, `DECONTAMINATE_HELP`, `FORCE_HELP` | `str` | Help texts shared by `distill build` and `distill gold`, so the two commands describe a shared flag identically. |
| `main` | `main() -> None` | Console-script entry point. Calls `ensure_utf8_streams()`, then `app()`; a `KumaruError` prints `error:`/`hint:` and raises `SystemExit(1)`. Its `KeyboardInterrupt` branch (`interrupted`, exit 130) does not run in practice; see [Exit codes](#errors-and-exit-codes). |
| `_build_overrides` | `_build_overrides(**kwargs: Any) -> dict[str, Any]` | Private, and the one place flags become config paths. Drops `None` values and unknown keys. `tests\test_cli.py` tests it directly. |
| `_load` | `_load(config: str \| None, **flags: Any)` | Private. Returns the `KumaruConfig` from `load_config(config, overrides=_build_overrides(**flags))`. |
| `_chat_loop` | `async _chat_loop(cfg: Any, single_prompt: str \| None, overrides: dict[str, Any]) -> None` | Private. The terminal REPL behind `chat`. Builds the `ChatAgent` before `backend.load()`, so a bad knowledge pack, documentation index or tool name fails before the model loads. Passes `overrides` to every `agent.stream` call. |

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
| `-c, --config` | `serve`, `chat`, `config`, `eval`, `bench run` | YAML layer | `serve` and `bench run`: "YAML file or profile name." A bare name means `configs\<name>.yaml`. |
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
- **A `KumaruError` during a turn ends the whole session** with `error:`/`hint:` and exit 1. For example, `agent.memory_policy: none` refuses a history over the budget. The loop has no per-turn recovery, and the history is not saved.

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
- the registered `backends`, `tools` and `benchmarks`;
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
  - `bench list` prints the benchmarks: `golden`, `gsm8k`, `gsm8k_train`, `humaneval`, `ifeval` and `mmlu_pro`. Any golden-set `.jsonl` file also works as a benchmark. `gsm8k_train` is GSM8K's training split: teacher data for `kumaru distill`, not a score.
  - `bench report RUN` shows one run in detail.
  - `bench compare BASELINE CANDIDATE [-e N]` answers "did B really beat A?" with paired differences, confidence intervals and p-values.
- **Records.** `bench run` saves every answer.
- **Further reading:**
  - methodology and results: `docs\BENCHMARKS.md`;
  - sampling, fingerprints, concurrency and the record format: `docs\howto\src\kumaru\eval\benchmarks\runner.md`.

`bench run BENCHMARKS...`. The argument help reads "Names from `kumaru bench list`, or golden-set .jsonl paths."

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
| `--allow-code-exec` | off | "Allow benchmarks that execute model-written code (humaneval)." |
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
```

- **Resuming.**
  - A rerun with the same `--out` reuses the finished answers.
  - A run whose fingerprint differs (model, decoding parameters, seed or prompt mode) is refused before anything is written.
  - On Ctrl+C the runner prints `Stopped. Finished answers are saved; resume with --out <dir>`.
- **`humaneval`** refuses to run without `--allow-code-exec`.

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
| The REPL ends with `error: ...` in the middle of a session. | A `KumaruError` during any turn ends `kumaru chat` with exit 1. | Fix the cause and restart. The history is lost. |
| `kumaru chat -q ""` opens the REPL. | An empty prompt is falsy. | Pass a non-empty question. |
| Ctrl+C ends `kumaru chat` with exit 130 and no output, at the prompt as well as during an answer. | `asyncio.run` turns Ctrl+C into a task cancellation, which Typer maps to a silent `Exit(130)`. `main()`'s `interrupted` branch never runs. | Leave the REPL with `/exit`. Scripts should treat 130 as "interrupted". |
| `Missing argument 'config'.` from `kumaru train`, exit 2. | `train` takes a positional path, not `-c`. | `kumaru train configs\train\lora-small.yaml`. |
| `error: Training config not found: <path>`. | A wrong path. | Start from a file in `configs\train\`. |
| `error: 'humaneval' grades answers by running model-written code on this machine.` | Benchmarks that execute code are opt-in. | Add `--allow-code-exec` after reading the safeguards in `docs\BENCHMARKS.md`, or drop `humaneval`. |
| `error: <dir> holds results from different settings (model, decoding parameters, seed or prompt mode).` | `--out` points at a run made with other settings, and resuming would mix them. | Omit `--out`, or choose a new directory. |
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
- `src\kumaru\agent\chat.py`: `ChatAgent.stream` and `resolve_params`, behind `chat` and `bench run` (`docs\howto\src\kumaru\agent\chat.md`).
- `src\kumaru\agent\knowledge.py`: `KnowledgePack`, `read_local_text` and `save_pack`, behind `learn` (`docs\howto\src\kumaru\agent\knowledge.md`).
- `src\kumaru\agent\doc_index.py`: `build_index` and `format_types`, behind `learn` for a folder (`docs\howto\src\kumaru\agent\doc_index.md`).
- `src\kumaru\server\app.py`: `run(cfg)`, called by `serve` (`docs\howto\src\kumaru\server\app.md`).
- `src\kumaru\train\runner.py`: `run_training`, called by `train` (`docs\TRAINING.md`).
- `src\kumaru\eval\harness.py`: `run_eval`, called by `eval` (`docs\howto\src\kumaru\eval\harness.md`).
- `src\kumaru\eval\benchmarks\runner.py`: everything behind `bench` (`docs\howto\src\kumaru\eval\benchmarks\runner.md`, `docs\BENCHMARKS.md`).
- `src\kumaru\train\distill.py`: everything behind `distill` (`docs\howto\src\kumaru\train\distill.md`).
- `tests\test_cli.py`: CliRunner tests and the `main()` error-format test.

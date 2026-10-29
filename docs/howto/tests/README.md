# `tests/`

> The offline regression suite. 990 tests, no GPU, no downloads, under a minute.

**Read this when:** you have changed something and want to know whether you broke it, or you are adding a feature and need to know where its test belongs.

---

## What it does

Exercises every layer that does not require a model: config resolution, the registry, core types, the backend contract, agent policies, the tool loop and its project folder, the HTTP API including SSE streaming, the training data pipeline and distillation data.

## Why it exists — and why it has no GPU tests

Most bugs are not in the model. They are in configuration precedence, prompt assembly, context trimming, JSON schemas, and stream framing. **All of those are testable as pure functions in milliseconds.**

If the suite required a GPU and 15 GB of weights, it would take minutes to run and nobody would run it. Instead the `echo` backend stands in for the model, and a stub tokenizer stands in for a real one, so the suite runs on any machine, in CI, on every save. Speed is what makes a test suite get used.

The parts that genuinely need a GPU are covered by `scripts\verify.ps1 -Full`, which is a separate, slower, manual check.

## Running it

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q          # everything
.\.venv\Scripts\python.exe -m pytest tests -v          # per-test names
.\.venv\Scripts\python.exe -m pytest tests\test_agent.py -v
.\.venv\Scripts\python.exe -m pytest tests -k memory   # by name substring
.\.venv\Scripts\python.exe -m pytest tests -x --lf     # stop at first failure, rerun last failures
```

`pyproject.toml` sets `testpaths = ["tests"]` and `asyncio_mode = "auto"`, so `async def` tests need no decorator.

## What each file protects

| File | Protects against |
|---|---|
| `conftest.py` | — shared fixtures (see below) |
| `test_config.py` | Config precedence silently changing; profiles failing to parse; typos being ignored; env var coercion breaking; **a bare `-c` name reading a file in the current folder** instead of the profile in `configs/` |
| `test_registry.py` | Plugin resolution breaking; aliases leaking into user-facing lists; duplicate registration passing silently |
| `test_types.py` | The wire vocabulary changing shape; `merged()` mutating shared state; division-by-zero in throughput maths |
| `test_backends.py` | `generate()` drifting from `stream()`; **stop sequences split across chunk boundaries being missed**; echo ignoring `max_tokens` or misreporting `length`; a batched greedy reply differing from the one the model gives alone, a batch spanning more than the context window, or a batch too big for the GPU failing instead of retrying in halves; a request or load slipping past `aclose()`; a server's `tool_calls` finish reason being reported as `stop` |
| `test_agent.py` | Prompt assembly losing the system message or dropping client system messages; memory policies overflowing the budget or dropping the system prompt |
| `test_tools.py` | A hung tool stalling the agent; a failing or unknown tool raising instead of returning `ERROR: ...` text; **a change being made without a yes**, or a tool that changes things being offered where nobody can approve it; a read asking for approval |
| `test_tool_loop.py` | The tool loop running a declined call, or one from a reply that was cut off, failed or was cancelled, going past `agent.max_tool_iterations` or 16 calls in one reply, or losing earlier rounds' results; arguments that `json.loads` cannot handle (a huge number, deep nesting) ending the chat; **a chat control token in a result reaching the model unescaped**, or the read→edit loop failing on a file that holds such tokens; trimming dropping the request, or a round whose results alone outgrow the budget; tools leaking into the system prompt instead of the request; a backend that cannot call tools ignoring them; openai_compat reassembling streamed calls wrongly, or turning an `error` or `cancelled` finish into `tool_calls`; `kumaru chat --workspace` changing anything without `y`, keys typed ahead answering its prompt, its prompt or streamed text letting control characters or line breaks steer the terminal, or a hangup or `kill` skipping the cleanups; the server offering a tool that changes anything |
| `test_workspace.py` | **A path leaving the project folder**: `..`, absolute, other-drive and UNC paths, symlinks, junctions (also one swapped in after the diff was shown), alternate data streams, device names, trailing dots and spaces; a file changed, created or deleted after its diff was shown being overwritten; an edit rewriting the line endings of lines it left alone; an edit copied with escaped control tokens missing the real ones, or text found as sent being read another way; listings and reads ignoring their limits |
| `test_commands.py` | **A command reaching a shell**: a program off the allowed list running, a shell such as `cmd` or PowerShell being allowed, or a batch-file argument `cmd` would act on; a line break inside an argument drawing lines of its own in the approval prompt; your secrets reaching the program; a folder or program swapped for a link while you decide being followed; a timeout, a cancel or Kumaru exiting leaving child processes running, even ones that detached themselves (a new session, a double fork); a command running where Windows' job object or Linux's child subreaper could not be set up; a time limit of `NaN` or `Infinity` being accepted; output ignoring its limit; `PATH` keeping a folder where a project file could pose as a program |
| `test_untrusted.py` | **A chat control token in a file or in a program's output reaching the model as a real token**: a template family's markers left unescaped, escaping that changes other text or changes again when repeated, restoring that does not give back what was escaped, or the module importing Kumaru and creating an import cycle |
| `test_knowledge.py` | Coding guidance missing from the CLI, API or UI path, or duplicated; learned references losing their whitespace, overwriting a file or running code; invalid packs failing without the reason, or only after the model has loaded; references overflowing the context, being cut anywhere but at an exact prefix, or all being read whatever the budget; chat control tokens in a reference reaching the prompt unescaped, even where the budget cuts one |
| `test_doc_chunks.py` | A documentation format losing its sections, code or link text, or keeping navigation and markup; code fences being split mid-line or closed early; indented code losing its markup; a `<!--` in a code span hiding the rest of the page; broken HTML losing the text before the break; **a crafted or deeply nested file taking quadratic or exponential time**, which would stall `kumaru learn` |
| `test_doc_index.py` | Indexing following a link out of the folder, even one swapped in after the folder was listed, overwriting a file, leaving a partial index or stopping at one bad page; oversized folders being read instead of refused, or a huge one being listed whole first; the same text under another heading being dropped as a duplicate; searches changing the index or ranking differently from run to run; question text acting as search syntax, or pasted code in any fence crowding out the question; **a crafted index running queries of its own, re-ranking passages or hanging startup**; excerpts overflowing their budget or the prompt's, even when shortened, losing their citation or keeping only it, being closed by a document, switching chat roles through control tokens, or reaching anything but the latest question; a tool round sent without the question's excerpts, or with passages found from a tool result; a missing index failing only after the model has loaded |
| `test_server.py` | API shape changing; SSE framing breaking; auth being bypassable; error responses losing their shape |
| `test_cli.py` | Flag handling, one-shot chat, `config` and the `bench` commands breaking; every command runs in-process on the echo backend |
| `test_llamacpp_server.py` | llama-server model resolution, command line, request payload and failure handling regressing; no process is started and nothing is downloaded |
| `test_bench_extract.py` | Answer extraction misreading a reply, which silently corrupts every benchmark score |
| `test_bench_graders.py` | Question sampling or a grader deciding right vs wrong incorrectly, including a shipped golden format row accepting a reply its prompt rules out |
| `test_bench_ifeval.py` | IFEval's instruction checks, strict/loose readings or its four metrics drifting |
| `test_bench_runner.py` | The files a run leaves behind, exact resume, refusing to mix settings or documentation indexes, surviving bad requests and graders, and the paired comparison of two runs |
| `test_bench_stats.py` | Wilson intervals, McNemar's test, the paired bootstrap or Holm's correction drifting from hand-checked values |
| `test_train_data.py` | **Loss masking regressing** — the highest-consequence silent bug in the project; collator padding labels with a real token id; corpus packing changing the `tokens.bin` format, holding the whole corpus in RAM, or leaving a partial file behind |
| `test_train_runner.py` | Training preflight missing a data file the task will read (LoRA `train_file` and `eval_file`; pretraining `text_file` or its `train_file` fallback), or accepting a directory as one, so a bad path fails at minute 0 instead of after the tokenizer and model load |
| `test_distill.py` | **Test questions leaking into training data**, including ones that open like other questions or are copied into several training prompts; wrong, cut-off or wrongly-stepped answers being kept; the two arms covering different questions or splits; the manifest's counts not adding up; the report's training advice failing when nothing was held out; the three distillation recipes differing in more than the factor each tests |
| `test_finetune_lora.py` | A recipe setting silently dropped on its way to transformers, as `warmup_ratio` was once (transformers 5.2 removed it, so runs trained without warmup); `peak_gpu_memory_gb` misreporting |

## Fixtures

| Fixture | Gives you |
|---|---|
| `echo_config` | A `KumaruConfig` on the `echo` backend with zero stream delay |
| `loaded_echo_backend` | An already-loaded `EchoBackend`, closed on teardown |
| `chat_agent` | A `ChatAgent` wired to the echo backend |
| `test_client` | A FastAPI `TestClient` with lifespan run, so the backend is loaded |
| `scratch_dir` | A temp directory under `tests/.scratch`, emptied before and after |

## The two tests that matter most

**Split stop sequences** (`test_backends.py`). A stop sequence can straddle two streamed chunks — `"</"` then `"s>"`. Naive per-chunk matching misses it, and the model's output leaks past where it should have stopped. This bug is invisible in short generations and appears intermittently in long ones, which makes it miserable to find in production.

**Loss masking** (`test_train_data.py`). If `labels` are not `-100` on non-assistant tokens, the model spends its gradient budget learning to predict questions. Training still converges, the loss curve looks perfect, and the resulting model is worse than the one you started with. There is no runtime symptom — only a test can catch it.

## Adding a test

Put it in the file matching the layer you changed. Name it after the behaviour asserted, not the function called:

```python
# good — states what must remain true
def test_sliding_window_keeps_system_message(): ...

# bad — restates the implementation
def test_apply(): ...
```

For anything touching a model, use the echo backend or a stub. If you find yourself wanting to download weights in a test, that test belongs in `scripts\verify.ps1` instead.

## Failure modes

| Symptom | Likely cause | Fix |
|---|---|---|
| `ModuleNotFoundError: kumaru` | Package not installed, or wrong interpreter | `pip install -e ".[dev]"`; use `.\.venv\Scripts\python.exe` |
| Async tests fail with `async def functions are not natively supported` | pytest-asyncio missing (pytest 8.4+ fails such tests instead of skipping them) | `pip install -e ".[dev]"` |
| Config tests fail only on your machine | A `KUMARU_*` env var is set and leaking in | Unset it, or pass `use_env=False` |
| Server tests hang | Lifespan not run | Use `TestClient` as a context manager |
| `test_train_data` fails after a transformers upgrade | `apply_chat_template` return type changed again | Check `_template_ids()` in `train/data.py` |

## Related files

- `scripts/verify.ps1` — the layered self-check that runs this suite as stage 3
- `configs/test.yaml` — the echo profile these tests mirror
- `src/kumaru/backends/echo.py` — the stand-in model
- `pyproject.toml` — pytest configuration and the `dev` extra

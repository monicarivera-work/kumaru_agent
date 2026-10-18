# `tests/`

> The offline regression suite. 511 tests, no GPU, no downloads, under a minute.

**Read this when:** you have changed something and want to know whether you broke it, or you are adding a feature and need to know where its test belongs.

---

## What it does

Exercises every layer that does not require a model: config resolution, the registry, core types, the backend contract, agent policies, the HTTP API including SSE streaming, the training data pipeline and distillation data.

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
| `test_config.py` | Config precedence silently changing; profiles failing to parse; typos being ignored; env var coercion breaking |
| `test_registry.py` | Plugin resolution breaking; aliases leaking into user-facing lists; duplicate registration passing silently |
| `test_types.py` | The wire vocabulary changing shape; `merged()` mutating shared state; division-by-zero in throughput maths |
| `test_backends.py` | `generate()` drifting from `stream()`; **stop sequences split across chunk boundaries being missed**; echo ignoring `max_tokens` or misreporting `length`; a batched greedy reply differing from the one the model gives alone, or a batch too big for the GPU failing instead of retrying in halves |
| `test_agent.py` | Prompt assembly losing the system message or dropping client system messages; memory policies overflowing the budget or dropping the system prompt |
| `test_tools.py` | A hung tool stalling the agent; a failing or unknown tool raising instead of returning `ERROR: ...` text |
| `test_server.py` | API shape changing; SSE framing breaking; auth being bypassable; error responses losing their shape |
| `test_cli.py` | Flag handling, one-shot chat, `config` and the `bench` commands breaking; every command runs in-process on the echo backend |
| `test_llamacpp_server.py` | llama-server model resolution, command line, request payload and failure handling regressing; no process is started and nothing is downloaded |
| `test_bench_extract.py` | Answer extraction misreading a reply, which silently corrupts every benchmark score |
| `test_bench_graders.py` | Question sampling or a grader deciding right vs wrong incorrectly, including a shipped golden format row accepting a reply its prompt rules out |
| `test_bench_ifeval.py` | IFEval's instruction checks, strict/loose readings or its four metrics drifting |
| `test_bench_runner.py` | The files a run leaves behind, exact resume, refusing to mix settings, surviving bad requests and graders, and the paired comparison of two runs |
| `test_bench_stats.py` | Wilson intervals, McNemar's test, the paired bootstrap or Holm's correction drifting from hand-checked values |
| `test_train_data.py` | **Loss masking regressing** — the highest-consequence silent bug in the project; collator padding labels with a real token id; corpus packing changing the `tokens.bin` format, holding the whole corpus in RAM, or leaving a partial file behind |
| `test_train_runner.py` | Training preflight missing a data file the task will read (LoRA `train_file` and `eval_file`; pretraining `text_file` or its `train_file` fallback), or accepting a directory as one, so a bad path fails at minute 0 instead of after the tokenizer and model load |
| `test_distill.py` | **Test questions leaking into training data**; wrong, cut-off or wrongly-stepped answers being kept; the two arms covering different questions or splits; the manifest's counts not adding up; the three distillation recipes differing in more than the factor each tests |
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

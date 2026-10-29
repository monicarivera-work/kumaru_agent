# `src\kumaru\eval\harness.py`

> The v0.1 `kumaru eval` harness: a seconds-long check of the agent path. It runs substring-checked quality cases at temperature 0 and reports latency, time to first token and throughput.

**Read this when:** you need a quick tripwire before and after a fine-tune, latency or throughput numbers for a backend or quantization change, a custom JSONL eval suite, or a more nuanced scorer. Also read it when you are choosing between `kumaru eval` and the v0.2 benchmarks (`kumaru bench`).

---

## What it does

`harness.py` defines evaluation cases, built-in smoke and speed suites, JSONL suite loading, and the async `run_eval()` entry point. The harness loads the configured backend, wraps it in `ChatAgent`, runs each prompt at `temperature=0.0`, checks required and forbidden substrings, prints a Rich table, optionally writes JSON, and returns an `EvalResult`.

The smoke suite is a regression tripwire, not a benchmark. The speed suite captures latency and tokens/second for operational comparisons.

Every prompt goes through `ChatAgent.complete`, the same path chat uses:

- `ChatAgent.prepare` adds the agent's system prompt: the configured persona, the shared quality directives and today's date. Enabled read-only tools, such as `current_time`, go with the request as tool declarations, and the agent runs their calls. Built without an approver, it never offers the tools that change things or use the project folder ([`tools.md`](../agent/tools.md));
- decoding parameters resolve as they do in chat;
- the request overrides only `temperature` (0) and `max_tokens` (the case's own value).

v0.2 left this file unchanged. It added the benchmarks package beside it: `src\kumaru\eval\benchmarks\`, run with `kumaru bench`. That package measures accuracy on public datasets and on your own golden questions, with confidence intervals and paired significance tests. [`kumaru eval` or `kumaru bench`?](#kumaru-eval-or-kumaru-bench) below says when to use which.

## Why it exists

Every prompt, quantization setting, backend, or fine-tune can feel better because the operator chose it. A repeatable harness turns subjective impressions into numbers and catches regressions before shipping.

Substring scoring is intentionally simple: it is deterministic, fast, and needs no judge model. For nuanced tasks, the data structures are already shaped so an LLM-judge scorer can be added later.

The harness stays useful next to `kumaru bench` for two reasons. It is the only one of the two that measures time to first token. And it sends the agent's own system prompt by default.

## Mental model

1. **A `Case` is a prompt plus string checks.** All `must_include` strings must appear; no `must_not_include` strings may appear.
2. **Evaluation is deterministic by default.** `temperature=0.0` avoids sampling noise.
3. **Speed and quality are both first-class.** Each case records latency, TTFT, completion tokens, and pass/fail details.
4. **Thinking follows the profile.**
   - The harness never sets `thinking`, so the profile's `generation.thinking` decides. No `kumaru eval` flag changes it, but the environment can: `$env:KUMARU_GENERATION__THINKING='false'`.
   - Reasoning tokens count against the case's `max_tokens`: 24–96 in `smoke`, 8–512 in `speed`.
   - Only the answer text (`message.content`) is checked and saved.
5. **One pass, one file.** There are no retries, no resume, no confidence intervals and no run directory: just the table and an optional JSON file. A backend error in any case aborts the run, and the JSON file is not written.

## `kumaru eval` or `kumaru bench`?

Both tools load backends from the same config. `kumaru bench` also resolves decoding parameters with the same `ChatAgent.resolve_params`. They answer different questions:

| | `kumaru eval` (this file) | `kumaru bench` (`src\kumaru\eval\benchmarks\`) |
|---|---|---|
| Question | Is this setup working, and how fast is it? | How accurate is it, and did a change really help? |
| Prompt | The agent's system prompt (persona, quality directives, today's date), then the case prompt. Enabled read-only tools go with it as tool declarations. | Only the benchmark's messages: the question, plus a golden row's `system` if it has one. `--use-agent-prompt` passes them through the same `ChatAgent.prepare`, still without tools. |
| Decoding | Temperature forced to 0. Per-case `max_tokens`: 24–96 (`smoke`), 8–512 (`speed`). Thinking from the profile, or `KUMARU_GENERATION__THINKING`. | The profile's preset (`-t` overrides), with a fixed seed per question. 8,192 answer tokens, plus the thinking budget. `--think/--no-think`, `--thinking-budget`. |
| Scoring | Case-insensitive substrings: every `must_include` present, no `must_not_include`. | Per-benchmark graders: number and letter extraction, IFEval's instruction checkers, HumanEval's and MultiPL-E's unit tests, seven golden graders. |
| Data | `smoke` (6 cases), `speed` (3), or a JSONL file of `Case` rows. | `golden` (27), GSM8K, MMLU-Pro, IFEval, HumanEval, MultiPL-E, and any golden-format JSONL. |
| Output | A table, plus one JSON file with `--output`. | A run directory: `run.json`, then `items.jsonl` and `summary.json` per benchmark. Resumable with `-o`. |
| Errors | The first backend error aborts the run, and nothing is saved. | Recorded per question and retried on resume. Five failures in a row stop the run; finished answers are kept. |
| Statistics | A pass rate. | A Wilson 95% CI on every accuracy. `compare` adds a paired bootstrap CI and an exact McNemar p-value, Holm-adjusted. |
| Speed numbers | Median time to first token and median tokens/second. | Tokens per answer and answers per minute (under `-j` concurrency). No TTFT. |
| Cost | Seconds. | Seconds (`golden`) to hours (`--full` public sets). |

Which to use:

- **New backend, quantization, context size or hardware.** Run `kumaru eval --suite speed` for TTFT and tokens/second. Then run `kumaru bench run golden` to confirm the answers are still right.
- **The persona, `QUALITY_DIRECTIVES` or enabled tools changed.** `kumaru eval --suite smoke` sends the system prompt the agent builds from them, and the enabled read-only tools. `kumaru bench` sends none of it unless you pass `--use-agent-prompt`, and never sends tools.
- **Before and after a fine-tune.** Run `kumaru eval --suite smoke` as the fast tripwire. Then run `kumaru bench run ... --adapter <dir>` and `kumaru bench compare` for a number you can defend.
- **"Is B better than A?"** Use only `kumaru bench`. Six substring cases cannot separate two decent models: 6/6 has a 95% Wilson interval of 61.0–100.0, and 5/6 one of 43.6–97.0.
- **Your own questions.** Write them in golden format and run `kumaru bench run my.jsonl`. You get stricter graders (`numeric`, `regex`, `choice`), CIs, resume and `compare`. The two file formats are not interchangeable (see Failure modes).
- **Thinking models.** Use `kumaru bench run --think --thinking-budget N`. The harness's small budgets can't hold the reasoning.

```powershell
kumaru eval -c qwen36 --suite speed --output runs\eval-qwen36-speed.json   # TTFT, tok/s; that profile keeps thinking off
kumaru bench run golden gsm8k -c qwen36                                     # accuracy with 95% CIs
kumaru bench compare runs\bench\<before> runs\bench\<after>                 # did it really change?
```

The benchmark pages start at [`benchmarks\__init__.md`](benchmarks/__init__.md). Run mechanics are in [`benchmarks\runner.md`](benchmarks/runner.md), and results are in [`docs\BENCHMARKS.md`](../../../../BENCHMARKS.md).

## Public API

| Name | Signature | What it's for |
|---|---|---|
| `Case` | `class Case(BaseModel)` | One eval prompt with `must_include`, `must_not_include`, `max_tokens`, and `tag`. |
| `CaseResult` | `class CaseResult(BaseModel)` | Per-case output, pass/fail details, latency, TTFT, and token count. |
| `CaseResult.tokens_per_second` | `property -> float` | Completion tokens divided by latency seconds. |
| `EvalResult` | `class EvalResult(BaseModel)` | Aggregate suite/model/backend metrics plus all `CaseResult` rows. |
| `SMOKE_SUITE` | `list[Case]` | Built-in regression tripwire for factual, arithmetic, format, code, and calibration behavior. |
| `SPEED_SUITE` | `list[Case]` | Built-in speed/latency prompts. |
| `load_suite` | `load_suite(suite: str) -> tuple[str, list[Case]]` | Resolves `smoke`, `speed`, or a JSONL path into cases. |
| `run_eval` | `async run_eval(config: KumaruConfig, *, suite: str = "smoke", output: str \| None = None) -> EvalResult` | Runs a suite against the configured backend, prints a report, writes optional JSON, and returns metrics. |

### `Case` fields

| Field | Type | Default | Meaning |
|---|---:|---:|---|
| `prompt` | string | required | User prompt sent as a single `Message.user(...)`. |
| `must_include` | list[string] | `[]` | Case-insensitive substrings that must appear in output. |
| `must_not_include` | list[string] | `[]` | Case-insensitive substrings that fail the case if present. |
| `max_tokens` | int | 256 | Completion token limit for this case. |
| `tag` | string | `general` | Label shown in the report table. |

## How to use it

Run the smoke suite:

```powershell
kumaru eval --suite smoke --output runs\eval-smoke.json
```

Run the speed suite:

```powershell
kumaru eval --suite speed --output runs\eval-speed.json
```

Run against a v0.2 llama.cpp profile. It starts `llama-server` the way `kumaru chat` does, and `configs\qwen36.yaml` keeps thinking off, which the harness needs:

```powershell
kumaru eval -c qwen36 --suite smoke --output runs\eval-qwen36-smoke.json
```

Evaluate an adapter trained by `finetune_lora.py`:

```powershell
kumaru eval --backend transformers --model Qwen/Qwen2.5-7B-Instruct --adapter runs\lora-qwen\adapter --suite smoke --output runs\lora-qwen\eval-smoke.json
```

Create a custom suite as JSONL:

```jsonl
{"prompt":"Answer with exactly OK.","must_include":["OK"],"must_not_include":["sorry"],"max_tokens":8,"tag":"format"}
{"prompt":"Compute 12 * 12. Reply with only the number.","must_include":["144"],"max_tokens":16,"tag":"arithmetic"}
```

Run it:

```powershell
kumaru eval --suite evals\custom.jsonl --output runs\custom-eval.json
```

Call from Python:

```python
import asyncio
from kumaru.core.config import load_config
from kumaru.eval.harness import load_suite, run_eval

name, cases = load_suite("smoke")
print(name, len(cases))

cfg = load_config(None, overrides={"backend": {"name": "echo", "model": "echo"}})
result = asyncio.run(run_eval(cfg, suite="smoke"))
print(result.n_passed, result.pass_rate)
```

## Tuning / operational notes

Use `smoke` before and after fine-tuning. Its built-in cases target failures that adapters often introduce: lost instruction following, arithmetic regression, invalid format output, broken code generation, and overconfident answers to unknowable questions.

Use `speed` when changing backend, quantization, model size, context settings, or serving hardware. The result records median TTFT and median tokens/second. Cases with very small completions are excluded from the speed median unless they produce more than four completion tokens.

`run_eval()` loads exactly the backend described by `KumaruConfig`, so all normal CLI overrides apply: `--backend`, `--model`, `--adapter`, and `--config`. It closes the backend in a `finally` block even when a case fails.

**Thinking profiles fail the harness.**

- **What happens.** The harness never sets `thinking`. With `generation.thinking: true`, or with `null` on a model whose chat template thinks by default, the reasoning uses up the case's 24–96 tokens. The answer comes back cut or empty, and the case fails with `missing: …`.
- **Where it comes from.** No shipped profile turns thinking on: `configs\qwen36.yaml` sets `false`, and the rest leave it at `null` (the `GenerationParams` default: the model's chat template decides). The risk is a profile of your own that sets `true`, or that leaves `null` for a model that thinks by default.
- **What to do.** Run the harness with a thinking-off profile, or force thinking off for the shell session. Environment variables override the YAML:

  ```powershell
  $env:KUMARU_GENERATION__THINKING = 'false'
  kumaru eval -c <your-profile> --suite smoke
  Remove-Item Env:KUMARU_GENERATION__THINKING
  ```
- **To measure a thinking model**, use `kumaru bench run --think --thinking-budget N`. It allows the thinking budget plus 8,192 answer tokens, and counts replies that still hit the cap as `cut off` (`n_truncated`).

**The small budgets are deliberate.** They keep a run to seconds and make a looping model fail fast. `kumaru bench` makes the opposite choice, 8,192 tokens for every answer: there, a cut answer would measure the cap rather than the model.

For training operations, keep JSON outputs under the run directory:

```powershell
kumaru eval --backend transformers --model Qwen/Qwen2.5-0.5B-Instruct --adapter runs\lora-small\adapter --suite smoke --output runs\lora-small\eval-smoke.json
```

## How to extend it

### Add a custom eval suite as JSONL

Create one JSON object per line using `Case` fields:

```jsonl
{"prompt":"Classify sentiment: Kumaru is useful.","must_include":["positive"],"must_not_include":["negative"],"max_tokens":16,"tag":"sentiment"}
{"prompt":"Return a JSON object with key status and value ok.","must_include":["status","ok"],"must_not_include":["```"],"max_tokens":64,"tag":"json"}
```

Then run:

```powershell
kumaru eval --suite evals\domain-smoke.jsonl --output runs\domain-smoke.json
```

### Add an LLM-judge scorer

Keep `Case` and `CaseResult` as the transport types, but add optional fields such as `judge_prompt` and `score`. A minimal approach is:

1. Extend `Case` with `judge_rubric: str | None = None`.
2. Extend `CaseResult` with `judge_score: float | None = None` and `judge_reason: str | None = None`.
3. In `run_eval()`, after substring checks, call a separate judge backend only when `case.judge_rubric` is set.
4. Keep substring checks for cheap hard failures, and make judge scoring additive rather than replacing deterministic tests.

Do not use sampled generation settings for judged answers; keep `temperature=0.0` for the candidate output so repeated runs are comparable.

## Failure modes

| Symptom | Likely cause | Fix |
|---|---|---|
| `Unknown eval suite` | `--suite` is not `smoke`, `speed`, or an existing file | Pass a valid built-in name or JSONL path. |
| Case unexpectedly fails | `must_include` is too strict or model uses a synonym | Use robust required substrings or add a judge scorer. |
| All cases fail after fine-tune | Adapter path wrong, model stopped following format, or backend misconfigured | Verify `--adapter`, run base model, inspect outputs in JSON. |
| Median TTFT is blank | Backend did not report time-to-first-token | Check backend instrumentation; latency and tok/s may still be present. |
| Tokens/second looks inflated on tiny outputs | Very short completions distort speed | Use `speed` suite; median speed ignores cases with <=4 completion tokens. |
| Eval results change between runs | Backend nondeterminism (GPU kernels, batching), or a different backend or model. The system prompt also carries today's date, so the prompt itself changes daily. The harness already forces temperature 0. | Compare on the same backend and model, ideally on the same day. Treat a one-case flip as noise, or measure with `kumaru bench` (no date in its prompt by default). |
| Thinking-model cases fail with `missing: …` and an empty or cut `output` | Thinking is on: the reasoning used the case's `max_tokens`. Only the answer text is checked and saved. | Use a thinking-off profile (`configs\qwen36.yaml`), set `$env:KUMARU_GENERATION__THINKING='false'`, or use `kumaru bench run --think --thinking-budget N`. |
| `FileNotFoundError: [Errno 2] No such file or directory: '<path>'` after the table prints | The `--output` folder doesn't exist. `run_eval` writes with `Path.write_text` and doesn't create folders. The run completed, but the JSON is lost. | Create the folder first (`New-Item -ItemType Directory <dir>`), then re-run. |
| `JSONDecodeError: Expecting value: line 1 column 1 (char 0)` from `--suite data\bench\golden.jsonl` | Golden files belong to `kumaru bench`. `load_suite` doesn't skip their `//` comment lines. | `kumaru bench run golden`, or `kumaru bench run <file>.jsonl`. |
| Every case of a custom suite passes | The file is golden-format (`answer`, `grader`). `Case` ignores unknown keys, so `must_include` is empty and nothing is checked. | Convert the rows to `must_include` / `must_not_include`, or run the file with `kumaru bench run`. |
| `kumaru bench run my-suite.jsonl` says `missing 'answer'` | A harness `Case` file was passed to `kumaru bench`. | Use `kumaru eval --suite`, or convert the rows to golden format ([`golden.jsonl.md`](../../../data/bench/golden.jsonl.md)). |
| A traceback instead of an `error:` line for a custom suite | A malformed line, or a row without `prompt`. `load_suite` raises `JSONDecodeError` or pydantic's `ValidationError`, without a file line number. | Validate each line; the "line 1" in a `JSONDecodeError` is within the row, not the file. |

## Related files

- `src\kumaru\eval\__init__.py` — re-exports `Case`, `EvalResult`, and `run_eval`.
- `src\kumaru\cli.py` — exposes `kumaru eval --suite ... --output ...`.
- `src\kumaru\agent\chat.py` — `ChatAgent.complete` and `resolve_params` (generation < thinking_mode < request), used to complete prompts.
- `src\kumaru\eval\benchmarks\` — the v0.2 benchmarks and `kumaru bench` ([`benchmarks\__init__.md`](benchmarks/__init__.md), [`benchmarks\runner.md`](benchmarks/runner.md)).
- `configs\qwen36.yaml` — a llama.cpp profile with thinking off by default, suitable for the harness.
- [`docs\BENCHMARKS.md`](../../../../BENCHMARKS.md) — benchmark methodology and results.
- `src\kumaru\core\registry.py` — creates configured backends.
- `src\kumaru\core\types.py` — defines `Message.user(...)` used for prompts.
- `src\kumaru\train\finetune_lora.py` — adapters produced there should be evaluated here.

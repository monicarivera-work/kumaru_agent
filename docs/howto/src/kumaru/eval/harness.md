# `src\kumaru\eval\harness.py`

> Deterministic evaluation harness for measuring Kumaru model quality, latency, time to first token, and throughput.

**Read this when:** you need to compare a base model and adapter, add a custom JSONL eval suite, investigate a regression, or add a more nuanced scorer.

---

## What it does

`harness.py` defines evaluation cases, built-in smoke and speed suites, JSONL suite loading, and the async `run_eval()` entry point. The harness loads the configured backend, wraps it in `ChatAgent`, runs each prompt at `temperature=0.0`, checks required and forbidden substrings, prints a Rich table, optionally writes JSON, and returns an `EvalResult`.

The smoke suite is a regression tripwire, not a benchmark. The speed suite captures latency and tokens/second for operational comparisons.

## Why it exists

Every prompt, quantization setting, backend, or fine-tune can feel better because the operator chose it. A repeatable harness turns subjective impressions into numbers and catches regressions before shipping.

Substring scoring is intentionally simple: it is deterministic, fast, and needs no judge model. For nuanced tasks, the data structures are already shaped so an LLM-judge scorer can be added later.

## Mental model

1. **A `Case` is a prompt plus string checks.** All `must_include` strings must appear; no `must_not_include` strings may appear.
2. **Evaluation is deterministic by default.** `temperature=0.0` avoids sampling noise.
3. **Speed and quality are both first-class.** Each case records latency, TTFT, completion tokens, and pass/fail details.

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
| `run_eval` | `async run_eval(config: KumaruConfig, *, suite: str = "smoke", output: str | None = None) -> EvalResult` | Runs a suite against the configured backend, prints a report, writes optional JSON, and returns metrics. |

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
| Eval results change between runs | Sampling or backend nondeterminism | Keep temperature at 0; avoid changing backend/model between comparisons. |

## Related files

- `src\kumaru\eval\__init__.py` — re-exports `Case`, `EvalResult`, and `run_eval`.
- `src\kumaru\cli.py` — exposes `kumaru eval --suite ... --output ...`.
- `src\kumaru\agent\chat.py` — `ChatAgent` used to complete prompts.
- `src\kumaru\core\registry.py` — creates configured backends.
- `src\kumaru\core\types.py` — defines `Message.user(...)` used for prompts.
- `src\kumaru\train\finetune_lora.py` — adapters produced there should be evaluated here.

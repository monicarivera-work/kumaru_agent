# `src\kumaru\eval\__init__.py`

> Public re-export surface for Kumaru's v0.1 evaluation harness (`kumaru eval`). The v0.2 benchmarks (`kumaru bench`) live in the subpackage `kumaru.eval.benchmarks`, which this file does not re-export.

**Read this when:** you want to import evaluation primitives from `kumaru.eval` instead of the deeper `kumaru.eval.harness` module, or `from kumaru.eval import …` can't find a benchmark name.

---

## What it does

This module imports `Case`, `EvalResult`, and `run_eval` from `kumaru.eval.harness` and exposes them in `__all__`. It gives callers a stable package-level import path.

It contains no evaluation logic itself. All suite loading, scoring, backend calls, tables, and JSON output live in `harness.py`.

Since v0.2 the package has two layers:

| Import from | Command | What it measures | Page |
|---|---|---|---|
| `kumaru.eval` (implemented in `kumaru.eval.harness`) | `kumaru eval` | Smoke and speed checks of the agent path: substring checks at temperature 0, plus time to first token and tokens/second. | [`harness.md`](harness.md) |
| `kumaru.eval.benchmarks`, and `kumaru.eval.benchmarks.runner` | `kumaru bench list\|run\|report\|compare` | Accuracy on GSM8K, MMLU-Pro, IFEval, HumanEval and golden-format question sets, with 95% confidence intervals and paired significance tests. | [`benchmarks\__init__.md`](benchmarks/__init__.md) |

This file re-exports only the first layer. The subpackage has its own `__all__`: `BenchItem`, `Benchmark`, `Grade`, `select_items`. The run machinery is imported from `kumaru.eval.benchmarks.runner`: `BenchRunner`, `BenchSettings`, `run_benchmarks`, `compare_runs`, `load_summaries`, `list_benchmarks` and the `render_*` functions.

## Why it exists

Package-level exports keep downstream code from depending on the internal file layout. If the harness implementation moves later, `from kumaru.eval import run_eval` can remain stable.

Without this file, callers would either import from `kumaru.eval.harness` directly or accidentally rely on non-public internals.

## Mental model

1. **This is an API shim.** It re-exports stable names.
2. **The implementation is in `harness.py`.** Debug behavior there.
3. **`__all__` is the contract.** Only listed names are intended package-level API.
4. **Two layers, no imports between them.** The harness and the benchmarks both build their backend from `KumaruConfig` and decode through `ChatAgent`, and neither imports the other.
   - Importing `kumaru.eval.benchmarks` runs this file first, because Python imports the parent package before a subpackage. That also loads `harness.py`, which is cheap.
   - It does not load `runner.py` or the `datasets` library.
   - The CLI imports both layers lazily, inside the command that needs them.

## Public API

| Name | Signature | What it's for |
|---|---|---|
| `Case` | `class Case(BaseModel)` | Re-export of one evaluation case: prompt plus required/forbidden substrings. |
| `EvalResult` | `class EvalResult(BaseModel)` | Re-export of aggregate evaluation results. |
| `run_eval` | `async run_eval(config: KumaruConfig, *, suite: str = "smoke", output: str \| None = None) -> EvalResult` | Re-export of the async harness entry point. |
| `__all__` | `list[str]` | Declares package-level public names: `Case`, `EvalResult`, `run_eval`. |

Nothing from `kumaru.eval.benchmarks` is available here. Its API is in [`benchmarks\__init__.md`](benchmarks/__init__.md#public-api) and [`benchmarks\runner.md`](benchmarks/runner.md#public-api).

## How to use it

Import from the package:

```python
import asyncio
from kumaru.core.config import load_config
from kumaru.eval import run_eval

cfg = load_config(None, overrides={"backend": {"name": "echo", "model": "echo"}})
result = asyncio.run(run_eval(cfg, suite="smoke", output="runs/eval-smoke.json"))
print(result.pass_rate)
```

Use the CLI instead when operating from a shell:

```powershell
kumaru eval --suite smoke --output runs\eval-smoke.json
kumaru eval --suite speed --output runs\eval-speed.json
```

Evaluate a trained adapter:

```powershell
kumaru eval --backend transformers --model Qwen/Qwen2.5-7B-Instruct --adapter runs\lora-qwen\adapter --suite smoke --output runs\lora-qwen\eval-smoke.json
```

The v0.2 benchmarks are imported from their own modules:

```python
from kumaru.eval.benchmarks import BenchItem, Benchmark, Grade, select_items     # the benchmark contract
from kumaru.eval.benchmarks.runner import BenchSettings, compare_runs, run_benchmarks
```

```powershell
kumaru bench run golden -c qwen36                        # the 27 hand-written golden questions
kumaru bench compare runs\bench\<run-A> runs\bench\<run-B>
```

[`benchmarks\__init__.md`](benchmarks/__init__.md#from-python) has a runnable offline Python example.

## Tuning / operational notes

This module has no runtime tuning knobs. Operational choices come from `run_eval()`:

- `suite="smoke"` checks deterministic quality tripwires.
- `suite="speed"` measures latency and throughput cases.
- `suite="path\to\cases.jsonl"` loads custom `Case` rows.
- `output="...json"` writes the full `EvalResult` payload.

For training operations, run eval before and after adapter changes. A quick smoke suite is a regression tripwire for instruction following, arithmetic, format compliance, code generation, factual recall, and calibrated refusal.

Choose the layer by the question you are asking. The full comparison is in [`harness.md`](harness.md#kumaru-eval-or-kumaru-bench).

- **`kumaru eval`**: a seconds-long tripwire, and the only one of the two layers that reports time to first token.
  - The built-in suites' per-case budgets are 8–512 tokens, and there is no thinking flag.
  - Run it with thinking off: use a thinking-off profile, or set `$env:KUMARU_GENERATION__THINKING='false'`.
- **`kumaru bench`**: accuracy with 95% confidence intervals, and A-vs-B comparisons.
  - Every answer gets 8,192 tokens, plus the thinking budget when thinking is on.
  - The report's **cut off** column (`n_truncated`) shows when that cap still binds.

## How to extend it

If `harness.py` adds a stable new public class, re-export it here and update `__all__`:

```python
from kumaru.eval.harness import Case, EvalResult, JudgeResult, run_eval

__all__ = ["Case", "EvalResult", "JudgeResult", "run_eval"]
```

Do not re-export helper functions such as `load_suite()` unless you want to support them as public API.

Do not re-export the benchmark API here either. Every `import kumaru.eval` would then also load `runner.py`, which adds over a hundred modules (Rich, asyncio, ssl, subprocess and more; 111 when measured on Windows). That cost would land on the `kumaru eval` path and on every `import kumaru.eval.benchmarks`. To add a benchmark, follow [`benchmarks\__init__.md`](benchmarks/__init__.md#add-a-benchmark-end-to-end).

## Failure modes

| Symptom | Likely cause | Fix |
|---|---|---|
| `ImportError: cannot import name ... from kumaru.eval` | Name is not re-exported in `__all__` | Import from `kumaru.eval.harness` or add a deliberate re-export. |
| `ImportError: cannot import name 'BenchRunner' from 'kumaru.eval'` (or `run_benchmarks`, `BenchSettings`, …) | The benchmark API lives in the subpackage and is not re-exported. | `from kumaru.eval.benchmarks.runner import BenchRunner`. The contract types come from `kumaru.eval.benchmarks`. |
| Eval behavior differs from expectation | Looking at this shim instead of implementation | Read `src\kumaru\eval\harness.py`. |
| Circular import after adding exports | New exported object imports `kumaru.eval` during module import | Keep implementation imports one-way from `__init__` to `harness`. |
| CLI eval fails | Backend/config issue, not this module | Check `kumaru eval` args and `harness.py` failure modes. |

## Related files

- `src\kumaru\eval\harness.py` — implementation of all re-exported names ([`harness.md`](harness.md)).
- `src\kumaru\eval\benchmarks\` — the v0.2 benchmarks package and `kumaru bench` ([`benchmarks\__init__.md`](benchmarks/__init__.md)).
- `src\kumaru\cli.py` — `kumaru eval` imports `run_eval` from `kumaru.eval.harness`, and `kumaru bench` imports from `kumaru.eval.benchmarks.runner`.
- `src\kumaru\core\config.py` — provides the `KumaruConfig` passed to `run_eval`.
- `src\kumaru\agent\chat.py` — harness uses `ChatAgent` to complete prompts.
- [`docs\BENCHMARKS.md`](../../../../BENCHMARKS.md) — benchmark methodology and results.

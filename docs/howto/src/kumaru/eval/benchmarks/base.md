# `src\kumaru\eval\benchmarks\base.py`

> The benchmark contract (`Benchmark`, `BenchItem`, `Grade`), the Hugging Face loader every public benchmark uses, and the deterministic `-n/--limit` sampler.

**Read this when:** you are writing or reviewing a benchmark, wondering which questions `-n 100` picks, or deciding why a run used the token budget it did.

---

## What it does

- **`BenchItem`** — one question: a stable `id`, the user `prompt`, the gold `answer` (any shape the grader expects), an optional `group` (subject/tag), an optional `system` message, an optional per-item `max_tokens` and free-form `meta`.
- **`Grade`** — one verdict: `correct`, optional partial-credit `score`, the `extracted` answer and grader `detail` such as the extraction method.
- **`Benchmark`** — abstract base class. Subclasses implement `load()` and `grade()` and set three descriptive class attributes. They may override six hook methods (`preflight`, `configure`, `fingerprint_extra`, `messages`, `generation_overrides`, `aggregate`) and four class switches (`answer_tokens`, `needs_code_execution`, `blocking_grader`, `training_data`).
- **`load_hf_rows(path, name, split, revision=None)`** — loads a Hugging Face dataset split as plain dicts, importing `datasets` lazily and turning failures into Kumaru errors with a hint. `revision` pins a dataset commit.
- **`select_items(items, limit, seed)`** — the sampler behind `-n/--limit` and `--seed`. It is deterministic, stratified by `group`, and returns items in dataset order.

## Why it exists

The module docstring states the design. A benchmark is three separate decisions:

1. **Which questions** (`load`) — a fixed public split, so numbers are comparable with anyone else who ran it.
2. **How they are asked** (`messages`) — the prompt is part of the measurement. Change it and you have a different benchmark.
3. **How answers are scored** (`grade`) — deterministic code that records what it extracted, so every verdict can be audited.

Everything else, from sampling, seeds, decoding and concurrency to persistence, resume and statistics, belongs to the runner. A benchmark is therefore small (gsm8k.py is about 60 lines) and cannot make the run irreproducible by accident.

## Mental model

The runner ([`runner.md`](runner.md)) drives each benchmark in this order:

1. `configure(sandbox=...)` passes on the run's `--sandbox`. Then `preflight()` runs before the model loads: IFEval uses it to fetch NLTK data, and the code benchmarks check for Docker or a compiler. Failing here costs nothing. A suite member whose preflight raises `ToolchainMissing` is skipped instead of stopping the run.
2. `load()` returns every item in a stable order. `select_items(items, settings.limit, settings.seed)` then picks the sample, also before the model loads.
3. The decoding parameters are the profile's resolved parameters `.merged(**generation_overrides())`. `merged` drops `None` values, so an override can change a parameter but cannot unset one. The settings fingerprint, which decides whether a saved run may be resumed, includes `fingerprint_extra()` when that returns anything.
4. `messages(item)` builds the conversation: `[system?, user]` by default.
5. The model answers. The **answer budget** is `item.max_tokens or answer_tokens` (see below).
6. `grade(item, content)` sees only the final answer text, never the reasoning (thinking) stream. If `blocking_grader` is true it runs in a worker thread. An exception is recorded as `grader failed: ...` against that question.
7. `aggregate(records, items)` adds extra metrics such as IFEval's four official scores. `records` include failed requests, which must count as wrong.

### The answer budget: one default, 8,192 tokens

`Benchmark.answer_tokens = 8192` applies to every shipped benchmark; none overrides it. The reason is in the code comment. A cut-off answer is graded as it stands and usually fails, because it never reached its final line, so a tight cap measures your settings, not the model. With caps of 1,024–2,048 tokens and thinking off, Qwen3.6's first run had 3 of 100 GSM8K answers and 16 of 100 MMLU-Pro answers cut mid-reasoning (12 of those 16 were graded wrong), plus 1 of 27 golden answers, which still passed. They were not loops.

- **Per item:** `BenchItem.max_tokens` overrides the default for that question. A golden-set row's `"max_tokens"` lands here.
- **Thinking on:** the runner adds `--thinking-budget` on top (budget + 8,192). With no thinking budget it uses the larger of 8,192 and the profile's `max_tokens`. The result is capped at 131,072.
- **`--max-tokens`:** replaces the whole computation. See `_max_tokens` in [`runner.md`](runner.md).
- **Spotting a binding cap:** the summary field `n_truncated`, shown in the **cut off** column. It counts answers that ended with `finish_reason == "length"`. If it is not zero, the report warns: "Some answers hit the token limit (cut off) and were graded as they stood, usually wrong; raise --max-tokens or --thinking-budget." Usually, not always: a fallback rule or a `contains` grader can find the right answer in the unfinished text, as it did for the other 4 of those 16 MMLU-Pro answers and for the golden one.

A long budget costs nothing on questions the model answers briefly; generation stops at end-of-sequence.

## Public API

| Name | Signature | What it's for |
|---|---|---|
| `BenchItem` | `BenchItem(id: str, prompt: str, answer: Any = None, group: str \| None = None, system: str \| None = None, max_tokens: int \| None = None, meta: dict = {})` | One question. `id` is joined on by paired comparisons, so it must be stable across runs and machines. By convention it is `<benchmark>/<dataset key>`. |
| `Grade` | `Grade(correct: bool, score: float \| None = None, extracted: str \| None = None, detail: dict = {})` | A verdict. `score=None` means 1.0/0.0 from `correct`. Accuracy always uses `correct`; `score` feeds `mean_score` in summary.json. |
| `Benchmark.name` | `str` (class attribute; an instance may rename itself) | Registry name and results sub-directory. A golden set loaded from another file becomes `golden-<stem>`. |
| `Benchmark.description` / `.source` | `ClassVar[str]` | Shown by `kumaru bench list`. `source` names the dataset id and paper. |
| `Benchmark.answer_tokens` | `ClassVar[int] = 8192` | Default answer budget, with thinking off. |
| `Benchmark.needs_code_execution` | `ClassVar[bool] = False` | True: the runner refuses the benchmark without `--allow-code-exec`. |
| `Benchmark.blocking_grader` | `ClassVar[bool] = False` | True: `grade` runs via `asyncio.to_thread`, so subprocess graders don't stall the event loop. |
| `Benchmark.training_data` | `ClassVar[bool] = False` | True only for questions from a dataset's *training* split (`gsm8k_train`), whose graded answers may become training data. `kumaru distill build` refuses runs of every other benchmark: training on an evaluation set inflates the score it measures. A split's *name* does not decide it; IFEval's only split is called `train`, yet IFEval is an evaluation set. |
| `Benchmark.preflight` | `preflight(self) -> None` | Raise early if a dependency or data file is missing. The default does nothing. Raise `codeexec.ToolchainMissing` for a missing compiler or runtime, so that a suite can skip this benchmark. |
| `Benchmark.configure` | `configure(self, *, sandbox: str = "none") -> None` | Run options, applied before `preflight`. `sandbox` is `--sandbox` (`codeexec.SANDBOXES`). The default ignores it; the code benchmarks raise `KumaruError("Unknown sandbox '...'")` for a value they don't know. |
| `Benchmark.fingerprint_extra` | `fingerprint_extra(self) -> dict[str, Any]` | Settings outside the run's own that change verdicts, such as the sandbox image or the compiler version. They join the settings fingerprint, so a resumed run with different ones is refused. The default `{}` adds nothing, which leaves the fingerprint as it was before the hook existed. |
| `Benchmark.load` | `load(self) -> list[BenchItem]` (abstract) | Every item of the evaluation split, in a stable order. |
| `Benchmark.messages` | `messages(self, item: BenchItem) -> list[Message]` | The conversation for one item. Override for few-shot or multi-turn formats. |
| `Benchmark.grade` | `grade(self, item: BenchItem, output: str) -> Grade` (abstract) | Score one answer. Must be deterministic. |
| `Benchmark.generation_overrides` | `generation_overrides(self) -> dict[str, Any]` | Decoding settings the task needs regardless of model. HumanEval sets `presence_penalty=0.0` and `repetition_penalty=1.0`, because repetition is correct in code. |
| `Benchmark.aggregate` | `aggregate(self, records: list[ItemRecord], items: dict[str, BenchItem]) -> dict[str, float]` | Extra metrics. `items` maps record id to question. |
| `load_hf_rows` | `load_hf_rows(path: str, name: str \| None, split: str, revision: str \| None = None) -> list[dict[str, Any]]` | `datasets.load_dataset(path, name, split=split, revision=revision)` as dicts. A pinned `revision` (a commit) means a dataset update can't silently change the questions; MultiPL-E pins one. A missing `datasets` raises `DependencyMissingError`; any other failure raises `KumaruError`. |
| `select_items` | `select_items(items: list[BenchItem], limit: int \| None, seed: int) -> list[BenchItem]` | Deterministic stratified sample (rules below). |

### What `-n/--limit` picks (`select_items`)

- `limit is None` (`--full`) or `limit >= len(items)` returns all items in order. `limit <= 0` returns `[]`.
- `random.Random(seed)` drives the choice, so the same `(items, limit, seed)` always gives the same questions on any machine. That is what makes two runs paired.
- **One group** (no item has a `group`): `rng.sample(items, limit)`.
- **Several groups** (`group or ""` is the key): each group gets a quota proportional to its size, using largest-remainder rounding. Ties go to the alphabetically first group. Each group is then sampled with the same RNG, in sorted key order. Small groups can get zero: `golden -n 10` drops the one-item `language` group.
- The result is re-sorted into the dataset's order.
- **A larger `-n` is not a superset of a smaller one.** For grouped benchmarks the quotas change, and with them the RNG stream. MMLU-Pro at `-n 100` and `-n 200` shared only 10 questions. For ungrouped sets the larger sample usually contains the smaller, but nothing guarantees it. Compare runs made with the same `-n` and `--seed`.

## How to use it

### Inspect the sample a run will use (no model needed)

```python
from kumaru.eval.benchmarks.golden import GoldenSet
from kumaru.eval.benchmarks.base import select_items

items = GoldenSet().load()                      # 27 items
print([i.id for i in select_items(items, 5, 1234)])
# ['golden/mult', 'golden/strawberry', 'golden/capital-au', 'golden/sum-squares', 'golden/banana-caps']
```

The public benchmarks work the same way: `GSM8K().load()` and so on. Each needs the dataset in the Hugging Face cache, and downloads it on first use.

### Try a benchmark without touching the registry module

`BENCHMARKS.register` is a decorator. A benchmark registered at runtime runs through the normal runner. This works offline with the echo profile:

```python
import asyncio
from kumaru.core.config import load_config
from kumaru.core.registry import BENCHMARKS
from kumaru.eval.benchmarks.base import BenchItem, Benchmark, Grade
from kumaru.eval.benchmarks.runner import BenchSettings, run_benchmarks

@BENCHMARKS.register("echo-check")
class EchoCheck(Benchmark):
    name = "echo-check"
    description = "Does the reply quote the prompt?"
    source = "inline"

    def load(self):
        return [BenchItem(id=f"echo-check/{w}", prompt=w, answer=w) for w in ("alpha", "beta", "gamma")]

    def grade(self, item, output):
        return Grade(correct=item.answer in output, extracted=output[:40])

out, [s] = asyncio.run(run_benchmarks(load_config("test"), ["echo-check"], BenchSettings(), profile="test"))
print(s.accuracy, s.params["max_tokens"])   # 1.0 8192
```

Registering the same name twice raises `RegistryError`. Restart the interpreter between experiments.

## How to extend it

- **The contract, as rules:**
  - `load` is stable and complete; let the runner do the sampling.
  - ids are unique and stable.
  - `grade` is pure and deterministic, and puts what it read into `extracted` and how it read it into `detail["method"]` (that feeds the report's `answers found by:` line). A grader that judges the whole reply, as `contains` does, records no method; the report counts those answers as `whole reply`. A code benchmark records its test outcome in `detail["status"]` (`passed`, `failed`, `timeout` or, for a compiled language, `compile_error`) instead, and the line becomes `tests: passed 80, failed 20`.
  - Expensive imports go inside methods.
- **Use the shared extractors** in [`extract.md`](extract.md) instead of new regexes. They are tested, and their method names are already understood by the report.
- **Few-shot or multi-turn prompts:** override `messages()`. Remember the prompt is part of the measurement; changing it changes every record key, so old results won't be reused.
- **A benchmark-wide budget other than 8,192:** override `answer_tokens`, but only with evidence; `n_truncated` is the evidence that a cap binds. Prefer per-item `max_tokens` for single outliers. The tests read the default from `GoldenSet.answer_tokens` (`BUDGET` in `tests\test_bench_runner.py`, `MINI_SET` in `tests\test_cli.py`), so changing it needs no test edits.
- **Partial credit:** return `Grade(score=...)`. The headline accuracy and the paired comparison still use `correct`. Report the graded metric through `aggregate` (as IFEval does).
- **Verdicts that depend on something outside the settings** (a compiler, a sandbox image): return it from `fingerprint_extra()`, as MultiPL-E does with the toolchain's version. Return `{}` when nothing applies, so existing runs keep their fingerprint and still resume.
- **Running model-written code:** set `needs_code_execution` and `blocking_grader`, and run it through [`codeexec.md`](codeexec.md), which handles the temp folder, limits, process-tree kill and `--sandbox docker`.
- Registration and tests: the end-to-end walkthrough is in [`__init__.md`](__init__.md#add-a-benchmark-end-to-end).

## Failure modes

| Symptom | Likely cause | Fix |
|---|---|---|
| `TypeError: Can't instantiate abstract class ...` | `load` or `grade` is not implemented. | Implement both. |
| `TypeError` from `BENCHMARKS.create` | `__init__` takes required arguments. | Give every constructor argument a default (`GoldenSet(path=None)` is the model). |
| `Optional dependency 'datasets' is not installed.` | The `bench` extra is missing. | `pip install -e ".[bench]"`. |
| `Could not load dataset '<id>' (split '<split>'): ...` | No internet on first use, a wrong id or split, or Hub trouble. | Fix the id or split. After one successful download, set `$env:HF_DATASETS_OFFLINE='1'` to force cache use. |
| Comparison shows many `unpaired` questions | The runs used different `-n`/`--seed`, or the ids are not stable. | Re-run with the same `-n` and `--seed`. Derive ids from dataset keys, not row order, where possible. |
| A subject is missing from `by group:` | Small `-n`: largest-remainder rounding gave that group zero items. | Raise `-n`, or use `--full`. |
| Many **cut off** answers (`n_truncated` > 0) | The answer budget binds. Those answers were graded as they stood, usually wrong. | Raise `--max-tokens` (thinking off) or `--thinking-budget` (thinking on). Check `finish_reason` in items.jsonl. |
| `grader failed: ...` in a record, or `Stopped <benchmark>: 5 questions failed in a row. Last error: grader failed: ...` | `grade` raised. One failure counts that question as wrong and the run continues; five in a row stop the benchmark. The hint says it: an error starting `grader failed` is the benchmark's own bug, not the backend's. | Fix the grader, then start a new run (omit `--out`). The fingerprint covers settings, not code, so resuming with `-o <run dir>` would re-ask only the failed questions and keep the old grader's verdicts on the rest. |

## Related files

- `src\kumaru\eval\benchmarks\runner.py` — calls these hooks in the order above ([`runner.md`](runner.md)).
- `src\kumaru\eval\benchmarks\extract.py` — shared answer extractors ([`extract.md`](extract.md)).
- `src\kumaru\core\registry.py` — `BENCHMARKS` (`register`, `register_lazy`, the `kumaru.benchmarks` entry-point group).
- `src\kumaru\core\types.py` — `Message`, `GenerationParams.merged`.
- `tests\test_bench_graders.py` — `TestSelectItems` pins the sampler (determinism, proportional quotas, tie-breaking).
- [`docs\BENCHMARKS.md`](../../../../../BENCHMARKS.md) — results and grader validation.

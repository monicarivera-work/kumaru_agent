# `src\kumaru\eval\benchmarks\__init__.py`

> Package entry for `kumaru bench`. It re-exports the benchmark contract (`Benchmark`, `BenchItem`, `Grade`, `select_items`) and its docstring maps the modules.

**Read this when:** you are new to the benchmarks package, want the shortest path from install to a compared A/B result, or are adding a benchmark end to end.

---

## What it does

The code is a single import plus `__all__`:

```python
from kumaru.eval.benchmarks.base import BenchItem, Benchmark, Grade, select_items
__all__ = ["BenchItem", "Benchmark", "Grade", "select_items"]
```

Everything else is imported from its own module: the runner, statistics and extractors (`kumaru.eval.benchmarks.runner`, `.stats`, `.extract`), and the individual benchmarks. The benchmarks are not imported here at all. `kumaru.core.registry.BENCHMARKS` registers them lazily, so `import kumaru.eval.benchmarks` and `kumaru bench list` never import `datasets` or download anything.

The package answers two questions: how good is the model, and did a change really help? It works on public datasets plus your own golden set. It keeps every answer, puts a 95% confidence interval on every accuracy, and tests every comparison for significance. The v0.1 harness (`kumaru eval`) answers a different question: is this setup working, and how fast? [`..\harness.md`](../harness.md) explains when to use which.

## Why it exists

A single accuracy on 100 questions moves by several points by chance. A model swap, a quantization change or a thinking budget will feel like an improvement because you chose it. This package exists to make such claims checkable:

- **Fixed samples.** The same `--seed` and `-n` give the same questions.
- **Per-question seeds.** Both runs of a comparison see identical inputs.
- **Every answer kept.** Each answer, extraction and verdict is on disk, so a surprising number can be read question by question.
- **Statistics that match the design.** Wilson intervals per run, and exact McNemar tests with a paired bootstrap per comparison.

## Mental model

1. **A benchmark is three decisions:**
   - which questions (`load`);
   - how they are asked (`messages`);
   - how answers are scored (`grade`, deterministic).

   See [`base.md`](base.md).
2. **The runner does the rest:**
   - samples questions and loads the model once;
   - asks with the same decoding `kumaru chat` uses, plus a per-question seed;
   - gives every question the same default answer budget of 8,192 tokens (`Benchmark.answer_tokens`), plus the thinking budget when thinking is on, because a cut-off answer usually fails (12 of the 16 MMLU-Pro answers cut at an old 2,048 cap were graded wrong). `n_truncated`, the **cut off** column, is how you spot a cap that binds;
   - grades each answer as it arrives and appends it to `items.jsonl`;
   - summarizes to `summary.json`.

   See [`runner.md`](runner.md).
3. **Every number is an estimate.** Accuracy comes with a Wilson 95% CI. A comparison pairs the two runs question by question and reports an exact McNemar p-value, Holm-corrected across benchmarks. See [`stats.md`](stats.md).
4. **Module map:**

| Module | Role | Page |
|---|---|---|
| `base` | `Benchmark` contract, `load_hf_rows`, `select_items` (the `-n` sampler) | [`base.md`](base.md) |
| `extract` | Final-answer extraction (numbers, letters, code) with a recorded method | [`extract.md`](extract.md) |
| `stats` | Wilson CI, exact McNemar, paired bootstrap, Holm | [`stats.md`](stats.md) |
| `runner` | Run, resume, summarize, compare, render | [`runner.md`](runner.md) |
| `golden` | Your own JSONL regression set, 7 graders | [`golden.md`](golden.md) |
| `gsm8k`, `mmlu_pro`, `ifeval`, `humaneval` | Public benchmarks | [`gsm8k.md`](gsm8k.md), [`mmlu_pro.md`](mmlu_pro.md), [`ifeval.md`](ifeval.md), [`humaneval.md`](humaneval.md) |
| `ifeval_checks` | IFEval's 25 instruction checkers, strict/loose | [`ifeval_checks.md`](ifeval_checks.md) |

## Public API

| Name | Signature | What it's for |
|---|---|---|
| `Benchmark` | `class Benchmark(ABC)` | Base class: subclass it to add a benchmark. |
| `BenchItem` | `class BenchItem(BaseModel)` | One question: `id`, `prompt`, `answer`, `group`, `system`, `max_tokens`, `meta`. |
| `Grade` | `class Grade(BaseModel)` | A verdict: `correct`, `score`, `extracted`, `detail`. |
| `select_items` | `select_items(items: list[BenchItem], limit: int \| None, seed: int) -> list[BenchItem]` | Deterministic, stratified subsample (what `-n/--limit` uses). |
| `__all__` | `list[str]` | Exactly the four names above. |

## How to use it

### Install the optional dependencies

```powershell
pip install -e ".[bench]"
```

The `bench` extra installs `datasets>=2.20`, `langdetect>=1.0.9`, `nltk>=3.9.1`, `numpy>=1.26` and `regex>=2023.0`. `datasets` downloads the public sets. `langdetect`, `nltk` and `regex` are for IFEval's checkers. numpy makes the bootstrap in `bench compare` about 1000× faster; without it the bootstrap still runs, in pure Python ([`stats.md`](stats.md)). The `all` extra includes `bench`.

The golden set needs none of these packages. A missing one surfaces as `Optional dependency '<pkg>' is not installed.` with the hint `Install it with: pip install -e ".[bench]" in the Kumaru folder` (the same on Windows: `setup.ps1` has no switch for this extra).

### See what is available

```powershell
kumaru bench list
```

```
│ name      │ what it measures                             │ source                                       │ runs code │
│ golden    │ Your own regression set: JSONL questions     │ data/bench/golden.jsonl (yours to extend)    │           │
│ gsm8k     │ Grade-school math word problems: multi-step  │ openai/gsm8k (main/test, 1,319 problems) -   │           │
│ humaneval │ Python function synthesis graded by hidden   │ openai/openai_humaneval (164 problems) -     │    yes    │
│ ifeval    │ Verifiable instruction following: 25         │ google/IFEval (541 prompts) - Zhou et al.    │           │
│ mmlu_pro  │ Hard 10-option multiple choice, 14 subjects: │ TIGER-Lab/MMLU-Pro (test, 12,032 questions)  │           │
Any .jsonl file in the golden-set format also works as a benchmark (see data/bench/golden.jsonl).
```

(Trimmed. `mmlu-pro` is a hidden alias of `mmlu_pro`.)

### Check the plumbing offline (no model, no downloads)

```powershell
kumaru bench run golden -c test
```

`configs\test.yaml` uses the echo backend. Echo only repeats the prompt, so expect a very low accuracy; that's correct. This run proves that loading, grading, the files on disk and `report` work. It writes `runs\bench\<YYYYmmdd-HHMMSS>-test-default\`.

### A real run, then a real comparison

```powershell
kumaru bench run golden gsm8k mmlu_pro ifeval -c qwen25-7b-gguf
kumaru bench run golden gsm8k mmlu_pro ifeval -c qwen36
kumaru bench compare runs\bench\<baseline-run> runs\bench\<candidate-run> --examples 3
```

Both runs use the defaults `-n 100 --seed 1234`, so they answer the same questions and the comparison is fully paired. Results and grader validation are in [`docs\BENCHMARKS.md`](../../../../../BENCHMARKS.md). Do not quote numbers from these pages.

### From Python

```python
import asyncio
from kumaru.core.config import load_config
from kumaru.eval.benchmarks.runner import BenchSettings, run_benchmarks

cfg = load_config("test")
out_dir, summaries = asyncio.run(
    run_benchmarks(cfg, ["golden"], BenchSettings(limit=10), profile="test")
)
print(out_dir, summaries[0].accuracy, summaries[0].ci95)
```

## How to extend it

### Add a benchmark end to end

1. **Write the module.** For example `src\kumaru\eval\benchmarks\mymath.py`. Subclass `Benchmark` and follow these rules:
   - The class must be constructible with no arguments, because `BENCHMARKS.create(name)` calls `cls()`.
   - Give items stable ids of the form `<name>/<dataset key>`.
   - `grade` must be deterministic and record what it read.
   - Keep the inherited 8,192-token `answer_tokens`. A tighter cap cuts answers, which then usually fail, so it measures the setting, not the model. On the first real run, check that `n_truncated` (**cut off**) is 0.

   ```python
   from kumaru.eval.benchmarks.base import BenchItem, Benchmark, Grade, load_hf_rows
   from kumaru.eval.benchmarks.extract import extract_final_number, numbers_equal

   PROMPT = 'Solve this. End with a last line "Answer: <number>".\n\n{question}'

   class MyMath(Benchmark):
       name = "mymath"
       description = "What it measures, in one line (shown by `kumaru bench list`)."
       source = "your-org/your-dataset (test) - citation"

       def load(self) -> list[BenchItem]:
           rows = load_hf_rows("your-org/your-dataset", None, "test")
           return [
               BenchItem(id=f"mymath/{i}", prompt=PROMPT.format(question=r["question"]), answer=r["answer"])
               for i, r in enumerate(rows)
           ]

       def grade(self, item: BenchItem, output: str) -> Grade:
           value, method = extract_final_number(output)
           return Grade(correct=numbers_equal(value, item.answer), extracted=value, detail={"method": method})
   ```
2. **Register it.** Add a lazy registration next to the others in `src\kumaru\core\registry.py`:

   ```python
   BENCHMARKS.register_lazy("mymath", "kumaru.eval.benchmarks.mymath:MyMath")
   ```

   A separate package can declare an entry point in the `kumaru.benchmarks` group instead, for example `mymath = "my_pkg.bench:MyMath"`. Entry-point benchmarks run by name but are not listed by `kumaru bench list`: `Registry.names()` does not scan entry points.
3. **Test the grader offline.** Add `tests\test_bench_mymath.py`. Build `BenchItem`s by hand, so no dataset is downloaded, and assert `grade()` on right answers, formatting variants and wrong answers. Copy the patterns in `tests\test_bench_graders.py` and `tests\test_bench_extract.py`. For a runner-level test, use the `echo_config` and `scratch_dir` fixtures the way `tests\test_bench_runner.py` does. Run only your file:

   ```powershell
   .\.venv\Scripts\python.exe -m pytest tests\test_bench_mymath.py -q -p no:cacheprovider
   ```
4. **Validate before trusting numbers.** Run the official scorer, if one exists, on the same outputs and compare the verdicts, as [`docs\BENCHMARKS.md`](../../../../../BENCHMARKS.md) does in its grader-validation section ("Are the graders right?"). Read the `answers found by:` line of a first real run.
5. **Document it.** Add `docs\howto\src\kumaru\eval\benchmarks\mymath.md`.

## Failure modes

| Symptom | Likely cause | Fix |
|---|---|---|
| `Optional dependency 'datasets' is not installed.` | The `bench` extra is not installed. | `pip install -e ".[bench]"`. |
| `Could not load dataset '<id>' (split '<split>'): ...` | First use without internet, or a Hugging Face Hub problem. | Connect once. Later runs are served from the HF cache (`HF_HOME`, default `~/.cache/huggingface`). |
| `Unknown benchmark: 'x'.` | Typo, or a plugin is not installed. | The hint lists the available names. A path ending in `.jsonl` is always treated as a golden set. |
| `'humaneval' grades answers by running model-written code on this machine.` | Code execution needs consent. | Read [`humaneval.md`](humaneval.md), then add `--allow-code-exec`, or drop it from the list. |
| A plugin benchmark runs but is missing from `kumaru bench list` | `names()` ignores entry points. | Run it by name. This is a known limitation. |
| `ImportError` for `run_benchmarks` from `kumaru.eval.benchmarks` | Only the four `base` names are re-exported. | Import from `kumaru.eval.benchmarks.runner`. |

## Related files

- `src\kumaru\eval\benchmarks\base.py` — the contract re-exported here.
- `src\kumaru\eval\benchmarks\runner.py` — runs, resumes and compares.
- `src\kumaru\core\registry.py` — `BENCHMARKS` and its lazy registrations.
- `src\kumaru\cli.py` — `kumaru bench list|run|report|compare`.
- `data\bench\golden.jsonl` — the shipped golden set ([`golden.jsonl.md`](../../../../data/bench/golden.jsonl.md)).
- `tests\test_bench_graders.py`, `test_bench_extract.py`, `test_bench_ifeval.py`, `test_bench_runner.py`, `test_bench_stats.py` — pin grader, extraction, statistics and runner behaviour.
- [`docs\BENCHMARKS.md`](../../../../../BENCHMARKS.md) — results and grader validation.
- [`..\harness.md`](../harness.md) — the v0.1 `kumaru eval` harness, and when to use which.

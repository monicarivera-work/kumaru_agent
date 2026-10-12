# `src\kumaru\eval\benchmarks\gsm8k.py`

> GSM8K: 1,319 grade-school maths word problems, asked zero-shot with "think step by step", graded by comparing the final number with the gold answer.

**Read this when:** you run `gsm8k`, a GSM8K score looks wrong, or you want a fast check that a model can chain several arithmetic steps.

---

## What it does

- **`GSM8K.load()`** reads `openai/gsm8k`, config `main`, split `test` (1,319 problems) through `load_hf_rows` (see [`base.md`](base.md)). Each problem becomes a `BenchItem`:
  - `id` = `gsm8k/<index>`, the 0-based position in the test split, e.g. `gsm8k/409`.
  - `prompt` = `PROMPT` with the question (stripped) filled in.
  - `answer` = `gold_answer(row["answer"])`: the text after the last `####`, stripped, commas removed. It is a string such as `"168000"`.
  - No `group`, so `-n` draws a plain seeded sample of the 1,319 (`select_items`).
- **`GSM8K.grade()`** runs `extract_final_number(output)`, then `numbers_equal(value, item.answer)`. The comparison is exact after removing `,` and `$`; only an absolute 1e-9 absorbs float noise. It records the extracted number text and `detail.method`.
  - Exact matters on large answers. An earlier relative tolerance of 1e-6 graded 1,450,001 correct for `gsm8k/611` (gold 1,450,000), and 2,880,001 for `gsm8k/796`. [`scripts\check_graders.py`](../../../../scripts/check_graders.py.md) caught both.
- Nothing else: no `preflight`, no `generation_overrides`, no `aggregate`. The only score is accuracy.

The exact prompt (part of the measurement; see [How to extend it](#how-to-extend-it)):

```
Solve the following math problem. Think step by step, then write the final answer on its own last line in the form "Answer: <number>", with no units.

{question}
```

## Why it exists

- **It tests chaining.** Each problem needs 2–8 steps of simple arithmetic, so the score drops whenever a model loses a step. That is the property that makes multi-step answers trustworthy.
- **It is cheap and unambiguous.** Every gold answer is an integer and grading is numeric, so "18", "18.00", "$18", "18 dollars" and "1,234" vs "1234" never decide a verdict. A wrong GSM8K answer is usually a real reasoning error or a cut-off, rarely a grader artefact.
- **It is a floor, not a ceiling.** Frontier models score above 95%. A local model far below that has a problem worth finding: chat template, thinking switch, sampling, or quantisation.

Zero-shot means no worked examples in the prompt. Published GSM8K numbers often use few-shot prompts and their own answer extraction, so compare Kumaru runs with each other (`kumaru bench compare`), not with model cards.

## Mental model

```
question -> PROMPT -> model reasons -> last line "Answer: 42" -> extract_final_number -> numbers_equal(gold)
```

- **Extraction order** (full rules in [`extract.md`](extract.md)): the last `Answer:` tag (the first number on that line or the next two), then the last `\boxed{}`, then the last "answer is", then the last number anywhere, else `none`. Units, `$`, `%` and `**bold**` around the number don't matter. `Answer: eighteen` has no number and is `none`, which is scored wrong.
- **`answers found by:`** in the report shows how often the model followed the format. `answer_tag 100` means every answer ended with the requested line. Many `last_number` or `none` answers mean the model ignores the format, which is a finding about the model or its template, not about maths.
- **A wrong answer has one of three causes**, and `items.jsonl` tells them apart:

  | What the record shows | Cause |
  |---|---|
  | `finish_reason: "length"` | Cut off by the token budget (counted in `n_truncated`, the **cut off** column). |
  | `detail.method` is `last_number` or `none` | The format was not followed; the grader may have read a stray number. |
  | `detail.method` is `answer_tag`, the number is wrong | A genuine reasoning error. Example from a real run: `gsm8k/409`, expected `168000`, the model's `Answer: 42000`. |

- **Budget:** the default 8,192 answer tokens (`Benchmark.answer_tokens`), plus the thinking budget when thinking is on. A cut answer is graded as it stands and, lacking its `Answer:` line, is usually wrong. With the earlier, tighter cap of 1,024 tokens, Qwen3.6 (thinking off) was cut mid-reasoning on 3 of 100 GSM8K answers, and all 3 were graded wrong. At 8,192 a cut-off should be rare. When **cut off** (`n_truncated`) is not zero, read those outputs: a repetition loop needs different sampling, while genuinely long reasoning needs a bigger `--max-tokens`.

## Public API

| Name | Signature | What it's for |
|---|---|---|
| `GSM8K` | `GSM8K()` | The benchmark, registered lazily as `gsm8k`. `name = "gsm8k"`. `description = "Grade-school math word problems: multi-step arithmetic reasoning."` `source = "openai/gsm8k (main/test, 1,319 problems) - Cobbe et al. 2021"`. |
| `GSM8K.load` | `load(self) -> list[BenchItem]` | All 1,319 test items, in dataset order. Needs `datasets` (the `bench` extra) and, the first time, the internet. |
| `GSM8K.grade` | `grade(self, item: BenchItem, output: str) -> Grade` | Numeric comparison. `extracted` is the number as written (e.g. `"1,234"`); `detail = {"method": ...}`. |
| `gold_answer` | `gold_answer(solution: str) -> str` | GSM8K's solution text -> the gold number: the part after the last `####`, with commas removed. |
| `PROMPT` | `str` with one `{question}` field | The user message template. |

## How to use it

### Run it

```powershell
kumaru bench run gsm8k -c qwen36                                  # 100 seeded problems, thinking off (qwen36's default)
kumaru bench run gsm8k -c qwen36 --full                           # all 1,319
kumaru bench run gsm8k -c qwen36 --think --thinking-budget 8192   # budget per question: 8,192 + 8,192
kumaru bench report runs\bench\<run>\gsm8k
```

A real report line (Qwen2.5-7B, illustrative only; results are in [`docs\BENCHMARKS.md`](../../../../../BENCHMARKS.md)):

```
│ gsm8k     │ 100 │       92.0 │ 85.0-95.9 │      0 │       0 │           244 │        47.8 │
gsm8k  answers found by: answer_tag 100
```

At n = 100 the 95% interval around 92% runs from 85.0 to 95.9, 11 points wide, and it is wider for scores nearer 50%. Use `--full` or a paired `compare` before you believe a difference of a few points ([`stats.md`](stats.md)).

### Check the grader on a reply (offline)

```python
from kumaru.eval.benchmarks.base import BenchItem
from kumaru.eval.benchmarks.gsm8k import GSM8K, gold_answer

gold_answer("She has 3 + 4 = 7 pens.\n#### 1,234")       # '1234'
item = BenchItem(id="gsm8k/0", prompt="p", answer="1234")
GSM8K().grade(item, "So...\nAnswer: 1,234")
# Grade(correct=True, score=None, extracted='1,234', detail={'method': 'answer_tag'})
GSM8K().grade(item, "Answer: $1,234.00")                  # correct, extracted '1,234.00'
GSM8K().grade(BenchItem(id="gsm8k/611", prompt="p", answer="1450000"), "Answer: 1,450,001")
# Grade(correct=False, score=None, extracted='1,450,001', detail={'method': 'answer_tag'})  - exact, however large
```

### See why answers were wrong

```powershell
Get-Content runs\bench\<run>\gsm8k\items.jsonl | ForEach-Object { $_ | ConvertFrom-Json } |
  Where-Object { -not $_.correct } |
  Select-Object id, expected, extracted, @{n='method'; e={$_.detail.method}}, finish_reason, error
```

Then read `output` for the ids that matter. The table above tells you which of the three causes you are looking at.

## How to extend it

- **Don't edit `PROMPT` casually.** The prompt is part of the measurement: change it and earlier runs are no longer comparable, even though ids still pair up in `compare`. Resuming an old run after a prompt change re-asks every question, because each record's `key` covers the messages sent.
- **A variant dataset with the same fields** (`question`, and `answer` ending in `#### <number>`): subclass `GSM8K` and override `name`, `description`, `source` and `load` (copy `load` with a different dataset id), reusing `grade`. Then register it and add tests, as in [`__init__.md`](__init__.md#add-a-benchmark-end-to-end).
- **Few-shot prompting:** override `messages(item)` to put worked examples before the question as earlier user/assistant turns. Give it a new `name`: it is a different benchmark.
- **Answers that aren't plain numbers** (fractions, expressions) need a new grader, not a looser `extract_final_number`. GSM8K's answers are integers, and its extractor is pinned by tests.

Pinned behaviour: `tests\test_bench_graders.py::TestGsm8kAndMmluPro` (`test_gsm8k_gold_answer_is_the_number_after_the_marker`, `test_gsm8k_grade_accepts_formatted_numbers`) and the number-extraction and matching cases in `tests\test_bench_extract.py`, including `test_numbers_equal_is_exact_near_large_values_unless_asked`.

## Failure modes

| Symptom | Likely cause | Fix |
|---|---|---|
| `Optional dependency 'datasets' is not installed.` | The `bench` extra is missing. | `pip install -e ".[bench]"` |
| `Could not load dataset 'openai/gsm8k' (split 'test'): ...` | First run without internet, or the Hub is unreachable. | Connect once. Later runs read the Hugging Face cache (`HF_HOME`, default `~\.cache\huggingface`). |
| Many wrong answers with method `last_number` or `none` | The model doesn't end with `Answer: <number>`. Often a broken chat template or a thinking-mode mismatch. | Read a few `output`s. Check the model in `kumaru chat` with the same flags. |
| `cut off` > 0 | Answers hit the 8,192-token budget (or your `--max-tokens`): a repetition loop, or genuinely long reasoning. A cut answer is graded as it stands and is usually wrong. | Read those outputs. If the reasoning is real, raise `--max-tokens` (or `--thinking-budget` with thinking on). If it loops, look at the profile's `presence_penalty` and temperature. |
| A correct number scored wrong | The extractor read another number: a restated quantity on the `Answer:` line, or no tag, so `last_number` took a check step. | Inspect `extracted` and `output`. A systematic pattern is worth a grader test, not a silent loosening. |
| Score far from the model card | Different protocol: few-shot, other extraction, other sampling. | Compare Kumaru runs with each other. |
| Two runs of one model differ by 3–5 points | Sampling noise at n = 100. | `kumaru bench compare A B` ([`runner.md`](runner.md#compare-two-runs)). |

## Related files

- `src\kumaru\eval\benchmarks\extract.py` — `extract_final_number`, `numbers_equal` ([`extract.md`](extract.md)).
- `src\kumaru\eval\benchmarks\base.py` — `Benchmark`, `load_hf_rows`, `select_items`, the 8,192 default ([`base.md`](base.md)).
- `src\kumaru\eval\benchmarks\runner.py` — budgets, records, report and compare ([`runner.md`](runner.md)).
- `src\kumaru\core\registry.py` — `BENCHMARKS.register_lazy("gsm8k", ...)`.
- `tests\test_bench_graders.py`, `tests\test_bench_extract.py` — pinned grading behaviour.
- [`docs\BENCHMARKS.md`](../../../../../BENCHMARKS.md) — results and grader validation.

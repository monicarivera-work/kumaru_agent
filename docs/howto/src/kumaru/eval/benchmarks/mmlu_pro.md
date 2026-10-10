# `src\kumaru\eval\benchmarks\mmlu_pro.py`

> MMLU-Pro: 12,032 hard multiple-choice questions with up to ten options in 14 subjects, asked zero-shot with "think step by step", graded by the letter the answer ends on.

**Read this when:** you run `mmlu_pro`, you want a knowledge-plus-reasoning score that separates strong models, or you need to know why an answer letter was read the way it was.

---

## What it does

- **`MMLUPro.load()`** reads `TIGER-Lab/MMLU-Pro`, split `test` (12,032 questions), through `load_hf_rows` ([`base.md`](base.md)). Each row becomes a `BenchItem`:
  - `id` = `mmlu_pro/<question_id>`, the dataset's own id, e.g. `mmlu_pro/1165`.
  - `prompt` = `PROMPT` with the subject, the question and the options as `A. text` lines (each option stripped, so stray newlines can't split the list).
  - `answer` = the gold letter, `LETTERS[answer_index]`.
  - `group` = the category, e.g. `law`. Accuracy is reported per group, and `-n` samples **stratified by subject**.
  - `meta = {"letters": "ABCD..."}`: the letters that exist for this question.
- **`clean_options(options, answer_index)`** removes the dataset's `"N/A"` padding options, as the reference script does, **unless** removing them would move the correct answer to another letter. Then all options are kept.
- **`MMLUPro.grade()`** calls `extract_choice(output, item.meta["letters"])` and compares the letter with `answer`. It records the letter as `extracted` and `detail.method`. A letter outside the question's options is never extracted.

The exact prompt:

```
The following is a multiple choice question about {subject}. Think step by step, then finish your answer with "The answer is (X)" where X is the letter of the correct option.

Question: {question}
Options:
A. ...
B. ...
```

## Why it exists

- **Plain MMLU no longer separates strong models; MMLU-Pro does.** Questions have ten options instead of four, so random guessing scores about 10%, not 25%. (Some have fewer real options; the dataset pads them with `"N/A"`.) Trivial and noisy questions were removed and reasoning-heavy ones added.
- **It covers breadth.** 14 subjects: biology, business, chemistry, computer science, economics, engineering, health, history, law, math, other, philosophy, physics, psychology. A model tuned for maths and code shows its gaps here.
- **It is still cheap to grade.** One letter, extracted in the reference script's order, no judge model.

## Mental model

- **Extraction order** (from the MMLU-Pro reference script; details in [`extract.md`](extract.md)):
  1. `answer_is`: the last "answer is (X)" (parentheses optional).
  2. `answer_tag`: "Answer: X".
  3. `boxed`: a letter inside the last `\boxed{}`.
  4. `last_letter`: the last standalone capital letter anywhere. **Weak:** it can catch the pronoun "I" or a stray initial.
  5. `none`: nothing found.
- **One deliberate difference from the reference.** When no letter can be extracted, the reference script scores a random guess; Kumaru scores the answer **wrong**. Results are therefore deterministic and slightly conservative. The bias is bounded by the share of `none` in `answers found by:`.
- **Read `answers found by:` first.** `answer_is 100` means every answer ended as asked. A sizeable `last_letter` count means some verdicts rest on the weak fallback; check those outputs before trusting the score.
  - An extracted `I` is not always the pronoun. In one real run, `mmlu_pro/1165` (law, gold `C`) extracted `I` by `answer_is`: the model really wrote "The answer is (I)." The `method` tells the two apart.
- **Per-subject accuracy is mostly noise at `-n 100`.** Stratification gives each subject a share proportional to its size: 3 to 11 questions per subject at n = 100. A subject at 6/6 has a Wilson 95% interval of 61.0–100%, and 1/8 has 2.2–47.1%. Use `by group:` to spot a broken subject, not to rank models on one. For subject-level claims, run `--full` or a much larger `-n`.
- **A larger `-n` is not a superset of a smaller one** for this grouped benchmark: changing `-n` changes the quotas and the random stream ([`base.md`](base.md#what--n--limit-picks-select_items)). Compare runs made with the same `-n` and `--seed`.
- **Budget:** the default 8,192 answer tokens (`Benchmark.answer_tokens`), plus the thinking budget when thinking is on. With the earlier, tighter cap of 2,048 tokens, Qwen3.6 (thinking off) was cut mid-reasoning on 16 of its 100 answers. A cut answer is graded as it stands and lacks the final "The answer is (X)" line, so it is usually wrong: 12 of those 16 were. The other 4 passed only because `last_letter` or an earlier "answer is" in the unfinished reasoning happened to hold the right letter. A non-zero **cut off** (`n_truncated`) shows when the cap binds.
- **Protocol caveat.** The MMLU-Pro paper's headline numbers use 5-shot chain of thought; Kumaru asks zero-shot. Treat comparisons with published numbers as approximate, and compare Kumaru runs with each other.

## Public API

| Name | Signature | What it's for |
|---|---|---|
| `MMLUPro` | `MMLUPro()` | The benchmark, registered lazily as `mmlu_pro`, with the alias `mmlu-pro`. `name = "mmlu_pro"` either way, which is also the results folder. `description = "Hard 10-option multiple choice, 14 subjects: knowledge plus reasoning."` `source = "TIGER-Lab/MMLU-Pro (test, 12,032 questions) - Wang et al. 2024"`. |
| `MMLUPro.load` | `load(self) -> list[BenchItem]` | All 12,032 items, in dataset order. |
| `MMLUPro.grade` | `grade(self, item: BenchItem, output: str) -> Grade` | `correct = letter == item.answer`; `extracted` = the letter or `None`; `detail = {"method": ...}`. |
| `clean_options` | `clean_options(options: list[str], answer_index: int) -> list[str]` | Drop `"N/A"` padding only if the answer keeps its letter. |
| `LETTERS` | `"ABCDEFGHIJ"` | Option letters; `LETTERS[: len(options)]` per question. |
| `PROMPT` | `str` with `{subject}`, `{question}`, `{options}` | The user message template. |

## How to use it

### Run it

```powershell
kumaru bench run mmlu_pro -c qwen36                    # 100 questions, stratified by subject
kumaru bench run mmlu_pro -c qwen36 -n 1000            # tighter interval (about ±3 points)
kumaru bench run mmlu_pro -c qwen36 --think --thinking-budget 8192
kumaru bench report runs\bench\<run>\mmlu_pro          # adds the "by group:" line
```

A real report (Qwen2.5-7B, illustrative only; results are in [`docs\BENCHMARKS.md`](../../../../../BENCHMARKS.md)):

```
│ mmlu_pro  │ 100 │       59.0 │ 49.2-68.1 │      0 │       0 │           416 │        25.3 │
mmlu_pro  answers found by: answer_is 100
    by group: biology 100.0 (n=6), business 42.9 (n=7), chemistry 66.7 (n=9), computer science 66.7 (n=3), economics 71.4 (n=7), engineering 12.5 (n=8), health 71.4 (n=7), history 100.0 (n=3), law 44.4 (n=9), math 72.7 (n=11), other 62.5 (n=8), philosophy 75.0 (n=4), physics 45.5 (n=11), psychology 42.9 (n=7)
```

An interval 19 points wide at n = 100 is why MMLU-Pro differences of a few points need a paired `compare` or more questions ([`stats.md`](stats.md)).

### Check the grader on a reply (offline)

```python
from kumaru.eval.benchmarks.base import BenchItem
from kumaru.eval.benchmarks.mmlu_pro import MMLUPro, clean_options

item = BenchItem(id="mmlu_pro/1", prompt="p", answer="C", meta={"letters": "ABCD"})
MMLUPro().grade(item, "...so the answer is (C).")         # correct, method 'answer_is'
MMLUPro().grade(item, "I think it's C but I am not sure")  # correct, method 'last_letter': "I" is not an option here
ten = BenchItem(id="mmlu_pro/2", prompt="p", answer="C", meta={"letters": "ABCDEFGHIJ"})
MMLUPro().grade(ten, "I think it's C but I am not sure")
# Grade(correct=False, score=None, extracted='I', detail={'method': 'last_letter'})   the pronoun, read as option I
MMLUPro().grade(item, "The answer is (E)")
# Grade(correct=False, score=None, extracted=None, detail={'method': 'none'})   E is not an option here
clean_options(["a", "b", "N/A"], 1)    # ['a', 'b']
clean_options(["a", "N/A", "b"], 2)    # ['a', 'N/A', 'b']  dropping N/A would move the answer from C to B
```

### Audit the weak extractions

```powershell
Get-Content runs\bench\<run>\mmlu_pro\items.jsonl | ForEach-Object { $_ | ConvertFrom-Json } |
  Where-Object { $_.detail.method -in 'last_letter', 'none' -or $_.finish_reason -eq 'length' } |
  Select-Object id, group, expected, extracted, @{n='method'; e={$_.detail.method}}, finish_reason
```

## How to extend it

- **Keep the prompt and the extraction order.** Both follow the reference so the numbers stay comparable; changing either makes a different benchmark. If you need a variant, such as 5-shot, subclass `MMLUPro`, override `messages(item)` and give it a new `name`, then register it ([`__init__.md`](__init__.md#add-a-benchmark-end-to-end)).
- **A subject subset** (say only `law` and `health`): subclass and filter in `load()`, under a new `name`. Don't filter by post-processing results; the sampler and the CI should see the subset as the benchmark.
- **Other multiple-choice datasets** can reuse the pattern: list options as `A. text`, put the available letters in `meta["letters"]`, and grade with `extract_choice`.

Pinned behaviour: `tests\test_bench_graders.py::TestGsm8kAndMmluPro` (`test_clean_options_drops_padding_only_when_the_answer_stays_put`, `test_mmlu_pro_grade_only_accepts_letters_that_exist`) and the `extract_choice` cases in `tests\test_bench_extract.py`.

## Failure modes

| Symptom | Likely cause | Fix |
|---|---|---|
| `Optional dependency 'datasets' is not installed.` | The `bench` extra is missing. | `pip install -e ".[bench]"` |
| `Could not load dataset 'TIGER-Lab/MMLU-Pro' (split 'test'): ...` | First run without internet, or the Hub is unreachable. | Connect once; later runs read the Hugging Face cache (`HF_HOME`). |
| `answers found by:` shows many `last_letter` | Answers don't end with "The answer is (X)". The fallback may be reading the pronoun "I". | Read those outputs. Check the chat template and thinking mode in `kumaru chat`. |
| `answers found by:` shows `none` | No letter at all, or a letter that isn't one of the options. Often a cut-off answer. | Check `finish_reason` and the **cut off** column. |
| `cut off` > 0 | Answers hit the 8,192-token budget (or your `--max-tokens`). A cut answer is graded as it stands and is usually wrong; when it is right, `last_letter` or a mid-reasoning "answer is" may have picked the letter by luck. | Read them. Raise `--max-tokens`, or `--thinking-budget` with thinking on, if the reasoning is real; fix the sampling if they loop. |
| Lower-case `(c)` scored wrong | Letters are matched in capitals only, as in the reference. | Expected. The prompt asks for "(X)". |
| A subject's accuracy swings wildly between runs | 3–11 questions per subject at n = 100. | Use `--full` or a larger `-n` for per-subject claims. |
| Score far from the model card | 5-shot vs zero-shot, different sampling, or a random guess on extraction failures. | Compare Kumaru runs with each other. |

## Related files

- `src\kumaru\eval\benchmarks\extract.py` — `extract_choice` ([`extract.md`](extract.md)).
- `src\kumaru\eval\benchmarks\base.py` — `select_items` stratification, `load_hf_rows`, the 8,192 default ([`base.md`](base.md)).
- `src\kumaru\eval\benchmarks\runner.py` — the `by group:` line, budgets and records ([`runner.md`](runner.md)).
- `src\kumaru\eval\benchmarks\stats.py` — Wilson intervals and paired comparisons ([`stats.md`](stats.md)).
- `src\kumaru\core\registry.py` — `register_lazy("mmlu_pro", ..., aliases=("mmlu-pro",))`.
- `tests\test_bench_graders.py`, `tests\test_bench_extract.py` — pinned grading behaviour.
- [`docs\BENCHMARKS.md`](../../../../../BENCHMARKS.md) — results and grader validation.

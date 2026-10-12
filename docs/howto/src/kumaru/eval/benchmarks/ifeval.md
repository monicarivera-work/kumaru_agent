# `src\kumaru\eval\benchmarks\ifeval.py`

> IFEval: 541 prompts, each with one to three machine-checkable instructions ("no commas", "at least 300 words", "end with this exact phrase"). Graded by the 25 checkers in `ifeval_checks.py` and reported as the four official metrics.

**Read this when:** you run `ifeval`, need to know what prompt-level, instruction-level, strict and loose mean, or an IFEval score moved after a prompt, template or fine-tuning change.

---

## What it does

- **`IFEval.preflight()`** runs before the model loads. It calls `detect_language(...)`, which needs `langdetect`, and `count_sentences(...)`, which needs `nltk`. On first use it downloads NLTK's `punkt_tab` sentence model. A missing extra or a failed download therefore fails in seconds, not after the model has answered.
- **`IFEval.load()`** reads `google/IFEval`, split `train`. That is the only split, and it is the evaluation set: 541 prompts. Each row becomes a `BenchItem`:
  - `id` = `ifeval/<key>`, the dataset key, e.g. `ifeval/1000`.
  - `prompt` = the row's prompt, verbatim. The instructions are part of the prompt text. There is no template and no system message, unless you pass `--use-agent-prompt`, which applies to every benchmark.
  - `meta = {"instructions": [...], "kwargs": [...]}`, with one kwargs dict per instruction and `None` values dropped. Arrow stores every possible argument on every row, mostly as `None`. For example, `ifeval/1000` has `['punctuation:no_comma', 'detectable_format:number_highlighted_sections', 'length_constraints:number_words']` with kwargs `[{}, {'num_highlights': 3}, {'relation': 'at least', 'num_words': 300}]`.
  - No `answer` and no `group`, so `-n` draws a plain seeded sample. It is not stratified by instruction type.
- **Unknown instruction ids fail the whole load.** An id that is not in `CHECKS` raises `KumaruError("IFEval prompt <key> uses unknown instructions: [...]")`, with the hint "The dataset changed upstream; add checkers in ifeval_checks.py." This happens before the model loads. Silently dropping those prompts would change the denominator and break comparability.
- **`IFEval.grade(item, output)`** calls `check_instructions` and returns:
  - `correct = all(strict)`;
  - `score = sum(strict) / len(strict)`;
  - `extracted = None`;
  - `detail = {"strict": [...], "loose": [...]}`, one flag per instruction, in prompt order.
- **`IFEval.aggregate(records, items)`** computes the four official metrics and `strict/<category>` per instruction category. The category is the id prefix before `:`.

The dataset has 541 prompts: 305 with one instruction, 179 with two and 57 with three, 834 instructions in all. By category: keywords 163, detectable_format 157, length_constraints 143, change_case 89, startend 67, punctuation 66, combination 65, detectable_content 53, language 31.

## Why it exists

- **It measures obedience, not knowledge.** Can the model follow explicit, checkable constraints on format, length, case and wording? Fine-tuning, a new system prompt or a chat-template change can break this while maths scores stay put.
- **No judge model.** Every verdict comes from a small deterministic program, so a run is cheap, reproducible and auditable instruction by instruction.
- **Comparable with published numbers.** The checkers reimplement the reference implementation's semantics, quirks included ([`ifeval_checks.md`](ifeval_checks.md)). The prompts are sent verbatim, as in the paper.

## Mental model

### The four metrics

| Metric | Counts | A unit passes when |
|---|---|---|
| `prompt_level_strict` | prompts | Every instruction passes on the reply as written. This is `correct` and the headline `accuracy`, and the only one `compare` tests. |
| `instruction_level_strict` | instructions | That instruction passes on the reply as written. |
| `prompt_level_loose` | prompts | Every instruction passes on at least one relaxed reading of the reply. |
| `instruction_level_loose` | instructions | That instruction passes on at least one relaxed reading. |

**Loose** tries 8 readings of the reply, and each instruction may pass on a different one:
- the reply as written;
- without its first line, without its last line, or without both (these three are stripped);
- the same four with every `*` removed.

This forgives "Sure, here it is:" preambles, sign-offs, and markdown bold around text that must start or end the reply. Three consequences:

- **Loose ≥ strict, always.** The first reading is the reply itself.
- **Loose also forgives real violations.** For example, `ifeval/1436` asks for no capital letters and bans "coop", "killings", "dead" and "night". The story opens with `in the dead of night , commandos crouched low…`. Its flags are strict `[True, False]` and loose `[True, True]`: dropping the first line drops the banned words. That is the upstream definition. Quote strict as the headline and read loose as a "formatting noise" gauge.
- **An empty reply follows nothing** in either reading. So does a failed request or a crashed grader: `aggregate` counts all of its instructions as not followed.

### Other numbers to know

- **Per-category strict** (`strict/keywords`, …) is the share of that category's instructions followed. `bench report` shows them. The summary printed by `bench run` shows only the four official metrics, the keys without `/`.
- **`mean_score`** in `summary.json` averages `score` per prompt (a macro average), so it is not `instruction_level_strict` (a micro average over instructions). In the run below they are 0.822 and 0.795. Quote the official metric.
- **Only `output` is graded.** That is the message content. Thinking text goes to `reasoning` and is never checked.
- **No `answers found by:` line, unless questions failed.** IFEval records no extraction method, so the line appears only when some questions errored, and then reads `answers found by: error 2, whole reply 98`.
- **Budget:** the default 8,192 answer tokens (`Benchmark.answer_tokens`), plus the thinking budget when thinking is on. IFEval replies are short; the run below averaged 321 tokens. A cut-off here therefore usually means a loop, not a long answer. A cut reply is graded as it stands, so it typically fails ending, length and format checks. A non-zero **cut off** (`n_truncated`) is how you spot it.
- **Noise:** the seed-1234 sample of 100 prompts holds 156 instructions. Per category that is 6 (language) to 29 (keywords), so `strict/language 100.0` there is 6/6, with a Wilson interval of 61.0–100. Instructions in one prompt share one reply and are not independent, so a binomial interval over 156 instructions would be too narrow. Use `--full` (all 541) for per-category claims.

## Public API

| Name | Signature | What it's for |
|---|---|---|
| `IFEval` | `IFEval()` | The benchmark, registered lazily as `ifeval`. `name = "ifeval"`. `description = "Verifiable instruction following: 25 instruction types, 541 prompts."` `source = "google/IFEval (541 prompts) - Zhou et al. 2023"`. |
| `IFEval.preflight` | `preflight(self) -> None` | Fails early on a missing `langdetect`/`nltk`; downloads `punkt_tab` if absent. |
| `IFEval.load` | `load(self) -> list[BenchItem]` | All 541 prompts in dataset order. Needs `datasets` and, the first time, internet. |
| `IFEval.grade` | `grade(self, item: BenchItem, output: str) -> Grade` | Strict and loose flags per instruction; `correct` is prompt-level strict. |
| `IFEval.aggregate` | `aggregate(self, records: list[ItemRecord], items: dict[str, BenchItem]) -> dict[str, float]` | The four official metrics, then `strict/<category>` (sorted). Each denominator is at least 1. |

## How to use it

### Run it

```powershell
pip install -e ".[bench]"                  # datasets, langdetect, nltk, numpy, regex
kumaru bench run ifeval -c qwen36           # 100 seeded prompts
kumaru bench run ifeval -c qwen36 --full    # all 541
kumaru bench report runs\bench\<run>        # a run, or one benchmark's dir such as runs\bench\<run>\ifeval
```

Real `report` output (Qwen2.5-7B, illustrative only; results are in [`docs\BENCHMARKS.md`](../../../../../BENCHMARKS.md)):

```
│ ifeval    │ 100 │       72.0 │ 62.5-79.9 │      0 │       0 │           321 │        34.4 │
ifeval  prompt_level_strict 72.0  instruction_level_strict 79.5  prompt_level_loose 76.0  instruction_level_loose 82.7  strict/change_case 50.0  strict/combination 71.4  strict/detectable_content 90.9  strict/detectable_format 92.3  strict/keywords 65.5  strict/language 100.0  strict/length_constraints 78.6  strict/punctuation 92.3  strict/startend 93.3
```

### Check the grader on a reply (offline)

These two checks need neither NLTK nor langdetect:

```python
from kumaru.eval.benchmarks.base import BenchItem
from kumaru.eval.benchmarks.ifeval import IFEval

item = BenchItem(id="ifeval/x", prompt="p", meta={
    "instructions": ["punctuation:no_comma", "startend:quotation"], "kwargs": [{}, {}]})
IFEval().grade(item, '"one, two"')
# Grade(correct=False, score=0.5, extracted=None, detail={'strict': [False, True], 'loose': [False, True]})
IFEval().grade(item, 'Sure, here it is:\n"no commas here"')
# Grade(correct=False, score=0.0, extracted=None, detail={'strict': [False, False], 'loose': [True, True]})
```

The second reply follows both instructions after its preamble, so loose forgives it and strict does not.

### Which instructions failed?

`items.jsonl` stores the flags but not the instruction ids; `ItemRecord` does not keep `meta`. Reload the items from the Hugging Face cache and pair them up:

```powershell
$env:HF_DATASETS_OFFLINE = '1'; $env:HF_HUB_OFFLINE = '1'   # read the cached dataset only
@'
from collections import Counter
from pathlib import Path
from kumaru.eval.benchmarks.ifeval import IFEval
from kumaru.eval.benchmarks.runner import load_results

items = {item.id: item for item in IFEval().load()}
summary, records = load_results(Path(r"runs\bench\<run>\ifeval"))
failed = Counter()
for record in records.values():
    ids = items[record.id].meta["instructions"]
    strict = record.detail.get("strict") or [False] * len(ids)  # a failed request followed nothing
    failed.update(i for i, ok in zip(ids, strict) if not ok)
print(failed.most_common(6))
'@ | .\.venv\Scripts\python.exe -
```

On the run above this prints `keywords:frequency` 5, `change_case:english_lowercase` 3, then four instructions with 2 failures each. Read those outputs next: a few kinds of instruction usually explain most of the gap.

## How to extend it

- **A new instruction type** (after an upstream change): add the checker and its `CHECKS` entry in `ifeval_checks.py` ([`ifeval_checks.md`](ifeval_checks.md#how-to-extend-it)). `load` accepts any id that is in `CHECKS`, and `aggregate` derives categories from the id prefix, so this file needs no change.
- **A dataset with the same columns** (`key`, `prompt`, `instruction_id_list`, `kwargs`), such as a translated IFEval: subclass `IFEval` and override `name`, `description`, `source` and `load` (copy `load` with the other dataset id). Register it and add tests, as in [`__init__.md`](__init__.md#add-a-benchmark-end-to-end).
- **Keep the strict/loose semantics.** They are what makes the numbers comparable with published ones. Report extra readings as extra `aggregate` metrics; don't change the four.

Pinned behaviour, in `tests\test_bench_ifeval.py`:
- `TestIFEvalBenchmark`: `grade` and `aggregate`, including a failed request counted as all-False;
- `TestStrictAndLoose`: preambles, asterisks, empty replies;
- the checker classes.

Tests that need NLTK data skip, rather than fail, when it is absent (the `nltk_ready` fixture).

## Failure modes

| Symptom | Likely cause | Fix |
|---|---|---|
| `Optional dependency 'langdetect' is not installed.` (or `'nltk'`) before the model loads | The `bench` extra is missing. | `pip install -e ".[bench]"` |
| `Could not download NLTK's 'punkt_tab' sentence model.` | First run without internet, or a proxy blocks NLTK's download. | Connect once, or run `python -m nltk.downloader punkt_tab`. It goes to NLTK's data directory, usually `%APPDATA%\nltk_data` on Windows. |
| `IFEval prompt <key> uses unknown instructions: [...]` | The dataset changed upstream. | Add the checkers ([`ifeval_checks.md`](ifeval_checks.md)). Until then, `$env:HF_DATASETS_OFFLINE='1'` keeps using a cached copy, if you have one. |
| `grader failed: ValueError: unknown IFEval relation: '...'` in a record | A row uses a relation other than `less than` or `at least`. The current dataset has none. | Extend `_compare` in `ifeval_checks.py`, then resume with `-o <run dir>`; failed questions are asked again. |
| Loose far above strict | Preambles, sign-offs, or `**bold**` around text that must start or end the reply. | Expected; loose exists for this. A large gap says the model wraps its answers, often because of a system prompt or template. |
| Many `english_lowercase` or `english_capital` failures | The whole reply must be in one case (`str.islower()`/`isupper()`). One capital letter, such as the pronoun "I" or an acronym, fails `english_lowercase`; one lowercase letter fails `english_capital`. | That is the instruction, as upstream defines it. Read the outputs. |
| Thinking text in `output`, with case, length or ending checks failing | The backend did not separate reasoning from content. | Check that records have `reasoning` filled. Fix the backend or profile, not the grader. |
| `cut off` > 0 | An answer hit the 8,192-token budget (or your `--max-tokens`). For IFEval that is almost always a repetition loop. | Read the output. Fix the sampling (`presence_penalty`, temperature) rather than raising the cap. |
| A category score jumps between runs | Only 6–29 instructions per category at `-n 100`. | Use `--full`. |

## Related files

- `src\kumaru\eval\benchmarks\ifeval_checks.py` — the 25 checkers, `loose_variants`, `check_instructions` ([`ifeval_checks.md`](ifeval_checks.md)).
- `src\kumaru\eval\benchmarks\base.py` — `Benchmark`, `load_hf_rows`, `select_items`, the 8,192 default ([`base.md`](base.md)).
- `src\kumaru\eval\benchmarks\runner.py` — records, `summarize`, report and compare ([`runner.md`](runner.md)).
- `src\kumaru\core\registry.py` — `BENCHMARKS.register_lazy("ifeval", ...)`.
- `tests\test_bench_ifeval.py` — pinned checker, strict/loose and metric behaviour.
- [`docs\BENCHMARKS.md`](../../../../../BENCHMARKS.md) — results and grader validation.

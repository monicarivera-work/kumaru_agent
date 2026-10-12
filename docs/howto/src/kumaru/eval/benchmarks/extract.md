# `src\kumaru\eval\benchmarks\extract.py`

> Reads the final answer (a number, a choice letter, or a block of Python) out of free-form model text, and reports which rule found it.

**Read this when:** a benchmark scores a correct-looking answer wrong, the report's `answers found by:` line looks odd, or you are writing a grader.

---

## What it does

Each benchmark has to find the model's final answer in free text before it can grade it. This module holds the four extractors the shipped graders use:

- `extract_final_number` — GSM8K and the golden `numeric` grader.
- `extract_choice` — MMLU-Pro and the golden `choice` grader.
- `extract_code` — HumanEval.
- `last_boxed` — used by the two above.

It also holds the number helpers `parse_number` and `numbers_equal`.

The number and choice extractors return `(answer, method)`. The method names the rule that fired. The graders store it in `Grade.detail["method"]`, and the runner counts it into `BenchSummary.extraction`. That count is the `answers found by:` line printed by `kumaru bench run` and `kumaru bench report`.

## Why it exists

A model can be right and still be scored wrong because the grader misread `1{,}080`, or took the "A" out of "A good approach is...". Two design choices limit this:

- **Explicit signal first, weak fallbacks last.** Each extractor follows the order of the benchmark's official script. The prompt asks for the strongest signal: `Answer: <number>` for GSM8K and `The answer is (X)` for MMLU-Pro.
- **The method is recorded.** A run where most answers came from `last_number` or `last_letter` says the model ignored the format. Its accuracy then partly measures the fallback, and you can see that without reading every answer. The share of answers found by the requested format is also a free instruction-following signal.

## Mental model

**`extract_final_number(text)`** first normalises LaTeX: `{,}` and `\,` become `,`, `\$` becomes `$`, and `\!` is removed. Then:

| Order | Method | Rule |
|---|---|---|
| 1 | `answer_tag` | The **last** `answer:` tag (any case; `**Answer:**`, `Final answer:`, the full-width `：` and `answer_:` count). Takes the first number on the rest of that line or the next two lines (3 lines in total). |
| 2 | `boxed` | The first number inside the **last** `\boxed{...}`. Nested braces are handled; unbalanced braces mean no match. |
| 3 | `answer_is` | The **last** "answer is". Takes the first number on the same line only. |
| 4 | `last_number` | The last number anywhere in the text. |
| 5 | `none` | No number at all. Returns `(None, "none")`. |

`NUMBER` is `-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|-?\.\d+`. It matches integers, decimals, a leading minus and correct thousands commas. It does not match fractions, scientific notation or decimal commas: `3/4` reads as `3`, and `3,5` reads as `3` and `5`.

**`numbers_equal(a, b, rel_tol=0.0)`** is exact by default. It parses both sides with `parse_number`, which strips `,` and `$` and passes ints and floats through, so formatting never matters: `"1,000"`, `"$5"` and `"2.50"` equal 1000, 5 and 2.5. It then compares with `math.isclose(..., rel_tol, abs_tol=1e-9)`. The absolute 1e-9 absorbs float noise such as `0.1 + 0.2`; nothing else is accepted. If either side is unparseable the result is `False`, never an exception.

- **Why exact:** the default used to be a relative `1e-6`, which accepts near-misses on large numbers. [`scripts\check_graders.py`](../../../../scripts/check_graders.py.md) found GSM8K items `gsm8k/611` (gold 1,450,000) and `gsm8k/796` (gold 2,880,000) grading the wrong answers 1,450,001 and 2,880,001 correct.
- **Pass `rel_tol` only for answers that are approximate by nature**, such as a rounded unit conversion. A golden row does this with `tolerance` ([`golden.md`](golden.md)).

**`extract_choice(text, letters="ABCDEFGHIJ")`** follows the MMLU-Pro reference order:

| Order | Method | Rule |
|---|---|---|
| 1 | `answer_is` | `answer is` [`:`] [`*`/`_`] [`(`]X[`)`], not followed by a letter or digit. Uses the last match. |
| 2 | `answer_tag` | `answer:` [`*`/`_`] [`(`]X[`)`], with the same rules. |
| 3 | `boxed` | The first standalone letter inside the last `\boxed{}`, with `\text` removed. |
| 4 | `last_letter` | The last standalone capital letter in the text. This rule is weak: it catches the pronoun "I" and stray initials. |
| 5 | `none` | Nothing found. |

- The phrases are case-insensitive, but **the letter must be a capital** in `letters`. `The answer is (c).` extracts nothing.
- An `answer is` anywhere outranks a later `Answer:` tag. This mirrors the reference script, and a test pins it.
- Letters outside `letters` are ignored. MMLU-Pro passes each question's own letters, so `K` never matches a 10-option question.

**`extract_code(text, entry_point=None)`**:

1. Collects fenced blocks (```` ```lang ... ``` ````).
2. Keeps the Python ones (language tag `""`, `python`, `py`, `python3`). If there are none, it keeps all blocks.
3. Prefers blocks that `def` / `async def` the entry point, and picks the longest of those. Otherwise it picks the longest candidate.
4. With no closed fence, it returns the code after an opening fence (the output was cut off). Otherwise it returns the whole text.

There is no `method` for code. HumanEval records the test outcome (`passed`, `failed`, `timeout`) in `detail["status"]` instead, and the report prints those counts on a `tests:` line rather than `answers found by:`.

## Public API

| Name | Signature | What it's for |
|---|---|---|
| `NUMBER` | `re.Pattern` | The number regex above. Reuse it rather than writing your own. |
| `last_boxed` | `last_boxed(text: str) -> str \| None` | Content of the last `\boxed{...}`. |
| `extract_final_number` | `extract_final_number(text: str) -> tuple[str \| None, str]` | `(number_text, method)`. The number is returned as text, e.g. `"1,080"`. |
| `parse_number` | `parse_number(text: str \| int \| float \| None) -> float \| None` | Tolerant `float()`. |
| `numbers_equal` | `numbers_equal(a, b, rel_tol: float = 0.0) -> bool` | Numeric equality, exact by default; `False` if either side is unparseable. |
| `extract_choice` | `extract_choice(text: str, letters: str = "ABCDEFGHIJ") -> tuple[str \| None, str]` | `(letter, method)`. |
| `extract_code` | `extract_code(text: str, entry_point: str \| None = None) -> str` | The program text to execute. |

## How to use it

### See what an extractor reads (real outputs)

```python
from kumaru.eval.benchmarks.extract import extract_final_number, extract_choice, extract_code, numbers_equal

extract_final_number("So the total is 1{,}080.\n\n**Answer:** $1,080")  # ('1,080', 'answer_tag')
extract_final_number("First guess 12.\nThe answer is 15 apples.")     # ('15', 'answer_is')
extract_final_number("We get \\boxed{\\frac{3}{4}}")                   # ('3', 'boxed')  - fractions are not parsed
extract_final_number("Answer:\n\nThe total is\n\n\n42")                # ('42', 'last_number') - tag too far above
extract_choice("The answer is (C).\nAnswer: D")                        # ('C', 'answer_is')
extract_choice("Option E fits best, so I'll go with it.")              # ('I', 'last_letter') - the pronoun
extract_choice("The answer is (c).")                                   # (None, 'none') - lower case
extract_code("```python\ndef f(x):\n    return", "f")                  # 'def f(x):\n    return' (cut-off fence)
numbers_equal("$1,450,000.00", 1450000)                                # True  - formatting never matters
numbers_equal("1,450,001", 1450000)                                    # False - exact by default
numbers_equal(0.1 + 0.2, "0.3")                                        # True  - float noise
numbers_equal("37.83", 37.8, rel_tol=0.001)                            # True  - only when you ask
```

### Read the extraction line of a run

```powershell
kumaru bench report runs\bench\<run>
```

Real output from a finished Qwen2.5-7B run (illustrative; not a result to quote):

```
golden  answers found by: answer_tag 14, whole reply 13
gsm8k  answers found by: answer_tag 98, boxed 2
humaneval  tests: passed 80, failed 20
mmlu_pro  answers found by: answer_is 100
```

How to read it:

- **The format the prompt asked for** (`answer_tag` for GSM8K, `answer_is` for MMLU-Pro) **near n** means the model followed the instructions, so the score is clean.
- **Many `last_number` or `last_letter`** means the model ignored the format. Open `items.jsonl` and check the `extracted` field against `output`.
- **`error N`** counts questions whose request or grader raised an error. They are scored wrong, and a resume asks them again.
- **`whole reply N`** counts rows whose grader judges the whole reply (golden's `contains`, `regex` and so on), which records no method. That is why golden's line adds up to its 27 questions, with only the `numeric` and `choice` rows under `answer_tag`.
- **`tests:`** replaces `answers found by:` for HumanEval, whose records hold a test outcome instead of a method. IFEval records neither, so it prints no line unless questions failed (`answers found by: error 2, whole reply 98`).

### Find the answers a weak rule picked

```powershell
Get-Content runs\bench\<run>\gsm8k\items.jsonl | ForEach-Object { $_ | ConvertFrom-Json } |
  Where-Object { $_.detail.method -eq 'last_number' } | Select-Object id, correct, extracted, expected
```

## How to extend it

- **New answer formats:**
  - Prefer changing the benchmark's prompt to ask for a format these extractors already read.
  - If you must parse something new (fractions, units), add a new function. Don't widen `NUMBER`, because every numeric grader and every past verdict depends on it.
  - Give the new rule its own method name so it shows up in the report.
- **Changing an existing extractor changes old scores.** Re-grading isn't automatic. Records store the verdict, and resuming reuses verdicts whose question key matches; the run fingerprint covers settings, not code, so a resume won't notice the change. Re-run into a new directory and say so in [`docs\BENCHMARKS.md`](../../../../../BENCHMARKS.md).
- **Pin the change in `tests\test_bench_extract.py`.** It has a class per extractor: `TestExtractFinalNumber`, `TestLastBoxed`, `TestNumbers`, `TestExtractChoice` and `TestExtractCode`. Add both the case you fixed and a case that must not change.

## Failure modes

| Symptom | Likely cause | Fix |
|---|---|---|
| A correct GSM8K answer scored wrong with method `last_number` | No `Answer:` line, or it was more than 3 lines from the number, and the text ends with another number such as a check step. | Inspect `output`. If many answers do this, the model ignores the requested format. That is a finding about the model; report it rather than loosening the extractor silently. |
| `extracted` is `3` for an answer of `3/4` | Fractions are not parsed. | Expected; GSM8K answers are integers. A golden item expecting a fraction should use `regex` or ask for a decimal. |
| MMLU-Pro `extracted` is `I` | The `last_letter` fallback caught the pronoun. | The answer had no "answer is"/"Answer:" line, which is often a cut-off. Check `finish_reason` and the **cut off** column. |
| Choice not found although the answer says `(c)` | Letters are matched in capitals only. | The prompt asks for "(X)"; that's the model's formatting failure. |
| HumanEval runs the wrong function | Several fenced blocks; the longest one defining the entry point won. | Read `output`. Graders can't guess intent; a single block is what the prompt asks for. |
| Many answers end in an unclosed fence | Output was cut by the token limit (`n_truncated` > 0). | Raise `--max-tokens` (thinking off) or `--thinking-budget`. The default answer budget is 8,192 tokens. |

## Related files

- `src\kumaru\eval\benchmarks\gsm8k.py`, `mmlu_pro.py`, `humaneval.py`, `golden.py` — the callers ([`gsm8k.md`](gsm8k.md), [`mmlu_pro.md`](mmlu_pro.md), [`humaneval.md`](humaneval.md), [`golden.md`](golden.md)).
- `src\kumaru\eval\benchmarks\runner.py` — `summarize` builds the `extraction` counts ([`runner.md`](runner.md)).
- `tests\test_bench_extract.py` — pins every rule above, e.g. `test_answer_tag_only_looks_three_lines_ahead`, `test_latex_thousands_separators_are_normalized`, `test_answer_is_outranks_a_later_answer_tag`, `test_letters_outside_the_option_range_are_ignored` and `test_numbers_equal_is_exact_near_large_values_unless_asked`.
- `scripts\check_graders.py` — runs these extractors over every gold answer of the full datasets, and caught the near-miss bug above ([`check_graders.py.md`](../../../../scripts/check_graders.py.md)).
- [`docs\BENCHMARKS.md`](../../../../../BENCHMARKS.md) — results, and the grader-validation section ("Are the graders right?") that compares these verdicts with the official scripts.

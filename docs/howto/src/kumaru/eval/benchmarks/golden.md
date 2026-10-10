# `src\kumaru\eval\benchmarks\golden.py`

> The golden set: your own questions in a JSONL file, scored by seven deterministic graders. The shipped file is benchmark `golden`; any other file becomes `golden-<stem>`.

**Read this when:** you want a regression test for *your* use case, you are adding a question a model once got wrong, you are writing a golden file for a project, or you need to know why a golden answer was marked wrong.

---

## What it does

- **`GoldenSet`** reads a JSONL file, one question per line, and validates every row. Errors name `file:line`. Each answer is graded with the grader the row names.
  - The default file is `data\bench\golden.jsonl`, with 27 questions in 8 tags. It is described in [`golden.jsonl.md`](../../../../data/bench/golden.jsonl.md).
- **`grade_answer`** is the grader dispatch, callable on its own to try a grader on a reply.

## Why it exists

- **Public benchmarks tell you how a model compares; a golden set tells you whether it does your job.** Add every question you care about, and especially every one a model got wrong that you then fixed. The file becomes a regression test that runs in seconds.
- **The graders are deterministic code, not an LLM judge.** A verdict never changes between runs, costs nothing, and can be checked by reading the row.
- **The shipped 27 questions are easy on purpose.** A healthy 7B instruct model gets nearly all of them right. A drop means the plumbing broke (chat template, thinking switch, answer format, token budget), not that the model is weaker. Treat it as a smoke test and regression canary, not a capability measure.
  - At n = 27, a Wilson 95% interval is 18–35 points wide, so it cannot rank similar models. Use the public benchmarks and [`stats.md`](stats.md) for that.

## Mental model

### Rows become questions

Each non-blank line not starting with `//` must be one JSON object:

```json
{"id": "pens", "prompt": "Pens cost $3 each, or $10 for a pack of 4. ... Finish with \"Answer: <number>\".", "grader": "numeric", "answer": 26, "tags": ["math"]}
```

| Key | Required | Default | Becomes |
|---|---|---|---|
| `prompt` | yes, non-empty | | The user message. |
| `answer` | yes, not empty (`null` is allowed but rarely what you want) | | `BenchItem.answer`, shown as "expected". Its shape depends on the grader (table below). `""`, `[]` and `[" "]` are errors: every reply contains the empty string, so `contains` or `regex` would pass anything. |
| `grader` | no | `contains` | One of `GRADERS`. |
| `id` | no, but set it | the file's line number | Item id `<benchmark name>/<id>`, e.g. `golden/pens`. Must be unique in the file. |
| `tags` | no | none | A list. The **first** tag is the group used for `by group:` accuracy and stratified `-n` sampling. Later tags stay in `meta["tags"]`, but nothing reports them and `items.jsonl` doesn't record them. |
| `tolerance` | no | `0` (exact) | Relative tolerance for `numeric`: a number >= 0, e.g. `0.001`. Only for answers that are approximate by nature. |
| `system` | no | none | Text: a system message sent before the prompt. |
| `max_tokens` | no | 8,192 | A whole number >= 1: this question's answer budget. With thinking on, the runner adds the thinking budget on top. |

These are the only keys (`ROW_KEYS`). Any other key stops the load with `unknown key 'grade'` and a hint listing the allowed keys, so a misspelt key cannot silently fall back to the default `contains` grader. Every row is checked when the file loads, before the model starts: the value rules above, every `regex` pattern compiles, a `numeric` answer parses as a number, and a `choice` answer is one letter A–J.

### The seven graders

The reply graded is the final answer text only; the model's reasoning (thinking) is never graded.

| Grader | `answer` | Correct when | Recorded |
|---|---|---|---|
| `exact` | string or list | The whole reply equals one of the answers after normalizing both: trim whitespace, then quote characters `"` `'` `` ` `` at both ends, then trailing periods; collapse inner whitespace; casefold. | `extracted` = reply (200 chars). |
| `contains` | string or list | **Every** string occurs in the reply. Case-insensitive substring match. | `detail.missing` |
| `contains_any` | string or list | **At least one** string occurs. | `detail.found` |
| `not_contains` | string or list | **None** of the strings occur. | `detail.forbidden` |
| `regex` | pattern string, or a list of patterns | `re.search(pattern, reply.strip())` matches; with a list, **any one** pattern. | `extracted` = the match (200 chars). |
| `numeric` | **one** number (a string like `"1,000"` works: `,` and `$` are dropped) | The final number in the reply equals `answer`: exactly by default, or within the row's relative `tolerance`. | `extracted` = the number; `detail.method` |
| `choice` | **one** letter, A–J (`CHOICE_LETTERS`) | The extracted multiple-choice letter (A–J) equals `answer`, upper-cased. | `extracted` = the letter; `detail.method` |

More on how these behave:

- **`regex`** is case-sensitive and runs with no flags, so `^` and `$` anchor the whole stripped reply, not a line. Use inline flags: `(?i)` for case, `(?s)` for `.` across lines, `(?m)` for per-line anchors. Each pattern in a list carries its own flags; the first one that matches is recorded.
- **`numeric` is exact by default.** Formatting never matters: `1,000`, `$5` and `2.50` equal 1000, 5 and 2.5, and an absolute 1e-9 absorbs float noise such as 0.1 + 0.2. Nothing else is accepted, so `1,450,001` is wrong for 1450000. Set `tolerance` only for an answer that is approximate by nature: the shipped `fahrenheit` row (answer 37.8) uses `0.001`, which accepts 37.78 and rejects 37.7.
- **`numeric` and `choice`** find the answer with [`extract.md`](extract.md)'s rules (`Answer: X` first, and so on). Prompts should therefore ask for `Finish with "Answer: <number>".`, as every shipped row does.
- **`answers found by:`** counts `numeric` and `choice` rows by the method that found the answer (`detail.method`). The other graders judge the whole reply and record no method, so the report counts them as `whole reply`. The shipped set reports `answer_tag 14, whole reply 13`.

### Naming and paths

- `GoldenSet()` or `GoldenSet(DEFAULT_PATH)` is named `golden`. Any other path is named `golden-<file stem>`: `evals\support.jsonl` gives `golden-support`, with ids `golden-support/<id>` and results in `<run>\golden-support\`.
  - Two files with the same stem in one run, e.g. `a\set.jsonl` and `b\set.jsonl`, would share `golden-set`'s results folder, so the run is refused before it starts ([`runner.md`](runner.md#1-what-one-run-does)). Rename one of them.
- `kumaru bench run` treats an argument as a golden file if it ends in `.jsonl`, or if it is an existing file that isn't a registered benchmark name. Relative paths resolve against the current directory. The default file is anchored at the repo root, so `golden` works from anywhere.
- **Paired comparisons join on ids.** Line-number ids shift when you insert a line above, and the comparison then pairs different questions under the same id. Give every row an `id`.

### Sampling and budget

- **`-n/--limit` applies** (default 100). The shipped 27 always run in full. A file with more than 100 rows is sampled with stratification by first tag (`select_items`, [`base.md`](base.md)), so add `--full` to run every row.
- **Every question gets the 8,192-token default answer budget** (`Benchmark.answer_tokens`) unless the row sets `max_tokens`. A non-zero `cut off` (`n_truncated`) in the report means a budget was hit; those answers were graded on their cut text, which usually fails.

## Public API

| Name | Signature | What it's for |
|---|---|---|
| `GoldenSet` | `GoldenSet(path: str \| Path \| None = None)` | The benchmark, registered as `golden`. `name` becomes `golden-<stem>` for non-default paths. `description` and `source` are shown by `bench list`. |
| `GoldenSet.load` | `load() -> list[BenchItem]` | Read the file as UTF-8 (a byte-order mark is fine), then parse and validate it. Raises `ConfigError` naming `file:line`, or `Cannot read golden set <path>: ...` for a file that is not UTF-8. |
| `GoldenSet.grade` | `grade(item: BenchItem, output: str) -> Grade` | Calls `grade_answer(item.meta["grader"], item.answer, output, tolerance=item.meta["tolerance"])`. |
| `grade_answer` | `grade_answer(grader: str, answer: Any, output: str, *, tolerance: float = 0.0) -> Grade` | One grader on one reply. An unknown grader raises `ConfigError("Unknown golden-set grader: 'x'")`. |
| `GRADERS` | `("exact", "contains", "contains_any", "not_contains", "regex", "numeric", "choice")` | The loader validates `grader` against this. |
| `ROW_KEYS` | `("id", "prompt", "grader", "answer", "tags", "tolerance", "system", "max_tokens")` | The keys a row may have; any other key is an error. |
| `CHOICE_LETTERS` | `"ABCDEFGHIJ"` | The letters `extract_choice` can find, so the only valid `choice` answers. |
| `DEFAULT_PATH` | `REPO_ROOT / "data" / "bench" / "golden.jsonl"` | The shipped set. |

`BenchItem.meta` holds `{"grader", "tags", "tolerance"}`, with `tolerance` as a float (0.0 unless the row sets it). `answer_tokens` is the base default of 8,192; GoldenSet does not override it.

## How to use it

### Run the shipped set

```powershell
kumaru bench run golden -c qwen36              # seconds; read the "by group" line with report
kumaru bench run golden -c test                # offline (echo backend): exercises the plumbing only, scores are meaningless
kumaru bench report runs\bench\<run>\golden    # per-tag accuracy and "answers found by"
```

### Run your own file

```powershell
kumaru bench run evals\support.jsonl -c qwen36                 # benchmark "golden-support"
kumaru bench run golden evals\support.jsonl gsm8k -c qwen36    # mix with other benchmarks
kumaru bench run evals\support.jsonl -c qwen36 --full          # more than 100 rows: run them all
```

### Check a file without a model

```powershell
.\.venv\Scripts\python.exe -c "from kumaru.eval.benchmarks.golden import GoldenSet as G; b = G(r'evals\support.jsonl'); print(b.name, len(b.load()))"
```

`load()` raises the same errors `bench run` would, every value check included, and writes nothing.

### Try a grader on a reply

```python
from kumaru.eval.benchmarks.golden import grade_answer

grade_answer("regex", r"(?i)^[\W_]*yes\b", "**Yes**, because every bloop is a razzie.")
# Grade(correct=True, score=None, extracted='**Yes', detail={})
grade_answer("regex", [r"^no\b", r"(?i)^yes\b"], "YES, both are.")
# Grade(correct=True, score=None, extracted='YES', detail={})
grade_answer("numeric", 37.8, "Answer: 37.80", tolerance=0.001)
# Grade(correct=True, score=None, extracted='37.80', detail={'method': 'answer_tag'})
grade_answer("numeric", 1450000, "Answer: 1,450,001")
# Grade(correct=False, score=None, extracted='1,450,001', detail={'method': 'answer_tag'})
grade_answer("contains", ["Canberra", "Australia"], "Canberra is the capital.")
# Grade(correct=False, score=None, extracted=None, detail={'missing': ['Australia']})
```

### See which questions failed and why

```powershell
Get-Content runs\bench\<run>\golden\items.jsonl | ForEach-Object { $_ | ConvertFrom-Json } |
  Where-Object { -not $_.correct } | Select-Object id, extracted, expected, @{n='detail'; e={$_.detail | ConvertTo-Json -Compress}}, output
```

## How to extend it

- **Add a question.** One line, with `id`, the prompt, the grader, the answer and one tag; authoring advice is in [`golden.jsonl.md`](../../../../data/bench/golden.jsonl.md).
  - Personal or project questions belong in your own file.
  - The **shipped** file's size is not pinned: `tests\test_bench_graders.py::TestGoldenSetLoader::test_shipped_golden_set_loads_with_unique_ids_and_known_graders` checks only that it loads, with unique ids and known graders. The set is meant to grow with every bug you fix.
- **Add a grader.**
  1. Add a branch to `grade_answer` that returns a `Grade`. Record `detail["method"]` if you extract something, so it appears in `answers found by:`; without one the row counts as `whole reply`.
  2. Add the name to `GRADERS`, which is what the loader validates against.
  3. If some `answer` values can never grade correct, reject them in `GoldenSet._item`, as `numeric` and `choice` do, so the mistake stops the load instead of failing every run.
  4. Add a case to `TestGoldenGraders` in `tests\test_bench_graders.py`.
  5. Document it in the graders table above.
- **Add a row key.** Add it to `ROW_KEYS` (otherwise the loader rejects it) and check its value in `GoldenSet._item`.
- **Anything more than a pattern check**, such as running code, calling a tool or multi-turn chat, is a benchmark of its own; see [`__init__.md`](__init__.md#add-a-benchmark-end-to-end).

Pinned behaviour lives in `tests\test_bench_graders.py`:

- `TestGoldenGraders` covers each grader's case, quote and period handling, a `regex` list (`test_regex_with_a_list_needs_any_one_pattern`), exact `numeric` matching (1,450,001 is not 1450000) and unknown graders.
- `TestGoldenSetLoader` covers ids and line-number ids, `golden-<stem>` naming, `system` and `max_tokens` pass-through, every malformed-file error (unknown keys, bad values, empty answers, non-numeric and non-letter answers, bad patterns), a file that is not UTF-8, a byte-order mark, and the shipped set.

## Failure modes

| Symptom | Likely cause | Fix |
|---|---|---|
| `support.jsonl:7: invalid JSON (...)` | Not one complete object on that line. Multi-line JSON isn't allowed. | One object per line; no trailing commas. |
| `<file>:<line>: every row needs a non-empty 'prompt'` | A missing or blank prompt, or the line is not an object. | Add `prompt`. |
| `<file>:<line>: unknown key 'grade'` | A misspelt or unsupported key. The hint lists the keys a row may have. | Fix the key name. |
| `<file>:<line>: unknown grader 'x'` | A misspelt grader value. | One of `GRADERS`. |
| `<file>:<line>: missing 'answer'` | No `answer` key. | Add it (`not_contains` takes the forbidden strings). |
| `<file>:<line>: duplicate id 'x'` | The same `id` twice, or an `id` equal to another row's line-number id. | Unique ids. |
| `<file>:<line>: 'answer' must not be empty (got '')` | `""`, `[]`, or a list holding a blank string. Every reply contains the empty string, so the row would pass anything. | Give the text to look for. |
| `<file>:<line>: the numeric grader needs a number as 'answer' (got ...)` | A list, or text that is not a number, e.g. `"about 26"`. | One number: `26` or `"1,000"`. |
| `<file>:<line>: the choice grader needs one letter A-J as 'answer' (got '(C)')` | Brackets, a word, or a letter past J. | The bare letter, `"C"`. |
| `<file>:<line>: invalid regex '(' (...)` | A pattern that does not compile, e.g. an unbalanced `(` or `[`. | Fix the pattern. |
| `<file>:<line>: 'tolerance' must be a number >= 0, e.g. 0.001 (got None)` | `"tolerance": null`, a string, or a negative number. | Omit the key for exact matching, or give a number. |
| `<file>:<line>: 'tags' must be a list, e.g. ["math"] (got 'math')` | A string rather than a list. | `"tags": ["math"]`. |
| `<file>:<line>: 'max_tokens' must be a whole number >= 1 (got ...)` or `'system' must be text` | A quoted number (`"512"`), a float (`512.0`), or a non-string system message. | A bare integer; a string. |
| `Golden set not found: <path>` | Wrong path. Relative paths resolve against the current directory. | Fix the path. |
| `Cannot read golden set <path>: 'utf-8' codec can't decode byte 0xff in position 0: invalid start byte` | The file is not UTF-8. Windows PowerShell 5's `>` writes UTF-16. | Save it as UTF-8, as the hint says. A byte-order mark is fine. |
| `Golden set <path> has no questions` | Only blank and `//` lines. | Add rows. |
| `Two benchmarks share the name 'golden-set' (the second is ...), so their results would share one folder.` | Two golden files with the same stem, e.g. `a\set.jsonl` and `b\set.jsonl`, in one run. | Rename one of the files. |
| A `regex` row with a list passes when only one pattern matches | A list needs **any one** pattern to match, not all. | For "all of", use `contains`, or one pattern with lookaheads: `"(?s)(?=.*a)(?=.*b)"`. |
| A `numeric` row is always wrong | The reply has no number the extractor accepts, or it is rounded differently: matching is exact, so `37.78` is not `37.8`. | Ask for `Answer: <number>` and say how to round; set `tolerance` only if the answer is approximate by nature. |
| `contains` with `26` passes on "260" | Substring match. | `numeric`, or `regex` with `\b26\b`. |
| Only 100 of your rows ran | Default `-n 100`. | `--full`. |
| `cut off` > 0 on a golden item | That row's `max_tokens` is too small, or thinking used the budget. | Raise or remove `max_tokens`; use `--thinking-budget`. |

File-encoding (ANSI, UTF-16) and JSON-escaping traps (`\b` is a backspace in JSON) are covered in [`golden.jsonl.md`](../../../../data/bench/golden.jsonl.md#failure-modes).

## Related files

- `data\bench\golden.jsonl` — the shipped set ([`golden.jsonl.md`](../../../../data/bench/golden.jsonl.md)).
- `src\kumaru\eval\benchmarks\extract.py` — the number and letter extraction behind `numeric` and `choice` ([`extract.md`](extract.md)).
- `src\kumaru\eval\benchmarks\base.py` — `Benchmark`, `BenchItem`, `select_items`, the 8,192 default ([`base.md`](base.md)).
- `src\kumaru\eval\benchmarks\runner.py` — path resolution (`resolve_benchmark`), budgets, records ([`runner.md`](runner.md)).
- `src\kumaru\eval\harness.py` — the older `kumaru eval` suites; a different file format ([`harness.md`](../harness.md)).
- [`docs\BENCHMARKS.md`](../../../../../BENCHMARKS.md) — results and grader validation.

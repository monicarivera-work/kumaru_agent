# `data\bench\golden.jsonl`

> The shipped golden set: 27 easy, hand-written questions in JSON Lines, each scored by a deterministic grader. It runs in seconds as a smoke test and regression canary for the benchmark path, and it is the template for your own question files.

**Read this when:** you are adding a question or writing a golden file for your own project, a `golden` score dropped and you need to know whether the model or the plumbing changed, or a row loads or grades differently from what you meant.

---

## What it does

`kumaru bench run golden` loads this file (`DEFAULT_PATH` in `src\kumaru\eval\benchmarks\golden.py`), asks each question once and grades the reply with the grader the row names. The loader and the exact semantics of the seven graders are documented in [`golden.md`](../../src/kumaru/eval/benchmarks/golden.md). This page covers the file itself: what is in it, how to write rows that grade what you mean, and how to run a file of your own.

The format is JSON Lines with comments:

- One JSON object per line, one question per object.
- Blank lines and lines starting with `//` are skipped. The file opens with two `//` header lines.
- The file must be UTF-8. A byte-order mark ("UTF-8 with BOM", as Notepad can save it) is fine. LF and CRLF line endings both work.

The 27 questions, grouped by first tag:

| Group | n | What it checks | Graders | Ids |
|---|---:|---|---|---|
| `math` | 8 | Arithmetic, a unit conversion, word problems with a built-in trap, and one multiple-choice question. | `numeric` ×7, `choice` ×1 | `pens`, `mult`, `seconds`, `fahrenheit`, `bat-ball`, `apples`, `speed`, `prime-choice` |
| `reasoning` | 3 | Classic trick questions: siblings, a syllogism, counting letters. | `numeric` ×2, `regex` ×1 | `sisters`, `syllogism`, `strawberry` |
| `dates` | 2 | A day count across a leap February, and a weekday. | `numeric`, `contains` | `leap-days`, `y2k-weekday` |
| `knowledge` | 4 | Well-known facts. | `contains` ×2, `regex`, `contains_any` | `capital-au`, `tungsten`, `austen`, `photosynthesis` |
| `honesty` | 2 | False-premise questions. The model must reject the premise instead of inventing an answer. | `contains_any` ×2 | `mars-walk`, `atlantis` |
| `code` | 3 | What a short Python snippet prints. | `numeric` ×3 | `list-aliasing`, `sum-squares`, `off-by-one` |
| `format` | 4 | Exact output shape: one capitalised word, three bare lines, JSON only, a sentence without the letter "e". | `regex` ×4 | `banana-caps`, `three-colors`, `json-ada`, `no-e` |
| `language` | 1 | A one-word translation. | `contains` | `spanish` |

Grader totals: `numeric` 13, `regex` 6, `contains` 4, `contains_any` 3, `choice` 1. No shipped row uses `exact`, `not_contains`, `system` or `max_tokens`. Only `fahrenheit` sets `tolerance`; the other 12 `numeric` rows must match exactly.

## Why it exists

- **It's a canary, not a measure.** The questions are easy on purpose: a healthy 7B instruct model gets nearly all of them right. When the score drops, suspect the plumbing first:
  - the chat template;
  - the thinking switch;
  - the answer format;
  - the token budget;
  - the backend.

  Only then suspect the model.
- **It's too small to rank models.** At n = 27, one question moves accuracy by 3.7 points. The 95% Wilson interval for 26/27 (96.3%) is 81.7–99.3. To rank models, use the public benchmarks at 100+ items and a paired `compare` ([`stats.md`](../../src/kumaru/eval/benchmarks/stats.md)).
- **It's deterministic and cheap.** There is no LLM judge. The same reply always gets the same verdict, and you can check any verdict by reading the row.
- **Each row is a regression test.** A golden file is meant to grow with the questions you care about, especially ones a model once got wrong. Keep project questions in your own file (see below). The shipped file is meant to grow too, with every bug you fix, so no test pins its size.

## Mental model

1. **The loader turns each row into a `BenchItem`.**
   - The id is `golden/<id>` (or `golden-<stem>/<id>` for your own file).
   - The group is the first tag.
   - `meta` holds `{grader, tags, tolerance}`.
   - A bad row stops the load with a `file:line` error, before the model starts: broken JSON, an unknown key, a value of the wrong type, or an answer its grader can never match.
2. **`-n/--limit` (default 100) picks the questions.**
   - All shipped rows run: there are 27 today, fewer than the default 100.
   - A bigger file is sampled, stratified by group ([`base.md`](../../src/kumaru/eval/benchmarks/base.md#what--n--limit-picks-select_items)).
   - `--full` runs every row.
3. **The runner sends `[system?, user]` with an answer budget.**
   - The budget is 8,192 tokens (`Benchmark.answer_tokens`), or the row's own `max_tokens`.
   - With thinking on, the thinking budget is added on top. With thinking on and no thinking budget, the profile's `max_tokens` is used when it is larger.
   - `--max-tokens` overrides all of this. Every budget is capped at 131,072.
4. **The grader checks the final reply text, never the reasoning.**
   - One record per question goes into `items.jsonl`: `id`, `group`, `prompt`, `correct`, `extracted`, `expected` (the row's `answer`), `output`, `finish_reason`, token counts and `detail`.
5. **`bench report` adds it up.** It shows:
   - accuracy with a Wilson CI;
   - `cut off` (`n_truncated`), the replies that stopped at the budget;
   - per-group accuracy;
   - `answers found by:`.

## Fields

| Field | Type | Default | What it controls | How to choose |
|---|---|---|---|---|
| `id` | string (numbers are converted with `str()`) | The row's physical line number, counting comment and blank lines (`pens` is line 3). | The item id `golden/<id>`. Resume and `compare` match questions by it. Must be unique in the file. | Always set it: short, kebab-case, stable. A missing or falsy id (`0`, `""`, `null`) falls back to the line number, which shifts when a line is inserted above. Give a question a new id when you change what it asks. |
| `prompt` | string | required, non-empty after trimming | The user message, sent as written. | One question with one checkable answer. State the answer format the grader needs. |
| `grader` | string | `contains` | Which of `exact`, `contains`, `contains_any`, `not_contains`, `regex`, `numeric`, `choice` scores the reply. | By answer type; see [Writing questions that grade what you mean](#writing-questions-that-grade-what-you-mean). |
| `answer` | depends on `grader` | required, not empty | What the grader checks against. It appears as `expected` in `items.jsonl` and in `compare` examples. | `numeric`: one number. `choice`: one letter A–J. `regex`: one pattern, or a list of patterns of which any one may match. Otherwise a string, or a list. `""`, `[]` and `[" "]` stop the load, because every reply contains the empty string. |
| `tags` | list of strings | none | The first tag is the group: per-group accuracy in `bench report`, and the strata when `-n` samples a large file. | One meaningful first tag. Later tags stay in `meta`, but nothing reports them and `items.jsonl` doesn't record them. A bare string such as `"math"` stops the load. |
| `tolerance` | number >= 0 | `0`: exact | Relative tolerance for `numeric`. At `0` formatting still never matters (`1,000`, `$1,000` and `1000.0` all equal 1000) and an absolute `1e-9` absorbs float noise, but 1001 is wrong. | Leave it out unless the answer is approximate by nature. `fahrenheit` uses `0.001` with answer `37.8`: it accepts 37.78 and rejects 37.7. |
| `system` | string | none | A system message sent before the prompt. With `--use-agent-prompt`, it is appended to the profile's agent prompt. | A rule one question needs. Without `system` or the flag, Kumaru sends no system message, though the model's chat template may insert its own default. |
| `max_tokens` | whole number >= 1 | none: the 8,192-token default | This question's answer budget. The thinking budget is added when thinking is on; `--max-tokens` overrides it. | Leave it out. A binding cap shows up as `cut off` (`n_truncated`), and the cut reply is graded as it stands, which usually fails. |

These are the only keys. Any other key stops the load: a typo such as `"grade": "numeric"` gives `<file>:<line>: unknown key 'grade'`, with the allowed keys as the hint, instead of quietly leaving the grader at `contains`. The values are checked at the same time, before the model starts; [`golden.md`](../../src/kumaru/eval/benchmarks/golden.md#failure-modes) lists every message.

## How to use it

### Run the shipped set

```powershell
kumaru bench run golden -c qwen36
kumaru bench report runs\bench\<run>\golden
```

Illustrative report from a finished Qwen2.5-7B run (`20261010-114917-qwen25-7b-gguf-default`):

```text
qwen25-7b-gguf-default  Qwen/Qwen2.5-7B-Instruct-GGUF:Q4_K_M · default
┌───────────┬────┬────────────┬───────────┬────────┬─────────┬───────────────┬─────────────┐
│ benchmark │  n │ accuracy % │    95% CI │ errors │ cut off │ tokens/answer │ answers/min │
├───────────┼────┼────────────┼───────────┼────────┼─────────┼───────────────┼─────────────┤
│ golden    │ 27 │       96.3 │ 81.7-99.3 │      0 │       0 │            86 │       125.2 │
└───────────┴────┴────────────┴───────────┴────────┴─────────┴───────────────┴─────────────┘
golden  answers found by: answer_tag 14, whole reply 13
    by group: code 100.0 (n=3), dates 100.0 (n=2), format 75.0 (n=4), honesty 100.0 (n=2), knowledge 100.0 (n=4), language 100.0 (n=1), math 100.0 (n=8), reasoning 100.0 (n=3)
```

- **`answer_tag 14, whole reply 13`.** Only the 13 `numeric` rows and the one `choice` row record how the answer was found, and all 14 used the `Answer:` tag. The other 13 rows (`contains`, `contains_any`, `regex`) judge the whole reply and count as `whole reply`. If the 14 shift towards `last_number` or `none`, the model stopped following the answer format. Check the chat template and thinking settings before blaming knowledge.
- **`cut off 0`.** No reply hit its budget. Anything above zero means a budget was binding, typically thinking on with a small `--thinking-budget`.
- **`format 75.0`.** The one miss was `golden/no-e`. The reply "The wavy brine bathes vast shores." contains four e's: a genuine model miss, not a plumbing fault.

### See what failed

```powershell
Get-Content runs\bench\<run>\golden\items.jsonl | ForEach-Object { $_ | ConvertFrom-Json } |
  Where-Object { -not $_.correct } | Select-Object id, expected, output, finish_reason
```

### Write and run your own file

Any golden-format `.jsonl` file is a benchmark named `golden-<file stem>`:

```powershell
New-Item -ItemType Directory -Force evals | Out-Null
Set-Content evals\support.jsonl '// Support-bot regression set. Format: docs/howto/data/bench/golden.jsonl.md'
Add-Content evals\support.jsonl '{"id": "vat", "prompt": "A coat costs £100 before 20% VAT. What does it cost with VAT? Finish with \"Answer: <number>\".", "grader": "numeric", "answer": 120, "tags": ["math"]}'
.\.venv\Scripts\python.exe -c "from kumaru.eval.benchmarks.golden import GoldenSet as G; b = G(r'evals\support.jsonl'); print(b.name, len(b.load()))"
# golden-support 1
```

- **Quoting.** Single quotes keep PowerShell away from `$`, backslashes and double quotes inside the JSON.
- **Encoding.** In PowerShell 7 (`pwsh`), `Set-Content` and `Add-Content` write UTF-8 without a BOM. Windows PowerShell 5.1 is different:
  - its cmdlets write the ANSI code page, so a non-ASCII character such as `£` stops the load with `Cannot read golden set <path>: 'utf-8' codec can't decode byte 0xa3 ...` and the hint `Save it as UTF-8 text, one JSON object per line.`;
  - its `>` writes UTF-16, which always stops the load the same way.

  Use an editor there.
- **Validation.** The `load()` one-liner raises the same `file:line` errors a run would, without loading a model.

```powershell
kumaru bench run evals\support.jsonl -c qwen36          # results in <run>\golden-support\, ids golden-support/<id>
kumaru bench run golden evals\support.jsonl -c qwen36   # the canary and your set in one run
kumaru bench run evals\support.jsonl -c qwen36 --full   # more than 100 rows: all of them, not a stratified 100
kumaru bench compare runs\bench\<before> runs\bench\<after> -e 5   # which of your questions flipped
```

How the command line treats files:

- An argument is treated as a golden file if it ends in `.jsonl`, or if it is an existing file that isn't a registered benchmark name.
- Relative paths resolve against the current directory. Only `golden` itself is anchored at the repo root, so it works from anywhere.
- Two different files with the same stem get the same name, so the run refuses to start: `Two benchmarks share the name 'golden-set' (the second is b\set.jsonl), so their results would share one folder.` Rename one. The same file named twice runs once.
- Renaming a file renames its benchmark, so older runs no longer pair with new ones in `compare`.

### Change the shipped file

- **Put project questions in your own file.** The shipped rows are a shared canary.
- **Adding shipped rows** is expected: the set is meant to grow with every bug you fix. No test pins its size; `tests\test_bench_graders.py::TestGoldenSetLoader::test_shipped_golden_set_loads_with_unique_ids_and_known_graders` checks that it loads with unique ids and known graders. Check with `.\.venv\Scripts\python.exe -m pytest tests\test_bench_graders.py -q -p no:cacheprovider`.
- **Editing a question** changes its record key, so resuming an older run asks it again. `compare` pairs on ids only and doesn't notice the change, so give a changed question a new id.

## How to extend it

### Writing questions that grade what you mean

| The answer is… | Grader | `answer` | Ask for | Shipped example |
|---|---|---|---|---|
| a number | `numeric` | `26` | `Finish with "Answer: <number>".` | `pens` |
| one of lettered options | `choice` | `"C"`: one letter A–J | Options as `A) …` lines, then `Finish with "Answer: <letter>".` | `prime-choice` |
| a word or name | `contains` | `"Canberra"`; a list means all must appear | — | `capital-au` |
| any of several phrasings | `contains_any` | `["carbon dioxide", "CO2", "CO₂"]` | — | `photosynthesis` |
| a rejected false premise | `contains_any` | many phrasings, with both apostrophe styles | — | `mars-walk`, `atlantis` |
| an exact output shape | `regex` | `"^BANANA\\.?$"`; a list passes if any one pattern matches | The shape, explicitly | `banana-caps`, `json-ada` |
| something that must not appear | `not_contains` | `["as an AI"]` | — | none shipped |
| a short reply and nothing else | `exact` | `"Paris"`, or a list | "Reply with only …" | none shipped |

Rules that decide whether a row measures what you think:

1. **Test the grader both ways before committing the row.** A right reply must pass, and a plausible wrong one must fail:
   ```powershell
   .\.venv\Scripts\python.exe -c "from kumaru.eval.benchmarks.golden import grade_answer as g; print(g('numeric', 120, 'Answer: 120').correct, g('numeric', 120, 'Answer: 100').correct)"
   # True False
   ```
2. **Watch out for substring matches.** `contains`, `contains_any` and `not_contains` are case-insensitive substring tests on the whole reply: `26` is in "260", and `no` is in "I don't know". Use `numeric`, or `regex` with `\b`.
3. **Regex is strict.**
   - It is case-sensitive and searches the trimmed reply with no flags, so `^` and `$` anchor the whole reply. Use `(?i)`, `(?s)` and `(?m)` inline.
   - A list of patterns passes if any one matches, each with its own inline flags. For "all of these", use `contains` with a list, or one pattern with lookaheads.
   - Allow harmless decoration, as the shipped rows do: Markdown bold before "yes" (`syllogism`: `(?i)^[\W_]*yes\b`), a trailing period (`banana-caps`), a code fence around JSON (`json-ada`).
4. **Double every backslash.** The pattern lives inside a JSON string, so `"\\bW\\b"` is the regex `\bW\b`.
   - A single `\b` is valid JSON for a backspace character: the row loads and then never matches.
   - A single `\d` fails to load with `Invalid \escape`.
5. **Ask for `Answer: <number>`.**
   - `numeric` takes the number after the last `Answer:` tag. Failing that, it tries `\boxed{}`, then "the answer is", then the last number in the reply ([`extract.md`](../../src/kumaru/eval/benchmarks/extract.md)).
   - Without the instruction, the last number in a worked solution may not be the answer. `answers found by:` shows which rule fired.
6. **`numeric` matches exactly.** `1,000`, `$1,000` and `1000.0` all match `1000`, but `1001` doesn't, and neither does `37.78` against a rounded `37.8`. Say how to round in the prompt; set `tolerance` only when the answer is approximate by nature (`fahrenheit`: `0.001`).
7. **Write refusals generously.** For false-premise questions, list many phrasings and both apostrophe styles (`hasn't`, `hasn’t`), because models often emit typographic quotes.
8. **Leave the budget alone.** Every question gets 8,192 tokens unless the row sets `max_tokens`. A non-zero `cut off` (`n_truncated`) in the report means a cap was binding.

New graders are added in `golden.py`; see [`golden.md`](../../src/kumaru/eval/benchmarks/golden.md#how-to-extend-it).

## Failure modes

Every loader error, such as a missing `prompt` or `answer`, an unknown grader, a duplicate id, an empty answer or a pattern that doesn't compile, is listed in [`golden.md`](../../src/kumaru/eval/benchmarks/golden.md#failure-modes). Authoring traps:

| Symptom | Likely cause | Fix |
|---|---|---|
| `Cannot read golden set <path>: 'utf-8' codec can't decode byte 0x…` | The file isn't UTF-8: ANSI or UTF-16 from Windows PowerShell 5.1. (A byte-order mark is not the problem; it is accepted.) | Re-save as UTF-8, as the hint says. PowerShell 7 cmdlets do this by default. |
| `<file>:<line>: invalid JSON (Invalid \escape)` | A regex backslash written once, e.g. `\d`. | `\\d`. |
| A `regex` row with `\b` never matches | `\b` written once is a JSON backspace character. | `\\b`. |
| `<file>:<line>: invalid JSON (...)` on a row that looks right | The object spans several lines, or has a trailing comma or single quotes. | One complete object per line, double quotes. |
| A substring grader accepts a wrong reply | `26` is in "260"; `no` is in "know". | `numeric`, or `regex` with `\b…\b`. |
| `<file>:<line>: unknown key 'grade'` | A misspelt key. The hint lists the keys a row may have. | Fix the key name. |
| `<file>:<line>: 'tags' must be a list, e.g. ["math"] (got 'math')` | `"tags": "math"`, a string rather than a list. | `"tags": ["math"]`. |
| `<file>:<line>: 'tolerance' must be a number >= 0, e.g. 0.001 (got None)` | `"tolerance": null`. | Omit the key for exact matching, or give a number. |
| A `numeric` row fails though the model's value looks right | Matching is exact: `37.78` is not `37.8`. | Say how to round in the prompt, or set `tolerance` if the answer is approximate by nature. |
| A `regex` row with a list passes a reply that misses some patterns | Any one pattern matching is enough. | For "all of these", `contains` with a list, or one pattern with lookaheads: `"(?is)^(?=.*\\bred\\b)(?=.*\\bblue\\b)"`. |
| `compare` pairs the wrong questions | Rows without `id` use line numbers, which shift when a line is inserted; or a question was edited under the same id. | Set every `id`; use a new id for a changed question. |
| `compare`: `The two runs share no benchmark.` | The file was renamed between runs, so the benchmark name changed. | Keep the file name stable. |
| Only 100 of your rows ran | `-n/--limit` defaults to 100. | `--full`. |
| `Two benchmarks share the name 'golden-<stem>' (the second is <path>), so their results would share one folder.` | Two different files with the same stem in one run. | Rename one: a golden set is named after its file. |
| `cut off` > 0 | A row's `max_tokens`, `--max-tokens`, or thinking used up the budget. The reply was graded as it stood, usually wrong. | Remove or raise `max_tokens`; raise `--thinking-budget`. |
| `test_shipped_golden_set_loads_with_unique_ids_and_known_graders` fails after you add a row | The new row breaks a load rule; the error names `golden.jsonl:<line>`. | Fix the row. The test doesn't pin the number of rows. |

## Related files

- `src\kumaru\eval\benchmarks\golden.py` — the loader and the graders ([`golden.md`](../../src/kumaru/eval/benchmarks/golden.md)).
- `src\kumaru\eval\benchmarks\extract.py` — how `numeric` and `choice` find the answer ([`extract.md`](../../src/kumaru/eval/benchmarks/extract.md)).
- `src\kumaru\eval\benchmarks\runner.py` — `resolve_benchmark`, budgets, the run layout, `compare` ([`runner.md`](../../src/kumaru/eval/benchmarks/runner.md)).
- `tests\test_bench_graders.py` — pins every grader and the loader's checks, and checks that the shipped file loads.
- [`docs\BENCHMARKS.md`](../../../BENCHMARKS.md) — see "Your own golden questions", plus results and grader validation.

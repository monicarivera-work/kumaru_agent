# `src\kumaru\eval\benchmarks\ifeval_checks.py`

> IFEval's 25 instruction checkers, each a small pure function `check(response, **kwargs) -> bool`, plus the strict/loose machinery (`loose_variants`, `check_instructions`) and the counting helpers built on NLTK, `regex` and langdetect. It reimplements the reference semantics, quirks included.

**Read this when:** an IFEval verdict looks wrong, you need the exact rule behind an instruction id, the dataset gained an instruction type, or you want to reuse a checker outside a benchmark run.

---

## What it does

- **`CHECKS`** maps each of the 25 IFEval instruction ids, such as `punctuation:no_comma`, to a checker function. A checker takes the reply plus that instruction's arguments from the dataset as keyword arguments. It accepts and ignores extra ones (`**_`).
- **`check_instructions(ids, kwargs_list, response)`** returns `(strict, loose)`, one flag per instruction:

  ```python
  strict = bool(response.strip()) and check(response, **kwargs)
  loose  = any(v.strip() and check(v, **kwargs) for v in loose_variants(response))
  ```
- **Counting helpers:**
  - `count_sentences` uses NLTK's pretrained Punkt model, `PunktTokenizer("english")`.
  - `count_words` counts `regex` `\w+` runs.
  - `count_capital_words` counts NLTK `word_tokenize` tokens that are entirely upper case.
  - `detect_language` uses langdetect with `DetectorFactory.seed = 0`, so the result is deterministic. It returns `None` when it cannot decide.
- **`ensure_nltk_data()`** downloads Punkt's `punkt_tab` data on first use. The tokenizer is built once per process (`lru_cache`).

## Why it exists

- **Counts decide verdicts at thresholds** ("at least 300 words", "fewer than 5 sentences"). The reference uses Punkt for sentences, `word_tokenize` for capital words and `\w+` runs for words. Hand-rolled regexes drift from these on lists, abbreviations ("Dr."), contractions and non-Latin scripts, so this module uses the same tools. Python's `re` would split Hindi or Thai words at every vowel sign; the `regex` engine keeps the combining marks inside the word.
- **The reference's quirks are kept on purpose**, so that scores stay comparable with published numbers. Examples: an undetectable language counts as followed, and `nth_paragraph_first_word` indexes the raw split. The table below marks each quirk.
- **One deliberate difference.** The module docstring records a check against the reference on GPT-4's 541 published responses: every instruction verdict agrees except two, and both come from `keywords:letter_frequency` with a non-letter (`#`, `!`). There the reference checks a *random* letter, a coin flip; Kumaru counts the character the prompt asked for. Validation results: [`docs\BENCHMARKS.md`](../../../../../BENCHMARKS.md).

## Mental model

### The 25 checks

All string matching is case-insensitive unless noted. "Compared" means `_compare(count, relation, target)`, which supports exactly two relations:
- `less than`: `count < target`;
- `at least`: `count >= target`.

Any other relation raises `ValueError("unknown IFEval relation: '...'")`. The current dataset uses only these two.

| Instruction id | Arguments | Followed when |
|---|---|---|
| `keywords:existence` | `keywords` | Every keyword occurs, as a substring: `cat` matches "category". |
| `keywords:frequency` | `keyword`, `frequency`, `relation` | The number of substring occurrences, compared with `frequency`. |
| `keywords:forbidden_words` | `forbidden_words` | None occurs as a whole word (`\b…\b`). |
| `keywords:letter_frequency` | `letter`, `let_frequency`, `let_relation` | `reply.lower().count(letter.lower())`, compared. A non-letter is counted as asked (the one deliberate difference). |
| `language:response_language` | `language` (ISO 639-1; 22 codes in the dataset) | langdetect returns that code, **or returns nothing** (reference behaviour). |
| `length_constraints:number_sentences` | `num_sentences`, `relation` | The Punkt sentence count, compared. |
| `length_constraints:number_paragraphs` | `num_paragraphs` | Split on the `***` divider (`\s?\*\*\*\s?`). The piece count equals `num_paragraphs` **exactly**. An empty first or last piece is ignored; an empty middle piece fails. |
| `length_constraints:number_words` | `num_words`, `relation` | The `\w+` run count, compared. "It's" counts as 2. |
| `length_constraints:nth_paragraph_first_word` | `num_paragraphs`, `nth_paragraph`, `first_word` | Split on blank lines (`\n\n`, not `***`). The non-empty count equals `num_paragraphs`, and piece *n* of the **raw** split starts with `first_word`. Leading quotes are stripped and the word is cut at `. , ? ! ' "`. Quirk: the raw split includes empty pieces, so an extra blank line shifts the index and fails. |
| `detectable_content:number_placeholders` | `num_placeholders` | At least that many `[...]`. |
| `detectable_content:postscript` | `postscript_marker` | The marker occurs **anywhere**, not only at a line start. `P.P.S` and `P.S.` also match with a space after each dot. |
| `detectable_format:number_bullet_lists` | `num_bullets` | Lines starting with `*` + a non-`*` character, plus lines starting with `-`, total **exactly** `num_bullets`. Quirk: a `---` rule counts as a bullet. A `**bold**` line does not. |
| `detectable_format:constrained_response` | — | Contains `My answer is yes.`, `My answer is no.` or `My answer is maybe.` (case-sensitive). |
| `detectable_format:number_highlighted_sections` | `num_highlights` | At least that many non-empty `*text*` or `**text**` spans, each within one line. |
| `detectable_format:multiple_sections` | `section_spliter` (sic, the upstream spelling), `num_sections` | Splitting on `<spliter> <digits>` yields at least `num_sections` sections. The splitter is matched **case-sensitively**: "Section 1" doesn't match `SECTION`. |
| `detectable_format:json_format` | — | The stripped reply parses with `json.loads` after removing a ```` ```json ````/```` ```Json ````/```` ```JSON ````/```` ``` ```` fence, only at the very start and end. A preamble before the fence fails strict. |
| `detectable_format:title` | — | Contains `<<title>>` with non-blank text, on one line. |
| `combination:two_responses` | — | Splitting on exactly six asterisks `******` gives exactly two non-empty parts that differ. An empty middle part fails. Seven asterisks leave a stray `*` that counts as a third part. |
| `combination:repeat_prompt` | `prompt_to_repeat` | The stripped reply starts with the stripped prompt. |
| `startend:end_checker` | `end_phrase` | The reply, stripped of whitespace and then of `"`, ends with the phrase. |
| `startend:quotation` | — | The stripped reply starts and ends with `"` and is longer than one character. |
| `change_case:capital_word_frequency` | `capital_frequency`, `capital_relation` | The number of all-caps NLTK tokens, compared. `I'M sure NASA said OK.` counts 4: `I`, `'M`, `NASA`, `OK`. |
| `change_case:english_capital` | — | `reply.isupper()`, and langdetect says `en` or nothing. |
| `change_case:english_lowercase` | — | `reply.islower()`, and langdetect says `en` or nothing. |
| `punctuation:no_comma` | — | No ASCII `,` anywhere. |

### Strict, loose and unsupported ids

- **Strict** runs the check on the reply as written. Each checker strips where its rule needs to. A blank reply fails every instruction, even `no_comma`.
- **Loose** passes if the check succeeds on any non-blank variant from `loose_variants`. There are 8 variants:
  - the reply;
  - the reply without its first line, without its last line, and without both, each `.strip()`ped;
  - the same four with every `*` removed.

  Lines are split on `\n`. A one-line reply has no line-dropping variants, so only asterisk removal can help it. Loose is therefore never stricter than strict, and it can forgive real violations as well as preambles ([`ifeval.md`](ifeval.md#the-four-metrics)).
- **Supported ids** are exactly the 25 above. An unsupported id never reaches grading in a benchmark run: `IFEval.load` refuses the whole dataset with `IFEval prompt <key> uses unknown instructions: [...]`. A direct `check_instructions` call raises `KeyError`. Arguments are matched by name, so a row missing a required one raises `TypeError`. The runner records that as `grader failed: TypeError: ...`.

## Public API

| Name | Signature | What it's for |
|---|---|---|
| `CHECKS` | `dict[str, Callable[..., bool]]` | Instruction id → checker. Exactly 25 entries. |
| `check_instructions` | `check_instructions(instruction_ids: list[str], kwargs_list: list[dict[str, Any]], response: str) -> tuple[list[bool], list[bool]]` | `(strict, loose)` flags in instruction order. The two lists must have equal length (`zip(strict=True)`). |
| `loose_variants` | `loose_variants(response: str) -> list[str]` | The 8 relaxed readings used by loose. |
| `count_sentences` | `count_sentences(text: str) -> int` | Punkt sentence count ("Dr. Smith arrived. He sat down." → 2). |
| `count_words` | `count_words(text: str) -> int` | `regex` `\w+` runs. Needs `regex`. |
| `count_capital_words` | `count_capital_words(text: str) -> int` | All-caps `word_tokenize` tokens. |
| `detect_language` | `detect_language(text: str) -> str \| None` | ISO 639-1 code, or `None` when langdetect cannot decide (e.g. `"12345 !!!"`). |
| `ensure_nltk_data` | `ensure_nltk_data() -> None` | Downloads `punkt_tab` if `nltk.data.find("tokenizers/punkt_tab/english/")` fails. |
| the 25 checkers | `name(response: str, *, <arguments>, **_: Any) -> bool` | `keywords_existence`, `keywords_frequency`, `keywords_forbidden_words`, `keywords_letter_frequency`, `response_language`, `number_sentences`, `number_paragraphs`, `number_words`, `nth_paragraph_first_word`, `number_placeholders`, `postscript`, `number_bullet_lists`, `constrained_response`, `number_highlighted_sections`, `multiple_sections`, `json_format`, `title`, `two_responses`, `repeat_prompt`, `end_checker`, `quotation`, `capital_word_frequency`, `english_capital`, `english_lowercase`, `no_comma`. |

A missing `langdetect`, `nltk` or `regex` raises `DependencyMissingError` with the `bench` extra as the fix. Only the checkers that need the package raise it, and only when called.

## How to use it

### Check a reply against instructions

```python
from kumaru.eval.benchmarks.ifeval_checks import check_instructions

check_instructions(["startend:quotation", "punctuation:no_comma"], [{}, {}],
                   'Sure, here it is:\n"The quoted reply"')
# ([False, False], [True, True])   the preamble fails strict; loose drops the first line

kw = {"forbidden_words": ["coop", "killings", "dead", "night"]}
check_instructions(["keywords:forbidden_words"], [kw],
                   "in the dead of night , commandos crouched low.\nthey moved through the sector.")
# ([False], [True])                loose forgives a real violation by dropping line 1
```

### Debug one instruction

Call the checker directly with the row's arguments. Take them from the dataset (`IFEval().load()`, `item.meta["kwargs"]`; see [`ifeval.md`](ifeval.md#which-instructions-failed)).

```python
from kumaru.eval.benchmarks.ifeval_checks import CHECKS, count_capital_words, count_sentences, count_words

CHECKS["combination:two_responses"]("Joke one ******\nJoke two *******")             # False: 7 asterisks
CHECKS["detectable_format:number_bullet_lists"]("* a\n* b\n---", num_bullets=2)       # False: '---' is a bullet
CHECKS["length_constraints:nth_paragraph_first_word"](
    "Intro\n\n\n\nSecond para", num_paragraphs=2, nth_paragraph=2, first_word="second")  # False: extra blank line
count_sentences("Dr. Smith arrived. He sat down.")                                   # 2: Punkt knows "Dr."
count_words("Hello, world! It's 2024.")                                           # 5: "It's" is two runs
count_capital_words("I'M sure NASA said OK.")                                     # 4: I, 'M, NASA, OK
```

The NLTK-based helpers need the `punkt_tab` data. The first call downloads it if it is missing.

## How to extend it

To add an instruction type, for example after an upstream dataset change:

1. **Write the checker** as a pure, deterministic function:

   ```python
   def my_rule(response: str, *, my_arg: int, **_: Any) -> bool:
       ...
   ```

   Rules for it:
   - Name its arguments exactly as the dataset's `kwargs` names them. `IFEval.load` drops `None` values, so give a parameter a default if a row may omit it.
   - Keep `**_`: an argument added upstream must not crash older checkers.
   - Use `_compare` for `relation`-style arguments.
   - Follow the reference implementation (google-research `instruction_following_eval`), including its quirks, unless you document a deliberate difference.
2. **Register it** in `CHECKS` under the exact dataset id, `category:name`. The category becomes a `strict/<category>` metric automatically. `IFEval.load` accepts any id in `CHECKS`.
3. **Test it** in `tests\test_bench_ifeval.py`:
   - cover followed, violated, and the exact threshold;
   - add a strict-versus-loose case if a preamble or asterisks matter;
   - update `test_all_25_instruction_types_are_registered`, which pins `len(CHECKS) == 25`;
   - put tests that need NLTK data under the `nltk_ready` fixture, so they skip rather than fail where the data is absent.
4. **Validate** against the reference on real responses, as [`docs\BENCHMARKS.md`](../../../../../BENCHMARKS.md) does, before trusting the new category's scores.

Run only that file:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_bench_ifeval.py -q -p no:cacheprovider
```

## Failure modes

| Symptom | Likely cause | Fix |
|---|---|---|
| `KeyError: 'x:y'` from `check_instructions` | The id is not in `CHECKS`. In a benchmark run, `IFEval.load` reports it earlier as `uses unknown instructions`. | Add the checker (above) or fix the id. |
| `grader failed: ValueError: unknown IFEval relation: '...'` | A relation other than `less than` or `at least`. | Extend `_compare`, with tests. |
| `grader failed: TypeError: ... missing 1 required keyword-only argument` | The row lacks an argument the checker requires. | Inspect `item.meta["kwargs"]`, then give the parameter a default if upstream allows omitting it. |
| `Optional dependency 'regex' is not installed.` (or `'nltk'`, `'langdetect'`) | The `bench` extra is missing. `regex` normally arrives with `nltk`. | `pip install -e ".[bench]"` |
| `Could not download NLTK's 'punkt_tab' sentence model.` | No internet on first use. | Connect once, or `python -m nltk.downloader punkt_tab`. |
| `response_language` passes on gibberish | langdetect returned `None`, which counts as followed. | Reference behaviour; keep it for comparability. |
| `number_bullet_lists` fails on a list that looks right | The count is exact. Every line starting with `-` counts, `---` rules included. | Read the reply. The verdict follows the reference. |
| `nth_paragraph_first_word` fails on a correct-looking reply | An extra blank line adds an empty piece that shifts the index, or the paragraph count is not exactly `num_paragraphs`. | Reference quirk; keep it. |
| `two_responses` fails with a visible separator | The separator must be exactly `******`. | Expected. |
| `end_checker`, `json_format` or a `less than` count fails on a reply that stops mid-sentence | The reply hit the answer budget: 8,192 tokens by default (`Benchmark.answer_tokens`), plus any thinking budget. Checkers see the cut text as it stands. | The record's `finish_reason` is `"length"`. `n_truncated`, the **cut off** column of the report, is how you spot a binding cap. With IFEval's short replies it usually means a loop. |
| Your verdict differs from the reference on `letter_frequency` | The letter is not a-z; the reference checks a random letter. | Deliberate. Kumaru counts the requested character. |

## Related files

- `src\kumaru\eval\benchmarks\ifeval.py` — loads the dataset, calls `check_instructions`, aggregates the metrics ([`ifeval.md`](ifeval.md)).
- `tests\test_bench_ifeval.py` — pins the 25 registrations, the checkers, strict/loose, and NLTK-based counting.
- `pyproject.toml` — the `bench` extra (`langdetect`, `nltk`, `regex`, `datasets`).
- [`docs\BENCHMARKS.md`](../../../../../BENCHMARKS.md) — results and grader validation.

# `src\kumaru\train\distill.py`

> Turns a teacher model's graded answers into training data for a small, fast student model (knowledge distillation), and builds the human-written control arm the student is measured against. Behind `kumaru distill build|gold`.

**Read this when:** you are distilling a large model into a small one, building or auditing a distillation dataset, wondering why an answer was dropped, or setting up a comparison you can trust.

---

## What it does

- **`build_from_runs`** (`kumaru distill build`) reads the graded answers of one or more `kumaru bench run` folders. It keeps only answers that are correct, complete and clean, and writes `train.jsonl`, `val.jsonl` and `manifest.json` to a new folder.
- **`build_gold`** (`kumaru distill gold`) builds the same kind of dataset from GSM8K's human-written training solutions: the **control arm**. With `--match`, it keeps exactly the questions of a teacher dataset, each on the same side of the train/val split.
- **Eleven checks**, applied in a fixed order (below). The first check that rejects an answer is the one counted, so in `manifest.json`, `kept` plus the `dropped` counts equals `read`.
- **`render`** prints what was kept, what was dropped and why, a few examples, and the next command.

## Why it exists

**Distillation in one paragraph.** A 35B model answers GSM8K's 7,473 training questions, and a 1.5B model is fine-tuned on those answers with ordinary SFT. That is *sequence-level* knowledge distillation (Kim & Rush, 2016): the student imitates the teacher's text, not its probabilities. Keeping only answers a grader marked correct turns it into rejection-sampling fine-tuning (RFT; Yuan et al., 2023), a filtered relative of STaR (Zelikman et al., 2022). The student learns the teacher's worked solutions, step by step, not just its final numbers. Yuan et al. found that more *distinct* correct solutions per question help more, and that weaker models gain most.

**A student copies everything, mistakes included.** Every check exists because of one way bad data teaches a bad habit:

- a wrong answer teaches a wrong method;
- a cut-off answer teaches stopping mid-solution;
- an answer without the `Answer: N` line teaches a format the grader reads less reliably;
- a right final number reached through a wrong step teaches the wrong step.

**Two safeguards against fooling yourself:**

1. **Decontamination.** Training on test questions inflates exactly the score you are measuring. Only benchmarks that declare themselves training splits are accepted as sources, and near-copies of evaluation questions are dropped.
2. **A control arm.** If the teacher-trained student beats the base model, was it the teacher, or just extra GSM8K practice? `build_gold` trains a second student on the same questions with human-written answers. Only the teacher-vs-gold comparison isolates the teacher's contribution.

**Provenance.** `manifest.json` records the sources, settings, counts per check, examples and the sha256 of each file: proof of exactly which data trained a model.

## Mental model

### 1. A dataset is a pure function of its inputs

The same source files and the same settings give the same rows. Nothing is sampled at build time: the split is a hash (§6), and the checks are deterministic. Only `created_at` and `git_commit` in the manifest change between two builds.

### 2. The checks, in order

The first six read each answer's benchmark record. The last five run in `_finish`, over the whole dataset.

| Check | Drops an answer when | Why |
|---|---|---|
| `error` | the request or the grader failed | There is nothing to learn from. |
| `wrong` | the grader marked it wrong | A wrong method, learned. |
| `unfinished` | `finish_reason` is not `stop` (usually: cut off by the token limit) | Teaches stopping mid-solution. `None` (the backend did not say) passes. |
| `format` | the grader found the number some way other than the `Answer: N` line (`detail.method` is not `answer_tag`) | Both arms must end their answers the same way. Graders that record no method pass. |
| `empty` | the answer is blank | |
| `too_long` | it has more than `--max-tokens` tokens | The trainer drops rows longer than `max_seq_length` anyway. |
| `not_matched` | `gold --match` only: the question is not in the teacher dataset | Both arms must cover the same questions. |
| `contaminated` | the question or the answer shares a 13-word run with an evaluation question (§4) | Training on the test inflates the score. |
| `arithmetic` | a simple step `a op b = c` has the wrong `c` (§5) | A right answer with a wrong step teaches the wrong step. |
| `duplicate` | the same answer (whitespace-normalised) to the same question is already kept | Copies only re-weight; distinct solutions are what help (§7). |
| `over_cap` | the question already has `--per-question` answers | Easy questions, which every sample gets right, would crowd out hard ones. |

### 3. Sources: training splits only, refused whole

A source is accepted only if its benchmark sets `Benchmark.training_data = True` ([`base.md`](../eval/benchmarks/base.md#public-api)). `gsm8k_train` does; `gsm8k`, every other benchmark and every golden set do not. Unknown names count as evaluation sets.

A bad source is **refused as a whole, before anything is written**, rather than silently dropped: a mistake in the inputs should stop the build, not quietly produce a smaller dataset. The same holds for thinking runs. The student would see the visible answer without the reasoning that produced it, so they are refused unless you pass `--allow-thinking`.

### 4. The near-copy check (`NearCopyIndex`)

Two texts that share a run of 13 consecutive words are near-copies. That is GPT-3's contamination test (Brown et al., 2020): long enough that a match is no coincidence, and short enough to catch a lightly edited copy. Both the question and the answer are checked against every question of each `--eval-set`.

**Boilerplate is masked first.** Every GSM8K prompt starts with the same instruction. A 13-word run found in more than 1% of the evaluation questions, or of the training questions, is boilerplate. It is cut out before runs are formed, so a run cannot straddle it either. Without that, "...with no units. John" would tie together every pair of questions that start with "John". An early smoke test caught exactly that bug.

**What it found in GSM8K.** Four training questions are template siblings of test questions: the same problem with new names and numbers.

```
gsm8k_train/1314  A train travels 270 miles in 3 hours. At the same rate, how many additional hours would it take to travel an additional 180 miles?
gsm8k/602         A plane travels 1200 miles in 3 hours. At the same rate, how many additional hours would it take to travel an additional 2000 miles?
```

The others are `gsm8k_train/20` (≈ `gsm8k/632`, the stamps problem), `gsm8k_train/406` (≈ `gsm8k/581`, two movies) and `gsm8k_train/5162` (≈ `gsm8k/602` again).

### 5. The arithmetic check (`wrong_steps`)

The check finds every step written as `a op b = c`: one operation on two plain numbers. It recomputes the result and flags `c` when it is off by at least one unit of its last digit.

- **Rounding and truncation pass.** `10 / 3 = 3.33` and `7 / 2 = 3` (whole boxes) are fine; `3 x 4 = 13` is not.
- **`%` may be read as a fraction**, so `1 / 4 = 25%` passes.
- **It is conservative on purpose**, because a false alarm throws away a good example. Skipped, not flagged:
  - steps inside longer expressions (`2 + 3 x 4 = 14`);
  - inputs with three or more decimals (4.333 is probably 13/3, rounded);
  - clock times (`0900`);
  - numbers that go on (`18 000`).

**What it found in GSM8K's own reference solutions.** Of 7,473 solutions, 24 are flagged.

- **17 are real errors**, for example `6*2.5 = 18.00` (gsm8k_train/168) and `6/3=.5` (gsm8k_train/198). In one, `192*100=1920` (gsm8k_train/167), the error carries through to the final label, so the official GSM8K answer to that question is wrong.
- **7 are false alarms**, where the text means something the arithmetic does not show:
  - unit changes: `$1 / 2 = 50 cents`, `$2.50 - $2.24 = 26 cents`, `5 * 3 = $15,000` (in thousands);
  - a mixed number: `12 x 1 and 1/4 = 15`;
  - typos in a number: `14, 400 / 80` and `. 84`;
  - a misreading: `day 1 * 2 = 20 pages`.

On 100 teacher answers it raised no false alarms.

### 6. The split: by question, by hash

`split_of(id, seed, val_fraction)` hashes `"<seed>:<question id>"` with sha256 and puts the question in `val` when the first 8 bytes, read as a fraction of 2^64, fall below `val_fraction`.

- **Hashing the question, not the answer,** keeps every answer to one question on the same side, so the validation loss never scores a question the model trained on.
- **The same seed holds out the same questions** in every dataset. `--match` goes further: the gold arm copies the teacher dataset's sides exactly.
- **`val.jsonl` is not the score.** The trainer uses it to pick the checkpoint with the lowest validation loss. The score comes from `kumaru bench run gsm8k`, GSM8K's 1,319 test questions, which no training row is drawn from.

### 7. Several answers per question

One teacher run gives one answer per question, so neither `duplicate` nor `over_cap` fires. Run the teacher again with another `--seed` (it samples at temperature 0.7) and pass both runs: each question can now have two different correct solutions. Identical answers count once, and no question keeps more than `--per-question` (default 4).

### 8. Train as you test

Rows carry no system message, like the requests of `kumaru bench run` (without `--use-agent-prompt`). Qwen's chat template adds its default system line in both cases, so training and evaluation see the same tokens.

**Caveat:** `kumaru chat` adds Kumaru's own system prompt, which the student never saw in training. Expect chat answers to differ from benchmark answers.

### 9. The output folder

```
runs\distill\data-teacher\
    train.jsonl     one row per line (below)
    val.jsonl       rows of the held-out questions
    manifest.json   Manifest: sources, settings, counts per check, examples, file hashes
```

One row:

```json
{"messages": [{"role": "user", "content": "Solve the following math problem. ..."}, {"role": "assistant", "content": "... \nAnswer: 72"}], "id": "gsm8k_train/0", "source": "qwen36-nothink"}
```

- **`source`** is the teacher run's label, or `gold:openai/gsm8k`.
- **`id`** is the benchmark's question id. `--match` uses it, and `kumaru train` ignores it.
- **Writes are atomic.** The old manifest is deleted first and the new one is written last, so a manifest exists only while the data beside it is complete.
- **Files are written as bytes** (`\n` line endings on every OS), so the hashes in the manifest match the files everywhere.

## Public API

| Name | Signature | What it's for |
|---|---|---|
| `build_from_runs` | `build_from_runs(runs: Sequence[str \| Path], out: str \| Path, settings: DistillSettings \| None = None, *, force: bool = False) -> Manifest` | Teacher runs → dataset. Each run is a whole run folder or one benchmark's folder in it. |
| `build_gold` | `build_gold(out: str \| Path, settings: DistillSettings \| None = None, *, match: str \| Path \| None = None, force: bool = False) -> Manifest` | GSM8K's human solutions → dataset. Sets `max_tokens=None`: human solutions carry no token count. |
| `DistillSettings` | `max_tokens=1024 (≥1 or None), per_question=4 (≥1), val_fraction=0.05 (0..0.5), seed=42, check_arithmetic=True, decontaminate=True, eval_sets=["gsm8k"], allow_thinking=False` | How a dataset is built. Recorded in full in the manifest. |
| `Manifest` | pydantic model: `kind` (`teacher`/`gold`), `created_at`, `kumaru_version`, `git_commit`, `settings`, `sources`, `read`, `kept`, `train_rows`, `val_rows`, `questions`, `val_questions`, `dropped`, `mean_answer_tokens`, `mean_answer_chars`, `examples`, `files` | `manifest.json`. `dropped` lists the non-zero checks in check order; `examples` keeps up to 5 per check; `files` maps each data file to its sha256. |
| `Candidate` | dataclass: `id`, `prompt`, `answer`, `source`, `tokens` | One answer before the dataset-level checks. |
| `wrong_steps` | `wrong_steps(text: str) -> list[str]` | Every simple step with a wrong result (§5). |
| `NearCopyIndex` | `NearCopyIndex(eval_prompts: dict[str, str], train_prompts: Iterable[str] = (), *, n: int = 13, share: float = 0.01)`; `.grams(text)`, `.find(text) -> str \| None`, `.boilerplate` | The near-copy check (§4). `find` returns the id of the first evaluation question `text` overlaps. |
| `split_of` | `split_of(question_id: str, seed: int, val_fraction: float) -> "train" \| "val"` | The split (§6). |
| `gold_rationale` | `gold_rationale(solution: str) -> str \| None` | A GSM8K solution in the teacher's format: calculator notes (`<<48/2=24>>`) removed, `#### 72` → `Answer: 72`. `None` when there is no `####` or no body. |
| `render` | `render(console: Console, manifest: Manifest, out: Path) -> None` | The console report. |
| Constants | `REASONS` (check → meaning, in check order), `SIDES = ("train", "val")`, `FILES`, `NGRAM = 13`, `COMMON_SHARE = 0.01`, `EXAMPLES_PER_REASON = 5`, `ROUNDED_DECIMALS = 3` | |

Every refusal is a `ConfigError` with a hint, raised before anything is written ([failure modes](#failure-modes)).

## How to use it

### The whole experiment

[`docs\TRAINING.md`](../../../../TRAINING.md#part-3--distillation) explains each step, what it costs and how to read the result. In brief:

```powershell
# 1. The teacher answers the training questions: a 200-question pilot, then the rest.
kumaru bench run gsm8k_train -c qwen36 --no-think -n 200 --max-tokens 1024 --context 8192 -o runs\distill\teacher-qwen36
kumaru bench run gsm8k_train -c qwen36 --no-think --full --max-tokens 1024 --context 8192 -o runs\distill\teacher-qwen36

# 2. The two datasets: teacher answers, and human answers to the same questions.
kumaru distill build runs\distill\teacher-qwen36 -o runs\distill\data-teacher
kumaru distill gold -o runs\distill\data-gold --match runs\distill\data-teacher

# 3. Train one student on each (check the data first with --dry-run).
kumaru train configs\train\distill-gsm8k.yaml
kumaru train configs\train\distill-gsm8k-gold.yaml

# 4. Score the base student and both trained ones on GSM8K's test set, then compare.
kumaru bench run gsm8k -c student --full --max-tokens 1024 --label student-base -o runs\distill\eval-base
kumaru bench run gsm8k -c student --full --max-tokens 1024 --adapter runs\distill\qwen1.5b-teacher\adapter --label student-teacher -o runs\distill\eval-teacher
kumaru bench run gsm8k -c student --full --max-tokens 1024 --adapter runs\distill\qwen1.5b-gold\adapter --label student-gold -o runs\distill\eval-gold
kumaru bench compare runs\distill\eval-gold runs\distill\eval-teacher -e 5
```

The second teacher command resumes the first: each question's sampling seed depends only on its id, so the pilot's 200 answers are reused. `--label` names each run in the comparison table; without it, all three runs would be called `student-default`.

### Build the datasets

```powershell
kumaru distill build runs\distill\teacher-qwen36 -o runs\distill\data-teacher
kumaru distill build runs\distill\teacher-qwen36 runs\distill\teacher-qwen36-s2 -o runs\distill\data-teacher-x2   # two samples per question
kumaru distill gold -o runs\distill\data-gold --match runs\distill\data-teacher
kumaru distill build runs\bench\<a gsm8k run> -o runs\distill\smoke --no-decontaminate                        # pipeline check only
```

Flags, verified against `kumaru distill build --help` and `kumaru distill gold --help`:

| Flag | Default | Effect |
|---|---|---|
| `RUNS...` | | `build` only. Run folders from `kumaru bench run`, or one benchmark's folder in one. |
| `-o/--out` | required | A new folder for the three files. |
| `--max-tokens` | 1024 | `build` only. Drop answers longer than this; keep it below the training `max_seq_length` minus the prompt (GSM8K prompts are at most ~280 tokens). |
| `--per-question` | 4 | `build` only. Distinct answers kept per question. |
| `--val-fraction` | 0.05 | 0..0.5. Share of *questions* held out. |
| `--seed` | 42 | Picks the held-out questions. Keep it equal across datasets you compare. |
| `--check-arithmetic/--no-check-arithmetic` | on | The `arithmetic` check. |
| `--eval-set` | `gsm8k` | Repeatable. A benchmark the student will be scored on; its questions never become training data. |
| `--decontaminate/--no-decontaminate` | on | Off skips both the source rule and the near-copy check, and prints a warning. Smoke tests only: never report a score from a model trained on such data. |
| `--allow-thinking` | off | `build` only. Accept thinking runs; the student learns the visible answers. |
| `--match` | | `gold` only. A teacher dataset folder: keep its questions, on the same sides (`--val-fraction` and `--seed` are then unused). |
| `--force` | off | Replace a dataset already in `--out`. |

### Read the report

Real output of `kumaru distill gold` (CPU only, about 4 seconds; paths shortened):

```
kept 7,445 of 7,473 answers (99.6%): 7,092 train rows, 353 val rows (353 held-out questions)
                             dropped
┌──────────────┬─────────┬──────────────────────────────────────┐
│ check        │ answers │ meaning                              │
├──────────────┼─────────┼──────────────────────────────────────┤
│ contaminated │       4 │ near-copy of an evaluation question  │
│ arithmetic   │      24 │ a step "a op b = c" with the wrong c │
└──────────────┴─────────┴──────────────────────────────────────┘
contaminated: gsm8k_train/20  gsm8k/632
contaminated: gsm8k_train/406  gsm8k/581
contaminated: gsm8k_train/1314  gsm8k/602
arithmetic: gsm8k_train/167  192*100=1920
arithmetic: gsm8k_train/168  6*2.5 = 18.00
arithmetic: gsm8k_train/198  6/3=.5

saved to runs\distill\data-gold
train: point data.train_file at runs\distill\data-gold\train.jsonl and data.eval_file at
runs\distill\data-gold\val.jsonl, then `kumaru train <config>` (see configs/train/distill-gsm8k.yaml)
```

- **First line:** answers kept out of answers read, and how they split. A teacher build adds `mean answer: N tokens`.
- **`dropped`:** one row per check that dropped anything, in check order.
- **Examples:** up to three per check for `contaminated` (which evaluation question it matches) and `arithmetic` (the first wrong step). The manifest keeps up to five per check.
- **`not decontaminated:`** in yellow, when you passed `--no-decontaminate`.

A teacher build keeps a little less than the teacher's accuracy, since wrong answers are the main loss. If it keeps much less, read `dropped` before you train; [Failure modes](#failure-modes) says what each large count usually means.

### From Python

```python
from kumaru.train.distill import DistillSettings, build_from_runs, split_of, wrong_steps

wrong_steps("3 x 4 = 13, so 10 / 3 = 3.33 and 7 / 2 = 3 boxes; 1 / 4 = 25%; 2 + 3 x 4 = 14")
# ['3 x 4 = 13']
split_of("gsm8k_train/0", 42, 0.05)
# 'train'

manifest = build_from_runs(
    ["runs/distill/teacher-qwen36"], "runs/distill/data-teacher-x1",
    DistillSettings(per_question=1),
)
manifest.kept, manifest.dropped   # the report's numbers: an int, and {check: count} in check order
```

## How to extend it

- **Another task.** `build_from_runs` works for any benchmark whose run records carry a verdict. Give the benchmark a training-split twin with `training_data = True` (as `GSM8KTrain` is to `GSM8K`, see [`gsm8k.md`](../eval/benchmarks/gsm8k.md)). For code or prose, pass `--no-check-arithmetic`: the check is meant for worked arithmetic. `build_gold` is GSM8K-only, so another task needs its own control-arm builder that returns `Candidate`s and calls `_finish`.
- **Another check.**
  - A check that needs only one answer's record goes in `_problem`; one that looks across answers goes in the loop in `_finish`.
  - Add its name and meaning to `REASONS` **in check order**: the report and the manifest follow that order.
  - Test it in `tests\test_distill.py`: it drops what it should, keeps what it should, and `kept + sum(dropped) == read` still holds.
- **Logit distillation** (Hinton et al., 2015). The student matches the teacher's full next-token distribution instead of its sampled text, which carries more information per token. It needs the teacher's per-token probabilities stored with each answer, a KL-divergence term in the trainer, and a tokenizer the two models share (or a mapping between the two).
- **On-policy distillation** (GKD; Agarwal et al., 2024). Training only on teacher text means the student never practises recovering from its own mistakes. In GKD the student generates and the teacher scores its tokens during training. This needs the teacher online throughout, so it is a trainer change rather than a data change.

## Failure modes

| Symptom | Likely cause | Fix |
|---|---|---|
| `<dir> holds answers to gsm8k, an evaluation set: training on them would inflate the score it measures.` | The source is a run of a test split, a golden set, or a benchmark named in `--eval-set`. | As the hint says: run the teacher on a training split (`gsm8k_train`) and pass that benchmark's own folder. `--no-decontaminate` is for pipeline smoke tests only. |
| `<dir> is a thinking run: the student would learn the visible answers without the reasoning that produced them.` | The teacher ran with thinking on. | Re-run the teacher with `--no-think`, or pass `--allow-thinking`. |
| `No benchmark results in <dir>` | The path is not a `kumaru bench run` folder. | Point at the folder `bench run` printed as `results:` (under `runs\`). |
| `<out> already holds a dataset (train.jsonl, val.jsonl, manifest.json)` | The folder was used before. | Another `--out`, or `--force` to replace it. |
| `<out> is a file, not a folder` | `-o` names an existing file. | Choose a folder. |
| `<match>\train.jsonl not found` | `--match` is not a distill dataset folder. | Pass the `-o` folder of `kumaru distill build`. |
| `No answer survived the checks (wrong 100).` | Every answer failed a check, here the grader. | Read the counts: they say which check. Make sure the runs are graded runs of the right benchmark. |
| Many `wrong` or `unfinished` | The teacher's `--max-tokens` cut answers off. About 5% of Qwen3.6's no-think GSM8K answers exceed 1,024 tokens ([BENCHMARKS.md §6.1](../../../../BENCHMARKS.md#61-the-truncation-confound)). A cut-off answer rarely reaches its `Answer:` line, so most are counted `wrong`; the few graded right are `unfinished`. | Expected at that rate: those answers would not fit `max_seq_length: 1536` anyway. `kumaru bench report <dir>` shows the `cut off` count. Far more than 5% means the teacher ran with a smaller budget than intended. |
| Many `format` | The teacher stopped writing the `Answer: N` line. | Read some records (`kumaru bench report <dir>`). A system prompt may be overriding the instruction (`--use-agent-prompt`). |
| Many `too_long` | The teacher ran with a larger budget than distill's `--max-tokens`. | Fine if few. Raising `--max-tokens` also needs a larger training `max_seq_length`. |
| `kumaru train` logs `dropped N/M example(s) (x%) longer than 1536 tokens` | Rows longer than `data.max_seq_length`. | Lower distill's `--max-tokens`, or raise `max_seq_length` (more VRAM). |
| `Training data not found: runs\distill\data-teacher\train.jsonl` | The dataset is not built yet, or was built elsewhere. | Build it, or point `data.train_file` and `data.eval_file` at it. |

## Known limitations

- **Paraphrases pass the near-copy check.** A reworded or translated test question shares no 13-word run. Decontamination here is a floor, not a guarantee.
- **The arithmetic check sees one operation at a time.** A chain like `2 + 3 x 4 = 14` is skipped, a wrong method with right arithmetic passes, and a step written in words ("twice 12 is 25") is not seen. Grading every step needs a process reward model ("Let's Verify Step by Step", Lightman et al., 2023).
- **Duplicates must match exactly** (after whitespace is collapsed). Two answers that differ by one word both count as distinct.
- **The grader trusts GSM8K's labels.** At least one is wrong (gsm8k_train/167). A teacher that gets that question right is graded wrong, and its answer is dropped.
- **`build_gold` is GSM8K-only.**

## Related files

- `src\kumaru\cli.py` — `kumaru distill build|gold` ([`cli.md`](../cli.md)).
- `src\kumaru\eval\benchmarks\gsm8k.py` — `gsm8k_train`, the prompt, `gold_answer` and `load_hf_rows` ([`gsm8k.md`](../eval/benchmarks/gsm8k.md)).
- `src\kumaru\eval\benchmarks\base.py` — `Benchmark.training_data` ([`base.md`](../eval/benchmarks/base.md)).
- `src\kumaru\eval\benchmarks\runner.py` — `find_results`, `load_results` and `resolve_benchmark`, and `bench compare` ([`runner.md`](../eval/benchmarks/runner.md)).
- `src\kumaru\train\finetune_lora.py` — the trainer the datasets feed ([`finetune_lora.md`](finetune_lora.md)).
- `configs\train\distill-gsm8k.yaml`, `distill-gsm8k-gold.yaml`, `distill-gsm8k-0.5b.yaml` — the training recipes ([`distill-gsm8k.yaml.md`](../../../configs/distill-gsm8k.yaml.md)).
- `configs\student.yaml` — the student's evaluation profile ([`student.yaml.md`](../../../configs/student.yaml.md)).
- `tests\test_distill.py` — pins every check, the split, the source rule, refusals before writing, the manifest and the configs. Among them: `test_every_check_drops_what_it_should_and_the_counts_add_up`, `test_near_copies_are_found_and_shared_instructions_are_not` (the straddling bug of §4) and `test_the_arms_and_the_pilot_differ_only_where_they_must`.
- [`docs\TRAINING.md`](../../../../TRAINING.md#part-3--distillation) — the experiment, step by step, with costs.

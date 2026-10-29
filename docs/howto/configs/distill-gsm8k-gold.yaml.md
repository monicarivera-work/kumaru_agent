# `configs/train/distill-gsm8k-gold.yaml`

> The control arm of the distillation experiment: the student and recipe of [`distill-gsm8k.yaml`](distill-gsm8k.yaml.md), trained on GSM8K's human-written solutions to the same questions instead of the teacher's answers.

**Read this when:** you are training the control arm, or deciding what a teacher-versus-gold result means.

---

## What it does

A student trained on the teacher's answers can beat the base model for two reasons. The teacher's answers may have taught it something; or any practice on GSM8K-style questions, in the answer format the grader expects, would have done the same. A comparison with the base model cannot tell these apart. This arm changes only who wrote the answers, so comparing the two arms isolates what the teacher added.

`kumaru distill gold --match runs\distill\data-teacher` writes its data. For every question in the teacher dataset, it takes GSM8K's own solution and rewrites it in the teacher's format: calculator notes such as `<<48/2=24>>` are removed, and `#### 72` becomes `Answer: 72`. Each question stays on the same side of the train/validation split. The arithmetic check runs on these solutions too. It flags 24 of them, [17 real errors](../src/kumaru/train/distill.md#5-the-arithmetic-check-wrong_steps) in GSM8K's reference answers among them, so up to 24 questions are in the teacher arm only.

## When to use this profile

- After the teacher dataset is built; `--match` reads it. Train it alongside the teacher arm: without it, a teacher-arm gain has two possible explanations.
- Not for the 0.5B pilot, which has no control arm. To add one, copy this file and change `base_model`, `output_dir`, `run_name` and the batch split as [`distill-gsm8k-0.5b.yaml`](distill-gsm8k-0.5b.yaml.md) does.

## Every field explained

Only four fields differ from the teacher arm. Every other field, and why it has its value, is in [`distill-gsm8k.yaml.md`](distill-gsm8k.yaml.md#every-field-explained). `test_the_arms_and_the_pilot_differ_only_where_they_must` (in `tests/test_distill.py`) fails if anything else differs.

| Field | Here | In the teacher arm | Why it differs |
|---|---|---|---|
| `data.train_file` | `runs/distill/data-gold/train.jsonl` | `runs/distill/data-teacher/train.jsonl` | The tested factor: who wrote the answers. |
| `data.eval_file` | `runs/distill/data-gold/val.jsonl` | `runs/distill/data-teacher/val.jsonl` | The same held-out questions, with human answers, so each arm picks its checkpoint on its own kind of answer. |
| `output_dir` | `runs/distill/qwen1.5b-gold` | `runs/distill/qwen1.5b-teacher` | Each arm keeps its own checkpoints and adapter. |
| `run_name` | `distill-gsm8k-gold` | `distill-gsm8k-teacher` | Names the run in logs. |

## Cost and length

GSM8K's solutions are short. These counts were measured with the trainer's own encoding, on the full gold dataset and on the teacher's answers to a 100-question sample:

| | Gold | Teacher |
|---|---|---|
| Answer, mean | 93 tokens (250 characters) | 333 tokens (1,040 characters) |
| Whole row, median | 206 tokens | 455 tokens |
| Longest row | 506 tokens | up to ~1,330 |

- **Steps:** 422, as in the teacher arm, when the teacher dataset holds one answer per question: 6,732 training rows against 6,751.
- **Time:** 134 minutes, measured on 2026-10-22. Shorter rows did not make it faster: at one row per pass, a step costs about the same whatever its length, and the first ~175 steps took about 16.6 seconds each, as in the teacher arm. A game then shared the GPU and slowed the rest by about a quarter. Alone, the run would have taken about two hours, as the teacher arm did.
- **Memory:** a peak of 6.4 GB reserved, measured, against 13.8 GB for the teacher arm. The longest row is 506 tokens, so the loss over the vocabulary needs under 1 GB (estimated; [the arithmetic](distill-gsm8k.yaml.md#cost)).

**The arms differ in length by design.** With equal rows and steps, the gold arm trains on about a quarter as many answer tokens: 0.62 million against 2.26 million in the 2026-10-22 run. That is part of what is being compared, since writing out every step is part of what a teacher gives. But it means a teacher win cannot say whether the teacher reasons better, or only writes more steps. The opposite can happen too. Li et al. (2025, *Small Models Struggle to Learn from Strong Reasoners*) found that students of 3B parameters or fewer often learn more from short reasoning than from a strong teacher's long reasoning. So the short human solutions can win.

## How to use it

From the Kumaru folder, after `kumaru distill build` and the base and teacher-arm evaluations ([Part 3 of TRAINING.md](../../TRAINING.md#part-3--distillation)):

```powershell
kumaru distill gold -o runs\distill\data-gold --match runs\distill\data-teacher
kumaru train configs\train\distill-gsm8k-gold.yaml --dry-run
kumaru train configs\train\distill-gsm8k-gold.yaml
kumaru bench run gsm8k -c student --full --max-tokens 1024 --adapter runs\distill\qwen1.5b-gold\adapter --label student-gold -o runs\distill\eval-gold
kumaru bench compare runs\distill\eval-gold runs\distill\eval-teacher -e 5
kumaru bench compare runs\distill\eval-base runs\distill\eval-gold
```

- **The build report** counts `not_matched` for every GSM8K question that is not in the teacher dataset: mostly the questions the teacher got wrong. The rows should then nearly equal the teacher dataset's.
- **The first comparison** is the experiment: the gold arm is the baseline (A), the teacher arm the candidate (B). The second shows what GSM8K practice alone did.
- **Never compare the arms' validation losses.** Each is measured on its own kind of answer, and the teacher's long, regular text is easier to predict per token. Only the GSM8K scores compare.

### Reading the result

"Above" and "below" mean `kumaru bench compare` reports a difference with a p-value below 0.05; "no difference", one without. "No difference" means none was detected, not that there was none. At 1,319 questions, a difference of about 3 points is the smallest it reliably detects. Train three seeds per arm before concluding ([how](distill-gsm8k.yaml.md#how-to-use-it)).

A teacher arm above gold says only that the teacher's answers beat human ones, not that either helped. Read the teacher arm against base first. Where a row turns on what human answers did on their own, read the gold arm against base (the second comparison above); never work it out from the other two. A teacher arm 2 points below base and 2 above gold can show no difference in either, while the gold arm, 4 points below base, shows one.

| Teacher arm against base | Teacher arm against gold | What it means |
|---|---|---|
| Above | Above | The teacher's answers taught more than human ones: distillation worked. |
| Above | No difference | A gain, but the teacher's answers added nothing measurable over human ones. If the gold arm is above base too, GSM8K-style practice alone was enough for a gain; if not, these runs cannot say whether the practice or the teacher's answers brought the teacher arm's gain. |
| Any | Below | Human solutions were better data for this student, likely for the reason above. Next: shorter teacher answers, or both datasets mixed. |
| No difference | Above | No measurable gain. Only the gold arm against base says why the arms differ. If the gold arm is below base, they differ at least partly because it lost points; if not, neither arm moved measurably on its own. [The 2026-10-22 run](../../TRAINING.md#results-2026-10-22) ended here: compared directly, the gold arm against base was −11.1 points (−13.8 to −8.3), p 2.19e-14, with answers a third as long. |
| No difference | No difference | The teacher's answers changed nothing measurable. Only the gold arm against base says whether human answers did. If neither did, check `example_preview.txt`, and whether the training loss fell (`log_history` in the last checkpoint's `trainer_state.json`). |
| Below | Above or no difference | Training on the teacher's answers hurt. Look at the `-e 5` examples, and at how many answers were cut off (`kumaru bench report`), as for [the pilot](distill-gsm8k-0.5b.yaml.md#reading-the-result). |

## Tuning it

Do not tune this file alone. Any change except the four fields above belongs in all three recipes: tune the teacher arm ([Tuning it](distill-gsm8k.yaml.md#tuning-it)) and copy the change here. Otherwise the arm test fails, and the comparison is no longer fair.

One case needs care. If the teacher dataset holds several answers per question, the gold arm still has one per question, so it gets fewer rows and fewer steps. Either report that as part of the result, or build the teacher dataset with `--per-question 1`.

## Failure modes

| Symptom | Likely cause | Fix |
|---|---|---|
| `...\data-teacher\train.jsonl not found` | The teacher dataset is not built yet, or `--match` points elsewhere. | Run `kumaru distill build` first; pass its `-o` folder to `--match`. |
| `runs\distill\data-gold already holds a dataset` | A previous build. | Add `--force` to replace it. |
| `not_matched` near 7,473 | `--match` points at a small dataset, such as one built from the 200-question pilot. | Rebuild the teacher dataset from the full run, then this one with `--force`. |
| `Training data not found: runs\distill\data-gold\train.jsonl` | This dataset is not built, or `kumaru train` ran in another folder. | Run `kumaru distill gold` (above), from the Kumaru folder. |
| `test_the_arms_and_the_pilot_differ_only_where_they_must` fails | One recipe was edited alone. | Make the same change in all three. |
| The gold arm's validation loss is far above the teacher arm's | Expected (see above). | Compare GSM8K scores. |

## Related files

- `configs/train/distill-gsm8k.yaml` — the teacher arm: every shared field, the cost breakdown and tuning ([`distill-gsm8k.yaml.md`](distill-gsm8k.yaml.md)).
- `src/kumaru/train/distill.py` — `build_gold`, `gold_rationale` and the arithmetic check ([`distill.md`](../src/kumaru/train/distill.md)).
- `src/kumaru/eval/benchmarks/gsm8k.py` — `gsm8k_train`, whose prompts and ids both arms share ([`gsm8k.md`](../src/kumaru/eval/benchmarks/gsm8k.md)).
- `tests/test_distill.py` — the arm test.
- [`docs/TRAINING.md`](../../TRAINING.md#part-3--distillation) — the experiment, step by step.

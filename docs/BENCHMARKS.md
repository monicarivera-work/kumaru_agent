# Benchmarks: is Kumaru actually smarter?

> You asked what would make Kumaru "incredibly smart". The biggest single lever is the model, and the only honest way to know a change helped is to measure it. This document is both: the measured result of replacing Qwen2.5-7B with Qwen3.6-35B-A3B, and the manual for measuring every change you make from now on.

**Read this when:** you want the numbers; you are about to change a model, prompt or setting and want to know whether it helped; or you want to add a benchmark.

---

## TL;DR

Accuracy (%) on the same questions, same engine, same harness. In parentheses: the 95% confidence interval.

| Benchmark (questions) | Qwen2.5-7B (before) | Qwen3.6-35B-A3B, Think off | Qwen3.6-35B-A3B, Think on |
|---|---|---|---|
| golden (27): your own questions | 96.3 (81.7–99.3) | 96.3 (81.7–99.3) | 100.0 (87.5–100.0) |
| GSM8K (100): grade-school math | 92.0 (85.0–95.9) | 95.0 (88.8–97.8) | 97.0 (91.5–99.0) |
| MMLU-Pro (100): knowledge + reasoning | 59.0 (49.2–68.1) | 84.0 (75.6–89.9) | 79.0 (70.0–85.8) |
| IFEval (100): instruction following | 72.0 (62.5–79.9) | 80.0 (71.1–86.7) | not run |
| HumanEval (100): Python coding | 81.0 (72.2–87.5) | 99.0 (94.6–99.8) | not run |
| tokens per answer (mean) | 283 | 611 | 2,039 (Think off: 856 on the same three benchmarks) |

**Verdicts** (paired significance tests, [§4](#4-reading-the-numbers)): Qwen3.6 with Think off beats the 7B on MMLU-Pro (+25 points) and HumanEval (+18), both p < 0.001; the golden, GSM8K and IFEval differences are within noise. Think on against Think off: no clear difference on golden (+3.7), GSM8K (+2.0) or MMLU-Pro (−5.0, p = 0.5), at 2.4 times the tokens. The MMLU-Pro dip is not evidence that thinking hurts: 24 of its 100 Think-on answers hit the 4,096-token thinking cap, including 5 of the 6 questions that Think off got right and Think on got wrong.

**What to use:** Qwen3.6 (`-c qwen36`) with Think off, the profile's default. It is the measured upgrade over the 7B, and in this run thinking bought no significant accuracy for 2.4 times the tokens. Think on is still worth trying on hard multi-step problems: chat does not cap thinking, so uncapped thinking is not yet measured. A rerun with a larger `--thinking-budget` would measure it.

The Think-on run was stopped after three benchmarks to free the GPU. To finish IFEval and HumanEval, run its original command again with `--out`; finished questions are skipped ([§9](#9-run-mechanics)):

```powershell
$env:KUMARU_BACKEND__MODELS_DIR = 'D:\kumaru\models'      # where that run's model is stored
$env:KUMARU_BACKEND__LLAMA_SERVER__N_CPU_MOE = '18'       # the GPU split that run used (§3)
kumaru bench run golden gsm8k mmlu_pro ifeval humaneval -c qwen36 --context 65536 --think --thinking-budget 4096 -n 100 --allow-code-exec --out runs\bench\20261010-134047-qwen36-think4096
```

Why a bigger model can be this fast: Qwen3.6-35B-A3B is a *mixture of experts*. It has 35B parameters (5x the 7B), and knowledge scales with total parameters, but a router activates only ~3B of them per token, and speed scales with *active* parameters. It is also two model generations newer, with an optional *thinking* mode that reasons privately before answering. The full explanation, including why it fits in 24 GB, is in [`configs/qwen36.yaml`](../configs/qwen36.yaml).

---

## 1. Measure something in five minutes

```powershell
pip install -e ".[bench]"                       # once: datasets + IFEval's checkers
kumaru bench list                               # what is available
kumaru bench run golden -c qwen36               # your 27 questions, about two minutes
kumaru bench run gsm8k mmlu_pro -c qwen36 -n 100
kumaru bench report  runs\bench\<run folder>
kumaru bench compare runs\bench\<run A> runs\bench\<run B> --examples 3
```

The first run of a public benchmark downloads its dataset (cached afterwards). `qwen36` needs `.\scripts\get-llama-server.ps1` once, and downloads the 17.7 GB model on first use; any profile works, including your fine-tuned ones.

`run` prints one progress line per benchmark and saves every answer. `report` shows a saved run in detail. `compare` answers the question you actually care about: *did B really beat A, or is that noise?*

## 2. The benchmarks

| Name | What it measures | Full size | Chance | Score |
|---|---|---|---|---|
| `golden` | Your own regression questions ([§8](#8-your-own-golden-questions)) | 27, grow it | - | accuracy |
| `gsm8k` | Multi-step grade-school arithmetic (Cobbe et al., 2021) | 1,319 | ~0% | final number correct |
| `mmlu_pro` | Knowledge plus reasoning across 14 subjects, up to 10 options (Wang et al., 2024) | 12,032 | ~11% | letter correct |
| `ifeval` | Following verifiable instructions: "no commas", "exactly 3 bullet points" (Zhou et al., 2023) | 541 prompts, 834 instructions, 25 types | - | prompt-level strict |
| `humaneval` | Writing a correct Python function from its docstring (Chen et al., 2021) | 164 | ~0% | pass@1: hidden unit tests pass |

What each is good for:

- **MMLU-Pro** separates models best: it is hard (10 options, many need calculation), broad, and far from saturated. Weight it most.
- **GSM8K** is nearly saturated by modern models (both score in the 90s). It is a smoke test: a big drop means something broke. Small models still have room: the Qwen2.5 report gives about 73% for Qwen2.5-1.5B, so [distillation](TRAINING.md#part-3--distillation) is measured on it.
- **IFEval** is the closest proxy for "does what I asked". It reports four scores: *prompt-level* (every instruction in the prompt followed) and *instruction-level* (share of individual instructions followed), each *strict* (checked literally) or *loose* (forgiving markdown asterisks or a "Sure, here it is" first line). The headline is prompt-level strict, the strictest and the one model cards usually quote.
- **HumanEval** runs the model's code against unit tests, so it measures working code, not plausible-looking code. It is opt-in (`--allow-code-exec`, [§7](#7-caveats-read-before-quoting-these-numbers)).
- **golden** is the only benchmark that cannot have leaked into a model's training data, and the only one that measures what *you* care about.

`kumaru bench list` also shows **`gsm8k_train`**, GSM8K's 7,473 training problems. It is not a score: a teacher model answers it to make distillation data, and `kumaru distill` accepts only runs of training splits like it ([TRAINING.md Part 3](TRAINING.md#part-3--distillation)).

## 3. How the comparison was run

Everything was held fixed except the model, and in the third column, thinking:

| | Setting |
|---|---|
| Engine | llama.cpp `llama-server` build b11541, started and managed by Kumaru (`backend.name: llamacpp_server`) |
| Weights | Qwen2.5-7B-Instruct Q4_K_M; Qwen3.6-35B-A3B UD-IQ4_XS. Both are ~4-bit quantizations |
| Questions | 100 per benchmark (golden: all 27), drawn with seed 1234. MMLU-Pro is stratified so all 14 subjects appear in proportion |
| Pairing | Every model answers the same questions with the same per-question sampling seed |
| Prompting | Zero-shot (no worked examples) and no system prompt: this measures the model, not Kumaru's chat persona. `--use-agent-prompt` adds the persona if you want to measure the whole agent |
| Answer budget | Up to 8,192 tokens per answer, plus 4,096 for thinking when it is on |
| Concurrency | Up to 4 questions at once, batched into one forward pass; fewer if they would not all fit in the KV cache (the 7B, with a 32k window, ran 3) |
| Hardware | RTX 3090 24 GB, Ryzen 7 5700X, 64 GB RAM, Windows 11 |

**Prompts.** Each benchmark asks for its answer in a fixed, parseable form:

| Benchmark | Prompt |
|---|---|
| GSM8K | `Solve the following math problem. Think step by step, then write the final answer on its own last line in the form "Answer: <number>", with no units.` + the problem |
| MMLU-Pro | `The following is a multiple choice question about {subject}. Think step by step, then finish your answer with "The answer is (X)" where X is the letter of the correct option.` + question and options |
| IFEval | The dataset prompt, unchanged |
| HumanEval | ``Complete the following Python function. Return the complete function, including its signature and any imports it needs, in a single ```python code block.`` + the stub |

The extractors accept the common ways models deviate (`**Answer:** $1,080`, `\boxed{C}`, a bare function body) and record which method found the answer; `report` shows the counts.

**Decoding.** Each model uses its publisher's recommended sampling preset, not one shared setting: models are tuned for their presets, and greedy decoding, the usual "neutral" choice, sends Qwen3 models into repetition loops.

| Run | temperature | top_p | top_k | min_p | presence penalty | repetition penalty |
|---|---|---|---|---|---|---|
| Qwen2.5-7B | 0.7 | 0.8 | 20 | 0 | 0 | 1.05 |
| Qwen3.6, Think off | 0.7 | 0.8 | 20 | 0 | 1.5 | 1.0 |
| Qwen3.6, Think on | 1.0 | 0.95 | 20 | 0 | 1.5 | 1.0 |

HumanEval turns both penalties off for every model, because code repeats tokens legitimately (indentation, names, brackets).

**The thinking budget** is enforced by llama-server. When a model has thought for 4,096 tokens, the server appends *"I have used my thinking budget, so I will now give my best final answer."* and makes it answer. Chat in the UI does not cap thinking (up to the profile's 32,768-token `max_tokens`); the benchmark cap exists to bound run time, so it understates what unlimited thinking can do on hard questions.

**Sharing the GPU.** The Qwen3.6 runs shared the GPU with a running game, so they kept 18 layers of experts in system RAM (`KUMARU_BACKEND__LLAMA_SERVER__N_CPU_MOE=18`) with a 64k context. That changes speed, not the model.

## 4. Reading the numbers

**A single score** is shown as `60.0 (50.2-69.1)`: accuracy and its 95% Wilson confidence interval. At 100 questions the interval is about ±10 points in the middle of the range and narrower near 0% or 100%. So two scores 5 points apart, taken alone, tell you almost nothing.

**Comparisons are paired, and that is where the power comes from.** Both runs answered the same questions with the same seeds, so `compare` looks question by question. Questions both runs got right, or both got wrong, say nothing about which is better. Only the disagreements do. A row of `kumaru bench compare <7B run> <Qwen3.6 run>`:

```
benchmark   n     A %    B %    B - A (95% CI)           only A / only B   p (Holm)   verdict    tokens A -> B
mmlu_pro    100   59.0   84.0   +25.0 (+15.0 to +36.0)   5 / 30            8.94e-05   B better   416 -> 1,472
```

The two runs agreed on 65 questions and disagreed on 35: the 7B alone was right on 5, Qwen3.6 alone on 30. If the two models were equally good, each disagreement would be a coin flip. McNemar's exact test asks how surprising the actual split is: 35 fair coin flips come out 5 / 30 or more lopsided about twice in 100,000 tries (p = 2.2 × 10⁻⁵; the Holm correction below multiplies it by 4, giving the 8.94e-05 shown). The interval on the difference comes from a paired bootstrap, which resamples the questions 10,000 times.

**Several benchmarks means several chances of a fluke.** `compare` applies Holm's correction across the benchmarks it compares, so the chance of *any* false "better" stays under 5%. The `p (Holm)` column is already corrected.

| Verdict | Meaning |
|---|---|
| `B better` / `A better` | Significant after Holm's correction (adjusted p < 0.05). Believe it |
| `B better? (not after correction)` | p < 0.05 for this benchmark alone. A lead: re-test with more questions |
| `no clear difference` | Either no difference, or too few questions to see one |

**Rule of thumb for 100 questions:** two runs typically disagree on 20-30 questions. To be significant, those must split at least 5-15 or 9-21, which is a 10- to 12-point difference. For smaller effects, use more questions (`-n 500`, or `--full`).

**The noise floor.** Running the *same* model twice does not give the same score ([§6.2](#62-the-noise-floor-the-same-model-twice)). Two identical 7B runs differed by up to 7 points, and `compare` called every one of those differences noise. Never act on a difference that `compare` does not call significant.

## 5. Are the graders right?

A benchmark score is only as trustworthy as its grader. One that misreads `$1,080`, or a function written without its signature, silently turns right answers into wrong ones and biases against models that format differently. Every grader was checked against the dataset's own answer key before any model was scored:

| Benchmark | Check | Result |
|---|---|---|
| GSM8K | All 1,319 reference answers, each phrased 5 ways (the dataset's own solution, `Answer: N`, `**Answer:** $N,NNN`, `\boxed{N{,}NNN}`, `The answer is N.`), must grade correct; the reference answer + 1 must grade wrong | 6,595 / 6,595 correct; 1,319 / 1,319 wrong answers rejected (after the fix described below) |
| MMLU-Pro | All 12,032 gold letters, each phrased 6 ways, must grade correct; a wrong letter must grade wrong; removing junk options must never move the gold one | 72,192 / 72,192 correct; no wrong letter accepted |
| IFEval | Kumaru's implementation against Google's reference implementation, on the 541 GPT-4 responses published with the paper | Identical inputs 541/541; instruction verdicts agree 833/834; prompt-level strict 77.08 vs 76.89 and loose 79.67 vs 79.48 (Kumaru vs reference); word and sentence counts match on 2,253 texts |
| HumanEval | All 164 reference solutions, in 3 forms (whole function, body only, unindented body), must pass; a wrong solution must fail; an infinite loop must time out; the guard must block `os.remove` | 492 / 492 pass (15 s on 8 cores); all three safety checks hold |

The one IFEval disagreement is deliberate. For "the letter *x* should appear at least *n* times" where *x* is not a letter, the reference implementation silently substitutes a random letter; Kumaru counts the character that was actually asked for.

The GSM8K check caught a real bug. The first grader compared numbers with a relative tolerance of one part in a million, so on the two largest answers in the test set, 1,450,000 and 2,880,000 (`gsm8k/611` and `gsm8k/796`), the wrong answer "reference + 1" graded correct. GSM8K answers are exact, so numbers now have to match exactly. Formatting such as `$1,450,000.00` still never matters, and a 10⁻⁹ allowance absorbs floating-point noise such as 0.1 + 0.2. Re-grading every saved GSM8K and golden answer in the runs reported here with the exact rule changed no verdict, so the bug inflated no score in this document.

Three golden format graders were later tightened, because each accepted a reply its prompt rules out: `banana-caps` took `BANANA.` although the prompt says "nothing else", `three-colors` took the same colour three times, and `json-ada` took a third key. Re-grading the 135 golden answers saved by the five runs reported here with the stricter patterns changed no verdict either.

Reproduce the first, second and fourth rows with:

```powershell
python scripts\check_graders.py        # exit code 0 = every check passed
```

The IFEval cross-check needs Google's reference code (`instruction_following_eval` in the `google-research` repository) in its own virtual environment: grade the GPT-4 responses published there (`data/input_response_data_gpt4_20231107_145030.jsonl`) with both implementations, then compare the verdicts instruction by instruction. The unit tests `tests/test_bench_extract.py`, `test_bench_graders.py` and `test_bench_ifeval.py` pin the edge cases found along the way.

## 6. Lab notebook: what we learned along the way

### 6.1 The truncation confound

The first full run (v1) gave each benchmark its own answer cap: 1,024 tokens for golden and GSM8K, 1,536 for HumanEval, 2,048 for MMLU-Pro and IFEval. The caps barely touched the 7B (one IFEval answer out of 427). They did touch Qwen3.6, which deliberates longer even with thinking off. On MMLU-Pro, 16 of its 100 answers were cut off mid-reasoning and 12 of those were scored wrong: more than half of its 23 misses. On GSM8K, 3 of its 7 misses were cut off. A cut-off answer usually never reaches its final line, so it is scored wrong because the model ran out of room, not because it was wrong. (A few are still graded right, when the answer appeared before the cut.)

That is a *confound*: a cap that is harmless for one model and binding for another biases the comparison against the more thorough model. The fix was one generous default for every benchmark (8,192 tokens, plus the thinking budget), and the run summary now shows a **cut off** column so a binding cap is visible instead of silent.

| Qwen3.6, Think off | v1 (capped) | v2 (8,192) | cut off, v1 → v2 |
|---|---|---|---|
| golden | 96.3 | 96.3 | 1 → 0 |
| GSM8K | 93.0 | 95.0 | 3 → 0 |
| MMLU-Pro | 77.0 | 84.0 | 16 → 1 |

(v1 was stopped during IFEval once the problem was found.)

A cap changes nothing until it binds, so the v2 answer lengths show how often each v1 cap would have bitten. For Qwen3.6, 17 MMLU-Pro answers ran past 2,048 tokens, 5 GSM8K answers past 1,024, 3 IFEval answers past 2,048 and 1 HumanEval answer past 1,536. For the 7B, one answer in 427 did (on IFEval), which is why the problem was invisible from its side.

**Follow the questions, not the score.** The headline change on MMLU-Pro, +7 points, is *not* significant on its own: the two runs disagreed on only 15 questions, split 4 / 11 (p = 0.12). The question-level records settle it:

| MMLU-Pro questions Qwen3.6 got wrong in v1 | Right in v2 |
|---|---|
| cut off by the v1 cap (12) | 10 |
| not cut off (11) | 1 |

Recovering 10 of 12 against 1 of 11 is not chance (Fisher's exact test, p = 0.0006). On the 84 questions v1 did not cut off, the two runs disagreed just 4 times (3 / 1), which is ordinary sampling noise. GSM8K shows the same pattern on a smaller scale: 2 of its 3 cut-off misses became right.

**Lesson:** check the cut-off column before believing any comparison. When an aggregate test is inconclusive but you suspect a cause, look at *which* questions changed: the records in `items.jsonl` can show a mechanism that a score hides.

### 6.2 The noise floor: the same model twice

The 7B reached a v1 cap only once (that one IFEval answer), so its v2 run is an *A/A test*: same weights, questions and seeds. Only the batch size differed (4 questions at a time in v1, 3 in v2).

| Qwen2.5-7B | v1 | v2 | only v1 right / only v2 right | verdict |
|---|---|---|---|---|
| golden | 96.3 | 96.3 | 0 / 0 | no clear difference |
| GSM8K | 96.0 | 92.0 | 5 / 1 | no clear difference |
| MMLU-Pro | 60.0 | 59.0 | 8 / 7 | no clear difference |
| IFEval | 65.0 | 72.0 | 3 / 10 | no clear difference |
| HumanEval | 80.0 | 81.0 | 2 / 3 | no clear difference |

Only 104 of the 427 answers came out identical word for word, and most of the rest had diverged within their first few hundred characters. Fixed seeds do not make GPU inference reproducible. Which requests share a batch changes the order of floating-point additions, which nudges the token probabilities, which occasionally flips a sampled token, and from there the answer diverges. (Every flipped grade came with a different answer, so the graders were not the cause.) Swings of 4 points on GSM8K and 7 on IFEval, from nothing at all, are exactly why a single run's difference means little. `compare` judged every row correctly: no clear difference.

## 7. Caveats (read before quoting these numbers)

- **Subsets.** 100 questions per benchmark detect big jumps, not small ones ([§4](#4-reading-the-numbers)). Use `--full` for numbers you want to publish.
- **Not comparable to model-card scores.** Publishers use their own prompts, few-shot examples, unquantized weights, larger budgets and sometimes several attempts per question; differences of several points are routine. Compare models *within* this harness.
- **Quantization.** Both models are ~4-bit. That costs a little accuracy against the original weights, equally for both, and it is how you run them.
- **Contamination.** GSM8K and HumanEval have been public since 2021 and MMLU-Pro since 2024; newer models may have seen them in training. Your golden set is immune; grow it.
- **The thinking cap.** Think-on scores used a 4,096-token thinking budget to bound run time ([§3](#3-how-the-comparison-was-run)). It cut short 24 of the 100 MMLU-Pro answers, so it likely understates Think on. Chat thinks without that cap.
- **One run per configuration**, with the run-to-run noise measured in [§6.2](#62-the-noise-floor-the-same-model-twice).
- **HumanEval runs model-written code** in a fresh Python process with a 15 s limit, an empty temporary folder, a scrubbed environment and a guard that disables destructive `os`/`shutil`/`subprocess` calls. That stops accidents, not a determined adversary: it is not a sandbox. Evaluate models you do not trust in a container or VM.

## 8. Your own golden questions

`data/bench/golden.jsonl` holds one question per line, graded by a rule rather than by a model:

```json
{"id": "pens", "prompt": "Pens cost $3 each, or $10 for a pack of 4. What is the least you can pay for exactly 10 pens? Finish with \"Answer: <number>\".", "grader": "numeric", "answer": 26, "tags": ["math"]}
```

Graders: `exact`, `contains`, `contains_any`, `not_contains`, `regex`, `numeric` and `choice`. `numeric` is exact unless the question sets a relative `tolerance`; use one only for answers that are approximate by nature, such as `0.001` for a rounded unit conversion (within 0.1%: for 37.8 it accepts 37.763 to 37.837, not 37.7 or 37.9). The first tag groups the question in `report`. Lines starting with `//` are comments. The full format is in [`docs/howto/src/kumaru/eval/benchmarks/golden.md`](howto/src/kumaru/eval/benchmarks/golden.md).

The habit that compounds: **whenever Kumaru gets something wrong that it should get right, add it as a golden question.** Rerun golden (about two minutes) after every change to a model, prompt or setting. Any other `.jsonl` file in the same format also works as a benchmark: `kumaru bench run my_questions.jsonl -c qwen36`.

## 9. Run mechanics

**What a run saves.** `runs\bench\<timestamp>-<label>\` holds:

- `run.json`: the model, every setting and the software versions;
- `<benchmark>\items.jsonl`: every question as sent, the full answer (and thinking), the extracted answer, the grade, token counts, latency and seed;
- `<benchmark>\summary.json`: the scores.

Every number in this document can be traced to individual answers.

**Resuming.** A run that dies (crash, reboot, Ctrl+C) resumes with `-o <its folder>` and the same flags. Answered questions are skipped, and a line half-written by a hard kill is repaired. Resuming with different settings (model, decoding parameters, seed, prompt mode or documentation indexes) is refused, so one folder never mixes two configurations.

**Useful flags for `kumaru bench run`:**

| Flag | Effect |
|---|---|
| `-c <profile>` | Which model and settings (`qwen36`, `qwen25-7b-gguf`, your own YAML) |
| `-n 100` / `--full` | Questions per benchmark, or all of them |
| `--seed 1234` | Picks the questions and each question's sampling seed. Keep it fixed across runs you will compare |
| `--think` / `--no-think` | Force thinking on or off (default: the profile's) |
| `--thinking-budget 4096` | Cap the thinking tokens per question |
| `--max-tokens N` | Override the output cap for every question |
| `--context 65536` | Shrink the context window (and the KV cache) to free VRAM |
| `-j N` | Questions in flight (default: as many as the backend can batch) |
| `--use-agent-prompt` | Include Kumaru's system prompt, as chat does |
| `--allow-code-exec` | Required for `humaneval` |
| `-o <folder>` | Write to (or resume) this folder |

**How long it takes** on the hardware above, for golden plus 100 questions of each public benchmark:

| Run | Wall time | Slowest benchmark |
|---|---|---|
| Qwen2.5-7B | 11 min | MMLU-Pro, 4 min |
| Qwen3.6, Think off | 100 min | MMLU-Pro, 56 min |
| Qwen3.6, Think on | 152 min for golden, GSM8K and MMLU-Pro (stopped there) | MMLU-Pro, 91 min |

The 9-fold gap to the 7B has two factors. Qwen3.6 generated about a quarter as many tokens per second in these runs (about 44 against 185, all requests together), partly because 18 layers of its experts ran from system RAM to leave room for a game ([§3](#3-how-the-comparison-was-run)). And it writes about twice as many tokens per answer (3.5 times as many on MMLU-Pro).

## 10. Where evaluation goes next

Evaluation has to scale with the model, or you lose the ability to tell whether changes help. In rough order of value:

1. **More questions.** `--full` runs all ~14,000 questions. It turns ±10-point intervals into about ±1 point on MMLU-Pro and GSM8K, ±4 on IFEval and ±6 on HumanEval (only 164 problems exist).
2. **Harder benchmarks**, as the current ones saturate: GPQA-Diamond (graduate-level science), MATH-500 and AIME (competition mathematics), LiveCodeBench (code problems published after a model's training cutoff, so contamination-resistant), BFCL (function calling, once Kumaru has tools) and RULER (long-context retrieval and reasoning).
3. **Open-ended quality.** Chat answers have no single right answer. The standard tool is a strong *LLM judge* comparing two answers pairwise, with positions swapped to cancel its bias toward the first one (Arena-Hard works this way).
4. **Test-time compute.** pass@k and majority voting over k samples measure how much extra compute buys. Thinking budgets are the first instance of this; the scores in the TL;DR are the first data point.
5. **A regression gate.** Run golden in CI on every change and fail the build if `compare` says the change made it worse.

**Adding a benchmark** takes about 50 lines: subclass `Benchmark` in `src/kumaru/eval/benchmarks/` (implement `load()` and `grade()`; optionally `aggregate()` for extra metrics and `generation_overrides()`), then register it in `src/kumaru/core/registry.py` with `BENCHMARKS.register_lazy(...)`. A separate package can also register through the `kumaru.benchmarks` entry point without touching Kumaru. Start from [`docs/howto/src/kumaru/eval/benchmarks/base.md`](howto/src/kumaru/eval/benchmarks/base.md). The evaluation axis of scaling is in [SCALABILITY.md §7](SCALABILITY.md#7-evaluation-scale).

## Related files

- `src/kumaru/eval/benchmarks/`: the benchmarks, the runner and the statistics, each with a how-to page under `docs/howto/src/kumaru/eval/benchmarks/`
- `data/bench/golden.jsonl`: your golden set
- `scripts/check_graders.py`: re-validates the graders
- `configs/qwen36.yaml`, `configs/qwen25-7b-gguf.yaml`: the two profiles compared here
- `docs/SCALABILITY.md` §7: evaluation as a scaling axis

# `scripts/check_graders.py`

> Proves the benchmark graders right before you trust a single score: every reference answer in the datasets, phrased the ways models phrase answers, must grade correct, and wrong answers must grade wrong. For MultiPL-E, which has no reference answers, Kumaru must reproduce its authors' own verdicts on graded completions.

**Read this when:** you changed an extractor or grader in `src/kumaru/eval/benchmarks/`; a score looks suspicious; you installed or updated a compiler that MultiPL-E uses; or you want to reproduce the "Are the graders right?" table in `docs/BENCHMARKS.md`.

---

## What it does

| Benchmark | Checks | Volume |
|---|---|---|
| `gsm8k` | Each of the 1,319 reference answers, in 5 phrasings (`Answer: 18`, `**Answer:** $1,080`, `\boxed{1{,}080}`, ...), grades correct; the reference + 1 grades wrong | 7,914 gradings |
| `mmlu_pro` | Each of the 12,032 gold letters, in 6 phrasings (`The answer is (C)`, `\boxed{\text{(C)}}`, "not (A). The answer is (C)", ...), grades correct; a wrong letter grades wrong; removing junk options never moves the gold option, which still appears in the prompt | 84,224 gradings |
| `humaneval` | All 164 reference solutions pass their hidden tests in 3 forms (whole function, body only, unindented body); a wrong solution fails; an infinite loop times out; the guard blocks `os.remove` | 495 Python processes |
| `multipl_e`, or one language: `multipl_e_<lang>` | MultiPL-E has no reference solutions in most languages, so this replays answers its authors already ran and graded with their own harness: StarCoder2-15B's (Python: replit-code-v1-3b's), from `nuprl/MultiPL-E-completions` at a pinned commit. Kumaru grades each through `kumaru bench`'s own path (answer reading, assembly, compile, run) and must reach the same pass/fail verdict, apart from five excused kinds of difference (below). `multipl_e` checks every language whose toolchain is installed and reports the rest as not checked; a language named on its own fails the check if its toolchain is missing | `--sample` (200) model-written programs per language |

With no names it runs `gsm8k`, `mmlu_pro` and `humaneval`, which need no compilers. It prints one line per benchmark (`passed/total` for each phrasing; for MultiPL-E, the agreement) and exits 0 only if everything held.

## Why it exists

A grader bug is invisible in a score. If the GSM8K extractor misread `$1,080`, a model that writes dollar signs would simply look 5 points worse than it is, and every comparison involving it would be wrong in a way no statistical test can detect. The only defence is to grade answers you *know* are right, in every format models actually produce, and check that they all come out right. The datasets' own answer keys provide those answers for free.

This mirrors what you would do for any measuring instrument: calibrate it against known references before measuring anything unknown.

## How to use it

```powershell
pip install -e ".[bench]"                                  # once
python scripts\check_graders.py                            # gsm8k, mmlu_pro, humaneval: about 1-2 minutes
python scripts\check_graders.py gsm8k                      # one benchmark
python scripts\check_graders.py multipl_e --workers 2      # every MultiPL-E language with a toolchain here
python scripts\check_graders.py multipl_e_rs --sample 50   # one language, a smaller sample
python scripts\check_graders.py humaneval multipl_e --sandbox docker
python scripts\check_graders.py --help
```

Run it with the project's virtual environment active (it imports `kumaru`). The datasets download on first use and are cached, the same as for `kumaru bench run`. MultiPL-E's check downloads one parquet file of 1–2 MB per language (11.5 MB for all eight).

| Option | Default | What it does |
|---|---|---|
| `--workers N` | `KUMARU_CODE_WORKERS`, else half the cores, at most 4 | How many programs run at once, for HumanEval and MultiPL-E. It also sets `codeexec`'s limit (`set_max_workers`). |
| `--sample N` | `200` | MultiPL-E completions replayed per language. |
| `--sandbox none\|docker` | `none` | Where the programs run. `docker` uses the sandbox image, which must be built first ([`Dockerfile.md`](../docker/sandbox/Dockerfile.md)); this is how to check that image before trusting scores from it. |

`--workers` or `--sample` below 1 prints `--workers and --sample must be at least 1` and exits 2. At `--workers 2`, a MultiPL-E language takes from under a minute (Python) to 9–13 minutes (C#, where every program is a `dotnet build`).

## Reading the output

```
gsm8k: dataset solution 1319/1319, Answer: N 1319/1319, ..., wrong answer rejected 1319/1319
mmlu_pro: The answer is (X) 12032/12032, ..., wrong letter rejected 12032/12032
  options per question: {3: 21, 4: 606, 5: 52, 6: 93, 7: 158, 8: 320, 9: 801, 10: 9981}
humaneval: whole function 164/164, body only 164/164, unindented body 164/164
  492 programs in 68.5s
  wrong solution fails, infinite loop times out, os.remove blocked: yes
all checks passed
```

If anything failed, `all checks passed` is replaced by `N problem(s):` and up to 20 indented lines, and the script exits 1 (2 for an unknown benchmark name). Each line is one of:

- `<id> [<phrasing>]: gold <gold>, extracted <value>`: a right answer, phrased that way, graded wrong.
- `<id>: the wrong answer <gold + 1> graded correct` (MMLU-Pro: `the wrong letter <X>`): the item id plus the wrong answer the grader accepted.
- `<id> [<form>]: <status> <end of stderr>`: a HumanEval reference solution that didn't pass.
- An MMLU-Pro consistency failure such as `<id>: option cleaning moved the gold option`, or a safety check, which has no id: `os.remove was not blocked: ...`.
- `multipl_e_<lang> <problem> #<n>: MultiPL-E <verdict>, Kumaru <outcome> (<mode>)`, then the last lines of each side's error output: a MultiPL-E verdict Kumaru didn't reproduce. `<n>` is the completion's index in the dataset row and `<mode>` how the answer was read: `whole` (it defines the function itself) or `continuation` (it carries on from the prompt).
- `multipl_e_<lang>: <why>`: that language was named on its own, and its toolchain is missing.

That is usually enough to write the failing case as a unit test in `tests/test_bench_extract.py` or `tests/test_bench_graders.py` (for MultiPL-E, `tests/test_bench_multipl_e.py` or `tests/test_bench_codeparse.py`) before fixing the grader.

### MultiPL-E's lines

```
multipl_e_py (python 3.12.6): 200/200 of MultiPL-E's verdicts on replit_code completions reproduced, 104 problems, 48s
  MultiPL-E Exception         48: Kumaru {'failed': 48}
  MultiPL-E OK               100: Kumaru {'passed': 100}
  MultiPL-E SyntaxError       48: Kumaru {'failed': 48}
  MultiPL-E Timeout            4: Kumaru {'timeout': 4}
multipl_e_ts (node v24.17.0; types stripped): 195/200 of MultiPL-E's verdicts on starcoder2_15b completions reproduced, 120 problems, 115s
  ...
  MultiPL-E SyntaxError       30: Kumaru {'failed': 25, 'passed': 5}
  MultiPL-E Timeout           10: Kumaru {'timeout': 10}
  4 expected difference(s): tsc failed on top-level statements Kumaru drops (not declarations), such as a file path
  1 expected difference(s): tsc rejected it; without tsc, types are not checked
multipl_e_go: not checked, Go needs the go command, which is not on PATH. Install Go: https://go.dev/dl/
```

The first line gives the toolchain, the agreement, how many problems the sample spread over, and the time. Each row below it is one of MultiPL-E's verdicts (`OK`, its error kinds, `SyntaxError` for code that didn't compile, `Timeout`) with what Kumaru made of the same answers.

The sample is fixed (seed 0): half passes, and the rest shared evenly between the ways MultiPL-E saw answers fail, at most 10 of them timeouts, each spread one per problem per round so that no problem dominates.

Each answer is graded on the tests as MultiPL-E published them, because those gave its verdicts: like with like. `kumaru bench run` grades C# on fixed tests instead (`multipl_e.TEST_FIXES`, [nuprl/MultiPL-E#71](https://github.com/nuprl/MultiPL-E/issues/71)), where a right List or Dictionary result passes; on the published tests, MultiPL-E and Kumaru fail it alike. So the check measures the grader, not the fix, which `tests/test_bench_multipl_e.py` pins with a real .NET build.

Five kinds of disagreement are excused and listed as `expected difference(s)`, because each comes from a known difference between Kumaru's setup and MultiPL-E's, not from a grading error:

- **TypeScript's stray lines.** MultiPL-E's `tsc` rejected the program for a line that Kumaru's program leaves out, because it is a top-level statement, not a declaration ([`multipl_e.md`](../src/kumaru/eval/benchmarks/multipl_e.md#how-each-languages-program-is-assembled)). StarCoder2 often ends an answer with a file path, such as `/src/codewars/7kyu/sum-of-odd-numbers.ts`, which `tsc` reads as a regular expression or a division, or with an example call cut off in the middle of a string. MultiPL-E kept `tsc`'s output, so the check reads which line it flagged first, and excuses the answer only if that line isn't in Kumaru's program. Later errors can follow from the first (an unterminated string runs into the next line, which Kumaru keeps), so only the first counts.
- **TypeScript without `tsc`.** MultiPL-E's `tsc` rejected the answer for its types, but with the types stripped it runs and passes. Install `tsc` to check types as well ([`multipl_e.md`](../src/kumaru/eval/benchmarks/multipl_e.md)).
- **A MultiPL-E `Timeout` that passes here.** Kumaru allows 15 s to run and 120 s to compile; MultiPL-E allowed as little as 5 s (`MULTIPL_E_LIMITS`), on a machine that may have been busy. A slow but correct answer passes here.
- **Go's `undefined: min`, `max` or `clear`.** Those are built into Go since 1.21, which is newer than MultiPL-E's Go.
- **C++'s 32-bit `long` on Windows.** The tests of `HumanEval_114` and `HumanEval_139` hold numbers beyond 32 bits in a `long`, which is 64 bits on Linux, where MultiPL-E grades, but 32 bits on Windows ([`multipl_e.md`](../src/kumaru/eval/benchmarks/multipl_e.md#problems-no-answer-can-pass)). So an answer MultiPL-E passed fails here, at an assertion that holds such a number. Only that failure, on those two problems (`CPP_NEEDS_64_BIT_LONG`), is excused, and only when programs run natively on Windows. Under `--sandbox docker`, which is Linux, nothing is excused: all 100 of MultiPL-E's answers to the two problems get its verdict. The default sample of 200 holds none of them, and a Windows run of a larger `--sample` that does hasn't been done yet.

Anything else is a problem. Some differences are not visible in the agreement at all, because both sides fail the answer for different reasons: [`BENCHMARKS.md`](../../BENCHMARKS.md) explains each language's.

## Things worth knowing

- **It has caught a real grader bug.** GSM8K answers used to match within a relative tolerance of 1e-6. Before the fix, the GSM8K line ended `wrong answer rejected 1317/1319` and the output finished with:
  ```
  2 problem(s):
     gsm8k/611: the wrong answer 1450001 graded correct
     gsm8k/796: the wrong answer 2880001 graded correct
  ```
  Each line names the item id plus the wrong answer the grader accepted. The gold answers are 1,450,000 and 2,880,000, and an error of 1 is less than a millionth of either, so a model answering 1,450,001 would have been scored correct. `numbers_equal` is now exact by default ([`extract.md`](../src/kumaru/eval/benchmarks/extract.md)), so gold + 1 can no longer pass and the line reads 1319/1319. `test_numbers_equal_is_exact_near_large_values_unless_asked` in `tests/test_bench_extract.py` keeps it that way.
- **HumanEval runs code.** It executes the dataset's own reference solutions, not model output, through the same guarded subprocess runner used for scoring (`humaneval.run_program`). `--workers` of them run at once.
- **MultiPL-E's check runs model-written code.** The completions it replays are StarCoder2's and replit-code's answers, failures included, and they run as `kumaru bench run multipl_e` runs answers: natively, through `codeexec` (a temp folder, time limits, a memory cap, the whole process tree killed), or in the container with `--sandbox docker`. The same cautions apply ([`codeexec.md`](../src/kumaru/eval/benchmarks/codeexec.md)).
- **MultiPL-E's check has caught real bugs.** An early Go run disagreed on two answers to HumanEval_8, which returns `[]interface{}`: the statement splitter took that `{}` for a block and cut the function in two. In the Docker sandbox, `tsc` failed two TypeScript answers that MultiPL-E had run. Their bodies name a local variable after the function (`let intersection = …`), which Kumaru took for the function's definition, so it left the prompt's function out. The agreement didn't show it, because both answers fail either way; the rows under it did (`MultiPL-E AssertionError 30: Kumaru {'compile_error': 1, 'failed': 29}`), so look into any `compile_error` there for an answer that MultiPL-E ran. Tests of whole answers, the form chat models give, found two more (a JavaScript `import { … } from` cut at its `}`, and a Java or C# `void main` missed, which left two `main`s); the replays can't, because MultiPL-E's completions only continue the prompt. All four are fixed and pinned in `tests/test_bench_codeparse.py` and `tests/test_bench_multipl_e.py`.
- **The `os.remove` check looks for `NoneType` in the error.** A real `os.remove` of a missing file would also fail, so only the guard's signature (calling a function it replaced with `None`) proves the guard fired.
- **IFEval is not covered here.** Its instructions have no "reference answer" to rephrase. It was validated against Google's reference implementation instead; `docs/BENCHMARKS.md` §5 describes how.
- **Golden is not covered here.** Its graders are exercised by `tests/test_bench_graders.py`.

## Extending it

Add a `check_<name>()` function that returns a list of problem strings, and register it in `CHECKS`. The pattern for a new benchmark: take each reference answer from the dataset, format it the ways a model plausibly would, assert it grades correct, and assert that a near-miss grades wrong. For numbers, gold + 1 is the near-miss that caught the tolerance bug above.

A new MultiPL-E language joins the replay by itself (`CHECKS` has an entry per `multipl_e.BY_LANGUAGE`). Give it a `COMPLETION_RUNS` entry (a run in `nuprl/MultiPL-E-completions` that covers the language) and its MultiPL-E time limit in `MULTIPL_E_LIMITS`. Excuse a new kind of disagreement in `expected_difference` only when Kumaru is demonstrably right, and document it in `docs/BENCHMARKS.md`.

## Related files

- `src/kumaru/eval/benchmarks/extract.py`: the answer extractors under test
- `src/kumaru/eval/benchmarks/gsm8k.py`, `mmlu_pro.py`, `humaneval.py`, `multipl_e.py`: the graders under test
- `src/kumaru/eval/benchmarks/codeexec.py`: runs the programs and enforces `--workers` and `--sandbox`
- `tests/test_bench_extract.py`, `tests/test_bench_graders.py`, `tests/test_bench_multipl_e.py`: the unit tests for edge cases; the last also pins this script's MultiPL-E sampling, and that it grades on the tests as published
- `docs/BENCHMARKS.md` §5: the results

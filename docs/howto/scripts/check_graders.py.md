# `scripts/check_graders.py`

> Proves the benchmark graders right before you trust a single score: every reference answer in the datasets, phrased the ways models phrase answers, must grade correct, and wrong answers must grade wrong.

**Read this when:** you changed an extractor or grader in `src/kumaru/eval/benchmarks/`; a score looks suspicious; or you want to reproduce the "Are the graders right?" table in `docs/BENCHMARKS.md`.

---

## What it does

| Benchmark | Checks | Volume |
|---|---|---|
| `gsm8k` | Each of the 1,319 reference answers, in 5 phrasings (`Answer: 18`, `**Answer:** $1,080`, `\boxed{1{,}080}`, ...), grades correct; the reference + 1 grades wrong | 7,914 gradings |
| `mmlu_pro` | Each of the 12,032 gold letters, in 6 phrasings (`The answer is (C)`, `\boxed{\text{(C)}}`, "not (A). The answer is (C)", ...), grades correct; a wrong letter grades wrong; removing junk options never moves the gold option, which still appears in the prompt | 84,224 gradings |
| `humaneval` | All 164 reference solutions pass their hidden tests in 3 forms (whole function, body only, unindented body); a wrong solution fails; an infinite loop times out; the guard blocks `os.remove` | 495 Python processes |

It prints one line per benchmark (`passed/total` for each phrasing) and exits 0 only if everything held.

## Why it exists

A grader bug is invisible in a score. If the GSM8K extractor misread `$1,080`, a model that writes dollar signs would simply look 5 points worse than it is, and every comparison involving it would be wrong in a way no statistical test can detect. The only defence is to grade answers you *know* are right, in every format models actually produce, and check that they all come out right. The datasets' own answer keys provide those answers for free.

This mirrors what you would do for any measuring instrument: calibrate it against known references before measuring anything unknown.

## How to use it

```powershell
pip install -e ".[bench]"                    # once
python scripts\check_graders.py              # all three, about 1-2 minutes
python scripts\check_graders.py gsm8k        # one benchmark
python scripts\check_graders.py --help
```

Run it with the project's virtual environment active (it imports `kumaru`). The datasets download on first use and are cached, the same as for `kumaru bench run`.

## Reading the output

```
gsm8k: dataset solution 1319/1319, Answer: N 1319/1319, ..., wrong answer rejected 1319/1319
mmlu_pro: The answer is (X) 12032/12032, ..., wrong letter rejected 12032/12032
  options per question: {3: 21, 4: 606, 5: 52, 6: 93, 7: 158, 8: 320, 9: 801, 10: 9981}
humaneval: whole function 164/164, body only 164/164, unindented body 164/164
  492 programs in 15.4s
  wrong solution fails, infinite loop times out, os.remove blocked: yes
all checks passed
```

If anything failed, `all checks passed` is replaced by `N problem(s):` and up to 20 indented lines, and the script exits 1 (2 for an unknown benchmark name). Each line is one of:

- `<id> [<phrasing>]: gold <gold>, extracted <value>`: a right answer, phrased that way, graded wrong.
- `<id>: the wrong answer <gold + 1> graded correct` (MMLU-Pro: `the wrong letter <X>`): the item id plus the wrong answer the grader accepted.
- `<id> [<form>]: <status> <end of stderr>`: a HumanEval reference solution that didn't pass.
- An MMLU-Pro consistency failure such as `<id>: option cleaning moved the gold option`, or a safety check, which has no id: `os.remove was not blocked: ...`.

That is usually enough to write the failing case as a unit test in `tests/test_bench_extract.py` or `tests/test_bench_graders.py` before fixing the extractor.

## Things worth knowing

- **It has caught a real grader bug.** GSM8K answers used to match within a relative tolerance of 1e-6. Before the fix, the GSM8K line ended `wrong answer rejected 1317/1319` and the output finished with:
  ```
  2 problem(s):
     gsm8k/611: the wrong answer 1450001 graded correct
     gsm8k/796: the wrong answer 2880001 graded correct
  ```
  Each line names the item id plus the wrong answer the grader accepted. The gold answers are 1,450,000 and 2,880,000, and an error of 1 is less than a millionth of either, so a model answering 1,450,001 would have been scored correct. `numbers_equal` is now exact by default ([`extract.md`](../src/kumaru/eval/benchmarks/extract.md)), so gold + 1 can no longer pass and the line reads 1319/1319. `test_numbers_equal_is_exact_near_large_values_unless_asked` in `tests/test_bench_extract.py` keeps it that way.
- **HumanEval runs code.** It executes the dataset's own reference solutions, not model output, through the same guarded subprocess runner used for scoring (`humaneval.run_program`). Eight run in parallel.
- **The `os.remove` check looks for `NoneType` in the error.** A real `os.remove` of a missing file would also fail, so only the guard's signature (calling a function it replaced with `None`) proves the guard fired.
- **IFEval is not covered here.** Its instructions have no "reference answer" to rephrase. It was validated against Google's reference implementation instead; `docs/BENCHMARKS.md` §5 describes how.
- **Golden is not covered here.** Its graders are exercised by `tests/test_bench_graders.py`.

## Extending it

Add a `check_<name>()` function that returns a list of problem strings, and register it in `CHECKS`. The pattern for a new benchmark: take each reference answer from the dataset, format it the ways a model plausibly would, assert it grades correct, and assert that a near-miss grades wrong. For numbers, gold + 1 is the near-miss that caught the tolerance bug above.

## Related files

- `src/kumaru/eval/benchmarks/extract.py`: the answer extractors under test
- `src/kumaru/eval/benchmarks/gsm8k.py`, `mmlu_pro.py`, `humaneval.py`: the graders under test
- `tests/test_bench_extract.py`, `tests/test_bench_graders.py`: the unit tests for edge cases
- `docs/BENCHMARKS.md` §5: the results

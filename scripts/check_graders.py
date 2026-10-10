"""Prove Kumaru's benchmark graders right, using each dataset's own answer key.

A benchmark score is only as trustworthy as its grader: one that misreads
"$1,080" or "**The answer is (C).**" silently turns right answers into wrong
ones. This script grades every reference answer, phrased the ways models
phrase answers, and requires each to grade correct and a wrong answer to grade
wrong. docs/BENCHMARKS.md ("Are the graders right?") reports the results.

    pip install -e ".[bench]"
    python scripts\\check_graders.py              # gsm8k, mmlu_pro, humaneval (~1-2 min)
    python scripts\\check_graders.py gsm8k        # one benchmark

Exit code 0 means every check passed. Datasets download on first use and are
cached, exactly as for `kumaru bench run`. The HumanEval check runs the
dataset's own reference solutions (not model output) in fresh Python processes.
IFEval is validated against Google's reference implementation instead; see
docs/BENCHMARKS.md.
"""

from __future__ import annotations

import os
import sys
import textwrap
import time
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

from kumaru.eval.benchmarks.base import load_hf_rows
from kumaru.eval.benchmarks.extract import parse_number
from kumaru.eval.benchmarks.gsm8k import GSM8K
from kumaru.eval.benchmarks.humaneval import HumanEval, assemble_program, run_program
from kumaru.eval.benchmarks.mmlu_pro import LETTERS, MMLUPro, clean_options


def show(name: str, total: int, passed: Counter[str]) -> None:
    print(f"{name}: " + ", ".join(f"{kind} {ok}/{total}" for kind, ok in passed.items()))


def check_gsm8k() -> list[str]:
    """Each reference answer, in five phrasings, grades correct; answer + 1 does not."""
    bench = GSM8K()
    items = bench.load()
    rows = load_hf_rows("openai/gsm8k", "main", "test")
    problems: list[str] = []
    passed: Counter[str] = Counter()
    for item, row in zip(items, rows, strict=True):
        reasoning, _, gold = row["answer"].rpartition("####")
        gold = gold.strip()
        if parse_number(item.answer) is None:
            problems.append(f"{item.id}: gold answer {item.answer!r} is not a number")
            continue
        number = float(gold.replace(",", ""))
        pretty = f"{number:,.0f}" if number.is_integer() else gold
        phrasings = {
            "dataset solution": row["answer"],
            "Answer: N": f"{reasoning}\nAnswer: {gold}",
            "**Answer:** $N,NNN": f"{reasoning}\n**Answer:** ${pretty}",
            "boxed{N{,}NNN}": f"{reasoning}\nSo the total is $\\boxed{{{pretty.replace(',', '{,}')}}}$.",
            "The answer is N.": f"{reasoning}\nThe answer is {gold}.",
        }
        for kind, text in phrasings.items():
            grade = bench.grade(item, text)
            passed[kind] += grade.correct
            if not grade.correct:
                problems.append(f"{item.id} [{kind}]: gold {gold}, extracted {grade.extracted!r}")
        wrong = format(number + 1, ".15g")
        if bench.grade(item, f"{reasoning}\nAnswer: {wrong}").correct:
            problems.append(f"{item.id}: the wrong answer {wrong} graded correct")
        else:
            passed["wrong answer rejected"] += 1
    show("gsm8k", len(items), passed)
    return problems


def check_mmlu_pro() -> list[str]:
    """Each gold letter, in six phrasings, grades correct; a wrong letter does not;
    and cleaning junk options never moves or drops the gold option."""
    bench = MMLUPro()
    items = bench.load()
    rows = load_hf_rows("TIGER-Lab/MMLU-Pro", None, "test")
    problems: list[str] = []
    passed: Counter[str] = Counter()
    option_counts: Counter[int] = Counter()
    for item, row in zip(items, rows, strict=True):
        gold_index = int(row["answer_index"])
        original = list(row["options"])
        letters = item.meta["letters"]
        option_counts[len(letters)] += 1
        if row["answer"] != LETTERS[gold_index] or item.answer not in letters:
            problems.append(f"{item.id}: inconsistent gold letter")
        if clean_options(original, gold_index)[gold_index] != original[gold_index]:
            problems.append(f"{item.id}: option cleaning moved the gold option")
        if f"\n{item.answer}. {original[gold_index].strip()}" not in item.prompt:
            problems.append(f"{item.id}: the gold option is missing from the prompt")
        gold = item.answer
        wrong = next(letter for letter in letters if letter != gold)
        phrasings = {
            "The answer is (X)": f"Let me think. A is tempting but wrong.\nThe answer is ({gold})",
            "**The answer is X.**": f"Reasoning...\n**The answer is {gold}.**",
            "Answer: X": f"Reasoning...\nAnswer: {gold}",
            "boxed{X}": f"Reasoning...\n$\\boxed{{{gold}}}$",
            "boxed{text{(X)}}": f"Reasoning...\n$\\boxed{{\\text{{({gold})}}}}$",
            "after ruling one out": f"The answer is not ({wrong}). The answer is ({gold}).",
        }
        for kind, text in phrasings.items():
            grade = bench.grade(item, text)
            passed[kind] += grade.correct
            if not grade.correct:
                problems.append(f"{item.id} [{kind}]: gold {gold}, extracted {grade.extracted!r}")
        if bench.grade(item, f"The answer is ({wrong})").correct:
            problems.append(f"{item.id}: the wrong letter {wrong} graded correct")
        else:
            passed["wrong letter rejected"] += 1
    show("mmlu_pro", len(items), passed)
    print(f"  options per question: {dict(sorted(option_counts.items()))}")
    return problems


def check_humaneval() -> list[str]:
    """All reference solutions pass in three formats; a wrong one fails; a hang
    times out; and the guard stops `os.remove`."""
    bench = HumanEval()
    items = bench.load()
    rows = load_hf_rows("openai/openai_humaneval", None, "test")
    jobs = []
    for item, row in zip(items, rows, strict=True):
        solution = row["canonical_solution"]
        jobs += [
            (item, "whole function", f"```python\n{row['prompt']}{solution}\n```"),
            (item, "body only", f"```python\n{solution}\n```"),
            (item, "unindented body", f"```python\n{textwrap.dedent(solution)}\n```"),
        ]
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 1)) as pool:
        grades = list(pool.map(lambda job: bench.grade(job[0], job[2]), jobs))
    elapsed = time.perf_counter() - start

    problems: list[str] = []
    passed: Counter[str] = Counter()
    for (item, kind, _), grade in zip(jobs, grades, strict=True):
        passed[kind] += grade.correct
        if not grade.correct:
            stderr = str(grade.detail.get("stderr", ""))[-200:]
            problems.append(f"{item.id} [{kind}]: {grade.extracted} {stderr}")
    show("humaneval", len(items), passed)
    print(f"  {len(jobs)} programs in {elapsed:.1f}s")

    item, row = items[0], rows[0]
    safety: list[str] = []
    if bench.grade(item, f"```python\n{row['prompt']}    return None\n```").correct:
        safety.append("a wrong solution passed its tests")
    hang = assemble_program(
        row["prompt"], "    while True:\n        pass\n", row["test"], row["entry_point"]
    )
    status, _ = run_program(hang, 3.0)
    if status != "timeout":
        safety.append(f"an infinite loop ended with {status!r}, not a timeout")
    delete = "    import os\n    os.remove('x')\n    return True\n"
    status, stderr = run_program(
        assemble_program(row["prompt"], delete, row["test"], row["entry_point"]), 15.0
    )
    # Only "NoneType" proves the guard fired: a real os.remove of a missing file also fails.
    if status != "failed" or "NoneType" not in stderr:
        safety.append(f"os.remove was not blocked: {status} {stderr[-200:]}")
    print(
        "  wrong solution fails, infinite loop times out, os.remove blocked: "
        + ("no" if safety else "yes")
    )
    return problems + safety


CHECKS: dict[str, Callable[[], list[str]]] = {
    "gsm8k": check_gsm8k,
    "mmlu_pro": check_mmlu_pro,
    "humaneval": check_humaneval,
}


def main(argv: list[str]) -> int:
    if any(arg in ("-h", "--help") for arg in argv):
        print(__doc__)
        return 0
    names = argv or list(CHECKS)
    unknown = [name for name in names if name not in CHECKS]
    if unknown:
        print(f"unknown benchmark(s): {', '.join(unknown)}; choose from {', '.join(CHECKS)}")
        return 2
    problems = [problem for name in names for problem in CHECKS[name]()]
    if not problems:
        print("all checks passed")
        return 0
    print(f"{len(problems)} problem(s):")
    for problem in problems[:20]:
        print("  ", problem)
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

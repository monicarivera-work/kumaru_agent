"""Prove Kumaru's benchmark graders right, using each dataset's own answer key.

A benchmark score is only as trustworthy as its grader: one that misreads
"$1,080" or "**The answer is (C).**" silently turns right answers into wrong
ones. This script grades every reference answer, phrased the ways models
phrase answers, and requires each to grade correct and a wrong answer to grade
wrong. docs/BENCHMARKS.md ("Are the graders right?") reports the results.

    pip install -e ".[bench]"
    python scripts\\check_graders.py              # gsm8k, mmlu_pro, humaneval (~1-2 min)
    python scripts\\check_graders.py gsm8k        # one benchmark
    python scripts\\check_graders.py multipl_e_rs # one MultiPL-E language (~2-10 min)
    python scripts\\check_graders.py multipl_e --workers 2 --sample 200

Exit code 0 means every check passed. Datasets download on first use and are
cached, exactly as for `kumaru bench run`. The HumanEval check runs the
dataset's own reference solutions (not model output) in fresh Python processes.
IFEval is validated against Google's reference implementation instead; see
docs/BENCHMARKS.md.

MultiPL-E has no reference solutions in most languages, so its check replays
completions that MultiPL-E's authors already ran and graded with their own
harness (nuprl/MultiPL-E-completions, pinned): StarCoder2-15B's, and for
Python replit-code-v1-3b's, the newest run with Python in it. Kumaru grades
each through `kumaru bench`'s own path (extraction, assembly, compile, run)
and must reach the same verdict. Like with like: the completions are graded
on the tests as MultiPL-E published them, which gave its verdicts, not with
the fixes `kumaru bench` applies (multipl_e.TEST_FIXES). A sample of
`--sample` completions per language (default 200) is spread across problems:
half passed, half failed in each way MultiPL-E saw (wrong answer, exception,
compile error; at most 10 timeouts, which cost 15 s each). `multipl_e`
checks every language whose toolchain is installed and reports the others as
skipped. These are model-written programs and run on this machine, as
`kumaru bench run` runs them without `--sandbox docker`. `--workers`
(default: KUMARU_CODE_WORKERS or half the cores, at most 4) caps how many
programs run at once.

`--sandbox docker` runs every check's programs (HumanEval's too) in the
sandbox image instead, which is how to check that image before trusting it:

    python scripts\\check_graders.py humaneval multipl_e --sandbox docker
"""

from __future__ import annotations

import argparse
import random
import re
import sys
import textwrap
import time
from collections import Counter, defaultdict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from kumaru.eval.benchmarks import codeexec, multipl_e
from kumaru.eval.benchmarks.base import BenchItem, Grade, load_hf_rows
from kumaru.eval.benchmarks.codeexec import ToolchainMissing
from kumaru.eval.benchmarks.extract import parse_number
from kumaru.eval.benchmarks.gsm8k import GSM8K
from kumaru.eval.benchmarks.humaneval import HumanEval, assemble_program, run_program
from kumaru.eval.benchmarks.mmlu_pro import LETTERS, MMLUPro, clean_options


@dataclass
class Settings:
    workers: int = field(default_factory=codeexec.max_workers)
    sample: int = 200
    sandbox: str = "none"


SETTINGS = Settings()


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
    bench.configure(sandbox=SETTINGS.sandbox)
    bench.preflight()
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
    with ThreadPoolExecutor(max_workers=SETTINGS.workers) as pool:
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
    status, _ = run_program(hang, 3.0, sandbox=SETTINGS.sandbox)
    if status != "timeout":
        safety.append(f"an infinite loop ended with {status!r}, not a timeout")
    delete = "    import os\n    os.remove('x')\n    return True\n"
    status, stderr = run_program(
        assemble_program(row["prompt"], delete, row["test"], row["entry_point"]),
        15.0,
        sandbox=SETTINGS.sandbox,
    )
    # Only "NoneType" proves the guard fired: a real os.remove of a missing file also fails.
    if status != "failed" or "NoneType" not in stderr:
        safety.append(f"os.remove was not blocked: {status} {stderr[-200:]}")
    print(
        "  wrong solution fails, infinite loop times out, os.remove blocked: "
        + ("no" if safety else "yes")
    )
    return problems + safety


COMPLETIONS = "nuprl/MultiPL-E-completions"
COMPLETIONS_REVISION = "64df476ca24ffa9e4205827bbdaaa8761df75a71"
# Whose graded completions each language replays (the parquet's folder name).
COMPLETION_RUNS = {
    lang: ("replit_code.0.2.reworded" if lang == "py" else "starcoder2_15b.0.2.reworded")
    for lang in multipl_e.LANGUAGES
}
TIMEOUT_CAP = 10
# MultiPL-E's time limits (its evaluation/src/eval_<lang>.py). Kumaru allows
# 15 s to run and 120 s to compile, so a correct answer that timed out there
# (slow, or graded on a busy machine) can pass here.
MULTIPL_E_LIMITS = {
    "py": "15 s",
    "js": "5 s",
    "ts": "15 s per step",
    "cs": "5 s, on Mono",
    "java": "15 s per step",
    "cpp": "15 s per step",
    "go": "30 s to build and run",
    "rs": "15 s to compile, 5 s to run",
}
# C++ problems whose tests hold numbers beyond 32 bits in a `long`: 64 bits on
# Linux, where MultiPL-E grades (and in the Docker sandbox), but 32 bits on
# Windows, so answers MultiPL-E passed fail there (multipl_e.md, "Problems no
# answer can pass").
CPP_NEEDS_64_BIT_LONG = frozenset({114, 139})
# How wide C++'s `long` is for programs run natively on this machine.
NATIVE_LONG_BITS = 32 if sys.platform == "win32" else 64


@dataclass(frozen=True)
class Replay:
    """One completion MultiPL-E graded, and its verdict."""

    problem: str
    index: int
    status: str
    completion: str
    stderr: str
    # The program MultiPL-E ran, and its output: where tsc reports its errors.
    program: str = ""
    stdout: str = ""


def completion_rows(lang: str) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        COMPLETIONS,
        f"humaneval.{lang}.{COMPLETION_RUNS[lang]}/test-00000-of-00001.parquet",
        repo_type="dataset",
        revision=COMPLETIONS_REVISION,
    )
    rows = pq.read_table(path).to_pylist()
    return [row for row in rows if int(row["problem"].split("_")[1]) not in multipl_e.EXCLUDED]


def quotas(counts: dict[str, int], size: int) -> dict[str, int]:
    """How many of each MultiPL-E status to replay: half passes, the rest shared
    evenly between the ways of failing, timeouts capped."""
    limits = {status: min(n, TIMEOUT_CAP) if status == "Timeout" else n for status, n in counts.items()}
    share = {status: 0 for status in limits}
    share["OK"] = min(limits.get("OK", 0), size // 2)
    budget = size - share["OK"]
    active = sorted(status for status in limits if status != "OK" and limits[status])
    while budget and active:
        each = max(1, budget // len(active))
        for status in list(active):
            take = min(each, limits[status] - share[status], budget)
            share[status] += take
            budget -= take
            if share[status] == limits[status]:
                active.remove(status)
            if not budget:
                break
    # Too few failures: more passes instead.
    share["OK"] += min(budget, limits.get("OK", 0) - share["OK"])
    return {status: n for status, n in share.items() if n}


def sample_replays(rows: list[dict[str, Any]], size: int, seed: int = 0) -> list[Replay]:
    """`size` graded completions, per `quotas`, each status's spread across problems."""
    rng = random.Random(seed)
    by_status: dict[str, dict[str, list[Replay]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        blank = [""] * len(row["completions"])
        verdicts = zip(
            row["completions"],
            row["statuses"],
            row["stderrs"],
            row.get("programs") or blank,
            row.get("stdouts") or blank,
            strict=True,
        )
        for index, (completion, status, stderr, program, stdout) in enumerate(verdicts):
            replay = Replay(row["problem"], index, status, completion, stderr or "", program or "", stdout or "")
            by_status[status][row["problem"]].append(replay)
    counts = {status: sum(map(len, problems.values())) for status, problems in by_status.items()}
    picked: list[Replay] = []
    for status, quota in sorted(quotas(counts, size).items()):
        pools = [rng.sample(pool, len(pool)) for _, pool in sorted(by_status[status].items())]
        rng.shuffle(pools)
        taken = 0
        # One per problem per round, so no problem dominates.
        while taken < quota:
            for pool in pools:
                if pool and taken < quota:
                    picked.append(pool.pop())
                    taken += 1
    return picked


# Where tsc reports an error: "<file>(<line>,<column>): error TS<code>: ...".
_TSC_ERROR_LINE = re.compile(r"\((\d+),\d+\): error TS\d+")


def dropped_by_kumaru(bench: multipl_e.MultiPLE, replay: Replay, item: BenchItem) -> bool:
    """Is the first line MultiPL-E's tsc rejected one that Kumaru's program leaves out?

    Kumaru keeps a TypeScript answer's declarations and drops its other
    top-level statements (multipl_e._assemble_js); MultiPL-E compiles them.
    StarCoder2 often writes a file path after the function, such as
    `/src/kata/7kyu/sum.ts`, which tsc reads as a regular expression, or an
    example call cut off mid-string. tsc lists errors in file order, and the
    later ones can follow from the first (an unterminated string runs into
    the next line), so only the first counts.
    """
    numbers = [int(n) for n in _TSC_ERROR_LINE.findall(replay.stdout)]
    lines = replay.program.splitlines()
    if bench.lang != "ts" or not numbers or max(numbers) > len(lines):
        return False
    meta, entry = item.meta, item.meta["entry_point"]
    code = multipl_e.extract_answer(replay.completion, "ts", entry)
    source, _ = multipl_e.solution_source(meta["prompt"], code, entry, "ts")
    program = multipl_e.assemble("ts", meta["prompt"], source, meta["tests"], "", entry)
    kept = {line.strip() for line in program.splitlines()}
    return lines[min(numbers) - 1].strip() not in kept


def expected_difference(
    bench: multipl_e.MultiPLE, replay: Replay, grade: Grade, item: BenchItem | None = None
) -> str:
    """Why Kumaru may rightly disagree with MultiPL-E here ("" if it may not)."""
    if replay.status == "SyntaxError" and grade.correct and item and dropped_by_kumaru(bench, replay, item):
        return "tsc failed on top-level statements Kumaru drops (not declarations), such as a file path"
    strip_only = bench.lang == "ts" and bench.toolchain.checker == "strip"
    if strip_only and replay.status == "SyntaxError" and grade.correct:
        return "tsc rejected it; without tsc, types are not checked"
    if replay.status == "Timeout" and grade.correct:
        limit = MULTIPL_E_LIMITS[bench.lang]
        return f"timed out under MultiPL-E's limit ({limit}), passes within Kumaru's"
    newer_go = re.search(r"undefined: (?:min|max|clear)\b", replay.stderr)
    if bench.lang == "go" and replay.status == "SyntaxError" and grade.correct and newer_go:
        return "uses min, max or clear, built into Go since 1.21 (MultiPL-E's Go is older)"
    long_is_32_bits = bench.lang == "cpp" and bench.sandbox == "none" and NATIVE_LONG_BITS == 32
    if long_is_32_bits and replay.status == "OK" and grade.extracted == "failed":
        # Only a failure at the assertion that holds such a number.
        failed = re.search(r"^Assertion failed: (.*)$", str(grade.detail.get("stderr", "")), re.M)
        too_big = failed is not None and any(int(n) >= 2**31 for n in re.findall(r"\d+", failed[1]))
        if too_big and int(replay.problem.split("_")[1]) in CPP_NEEDS_64_BIT_LONG:
            return "its test needs a 64-bit long; C++'s long is 32 bits on Windows"
    return ""


def check_multipl_e_language(lang: str, *, skip_missing: bool = False) -> list[str]:
    """MultiPL-E's own verdicts on a sample of graded completions are reproduced."""
    bench = multipl_e.BY_LANGUAGE[lang]()
    bench.configure(sandbox=SETTINGS.sandbox)
    try:
        bench.preflight()
    except ToolchainMissing as exc:
        print(f"{bench.name}: not checked, {exc.why}")
        return [] if skip_missing else [f"{bench.name}: {exc.why}"]
    rows = completion_rows(lang)
    replays = sample_replays(rows, SETTINGS.sample)
    # The tests as published, not as `load` fixes them (multipl_e.TEST_FIXES):
    # MultiPL-E's verdicts come from these.
    items = {
        row["problem"]: BenchItem(
            id=row["problem"],
            prompt="",
            meta={
                "prompt": row["prompt"],
                "tests": row["tests"],
                "entry_point": (
                    row["problem"].split("_", 2)[2]
                    if lang == "py"
                    else multipl_e.entry_point(row["prompt"], lang)
                ),
            },
        )
        for row in rows
    }
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=SETTINGS.workers) as pool:
        grades = list(pool.map(lambda r: bench.grade(items[r.problem], r.completion), replays))
    elapsed = time.perf_counter() - start

    problems: list[str] = []
    table: dict[str, Counter[str]] = defaultdict(Counter)
    explained: Counter[str] = Counter()
    agree = 0
    for replay, grade in zip(replays, grades, strict=True):
        table[replay.status][grade.extracted] += 1
        if grade.correct == (replay.status == "OK"):
            agree += 1
            continue
        reason = expected_difference(bench, replay, grade, items[replay.problem])
        if reason:
            explained[reason] += 1
            continue
        ours = str(grade.detail.get("stderr", "")).strip().splitlines()[-3:]
        theirs = replay.stderr.strip().splitlines()[-3:]
        problems.append(
            f"{bench.name} {replay.problem} #{replay.index}: MultiPL-E {replay.status}, "
            f"Kumaru {grade.extracted} ({grade.detail.get('mode')})\n"
            f"      Kumaru: {' | '.join(ours)[-300:]}\n      MultiPL-E: {' | '.join(theirs)[-300:]}"
        )
    run = COMPLETION_RUNS[lang].split(".")[0]
    print(
        f"{bench.name} ({bench.toolchain.version}): {agree}/{len(replays)} of MultiPL-E's verdicts "
        f"on {run} completions reproduced, {len({r.problem for r in replays})} problems, {elapsed:.0f}s"
    )
    for status, kinds in sorted(table.items()):
        print(f"  MultiPL-E {status:<15} {sum(kinds.values()):>4}: Kumaru {dict(sorted(kinds.items()))}")
    for reason, n in explained.items():
        print(f"  {n} expected difference(s): {reason}")
    return problems


def check_multipl_e() -> list[str]:
    """Every MultiPL-E language whose toolchain is installed."""
    return [p for lang in multipl_e.LANGUAGES for p in check_multipl_e_language(lang, skip_missing=True)]


CHECKS: dict[str, Callable[[], list[str]]] = {
    "gsm8k": check_gsm8k,
    "mmlu_pro": check_mmlu_pro,
    "humaneval": check_humaneval,
    "multipl_e": check_multipl_e,
    **{
        cls.name: (lambda lang=lang: check_multipl_e_language(lang))
        for lang, cls in multipl_e.BY_LANGUAGE.items()
    },
}
# What runs when no benchmark is named: the quick checks, which need no compilers.
DEFAULT_CHECKS = ("gsm8k", "mmlu_pro", "humaneval")


def main(argv: list[str]) -> int:
    if any(arg in ("-h", "--help") for arg in argv):
        print(__doc__)
        return 0
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("names", nargs="*")
    parser.add_argument("--workers", type=int, default=SETTINGS.workers)
    parser.add_argument("--sample", type=int, default=SETTINGS.sample)
    parser.add_argument("--sandbox", choices=codeexec.SANDBOXES, default=SETTINGS.sandbox)
    args = parser.parse_args(argv)
    if args.workers < 1 or args.sample < 1:
        print("--workers and --sample must be at least 1")
        return 2
    SETTINGS.workers, SETTINGS.sample, SETTINGS.sandbox = args.workers, args.sample, args.sandbox
    codeexec.set_max_workers(SETTINGS.workers)
    names = args.names or list(DEFAULT_CHECKS)
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

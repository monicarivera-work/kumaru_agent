"""Distillation data: turn a teacher model's graded answers into training examples.

Knowledge distillation trains a small, fast *student* model to imitate a large
*teacher*. This module builds the data for the sequence-level form (Kim & Rush,
2016): the teacher answers every training question, and the student is
fine-tuned on those answers with ordinary SFT (`kumaru train`). Keeping only
answers a grader marked correct makes it rejection-sampling fine-tuning (RFT;
Yuan et al., 2023), a filtered relative of STaR (Zelikman et al., 2022). The
student learns the teacher's worked solutions, not just its final numbers, and
that is where the gain comes from::

    kumaru bench run gsm8k_train -c qwen36 --no-think --full --max-tokens 1024 -o runs/distill/teacher-qwen36
    kumaru distill build runs/distill/teacher-qwen36 -o runs/distill/data-teacher
    kumaru distill gold -o runs/distill/data-gold --match runs/distill/data-teacher
    kumaru train configs/train/distill-gsm8k.yaml

Filters
-------
A student copies everything, mistakes included, so an answer is kept only if
it passes every check below, in this order. The first check that rejects an
answer is the one counted, so in manifest.json `kept` plus the `dropped`
counts equals `read`.

* error, wrong, unfinished, format, empty, too_long: read straight from the
  benchmark records. Only complete, correct answers that end with the
  "Answer: N" line the prompt asked for survive.
* contaminated: the question or the answer copies an evaluation question, or
  shares a 13-word run with one outside their instruction text (`NearCopyIndex`).
* arithmetic: a right final answer can still contain a wrong step
  ("12 x 3 = 35"). `wrong_steps` recomputes the simple ones.
* duplicate, over_cap: identical answers to one question count once, and a
  question keeps at most `per_question` answers, so easy questions (which
  every sample gets right) do not crowd out hard ones.

Two safeguards against fooling yourself
---------------------------------------
* **Decontamination.** Training on test questions inflates exactly the score
  you are trying to measure. Only benchmarks that declare themselves training
  splits (`Benchmark.training_data`, e.g. gsm8k_train) are accepted as
  sources, and near-copies of evaluation questions are dropped.
* **A control arm.** `build_gold` makes the same dataset from GSM8K's own
  human-written solutions. If the teacher arm beats the gold arm, the
  teacher's answers taught more than human ones. Only the gold arm against
  the untrained student says whether GSM8K-style practice alone helped.

Output, all in one folder::

    train.jsonl    {"messages": [user, assistant], "id": ..., "source": ...} per line
    val.jsonl      held-out questions, scored for eval loss during training
    manifest.json  sources, settings, counts per filter, examples, file hashes

Rows carry no system message, like the requests of `kumaru bench run`
(without --use-agent-prompt): the student is trained the way it is tested.
"""

from __future__ import annotations

import hashlib
import json
import operator
import os
import re
import statistics
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, Field

from kumaru import __version__
from kumaru.core.errors import ConfigError

if TYPE_CHECKING:
    from rich.console import Console

    from kumaru.eval.benchmarks.base import Benchmark
    from kumaru.eval.benchmarks.runner import BenchSummary, ItemRecord

Split = Literal["train", "val"]
SIDES: tuple[Split, ...] = ("train", "val")
FILES = ("train.jsonl", "val.jsonl", "manifest.json")

# Word-run length of the near-copy check: the contamination test of GPT-3.
NGRAM = 13
# A run of words found in more than this share of one benchmark's prompts is
# instruction text: the template is in every prompt, a question's words in few.
INSTRUCTION_SHARE = 0.5
EXAMPLES_PER_REASON = 5

# Why an answer was dropped, in the order the checks run.
REASONS: dict[str, str] = {
    "error": "the request or the grader failed",
    "wrong": "graded wrong",
    "unfinished": "stopped before the end (cut off by the token limit)",
    "format": 'no "Answer: N" line (the grader found the number another way)',
    "empty": "empty answer",
    "too_long": "longer than --max-tokens",
    "not_matched": "question not in the --match dataset",
    "contaminated": "near-copy of an evaluation question",
    "arithmetic": 'a step "a op b = c" with the wrong c',
    "duplicate": "same answer to the same question already kept",
    "over_cap": "question already has --per-question answers",
}


class DistillSettings(BaseModel):
    """How a dataset is built. Every field is recorded in manifest.json."""

    # Drop answers longer than this many (teacher) tokens. Keep it below the
    # training max_seq_length minus the prompt, or the trainer drops them.
    max_tokens: int | None = Field(default=1024, ge=1)
    # Distinct answers kept per question. Several help (RFT), but uncapped,
    # the questions every sample gets right would dominate the data.
    per_question: int = Field(default=4, ge=1)
    # Share of *questions* held out for validation loss.
    val_fraction: float = Field(default=0.05, ge=0.0, le=0.5)
    seed: int = 42
    check_arithmetic: bool = True
    decontaminate: bool = True
    # Benchmarks the student will be scored on (registry names or .jsonl paths).
    eval_sets: list[str] = Field(default_factory=lambda: ["gsm8k"])
    allow_thinking: bool = False


class Manifest(BaseModel):
    """manifest.json: what is needed to trust, reproduce and compare a dataset."""

    kind: Literal["teacher", "gold"]
    created_at: str
    kumaru_version: str = __version__
    git_commit: str | None = None
    settings: DistillSettings
    sources: list[dict[str, Any]] = Field(default_factory=list)
    read: int = 0
    kept: int = 0
    train_rows: int = 0
    val_rows: int = 0
    questions: int = 0
    val_questions: int = 0
    # Answers rejected per check, in check order: kept + sum(dropped) == read.
    dropped: dict[str, int] = Field(default_factory=dict)
    mean_answer_tokens: float | None = None
    mean_answer_chars: float = 0.0
    examples: dict[str, list[dict[str, str]]] = Field(default_factory=dict)
    # sha256 of each data file: proof of exactly which data trained a model.
    files: dict[str, str] = Field(default_factory=dict)


@dataclass
class Candidate:
    """One answer to one question, before the dataset-level checks."""

    id: str
    prompt: str
    answer: str
    source: str
    tokens: int | None = None


# ---------------------------------------------------------------------------
# Arithmetic check
# ---------------------------------------------------------------------------

# Numbers as answers write them: 12, 3.5, .5, 1,500, 1,500.25
_NUM = r"(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+"
_STEP = re.compile(
    rf"(?P<a>{_NUM})[ \t]*(?P<op>[-+*/×÷·xX])[ \t]*(?P<b>{_NUM})[ \t]*=[ \t]*(?P<c>-?(?:{_NUM}))"
)
_OPS: dict[str, Callable[[float, float], float]] = {
    "+": operator.add,
    "-": operator.sub,
    "*": operator.mul,
    "×": operator.mul,
    "·": operator.mul,
    "x": operator.mul,
    "X": operator.mul,
    "/": operator.truediv,
    "÷": operator.truediv,
}
# What answers write instead of plain symbols: LaTeX operators and thousands
# separators, $ for money or for LaTeX math mode, ** for markdown bold.
_REWRITES = (
    ("\\times", "×"),
    ("\\cdot", "·"),
    ("\\div", "÷"),
    ("\\%", "%"),
    ("{,}", ","),
    ("\\,", ","),
    ("\\!", ""),
    ("\\$", ""),
    ("$", ""),
    ("**", " "),
    ("\u2212", "-"),  # the minus sign
)
# Next to a step (spaces skipped), these mean it is part of something longer:
# "2 + 3 x 4 = 14" before it, "17 x 16 = 17 x (10 + 6)" or "18 000" after it.
_OPERATORS = "+-*/×÷·xX^"
_BEFORE = frozenset(_OPERATORS + "0123456789)]}")
_AFTER = frozenset(_OPERATORS + "0123456789")
# An input with this many decimals is probably rounded: 4.333 for 13/3.
ROUNDED_DECIMALS = 3
_LEADING_ZERO = re.compile(r"-?0\d")  # "0900" is a clock time, not nine hundred
_PERCENT = re.compile(r"[ \t]*(?:%|percent\b)", re.IGNORECASE)


def wrong_steps(text: str) -> list[str]:
    """Every simple step "a op b = c" in `text` whose c is wrong.

    Conservative on purpose: a false alarm throws away a good example, so a
    step is checked only when it is one operation on two plain numbers and c
    is the whole right-hand side. Skipped, not flagged: a step inside a longer
    expression ("2 + 3 x 4 = 14", "17 x 16 = 17 x (10 + 6)"), an input with 3+
    decimals (4.333 is probably 13/3 rounded), clock times ("0900") and
    numbers that go on ("18 000"). A c rounded or truncated to the digits it
    shows is right: "10 / 3 = 3.33" and "7 / 2 = 3" (whole boxes) pass,
    "3 x 4 = 13" does not. A c followed by % may also be read as a fraction:
    "1 / 4 = 25%" passes.
    """
    for old, new in _REWRITES:
        text = text.replace(old, new)
    wrong: list[str] = []
    pos = 0
    while match := _STEP.search(text, pos):
        # c may be the first number of the next step: "2 + 3 = 5 + 4 = 9".
        pos = match.start("c")
        if not _checkable(text, match):
            continue
        try:
            value = _OPS[match["op"]](_number(match["a"]), _number(match["b"]))
        except ZeroDivisionError:
            continue
        if not _agrees(value, match["c"], percent=bool(_PERCENT.match(text, match.end("c")))):
            wrong.append(" ".join(match.group(0).split()))
    return wrong


def _checkable(text: str, match: re.Match[str]) -> bool:
    """True when `match` is a whole step on plain quantities (see wrong_steps)."""
    if any(_LEADING_ZERO.match(match[name]) for name in "abc"):
        return False
    if any(len(match[name].partition(".")[2]) >= ROUNDED_DECIMALS for name in "ab"):
        return False
    start, end = match.start("a"), match.end("c")
    # Glued to a word or a number: "Q3", the 5 of "3.5", the 30 of "1:30".
    if start > 0 and (text[start - 1].isalnum() or text[start - 1] in ".,:"):
        return False
    # c must be a plain number: not "3/4", "2^3", "1:30" or "15km".
    if end < len(text) and (text[end].isalnum() or text[end] in "/^:"):
        return False
    return _neighbour(text, start - 1, -1) not in _BEFORE and _neighbour(text, end, 1) not in _AFTER


def _neighbour(text: str, index: int, step: int) -> str:
    """The first character from `index`, moving by `step`, that is not a space or tab."""
    while 0 <= index < len(text) and text[index] in " \t":
        index += step
    return text[index] if 0 <= index < len(text) else ""


def _number(text: str) -> float:
    return float(text.replace(",", ""))


def _agrees(value: float, stated: str, *, percent: bool) -> bool:
    """True when `stated` is `value` to within one unit of its last digit."""
    unit = 10.0 ** -len(stated.partition(".")[2])
    readings = [(_number(stated), unit)]
    if percent:
        readings.append((_number(stated) / 100, unit / 100))
    # Strictly less than one unit (less a hair for float noise): rounding and
    # truncation both pass, an off-by-one does not.
    return any(abs(value - reading) < step * (1 - 1e-6) for reading, step in readings)


# ---------------------------------------------------------------------------
# Near-copy check (decontamination)
# ---------------------------------------------------------------------------

_WORD = re.compile(r"\w+")


def _words(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def _grams(words: Sequence[str], n: int) -> list[str]:
    return [" ".join(words[i : i + n]) for i in range(len(words) - n + 1)]


def _benchmark_of(item_id: str) -> str:
    """The benchmark an item id names ("gsm8k/12" -> "gsm8k"), or "" if none."""
    name, slash, _ = item_id.partition("/")
    return name if slash else ""


def _instruction(
    prompts: Mapping[str, str], n: int, share: float, within: set[str] | None = None
) -> set[str]:
    """Runs found in more than `share` of one benchmark's prompts, and in at least two.

    With `within`, only the runs in it are counted.
    """
    benchmarks: defaultdict[str, set[str]] = defaultdict(set)
    for item_id, prompt in prompts.items():
        benchmarks[_benchmark_of(item_id)].add(prompt)
    found: set[str] = set()
    for texts in benchmarks.values():
        counts: Counter[str] = Counter()
        for text in texts:
            grams = set(_grams(_words(text), n))
            counts.update(grams if within is None else grams & within)
        limit = max(1.0, share * len(texts))
        found |= {gram for gram, count in counts.items() if count > limit}
    return found


class NearCopyIndex:
    """Finds text that copies an evaluation question or shares a 13-word run with one.

    A 13-word overlap is the contamination test of GPT-3 (Brown et al., 2020),
    since used by many papers: long enough that a match is no coincidence,
    short enough to catch a lightly edited copy.

    Prompts also share instruction text: a benchmark asks every question with
    one template (GSM8K's starts with the same instruction each time). So a
    run found in more than half of one benchmark's prompts, evaluation or
    training, is instruction text; a question's own words are in a few at
    most. Instruction text is cut out before runs are formed, so a run cannot
    straddle it either: otherwise "...with no units. John" would tie together
    every pair of questions that start with "John". Prompts are grouped by
    the benchmark their id names ("gsm8k/12"), so mixing benchmarks with
    different templates hides none of them.

    An exact copy of an evaluation prompt (the same words, ignoring case and
    punctuation) is found whatever was cut.

    It finds copies, not every leak: a paraphrased or translated question
    shares no 13-word run, and a question under 13 words is found only as an
    exact copy. A benchmark with very few prompts cannot show its template,
    so text they all share is taken for instruction text.
    """

    def __init__(
        self,
        eval_prompts: Mapping[str, str],
        train_prompts: Mapping[str, str] | None = None,
        *,
        n: int = NGRAM,
        share: float = INSTRUCTION_SHARE,
    ) -> None:
        self.n = n
        self.instruction = _instruction(eval_prompts, n, share)
        # Only runs that occur in an evaluation question can cause a match, so
        # only those are counted on the training side.
        seen = {gram for prompt in set(eval_prompts.values()) for gram in _grams(_words(prompt), n)}
        self.instruction |= _instruction(train_prompts or {}, n, share, within=seen)
        self._exact: dict[str, str] = {}
        self._owner: dict[str, str] = {}
        for eval_id, prompt in eval_prompts.items():
            words = _words(prompt)
            if words:
                self._exact.setdefault(" ".join(words), eval_id)
            for gram in self._outside_instruction(words):
                self._owner.setdefault(gram, eval_id)

    def grams(self, text: str) -> list[str]:
        """The n-word runs of `text` that lie wholly outside instruction text."""
        return self._outside_instruction(_words(text))

    def find(self, text: str) -> str | None:
        """The id of an evaluation question `text` copies or overlaps, or None."""
        words = _words(text)
        if (owner := self._exact.get(" ".join(words))) is not None:
            return owner
        for gram in self._outside_instruction(words):
            if (owner := self._owner.get(gram)) is not None:
                return owner
        return None

    def _outside_instruction(self, words: Sequence[str]) -> list[str]:
        n = self.n
        grams = _grams(words, n)
        masked = [False] * len(words)
        for i, gram in enumerate(grams):
            if gram in self.instruction:
                masked[i : i + n] = [True] * n
        clean: list[str] = []
        run = 0  # unmasked words ending at position p
        for p, hidden in enumerate(masked):
            run = 0 if hidden else run + 1
            if run >= n:
                clean.append(grams[p - n + 1])
        return clean


# ---------------------------------------------------------------------------
# Building
# ---------------------------------------------------------------------------


def split_of(question_id: str, seed: int, val_fraction: float) -> Split:
    """Which side of the train/val split a question falls on.

    A pure function of (seed, question id). Hashing the *question*, not the
    answer, keeps every answer to one question on the same side, so eval loss
    never scores a question the model trained on; and every dataset built with
    the same seed holds out the same questions.
    """
    digest = hashlib.sha256(f"{seed}:{question_id}".encode()).digest()
    return "val" if int.from_bytes(digest[:8], "big") / 2**64 < val_fraction else "train"


_CALCULATOR = re.compile(r"<<[^<>]*>>")


def gold_rationale(solution: str) -> str | None:
    """A GSM8K reference solution in the format the teacher was asked for.

    GSM8K writes calculator notes inline ("48/2 = <<48/2=24>>24") and ends with
    "#### 72". The notes are removed and the last line becomes "Answer: 72", so
    both arms of the experiment end their answers the same way.
    """
    from kumaru.eval.benchmarks.gsm8k import gold_answer

    body, marker, _ = solution.rpartition("####")
    body = _CALCULATOR.sub("", body).strip()
    if not marker or not body:
        return None
    return f"{body}\nAnswer: {gold_answer(solution)}"


def build_from_runs(
    runs: Sequence[str | Path],
    out: str | Path,
    settings: DistillSettings | None = None,
    *,
    force: bool = False,
) -> Manifest:
    """Build a dataset from teacher runs of `kumaru bench run`.

    Each of `runs` is a whole run folder or one benchmark's folder in it.
    """
    from kumaru.eval.benchmarks.runner import find_results, load_results

    settings = settings or DistillSettings()
    out = Path(out)
    _check_out(out, force=force)
    evals = _eval_benchmarks(settings)
    eval_names = {bench.name for bench in evals}

    candidates: list[Candidate] = []
    dropped: Counter[str] = Counter()
    sources: list[dict[str, Any]] = []
    for run in runs:
        for bench_dir in find_results(Path(run)).values():
            summary, records = load_results(bench_dir)
            _check_source(bench_dir, summary, records, settings, eval_names)
            sources.append({
                "path": str(bench_dir),
                "benchmark": summary.benchmark,
                "label": summary.label,
                "model": summary.model,
                "backend": summary.backend,
                "thinking": summary.thinking,
                "params": summary.params,
                "n": summary.n,
                "accuracy": summary.accuracy,
                "complete": summary.complete,
                "fingerprint": summary.fingerprint,
                "git_commit": summary.git_commit,
            })  # fmt: skip
            for record in records.values():
                if reason := _problem(record, settings):
                    dropped[reason] += 1
                    continue
                candidates.append(Candidate(
                    id=record.id,
                    prompt=record.prompt,
                    answer=record.output.strip(),
                    source=summary.label,
                    tokens=record.completion_tokens or None,  # 0: not reported
                ))  # fmt: skip
    return _finish("teacher", out, candidates, settings, sources=sources, dropped=dropped, evals=evals)


def build_gold(
    out: str | Path,
    settings: DistillSettings | None = None,
    *,
    match: str | Path | None = None,
    force: bool = False,
) -> Manifest:
    """The control arm: GSM8K's training solutions, written by people.

    Prompts and ids are exactly those of the `gsm8k_train` benchmark, so the
    rows line up with a teacher dataset question for question. With `match`,
    only the questions of that dataset are kept, on the same side of the
    split: the two arms then differ only in who wrote the answers.
    """
    from kumaru.eval.benchmarks import gsm8k

    # Human solutions carry no token count, so no length limit is applied
    # (they are short: about 100 tokens); the manifest records exactly that.
    settings = (settings or DistillSettings()).model_copy(update={"max_tokens": None})
    out = Path(out)
    _check_out(out, force=force)
    splits = _splits_of(Path(match)) if match else None
    evals = _eval_benchmarks(settings)

    bench = gsm8k.GSM8KTrain()
    rows = gsm8k.load_hf_rows("openai/gsm8k", "main", bench.split)
    candidates: list[Candidate] = []
    dropped: Counter[str] = Counter()
    for item, row in zip(bench.items(rows), rows, strict=True):
        answer = gold_rationale(row["answer"])
        if answer is None:
            dropped["empty"] += 1
            continue
        candidates.append(Candidate(item.id, item.prompt, answer, "gold:openai/gsm8k"))
    sources = [{
        "dataset": "openai/gsm8k",
        "config": "main",
        "split": bench.split,
        "n": len(rows),
        "match": str(match) if match else None,
    }]  # fmt: skip
    return _finish(
        "gold", out, candidates, settings,
        sources=sources, dropped=dropped, evals=evals, splits=splits,
    )  # fmt: skip


def _problem(record: ItemRecord, settings: DistillSettings) -> str | None:
    """Why a teacher's answer is unusable, judged from its record alone."""
    if record.error:
        return "error"
    if not record.correct:
        return "wrong"
    # None means the backend did not say; a cut-off answer still fails "format".
    if record.finish_reason not in (None, "stop"):
        return "unfinished"
    method = record.detail.get("method")
    if method is not None and method != "answer_tag":
        return "format"
    if not record.output.strip():
        return "empty"
    if settings.max_tokens is not None and record.completion_tokens > settings.max_tokens:
        return "too_long"
    return None


def _check_source(
    path: Path,
    summary: BenchSummary,
    records: dict[str, ItemRecord],
    settings: DistillSettings,
    eval_names: set[str],
) -> None:
    if settings.decontaminate and (
        summary.benchmark in eval_names or not _is_training_data(summary.benchmark)
    ):
        raise ConfigError(
            f"{path} holds answers to {summary.benchmark}, an evaluation set: training "
            "on them would inflate the score it measures.",
            hint="Run the teacher on a training split (e.g. gsm8k_train) and pass that "
            "benchmark's own folder. For a pipeline smoke test only, pass --no-decontaminate.",
        )
    if not settings.allow_thinking and (summary.thinking or any(r.reasoning for r in records.values())):
        raise ConfigError(
            f"{path} is a thinking run: the student would learn the visible answers "
            "without the reasoning that produced them.",
            hint="Run the teacher with --no-think, or pass --allow-thinking to train on "
            "the visible answers anyway.",
        )


def _is_training_data(benchmark: str) -> bool:
    """Whether a benchmark declares itself a training split (`Benchmark.training_data`).

    Unknown names, golden sets among them, count as evaluation sets.
    """
    from kumaru.core.errors import RegistryError
    from kumaru.core.registry import BENCHMARKS

    try:
        return bool(BENCHMARKS.get(benchmark).training_data)
    except RegistryError:
        return False


def _eval_benchmarks(settings: DistillSettings) -> list[Benchmark]:
    if not settings.decontaminate:
        return []
    from kumaru.eval.benchmarks.runner import resolve_benchmark

    return [resolve_benchmark(name) for name in settings.eval_sets]


def _splits_of(folder: Path) -> dict[str, Split]:
    """Question id -> side of the split, for every row of an existing dataset."""
    splits: dict[str, Split] = {}
    for side in SIDES:
        path = folder / f"{side}.jsonl"
        if not path.exists():
            raise ConfigError(
                f"{path} not found",
                hint="--match takes a folder written by `kumaru distill build`.",
            )
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    splits[json.loads(line)["id"]] = side
    return splits


def _check_out(out: Path, *, force: bool) -> None:
    if out.exists() and not out.is_dir():
        raise ConfigError(f"{out} is a file, not a folder")
    existing = [name for name in FILES if (out / name).exists()]
    if existing and not force:
        raise ConfigError(
            f"{out} already holds a dataset ({', '.join(existing)})",
            hint="Choose another --out, or pass --force to replace it.",
        )


def _finish(
    kind: Literal["teacher", "gold"],
    out: Path,
    candidates: list[Candidate],
    settings: DistillSettings,
    *,
    sources: list[dict[str, Any]],
    dropped: Counter[str],
    evals: list[Benchmark],
    splits: dict[str, Split] | None = None,
) -> Manifest:
    """Run the dataset-level checks, split, and write the three files."""
    read = len(candidates) + sum(dropped.values())
    examples: dict[str, list[dict[str, str]]] = defaultdict(list)

    def reject(reason: str, candidate: Candidate, **detail: str) -> None:
        dropped[reason] += 1
        if len(examples[reason]) < EXAMPLES_PER_REASON:
            examples[reason].append({"id": candidate.id, **detail})

    index = None
    if evals:
        eval_prompts = {item.id: item.prompt for bench in evals for item in bench.load()}
        index = NearCopyIndex(eval_prompts, {c.id: c.prompt for c in candidates})
    prompt_hits: dict[str, str | None] = {}
    seen: set[tuple[str, str]] = set()
    per_question: Counter[str] = Counter()
    kept: list[Candidate] = []
    for c in candidates:
        if splits is not None and c.id not in splits:
            reject("not_matched", c)
            continue
        if index is not None:
            if c.prompt not in prompt_hits:
                prompt_hits[c.prompt] = index.find(c.prompt)
            if hit := prompt_hits[c.prompt] or index.find(c.answer):
                reject("contaminated", c, matches=hit)
                continue
        if settings.check_arithmetic and (steps := wrong_steps(c.answer)):
            reject("arithmetic", c, step=steps[0])
            continue
        key = (c.id, " ".join(c.answer.split()))
        if key in seen:
            reject("duplicate", c)
            continue
        if per_question[c.id] >= settings.per_question:
            reject("over_cap", c)
            continue
        seen.add(key)
        per_question[c.id] += 1
        kept.append(c)

    counts = {reason: dropped[reason] for reason in REASONS if dropped[reason]}
    if not kept:
        summary = ", ".join(f"{reason} {count:,}" for reason, count in counts.items())
        raise ConfigError(
            f"No answer survived the checks ({summary or 'nothing to read'}).",
            hint="Check that the sources are graded runs of the right benchmark.",
        )

    rows: dict[Split, list[dict[str, Any]]] = {side: [] for side in SIDES}
    for c in kept:
        side = splits[c.id] if splits is not None else split_of(c.id, settings.seed, settings.val_fraction)
        rows[side].append({
            "messages": [
                {"role": "user", "content": c.prompt},
                {"role": "assistant", "content": c.answer},
            ],
            "id": c.id,
            "source": c.source,
        })  # fmt: skip

    out.mkdir(parents=True, exist_ok=True)
    # The manifest exists only while the data beside it is complete.
    (out / "manifest.json").unlink(missing_ok=True)
    files: dict[str, str] = {}
    for side in SIDES:
        text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows[side])
        files[f"{side}.jsonl"] = _write_atomic(out / f"{side}.jsonl", text)

    tokens = [c.tokens for c in kept if c.tokens is not None]
    manifest = Manifest(
        kind=kind,
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        git_commit=_git_commit(),
        settings=settings,
        sources=sources,
        read=read,
        kept=len(kept),
        train_rows=len(rows["train"]),
        val_rows=len(rows["val"]),
        questions=len({c.id for c in kept}),
        val_questions=len({row["id"] for row in rows["val"]}),
        dropped=counts,
        mean_answer_tokens=round(statistics.fmean(tokens), 1) if tokens else None,
        mean_answer_chars=round(statistics.fmean(len(c.answer) for c in kept), 1),
        examples=dict(examples),
        files=files,
    )
    _write_atomic(out / "manifest.json", manifest.model_dump_json(indent=2) + "\n")
    return manifest


def _write_atomic(path: Path, text: str) -> str:
    """Write `text` as UTF-8 bytes via a temporary file; return its sha256.

    Bytes rather than text mode, so "\\n" stays "\\n" on Windows and the hash
    in the manifest matches the file on every OS.
    """
    data = text.encode("utf-8")
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)
    return hashlib.sha256(data).hexdigest()


def _git_commit() -> str | None:
    from kumaru.eval.benchmarks.runner import git_commit

    return git_commit()


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def render(console: Console, manifest: Manifest, out: Path) -> None:
    """Print what a build kept, what it dropped and why, and what to run next."""
    from rich.markup import escape
    from rich.table import Table

    share = manifest.kept / manifest.read if manifest.read else 0.0
    console.print(
        f"kept [bold]{manifest.kept:,}[/] of {manifest.read:,} answers ({share:.1%}): "
        f"{manifest.train_rows:,} train rows, {manifest.val_rows:,} val rows "
        f"({manifest.val_questions:,} held-out questions)"
    )
    if manifest.mean_answer_tokens is not None:
        console.print(f"[dim]mean answer:[/] {manifest.mean_answer_tokens:,.0f} tokens")
    if manifest.dropped:
        table = Table(border_style="dim", title="dropped")
        table.add_column("check", style="cyan", no_wrap=True)
        table.add_column("answers", justify="right")
        table.add_column("meaning", style="dim")
        for reason, count in manifest.dropped.items():
            table.add_row(reason, f"{count:,}", escape(REASONS.get(reason, "")))
        console.print(table)
    for reason in ("contaminated", "arithmetic"):
        for example in manifest.examples.get(reason, [])[:3]:
            detail = example.get("matches") or example.get("step") or ""
            console.print(f"[dim]{reason}:[/] {escape(example['id'])}  {escape(detail)}")
    if not manifest.settings.decontaminate:
        console.print(
            "[yellow]not decontaminated:[/] fine for a pipeline smoke test; never "
            "report a score from a model trained on it."
        )
    console.print(f"\n[dim]saved to[/] {out}")
    config = "then `kumaru train <config>` (see configs/train/distill-gsm8k.yaml)"
    if manifest.val_rows:
        console.print(
            f"[dim]train:[/] point data.train_file at {out / 'train.jsonl'} and data.eval_file "
            f"at {out / 'val.jsonl'}, {config}"
        )
        return
    console.print(
        f"[dim]train:[/] point data.train_file at {out / 'train.jsonl'}, set data.eval_file: null "
        f"and data.eval_split: 0, {config}"
    )
    console.print(
        "[yellow]no held-out questions:[/] val.jsonl is empty, which the trainer refuses, and a "
        "row split (data.eval_split) would put answers to one question on both sides. For an "
        "eval loss, hold out questions with --val-fraction (with --match: in the matched dataset)."
    )

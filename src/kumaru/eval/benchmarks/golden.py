"""Golden set — your own questions, graded deterministically.

Public benchmarks tell you how a model compares with others. A golden set tells
you whether it does *your* job. Add a line to `data/bench/golden.jsonl` for
every question you care about — especially every question it ever got wrong
that you then fixed — and you have a regression test for your use case.

Each JSONL row::

    {"id": "pens", "prompt": "...", "grader": "numeric", "answer": 26,
     "tags": ["math"], "tolerance": 0, "system": null, "max_tokens": 512}

Graders (all case-insensitive unless noted):

* ``exact``        — whole reply equals ``answer`` (or one of a list), ignoring
  surrounding whitespace, quotes and a trailing period;
* ``contains``     — every string in ``answer`` appears in the reply;
* ``contains_any`` — at least one string in ``answer`` appears;
* ``not_contains`` — none of the strings in ``answer`` appear;
* ``regex``        — ``re.search(answer, reply.strip())`` succeeds; with a list,
  any one pattern (case-sensitive; prefix a pattern with ``(?i)`` to ignore case);
* ``numeric``      — the final number (see `extract_final_number`) equals
  ``answer``: exactly by default (``1,000`` and ``1000.0`` still match), or
  within relative ``tolerance`` when the row sets one, e.g. 0.001;
* ``choice``       — the extracted multiple-choice letter equals ``answer``.

The first tag becomes the item's group, so summaries report accuracy per tag.
Lines that are blank or start with ``//`` are ignored. Any other key is an
error, so a misspelled ``"grade"`` cannot silently fall back to ``contains``;
so is an empty answer (every reply contains ""), a ``numeric`` answer that is
not a number and a ``choice`` answer that is not one letter A-J. Mistakes stop
the load, before the model starts, instead of quietly scoring every reply.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

from kumaru.core.config import REPO_ROOT
from kumaru.core.errors import ConfigError
from kumaru.eval.benchmarks.base import BenchItem, Benchmark, Grade
from kumaru.eval.benchmarks.extract import (
    extract_choice,
    extract_final_number,
    numbers_equal,
    parse_number,
)

DEFAULT_PATH = REPO_ROOT / "data" / "bench" / "golden.jsonl"
GRADERS = ("exact", "contains", "contains_any", "not_contains", "regex", "numeric", "choice")
ROW_KEYS = ("id", "prompt", "grader", "answer", "tags", "tolerance", "system", "max_tokens")
CHOICE_LETTERS = "ABCDEFGHIJ"  # what extract_choice can find


def _as_list(answer: Any) -> list[str]:
    if isinstance(answer, list):
        return [str(a) for a in answer]
    return [str(answer)]


def _normalize(text: str) -> str:
    return " ".join(text.strip().strip("\"'`").rstrip(".").split()).casefold()


def grade_answer(grader: str, answer: Any, output: str, *, tolerance: float = 0.0) -> Grade:
    """Apply one golden-set grader to a reply."""
    low = output.casefold()
    if grader == "exact":
        ok = _normalize(output) in {_normalize(a) for a in _as_list(answer)}
        return Grade(correct=ok, extracted=output.strip()[:200])
    if grader == "contains":
        missing = [a for a in _as_list(answer) if a.casefold() not in low]
        return Grade(correct=not missing, detail={"missing": missing})
    if grader == "contains_any":
        found = [a for a in _as_list(answer) if a.casefold() in low]
        return Grade(correct=bool(found), detail={"found": found})
    if grader == "not_contains":
        found = [a for a in _as_list(answer) if a.casefold() in low]
        return Grade(correct=not found, detail={"forbidden": found})
    if grader == "regex":
        reply = output.strip()
        match = next((m for p in _as_list(answer) if (m := re.search(p, reply))), None)
        return Grade(correct=match is not None, extracted=match.group(0)[:200] if match else None)
    if grader == "numeric":
        value, method = extract_final_number(output)
        ok = numbers_equal(value, answer, rel_tol=tolerance)
        return Grade(correct=ok, extracted=value, detail={"method": method})
    if grader == "choice":
        letter, method = extract_choice(output, CHOICE_LETTERS)
        return Grade(correct=letter == str(answer).strip().upper(), extracted=letter, detail={"method": method})
    raise ConfigError(f"Unknown golden-set grader: {grader!r}", hint=f"Use one of: {', '.join(GRADERS)}")


class GoldenSet(Benchmark):
    name = "golden"
    description = "Your own regression set: JSONL questions with deterministic graders."
    source = "data/bench/golden.jsonl (yours to extend)"

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else DEFAULT_PATH
        if self.path.resolve() != DEFAULT_PATH.resolve():
            # Distinct results directory per file, e.g. "golden-support".
            self.name = f"golden-{self.path.stem}"

    def load(self) -> list[BenchItem]:
        if not self.path.exists():
            raise ConfigError(
                f"Golden set not found: {self.path}",
                hint="Create it, or pass the path of a JSONL file: kumaru bench run my_set.jsonl",
            )
        items: list[BenchItem] = []
        seen: set[str] = set()
        try:
            # utf-8-sig: Notepad and PowerShell may start the file with a BOM.
            text = self.path.read_text(encoding="utf-8-sig")
        except (OSError, UnicodeDecodeError) as exc:
            raise ConfigError(
                f"Cannot read golden set {self.path}: {exc}",
                hint="Save it as UTF-8 text, one JSON object per line.",
            ) from exc
        for line_no, line in enumerate(text.splitlines(), 1):
            if not line.strip() or line.lstrip().startswith("//"):
                continue
            where = f"{self.path.name}:{line_no}"
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ConfigError(f"{where}: invalid JSON ({exc.msg})") from exc
            if not isinstance(row, dict) or not str(row.get("prompt", "")).strip():
                raise ConfigError(f"{where}: every row needs a non-empty 'prompt'")
            unknown = [key for key in row if key not in ROW_KEYS]
            if unknown:
                raise ConfigError(
                    f"{where}: unknown key {unknown[0]!r}", hint=f"Keys a row may have: {', '.join(ROW_KEYS)}."
                )
            grader = row.get("grader", "contains")
            if grader not in GRADERS:
                raise ConfigError(f"{where}: unknown grader {grader!r}", hint=f"Use one of: {', '.join(GRADERS)}")
            if "answer" not in row:
                raise ConfigError(f"{where}: missing 'answer'")
            item_id = str(row.get("id") or line_no)
            if item_id in seen:
                raise ConfigError(f"{where}: duplicate id {item_id!r}")
            seen.add(item_id)
            items.append(self._item(row, grader, item_id, where))
        if not items:
            raise ConfigError(f"Golden set {self.path} has no questions")
        return items

    def _item(self, row: dict[str, Any], grader: str, item_id: str, where: str) -> BenchItem:
        """Check the optional fields here, before the model loads, not mid-run."""
        tolerance = row.get("tolerance", 0.0)
        if type(tolerance) not in (int, float) or not 0 <= tolerance < math.inf:
            raise ConfigError(f"{where}: 'tolerance' must be a number >= 0, e.g. 0.001 (got {tolerance!r})")
        max_tokens = row.get("max_tokens")
        if max_tokens is not None and (type(max_tokens) is not int or max_tokens < 1):
            raise ConfigError(f"{where}: 'max_tokens' must be a whole number >= 1 (got {max_tokens!r})")
        tags = row.get("tags") or []
        if not isinstance(tags, list):
            raise ConfigError(f"{where}: 'tags' must be a list, e.g. [\"math\"] (got {tags!r})")
        system = row.get("system")
        if system is not None and not isinstance(system, str):
            raise ConfigError(f"{where}: 'system' must be text (got {system!r})")
        answers = _as_list(row["answer"])
        if not answers or not all(a.strip() for a in answers):
            # "" is in every reply: contains/regex would pass anything.
            raise ConfigError(f"{where}: 'answer' must not be empty (got {row['answer']!r})")
        if grader == "numeric" and parse_number(row["answer"]) is None:
            raise ConfigError(f"{where}: the numeric grader needs a number as 'answer' (got {row['answer']!r})")
        if grader == "choice" and str(row["answer"]).strip().upper() not in tuple(CHOICE_LETTERS):
            raise ConfigError(
                f"{where}: the choice grader needs one letter {CHOICE_LETTERS[0]}-{CHOICE_LETTERS[-1]} "
                f"as 'answer' (got {row['answer']!r})"
            )
        if grader == "regex":
            for pattern in answers:
                try:
                    re.compile(pattern)
                except re.error as exc:
                    raise ConfigError(f"{where}: invalid regex {pattern!r} ({exc})") from exc
        tags = [str(t) for t in tags]
        return BenchItem(
            id=f"{self.name}/{item_id}",
            prompt=str(row["prompt"]),
            answer=row["answer"],
            group=tags[0] if tags else None,
            system=system,
            max_tokens=max_tokens,
            meta={"grader": grader, "tags": tags, "tolerance": float(tolerance)},
        )

    def grade(self, item: BenchItem, output: str) -> Grade:
        return grade_answer(item.meta["grader"], item.answer, output, tolerance=item.meta["tolerance"])

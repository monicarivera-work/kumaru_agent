"""Answer extraction: reading the final answer out of free-form model text.

This is the least glamorous and most error-prone part of evaluation. A model can
be right and still be scored wrong because the grader misread "1{,}080" or
picked the "A" out of "A good approach is...". Every extractor here therefore
returns *how* it found the answer as well as the answer, so a summary can say
"92% of answers used the requested format" — a real signal of instruction
following — and so suspicious verdicts can be audited.

Each extractor tries the most explicit signal first and falls back to weaker
ones, mirroring the official evaluation scripts of each benchmark.
"""

from __future__ import annotations

import math
import re

# An integer or decimal, optionally negative, optionally with thousands commas.
NUMBER = re.compile(r"-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|-?\.\d+")
# "Answer:", "**Answer:**", "Final answer:" — the format the prompts ask for.
_ANSWER_TAG = re.compile(r"answer[\s*_]*[:：]", re.IGNORECASE)
_ANSWER_IS = re.compile(r"answer\s+is\b", re.IGNORECASE)
_FENCE = re.compile(r"```[ \t]*([\w+-]*)[^\n]*\n(.*?)```", re.DOTALL)
_OPEN_FENCE = re.compile(r"```[ \t]*[\w+-]*[^\n]*\n(.*)$", re.DOTALL)


def _normalize_math(text: str) -> str:
    # LaTeX thousands separators ("1{,}080", "1\,080") and "\$" escapes are
    # common in model output and invisible once rendered.
    return text.replace("{,}", ",").replace("\\,", ",").replace("\\$", "$").replace("\\!", "")


def last_boxed(text: str) -> str | None:
    """Content of the last `\\boxed{...}`, with nested braces handled."""
    start = text.rfind("\\boxed")
    if start < 0:
        return None
    open_at = text.find("{", start)
    if open_at < 0:
        return None
    depth = 0
    for index in range(open_at, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[open_at + 1 : index]
    return None


def _number_after(text: str, start: int, max_lines: int = 3) -> str | None:
    """First number on the line starting at `start`, or on the next few lines."""
    for line in text[start:].splitlines()[:max_lines]:
        match = NUMBER.search(line)
        if match:
            return match.group(0)
    return None


def extract_final_number(text: str) -> tuple[str | None, str]:
    """Return `(number_text, method)` for the final numeric answer.

    Order: an "Answer:" tag (what the prompt asked for), then `\\boxed{}` (a
    habit of math-tuned models), then "the answer is", then the last number in
    the text. `method` is "none" when nothing numeric was found.
    """
    text = _normalize_math(text or "")

    tags = list(_ANSWER_TAG.finditer(text))
    if tags:
        found = _number_after(text, tags[-1].end())
        if found:
            return found, "answer_tag"

    boxed = last_boxed(text)
    if boxed is not None:
        match = NUMBER.search(boxed)
        if match:
            return match.group(0), "boxed"

    phrases = list(_ANSWER_IS.finditer(text))
    if phrases:
        found = _number_after(text, phrases[-1].end(), max_lines=1)
        if found:
            return found, "answer_is"

    numbers = NUMBER.findall(text)
    if numbers:
        return numbers[-1], "last_number"
    return None, "none"


def parse_number(text: str | int | float | None) -> float | None:
    if text is None:
        return None
    if isinstance(text, int | float):
        return float(text)
    try:
        return float(str(text).replace(",", "").replace("$", "").strip())
    except ValueError:
        return None


def numbers_equal(a: str | int | float | None, b: str | int | float | None, rel_tol: float = 0.0) -> bool:
    """True if `a` and `b` are the same number.

    Exact by default: formatting ("1,000", "$5", "2.50") never matters, and an
    absolute 1e-9 absorbs float noise such as 0.1 + 0.2, but nothing else does.
    A relative tolerance accepts wrong answers near large values: 1e-6 graded
    1,450,001 correct for 1,450,000. Pass `rel_tol` only for answers that are
    approximate by nature, such as a rounded unit conversion.
    """
    x, y = parse_number(a), parse_number(b)
    if x is None or y is None:
        return False
    return math.isclose(x, y, rel_tol=rel_tol, abs_tol=1e-9)


def extract_choice(text: str, letters: str = "ABCDEFGHIJ") -> tuple[str | None, str]:
    """Return `(letter, method)` for a multiple-choice answer.

    Follows the MMLU-Pro reference script: "the answer is (X)", then
    "Answer: X", then `\\boxed{X}`, then the last standalone letter. The last
    fallback is weak (it can catch the pronoun "I"), which is why the method is
    recorded and summarized.
    """
    text = text or ""
    choice = f"[{letters}]"
    patterns = (
        ("answer_is", rf"(?i:answer\s+is)\s*:?\s*[*_]*\s*\(?\s*({choice})\s*\)?(?![A-Za-z0-9])"),
        ("answer_tag", rf"(?i:answer)[\s*_]*[:：]\s*[*_]*\s*\(?\s*({choice})\s*\)?(?![A-Za-z0-9])"),
    )
    for method, pattern in patterns:
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1], method

    boxed = last_boxed(text)
    if boxed is not None:
        match = re.search(rf"(?<![A-Za-z])({choice})(?![A-Za-z])", boxed.replace("\\text", ""))
        if match:
            return match.group(1), "boxed"

    matches = re.findall(rf"\b({choice})\b", text)
    if matches:
        return matches[-1], "last_letter"
    return None, "none"


def extract_code(text: str, entry_point: str | None = None) -> str:
    """The Python code in a reply.

    Prefers fenced blocks that define `entry_point`, then the longest Python
    block. An unterminated fence (output cut off by the token limit) still
    yields the code after it; unfenced replies are returned whole.
    """
    text = text or ""
    blocks = [(lang.lower(), body) for lang, body in _FENCE.findall(text)]
    if not blocks:
        match = _OPEN_FENCE.search(text)
        return match.group(1) if match else text

    python = [body for lang, body in blocks if lang in ("", "python", "py", "python3")]
    candidates = python or [body for _, body in blocks]
    if entry_point:
        pattern = re.compile(rf"^\s*(?:async\s+)?def\s+{re.escape(entry_point)}\s*\(", re.MULTILINE)
        defining = [body for body in candidates if pattern.search(body)]
        if defining:
            return max(defining, key=len)
    return max(candidates, key=len)

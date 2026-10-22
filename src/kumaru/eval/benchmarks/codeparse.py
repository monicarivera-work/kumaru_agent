"""Find the top-level structure of source code without parsing it.

MultiPL-E grades a model's code by pasting it next to hidden tests. Chat models
return more than one function: imports, helpers, a `main`, example calls,
sometimes tests of their own. Turning that into a program that compiles beside
the hidden tests means knowing where each top-level piece starts and ends, in
seven languages. A parser per language would mean seven dependencies; two
small ideas get far enough:

1. `mask` blanks comments and turns every string and character literal into
   zeros of the same length, so a "}" inside a string or a comment can no
   longer pass for the end of a function, and every offset into the masked
   text is also an offset into the original.
2. `chunks` cuts the masked text at the top level: a piece ends at ";", at the
   "}" that closes its block, or (in JavaScript, TypeScript and Go, which
   insert semicolons at line ends) at a line break that ends a complete
   statement. The comments before a piece belong to it.

Both are approximations. They know each language's literal syntax (template
literals, raw and verbatim strings, interpolation holes, Rust's chars versus
lifetimes, C++ digit separators, JavaScript regular expressions), which covers
what models write. Code that confuses them is usually broken anyway, and then
fails to compile either way.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

LANGUAGES = ("js", "ts", "java", "cs", "cpp", "go", "rs")

_PAIRS = {"(": ")", "[": "]", "{": "}"}
# After these words a JavaScript "/" starts a regular expression, not a division.
_JS_REGEX_AFTER = frozenset({
    "return", "typeof", "instanceof", "in", "of", "new", "delete", "void", "throw",
    "case", "do", "else", "yield", "await",
})  # fmt: skip
_CPP_STRING_PREFIXES = frozenset({"u8", "u", "U", "L"})
_CPP_RAW_PREFIXES = frozenset({"R", "u8R", "uR", "UR", "LR"})
# A word before `name(` that makes it a call or an expression, not a definition
# (C++, Java, C#: `void` is not here, as `void main(` is a definition).
_NOT_A_TYPE = frozenset({
    "return", "new", "else", "case", "throw", "await", "yield", "goto", "co_return",
    "co_yield", "in", "of", "typeof", "delete", "sizeof", "decltype", "using",
    "not", "and", "or", "is", "as", "do", "default",
})  # fmt: skip
# Tokens that, right after a top-level "}", continue the same statement. In Go
# and TypeScript a "}" can end a type: `func f() interface{} {`, `var x struct{} = ...`.
_AFTER_BRACE = {
    "js": ",.?:=", "ts": ",.?:=|&{", "java": ",.=", "cs": ",.=", "cpp": ",.=", "go": ",.={", "rs": ",.=",
}
_AFTER_BRACE_WORDS = frozenset({"else", "catch", "finally", "while", "as", "satisfies"})
# And in JavaScript/TypeScript: `import { a } from "m"`, `export { a } from "m"`.
_JS_AFTER_BRACE_WORDS = _AFTER_BRACE_WORDS | {"from"}
# JavaScript/TypeScript: a line starting with one of these continues the line before.
_JS_CONTINUES = ".,?:=+-*/%&|^<>{"


def _ident(ch: str, lang: str) -> bool:
    return ch.isalnum() or ch == "_" or (ch == "$" and lang != "cs")


class _Scanner:
    """Records where comments and literals are; `mask` turns that into text."""

    def __init__(self, text: str, lang: str) -> None:
        if lang not in LANGUAGES:
            raise ValueError(f"unsupported language {lang!r}; expected one of {', '.join(LANGUAGES)}")
        self.text, self.lang, self.n = text, lang, len(text)
        # (start, end, is_comment)
        self.spans: list[tuple[int, int, bool]] = []

    def scan(self, i: int = 0, *, hole: bool = False) -> int:
        """Record spans from `i`. In a `hole` (an interpolation whose "{" the
        caller consumed), stop after the "}" that closes it."""
        text, n, lang = self.text, self.n, self.lang
        depth = 0
        regex_ok = True  # would a "/" here start a JavaScript regular expression?
        while i < n:
            ch = text[i]
            nxt = text[i + 1] if i + 1 < n else ""
            if ch == "/" and nxt == "/":
                end = text.find("\n", i)
                end = n if end < 0 else end
                self.spans.append((i, end, True))
                i = end
                continue
            if ch == "/" and nxt == "*":
                end = self._block_comment(i)
                self.spans.append((i, end, True))
                i = end
                continue
            end = self._literal(i, regex_ok)
            if end is not None:
                self.spans.append((i, end, False))
                i, regex_ok = end, False
                continue
            if _ident(ch, lang):
                j = self._word_end(i)
                regex_ok = text[i:j] in _JS_REGEX_AFTER
                i = j
                continue
            if not ch.isspace():
                if hole:
                    if ch == "{":
                        depth += 1
                    elif ch == "}":
                        if depth == 0:
                            return i + 1
                        depth -= 1
                regex_ok = ch not in ")]"
            i += 1
        return n

    # -- pieces ---------------------------------------------------------------

    def _word_end(self, i: int) -> int:
        text, n, lang = self.text, self.n, self.lang
        number = text[i].isdigit()
        j = i + 1
        while j < n:
            c = text[j]
            if _ident(c, lang):
                j += 1
            elif number and c == "'" and lang == "cpp" and j + 1 < n and text[j + 1].isalnum():
                j += 1  # a digit separator: 1'000'000
            else:
                break
        return j

    def _literal(self, i: int, regex_ok: bool) -> int | None:
        """End of the literal starting at `i`, or None if none starts there."""
        text, n, lang = self.text, self.n, self.lang
        ch = text[i]
        if _ident(ch, lang):
            j = self._word_end(i)
            word, q = text[i:j], text[j] if j < n else ""
            if lang == "rs":
                if word in ("r", "br", "cr") and q in ('"', "#"):
                    return self._rust_raw(j)
                if word in ("b", "c") and q == '"':
                    return self._quoted(j, '"', multiline=True)
                if word == "b" and q == "'":
                    return self._quoted(j, "'", multiline=False)
            elif lang == "cpp" and q == '"' and word in _CPP_RAW_PREFIXES:
                return self._cpp_raw(j)
            elif lang == "cpp" and q in ('"', "'") and word in _CPP_STRING_PREFIXES:
                return self._quoted(j, q, multiline=False)
            return None
        if ch == '"':
            if lang == "java" and text.startswith('"""', i):
                return self._until(i + 3, '"""', escapes=True)
            if lang == "cs" and text.startswith('"""', i):
                return self._cs_raw(i)
            return self._quoted(i, '"', multiline=lang == "rs")
        if ch == "'":
            if lang == "rs":
                return self._rust_char(i)
            return self._quoted(i, "'", multiline=False)
        if ch == "`":
            if lang in ("js", "ts"):
                return self._template(i)
            if lang == "go":
                return self._until(i + 1, "`", escapes=False)
            return None
        if lang == "cs" and ch in "@$":
            return self._cs_prefixed(i)
        if ch == "/" and regex_ok and lang in ("js", "ts"):
            return self._regex(i)
        return None

    def _quoted(self, i: int, quote: str, *, multiline: bool) -> int:
        text, n = self.text, self.n
        j = i + 1
        while j < n:
            c = text[j]
            if c == "\\":
                j += 2
                continue
            if c == quote:
                return j + 1
            if c == "\n" and not multiline:
                return j  # unterminated: it ends with its line
            j += 1
        return n

    def _until(self, j: int, closer: str, *, escapes: bool) -> int:
        text, n = self.text, self.n
        while j < n:
            if escapes and text[j] == "\\":
                j += 2
                continue
            if text.startswith(closer, j):
                return j + len(closer)
            j += 1
        return n

    def _block_comment(self, i: int) -> int:
        text, n = self.text, self.n
        nested = self.lang == "rs"
        depth, j = 0, i
        while j < n:
            if text.startswith("/*", j) and (nested or depth == 0):
                depth += 1
                j += 2
            elif text.startswith("*/", j):
                depth -= 1
                j += 2
                if depth == 0:
                    return j
            else:
                j += 1
        return n

    def _template(self, i: int) -> int:
        text, n = self.text, self.n
        j = i + 1
        while j < n:
            c = text[j]
            if c == "\\":
                j += 2
            elif c == "`":
                return j + 1
            elif c == "$" and text.startswith("{", j + 1):
                j = self.scan(j + 2, hole=True)
            else:
                j += 1
        return n

    def _regex(self, i: int) -> int | None:
        text, n = self.text, self.n
        j, in_class = i + 1, False
        while j < n:
            c = text[j]
            if c == "\\":
                j += 2
                continue
            if c == "\n":
                return None  # no closing "/" on this line: it was a division
            if in_class:
                in_class = c != "]"
            elif c == "[":
                in_class = True
            elif c == "/":
                j += 1
                while j < n and text[j].isalpha():
                    j += 1  # flags
                return j
            j += 1
        return None

    def _rust_raw(self, j: int) -> int | None:
        text, n = self.text, self.n
        k = j
        while k < n and text[k] == "#":
            k += 1
        if k >= n or text[k] != '"':
            return None  # r#ident is a raw identifier
        closer = '"' + "#" * (k - j)
        end = text.find(closer, k + 1)
        return n if end < 0 else end + len(closer)

    def _rust_char(self, i: int) -> int | None:
        text, n = self.text, self.n
        if text.startswith("\\", i + 1):
            return self._quoted(i, "'", multiline=False)
        if i + 2 < n and text[i + 2] == "'" and text[i + 1] != "\n":
            return i + 3
        return None  # a lifetime ('a) or a loop label

    def _cpp_raw(self, j: int) -> int:
        text, n = self.text, self.n
        paren = text.find("(", j + 1, j + 18)
        if paren < 0:
            return self._quoted(j, '"', multiline=False)
        closer = ")" + text[j + 1 : paren] + '"'
        end = text.find(closer, paren + 1)
        return n if end < 0 else end + len(closer)

    def _cs_raw(self, j: int) -> int:
        text, n = self.text, self.n
        k = j
        while k < n and text[k] == '"':
            k += 1
        closer = '"' * (k - j)
        end = text.find(closer, k)
        return n if end < 0 else end + len(closer)

    def _cs_prefixed(self, i: int) -> int | None:
        """C# @"verbatim", $"interpolated {x}", $@"both" and $\"\"\"raw\"\"\" strings."""
        text, n = self.text, self.n
        j, interpolated, verbatim = i, False, False
        while j < n and text[j] in "@$":
            interpolated |= text[j] == "$"
            verbatim |= text[j] == "@"
            j += 1
        if j >= n or text[j] != '"':
            return None  # @identifier
        if text.startswith('"""', j) and not verbatim:
            return self._cs_raw(j)
        k = j + 1
        while k < n:
            c = text[k]
            if c == "\\" and not verbatim:
                k += 2
            elif c == '"':
                if verbatim and text.startswith('"', k + 1):
                    k += 2  # "" is an escaped quote
                    continue
                return k + 1
            elif c == "\n" and not verbatim:
                return k
            elif c == "{" and interpolated:
                k = k + 2 if text.startswith("{", k + 1) else self.scan(k + 1, hole=True)
            else:
                k += 1
        return n


def mask(text: str, lang: str) -> str:
    """`text` with comments blanked and literals zeroed, character for character.

    Comments become spaces (their line breaks stay, so lines still end where
    they did); string and character literals become "0"s, line breaks
    included, so code around a literal still reads as a complete expression.
    """
    scanner = _Scanner(text, lang)
    scanner.scan()
    chars = list(text)
    for start, end, comment in scanner.spans:
        for k in range(start, min(end, len(chars))):
            if not comment:
                chars[k] = "0"
            elif chars[k] != "\n":
                chars[k] = " "
    return "".join(chars)


def uncomment(text: str, lang: str, *, masked: str | None = None) -> str:
    """`text` with comments blanked but literals kept, character for character."""
    masked = mask(text, lang) if masked is None else masked
    return "".join(" " if m == " " and not c.isspace() else c for c, m in zip(text, masked, strict=True))


def matching(masked: str, i: int) -> int:
    """Index of the bracket that closes the one at `masked[i]`, or -1."""
    opener = masked[i]
    closer = _PAIRS[opener]
    depth = 0
    for j in range(i, len(masked)):
        ch = masked[j]
        if ch == opener:
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0:
                return j
    return -1


@dataclass(frozen=True)
class Chunk:
    """One top-level piece of source, with the comments and blank lines before it."""

    start: int
    end: int
    text: str
    # The same span, masked: what classification reads.
    code: str

    @property
    def head(self) -> str:
        return self.code.strip()


def chunks(
    text: str, lang: str, *, masked: str | None = None, start: int = 0, end: int | None = None
) -> list[Chunk]:
    """Cut `text[start:end]` into top-level pieces. Joined, they give back the
    text, less any comments after the last piece."""
    masked = mask(text, lang) if masked is None else masked
    end = len(text) if end is None else end
    asi = lang in ("js", "ts", "go")
    after_words = _JS_AFTER_BRACE_WORDS if lang in ("js", "ts") else _AFTER_BRACE_WORDS
    out: list[Chunk] = []
    begin, started, depth, i = start, False, 0, start

    def emit(stop: int) -> None:
        nonlocal begin, started
        if masked[begin:stop].strip():
            out.append(Chunk(begin, stop, text[begin:stop], masked[begin:stop]))
        begin, started = stop, False

    while i < end:
        ch = masked[i]
        if depth == 0 and not started:
            if ch.isspace():
                i += 1
                continue
            started = True
            if ch == "#" and lang in ("cpp", "cs"):  # a preprocessor line is a piece of its own
                stop = _line_end(masked, i, end)
                emit(stop)
                i = stop
                continue
            if lang == "rs" and masked.startswith("#![", i):  # an inner attribute, too
                close = matching(masked, i + 2)
                stop = end if close < 0 or close >= end else close + 1
                emit(stop)
                i = stop
                continue
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            if depth == 0:  # a stray closer: a piece of its own
                emit(i)
                emit(i + 1)
                i += 1
                continue
            depth -= 1
            if depth == 0 and ch == "}":
                j = _skip_space(masked, i + 1, end)
                if j < end and masked[j] == ";":
                    emit(j + 1)
                    i = j + 1
                    continue
                if j < end and (masked[j] in _AFTER_BRACE[lang] or _word_at(masked, j) in after_words):
                    i += 1
                    continue
                emit(i + 1)
        elif depth == 0 and (
            ch == ";" or (ch == "\n" and asi and started and _statement_ends(masked, begin, i, end, lang))
        ):
            emit(i + 1)
        i += 1
    if started:
        emit(end)
    return out


def _skip_space(masked: str, j: int, end: int) -> int:
    while j < end and masked[j].isspace():
        j += 1
    return j


def _word_at(masked: str, j: int) -> str:
    match = re.match(r"\w+", masked[j : j + 16])
    return match.group(0) if match else ""


def _line_end(masked: str, i: int, end: int) -> int:
    """End of a preprocessor line, following backslash continuations."""
    while True:
        newline = masked.find("\n", i, end)
        if newline < 0:
            return end
        if not masked[i:newline].rstrip().endswith("\\"):
            return newline + 1
        i = newline + 1


def _statement_ends(masked: str, begin: int, i: int, end: int, lang: str) -> bool:
    """Does the line break at `i` end the statement, as automatic semicolons would?"""
    before = masked[begin:i].rstrip()
    if not before:
        return False
    last = before[-1]
    complete = last.isalnum() or last in "_$)]}" or before.endswith(("++", "--"))
    if lang == "go" or not complete:
        return complete
    j = _skip_space(masked, i + 1, end)
    if j >= end:
        return True
    if masked[j] in _JS_CONTINUES and not masked.startswith(("++", "--"), j):
        return False
    return _word_at(masked, j) not in _AFTER_BRACE_WORDS | {"extends", "implements"}


def defines(code: str, name: str, lang: str, *, masked: str | None = None) -> bool:
    """Does `code` define a function (or, in JavaScript, a binding) called `name`?"""
    masked = mask(code, lang) if masked is None else masked
    word = re.escape(name)
    if lang in ("js", "ts"):
        pattern = (
            rf"(?<![\w$.])function\s*\*?\s*{word}\s*[<(]"
            rf"|(?<![\w$.])(?:const|let|var)\s+{word}\s*(?::[^=;]*)?=(?![=>])"
        )
        return re.search(pattern, masked) is not None
    if lang == "go":
        return re.search(rf"(?m)^[ \t]*func\s+{word}\s*[\[(]", masked) is not None
    if lang == "rs":
        return re.search(rf"(?<![\w.])fn\s+{word}\s*[<(]", masked) is not None
    return find_definition(masked, name) >= 0


def find_definition(masked: str, name: str) -> int:
    """C++, Java, C#: where a typed function definition `T name(...) {` starts, or -1.

    `name` must follow a type (a word that is not a keyword, or a closing ">",
    "]", "*", "&" or "?") and its parameter list must be followed by a body,
    so calls and prototypes do not count.
    """
    for match in re.finditer(rf"(?<![\w$]){re.escape(name)}\s*\(", masked):
        before = masked[: match.start()].rstrip()
        if not before:
            continue
        prev = before[-1]
        if prev.isalnum() or prev == "_":
            word = re.search(r"\w+$", before)
            if word is None or word.group(0) in _NOT_A_TYPE:
                continue
        elif prev not in ">]*&?":
            continue
        close = matching(masked, match.end() - 1)
        if close >= 0 and re.match(r"[^;{}=]*(?:\{|=>)", masked[close + 1 : close + 400]):
            return match.start()
    return -1

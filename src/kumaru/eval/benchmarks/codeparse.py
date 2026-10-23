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

Everything here takes time linear in the text's length, because a reply can
be tens of thousands of characters of junk ("@@@@...", "/[/[/[...") and grading
it must still take moments, not hours: a probe for a literal that fails
remembers what it read, and interpolation holes nest at most 32 deep (deeper
ones run to the end of the text).
"""

from __future__ import annotations

import bisect
import functools
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
# ...unless it starts with one of these words.
_JS_NOT_CONTINUED = _AFTER_BRACE_WORDS | {"extends", "implements"}
# A word: letters, digits, "_" and (except in C#) "$". `\w` is exactly
# `str.isalnum()` plus "_".
_WORD = re.compile(r"[\w$]+")
_WORD_CS = re.compile(r"\w+")
# Interpolation holes nested deeper than this run to the end of the text.
_MAX_NESTING = 32
_BRACKETS = re.compile(r"[()\[\]{}]")
_OPENERS = {")": "(", "]": "[", "}": "{"}
# After a parameter list: a body, not a prototype or a call.
_BODY = re.compile(r"[^;{}=]*(?:\{|=>)")


def _ident(ch: str, lang: str) -> bool:
    return ch.isalnum() or ch == "_" or (ch == "$" and lang != "cs")


class _Scanner:
    """Records where comments and literals are; `mask` turns that into text."""

    def __init__(self, text: str, lang: str) -> None:
        if lang not in LANGUAGES:
            raise ValueError(f"unsupported language {lang!r}; expected one of {', '.join(LANGUAGES)}")
        self.text, self.lang, self.n = text, lang, len(text)
        # (start, end, is_comment) of the outermost comments and literals, in
        # order. Those inside an interpolation hole are not kept: the literal
        # around them covers them.
        self.spans: list[tuple[int, int, bool]] = []
        self._word = _WORD_CS if lang == "cs" else _WORD
        self._nesting = 0
        # The run of C# "@"/"$" that last failed to start a string: its
        # suffixes fail too, so `scan` need not read it again from each one.
        self._cs_fail = (-1, -1)
        # JavaScript regular-expression probes are deterministic walks over
        # (position, inside a [class]) states; those that reached a line end
        # mark every state they passed as dead (bit 1 outside a class, bit 2
        # inside), so a later probe stops on reaching one.
        self._dead: bytearray | None = None

    def scan(self, i: int = 0, *, hole: bool = False) -> int:
        """Record spans from `i`. In a `hole` (an interpolation whose "{" the
        caller consumed), stop after the "}" that closes it."""
        if not hole:
            return self._scan(i, hole=False)
        if self._nesting >= _MAX_NESTING:
            return self.n
        self._nesting += 1
        try:
            return self._scan(i, hole=True)
        finally:
            self._nesting -= 1

    def _scan(self, i: int, *, hole: bool) -> int:
        text, n, lang = self.text, self.n, self.lang
        depth = 0
        regex_ok = True  # would a "/" here start a JavaScript regular expression?
        while i < n:
            ch = text[i]
            if ch.isspace():
                i += 1
                continue
            comment = ch == "/" and i + 1 < n and text[i + 1] in "/*"
            end: int | None
            if comment:
                if text[i + 1] == "/":
                    end = text.find("\n", i)
                    end = n if end < 0 else end
                else:
                    end = self._block_comment(i)
            elif _ident(ch, lang):
                j = self._word_end(i)
                end = self._prefixed(i, j)
                if end is None:
                    regex_ok = text[i:j] in _JS_REGEX_AFTER
                    i = j
                    continue
            else:
                end = self._literal(i, regex_ok)
            if end is not None:
                if self._nesting == 0:
                    self.spans.append((i, end, comment))
                i = end
                if not comment:
                    regex_ok = False
                continue
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
        """End of the word whose first character is `text[i]`."""
        text, n = self.text, self.n
        number = self.lang == "cpp" and text[i].isdigit()
        j = i + 1
        while True:
            match = self._word.match(text, j)
            if match:
                j = match.end()
            if number and j + 1 < n and text[j] == "'" and text[j + 1].isalnum():
                j += 1  # a digit separator: 1'000'000
                continue
            return j

    def _prefixed(self, i: int, j: int) -> int | None:
        """End of a literal whose prefix is the word `text[i:j]` (Rust's
        b"x", r#"x"#, C++'s u8"x", R"(x)"), or None if no literal starts there."""
        text, n, lang = self.text, self.n, self.lang
        if lang not in ("rs", "cpp") or j >= n or text[j] not in "\"'#":
            return None
        word, q = text[i:j], text[j]
        if lang == "rs":
            if word in ("r", "br", "cr") and q in ('"', "#"):
                return self._rust_raw(j)
            if word in ("b", "c") and q == '"':
                return self._quoted(j, '"', multiline=True)
            if word == "b" and q == "'":
                return self._quoted(j, "'", multiline=False)
        elif q == '"' and word in _CPP_RAW_PREFIXES:
            return self._cpp_raw(j)
        elif q in ('"', "'") and word in _CPP_STRING_PREFIXES:
            return self._quoted(j, q, multiline=False)
        return None

    def _literal(self, i: int, regex_ok: bool) -> int | None:
        """End of the unprefixed literal starting at `i`, or None if none starts there."""
        text, lang = self.text, self.lang
        ch = text[i]
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
        dead = self._dead
        passed: list[int] = []  # states read, as position * 2 + in_class
        j, in_class = i + 1, False
        while j < n:
            if dead is not None and dead[j] & (1 << in_class):
                break  # an earlier probe went on from here to a line end
            passed.append(j * 2 + in_class)
            c = text[j]
            if c == "\\":
                j += 2
                continue
            if c == "\n":
                break  # no closing "/" on this line: it was a division
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
        if dead is None:
            dead = self._dead = bytearray(n)
        for state in passed:
            dead[state >> 1] |= 1 << (state & 1)
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
        if self._cs_fail[0] <= i < self._cs_fail[1]:
            return None
        j, interpolated, verbatim = i, False, False
        while j < n and text[j] in "@$":
            interpolated |= text[j] == "$"
            verbatim |= text[j] == "@"
            j += 1
        if j >= n or text[j] != '"':
            self._cs_fail = (i, j)
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
    pieces: list[str] = []
    done = 0
    for start, end, comment in scanner.spans:
        end = min(end, len(text))
        piece = text[start:end]
        pieces.append(text[done:start])
        pieces.append("\n".join(" " * len(line) for line in piece.split("\n")) if comment else "0" * len(piece))
        done = end
    pieces.append(text[done:])
    return "".join(pieces)


def uncomment(text: str, lang: str, *, masked: str | None = None) -> str:
    """`text` with comments blanked but literals kept, character for character."""
    masked = mask(text, lang) if masked is None else masked
    return "".join(" " if m == " " and not c.isspace() else c for c, m in zip(text, masked, strict=True))


def matching(masked: str, i: int) -> int:
    """Index of the bracket that closes the one at `masked[i]`, or -1.

    Only brackets of the same kind count, as the kinds need not nest properly
    in code `mask` got wrong. Every bracket of a text is matched at once, and
    the last few texts are remembered, so a caller can ask about each bracket.
    """
    if masked[i] not in _PAIRS:
        raise KeyError(masked[i])
    return _closers(masked).get(i, -1)


@functools.lru_cache(maxsize=8)
def _closers(masked: str) -> dict[int, int]:
    """Each opening bracket's index -> the index of the bracket that closes it."""
    found: dict[int, int] = {}
    open_at: dict[str, list[int]] = {opener: [] for opener in _PAIRS}
    for match in _BRACKETS.finditer(masked):
        ch, j = match.group(), match.start()
        if ch in open_at:
            open_at[ch].append(j)
        elif open_at[_OPENERS[ch]]:
            found[open_at[_OPENERS[ch]].pop()] = j
    return found


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
    last = -1  # the current piece's last non-space character so far
    spaces = (0, -1)  # the last run of whitespace measured: masked[a:b], b at a non-space or `end`

    def emit(stop: int) -> None:
        nonlocal begin, started, last
        if masked[begin:stop].strip():
            out.append(Chunk(begin, stop, text[begin:stop], masked[begin:stop]))
        begin, started, last = stop, False, -1

    def skip_space(j: int) -> int:
        # Asked with ever larger `j`: each run of whitespace is read once.
        nonlocal spaces
        if not spaces[0] <= j <= spaces[1]:
            spaces = (j, _skip_space(masked, j, end))
        return spaces[1]

    while i < end:
        ch = masked[i]
        if not ch.isspace():
            last = i
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
                j = skip_space(i + 1)
                if j < end and masked[j] == ";":
                    emit(j + 1)
                    i = j + 1
                    continue
                if j < end and (masked[j] in _AFTER_BRACE[lang] or _word_at(masked, j) in after_words):
                    i += 1
                    continue
                emit(i + 1)
        elif depth == 0 and (
            ch == ";"
            or (
                ch == "\n"
                and asi
                and started
                and _statement_ends(masked, begin, last, skip_space(i + 1), end, lang)
            )
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


def _statement_ends(masked: str, begin: int, last: int, after: int, end: int, lang: str) -> bool:
    """Does a line break end the statement, as automatic semicolons would?

    The statement started at `begin`; `last` is its last non-space character
    before the break (-1 if none) and `after` the first one after it (or `end`).
    """
    if last < begin:
        return False
    c = masked[last]
    complete = c.isalnum() or c in "_$)]}" or (last > begin and masked[last - 1 : last + 1] in ("++", "--"))
    if lang == "go" or not complete:
        return complete
    if after >= end:
        return True
    if masked[after] in _JS_CONTINUES and not masked.startswith(("++", "--"), after):
        return False
    return _word_at(masked, after) not in _JS_NOT_CONTINUED


def defines(code: str, name: str, lang: str, *, masked: str | None = None) -> bool:
    """Does `code` define a function (or, in JavaScript, a binding) called `name`?"""
    masked = mask(code, lang) if masked is None else masked
    word = re.escape(name)
    if lang in ("js", "ts"):
        return _js_defines(masked, word)
    if lang == "go":
        return re.search(rf"(?m)^[ \t]*func\s+{word}\s*[\[(]", masked) is not None
    if lang == "rs":
        return re.search(rf"(?<![\w.])fn\s+{word}\s*[<(]", masked) is not None
    return find_definition(masked, name) >= 0


def _js_defines(masked: str, word: str) -> bool:
    """`function word(`, or `const word = ...` (`let`, `var`, maybe with a
    `: Type` before the "=") at the code's top level, but not `word == ...`
    or `word => ...`.

    The top level is the shallowest brace depth. Code that closes a "}" it
    never opened is the body of a function, and a binding in that body is a
    local variable that shares the function's name (`let intersection = ...`
    in `intersection`), not the function.
    """
    if re.search(rf"(?<![\w$.])function\s*(?:\*\s*)?{word}\s*[<(]", masked):
        return True
    stops: list[int] | None = None  # where each "=" and ";" is, once a type needs them
    levels: tuple[list[int], list[int], int] | None = None  # once a binding needs them
    for match in re.finditer(rf"(?=(?<![\w$.])(?:const|let|var)\s+{word}\s*([:=]))", masked):
        at = match.start(1)
        if masked[at] == ":":  # the type runs to the first "=" or ";"
            if stops is None:
                stops = [stop.start() for stop in re.finditer(r"[=;]", masked)]
            k = bisect.bisect_right(stops, at)
            if k == len(stops) or masked[stops[k]] != "=":
                continue
            at = stops[k]
        if masked[at + 1 : at + 2] in ("=", ">"):
            continue
        if levels is None:
            levels = _brace_depths(masked)
        braces, depths, top = levels
        k = bisect.bisect_left(braces, match.start())
        if (depths[k - 1] if k else 0) == top:
            return True
    return False


def _brace_depths(masked: str) -> tuple[list[int], list[int], int]:
    """Where each "{" and "}" is, the depth after it, and the shallowest depth (0 or less)."""
    braces: list[int] = []
    depths: list[int] = []
    depth = top = 0
    for brace in re.finditer(r"[{}]", masked):
        depth += 1 if brace.group() == "{" else -1
        top = min(top, depth)
        braces.append(brace.start())
        depths.append(depth)
    return braces, depths, top


def find_definition(masked: str, name: str) -> int:
    """C++, Java, C#: where a typed function definition `T name(...) {` starts, or -1.

    `name` must follow a type (a word that is not a keyword, or a closing ">",
    "]", "*", "&" or "?") and its parameter list must be followed by a body,
    so calls and prototypes do not count.
    """
    for match in re.finditer(rf"(?<![\w$]){re.escape(name)}\s*\(", masked):
        k = match.start()
        while k > 0 and masked[k - 1].isspace():
            k -= 1
        if k == 0:
            continue
        prev = masked[k - 1]
        if prev.isalnum() or prev == "_":
            w = k - 1
            while w > 0 and (masked[w - 1].isalnum() or masked[w - 1] == "_"):
                w -= 1
            if masked[w:k] in _NOT_A_TYPE:
                continue
        elif prev not in ">]*&?":
            continue
        close = matching(masked, match.end() - 1)
        if close >= 0 and _BODY.match(masked, close + 1, close + 400):
            return match.start()
    return -1

"""Turn documentation files into short, cited passages for keyword retrieval.

Pure functions over strings: no file access, network or execution. Each
converter turns one format (Markdown/MDX, reStructuredText, HTML or plain
text) into a flat list of `Block`s (headings, prose and code), and
`chunk_blocks` packs those into `Chunk`s of at most `MAX_CHUNK_CHARS`, each
labelled with its section path, such as "Physics > Using CharacterBody2D".

Markup is stripped so that search matches words rather than syntax and the
model reads clean prose. Code is kept verbatim in Markdown fences, because its
indentation and punctuation carry meaning. Includes and links are never
followed: a passage holds only what its own file says.
"""

from __future__ import annotations

import html
import re
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from functools import partial
from html.parser import HTMLParser
from pathlib import PurePosixPath
from typing import Literal

# A passage is closed at the next heading once it holds MIN_CHUNK_CHARS;
# smaller sections share a passage, with their headings kept inline.
MIN_CHUNK_CHARS = 400
MAX_CHUNK_CHARS = 1500
MAX_HEADING_CHARS = 120
MAX_SECTION_CHARS = 300
SECTION_SEPARATOR = " > "
# Each reStructuredText nesting level (a directive, a list in a list) rescans
# its lines; anything deeper is kept as plain text, so time stays linear.
MAX_RST_DEPTH = 16


@dataclass(frozen=True)
class Block:
    """One heading, paragraph or code block of a converted document."""

    kind: Literal["heading", "text", "code"]
    text: str
    level: int = 0
    lang: str = ""


@dataclass(frozen=True)
class Chunk:
    """A passage of at most `MAX_CHUNK_CHARS` and the section path it sits under."""

    section: str
    body: str


# -- shared helpers --------------------------------------------------------

_LANG = re.compile(r"[\w+#.-]{1,30}")
_BLANK_LINES_AT_START = re.compile(r"\A(?:[ \t]*\n)+")
_SENTENCE_END = re.compile(r"(?<=[.!?\u3002])\s+")
_CLAUSE_END = re.compile(r"(?<=[,;:])\s+")
_TEXT_SEPARATORS = ((re.compile(r"\n"), "\n"), (_SENTENCE_END, " "), (_CLAUSE_END, " "))


def code_lang(word: str) -> str:
    """The leading language name of a fence info string or class, e.g. `rust,ignore` -> `rust`."""
    match = _LANG.match(word.strip())
    lang = match.group(0).lower() if match else ""
    return "" if lang in {"default", "none", "text"} else lang


def fence(code: str, lang: str = "") -> str:
    """Wrap code in a Markdown fence longer than any backtick fence inside it."""
    runs = re.findall(r"^[ \t]*(`{3,})", code, flags=re.M)
    ticks = "`" * max([3, *(len(run) + 1 for run in runs)])
    return f"{ticks}{lang}\n{code}\n{ticks}"


def clean_heading(text: str) -> str:
    text = re.sub(r"\{#[^{}]*\}\s*$", "", text)
    text = " ".join(text.replace("`", "").replace("*", "").split())
    if len(text) > MAX_HEADING_CHARS:
        text = text[: MAX_HEADING_CHARS - 1].rstrip() + "\u2026"
    return text


def section_path(headings: Sequence[str]) -> str:
    path = SECTION_SEPARATOR.join(headings)
    if len(path) > MAX_SECTION_CHARS and len(headings) > 2:
        path = SECTION_SEPARATOR.join((headings[0], "\u2026", headings[-1]))
    return path[:MAX_SECTION_CHARS]


def _trim_lines(text: str) -> str:
    """Drop blank lines at both ends, keeping the first line's indentation."""
    return _BLANK_LINES_AT_START.sub("", text).rstrip()


def _indent(line: str) -> int:
    expanded = line.expandtabs(8)
    return len(expanded) - len(expanded.lstrip())


def _strip_indent(line: str, columns: int) -> str:
    """Remove up to `columns` columns of leading whitespace, keeping deeper tabs."""
    column = 0
    for position, char in enumerate(line):
        if column >= columns or char not in " \t":
            return line[position:]
        column = column + 1 if char == " " else (column // 8 + 1) * 8
    return ""


def _pack(units: Sequence[str], limit: int, joiner: str) -> list[str]:
    """Greedily join units into pieces of at most `limit` characters."""
    pieces: list[str] = []
    current: str | None = None
    for unit in units:
        if current is None:
            current = unit
        elif len(current) + len(joiner) + len(unit) <= limit:
            current += joiner + unit
        else:
            pieces.append(current)
            current = unit
    if current is not None:
        pieces.append(current)
    return pieces


def _hard_split(text: str, limit: int) -> list[str]:
    pieces = []
    start = 0
    while len(text) - start > limit:
        cut = text.rfind(" ", start + limit // 2, start + limit)
        cut = cut if cut > start else start + limit
        pieces.append(text[start:cut])
        start = cut
        while start < len(text) and text[start] == " ":
            start += 1
    return [*pieces, text[start:]] if start < len(text) else pieces


def _split(text: str, limit: int, separators: Sequence[tuple[re.Pattern[str], str]]) -> list[str]:
    """Split at lines, then sentences, then clauses, then spaces, as needed."""
    if len(text) <= limit:
        return [text]
    if not separators:
        return _hard_split(text, limit)
    pattern, joiner = separators[0]
    units: list[str] = []
    for part in pattern.split(text):
        units.extend(_split(part, limit, separators[1:]))
    return _pack(units, limit, joiner)


def text_pieces(text: str, limit: int = MAX_CHUNK_CHARS) -> list[str]:
    return [piece for piece in (part.strip() for part in _split(text, limit, _TEXT_SEPARATORS)) if piece]


def code_pieces(code: str, lang: str = "", limit: int = MAX_CHUNK_CHARS) -> list[str]:
    """Fence code, splitting it at line ends when it is too long for one passage."""
    whole = fence(code, lang)
    if len(whole) <= limit:
        return [whole]
    room = max(limit - (len(whole) - len(code)), 100)
    units: list[str] = []
    for line in code.split("\n"):
        units.extend(line[start : start + room] for start in range(0, len(line), room) or [0])
    pieces = (_trim_lines(piece) for piece in _pack(units, room, "\n"))
    return [fence(piece, lang) for piece in pieces if piece.strip()]


# -- chunking --------------------------------------------------------------


class _Packer:
    """Accumulates blocks into chunks while tracking the heading stack."""

    def __init__(self, title: str, root: bool) -> None:
        self.title = title
        self.stack: list[tuple[int, str]] = [(1, title)] if root else []
        self.section = self.path()
        self.parts: list[str] = []
        self.size = 0
        self.chunks: list[Chunk] = []

    def path(self) -> str:
        return section_path([text for _, text in self.stack]) or self.title

    def close(self) -> None:
        if self.parts:
            self.chunks.append(Chunk(self.section, "\n\n".join(self.parts)))
        self.parts, self.size = [], 0

    def add(self, piece: str) -> None:
        cost = len(piece) + (2 if self.parts else 0)
        if self.parts and self.size + cost > MAX_CHUNK_CHARS:
            self.close()
            self.section = self.path()
            cost = len(piece)
        self.parts.append(piece)
        self.size += cost

    def heading(self, level: int, text: str) -> None:
        while self.stack and self.stack[-1][0] >= level:
            self.stack.pop()
        self.stack.append((level, text))
        if not self.parts or self.size >= MIN_CHUNK_CHARS:
            self.close()
            self.section = self.path()
        else:
            self.add(f"{'#' * level} {text}")


def chunk_blocks(blocks: Sequence[Block], title: str) -> list[Chunk]:
    """Pack blocks into passages labelled with their section path.

    `title` names the document; it roots the section path when the document
    has no level-1 heading of its own.
    """
    title = clean_heading(title) or "Untitled"
    has_title = any(block.kind == "heading" and block.level == 1 and clean_heading(block.text) for block in blocks)
    packer = _Packer(title, root=not has_title)
    for block in blocks:
        if block.kind == "heading":
            text = clean_heading(block.text)
            if text:
                packer.heading(min(max(block.level, 1), 6), text)
        elif block.kind == "code":
            code = _trim_lines(block.text)
            if code.strip():
                for piece in code_pieces(code, block.lang):
                    packer.add(piece)
        else:
            for piece in text_pieces(block.text):
                packer.add(piece)
    packer.close()
    return packer.chunks


# -- Markdown and MDX ------------------------------------------------------

# These patterns see whole lines and paragraphs of files up to 4 MiB, so each must stay
# linear: a repeat stops at the next opener ([, {, <) instead of rescanning the rest of
# the text from every one, and no two parts can match the same character, such as a
# space or the "/" of "<a b />", or a failed match retries every split between them.
_MD_FENCE = re.compile(r"^([ \t]*)(`{3,}|~{3,})(.*)$")
_MD_FENCE_CLOSE = re.compile(r"^[ \t]*(`{3,}|~{3,})[ \t]*$")
_MD_ATX = re.compile(r"^ {0,3}(#{1,6})(?:[ \t]+(.*))?$")
_MD_SETEXT = re.compile(r"^ {0,3}(=+|-+)[ \t]*$")
_MD_BREAK = re.compile(r"^ {0,3}([-*_])(?:[ \t]*\1){2,}[ \t]*$")
_MD_LIST_OR_TABLE = re.compile(r"^[ \t]*(?:[-*+][ \t]|\d+[.)][ \t]|\|)")
_MD_TABLE_RULE = re.compile(r"^\|?[ \t]*:?-{2,}:?[ \t]*(?:\|[ \t]*:?-{2,}:?[ \t]*)*\|?$")
_MD_LINK_DEFINITION = re.compile(r"^ {0,3}\[[^\]]+\]:[ \t]*\S")
_MD_CONTAINER_FENCE = re.compile(r"^[ \t]*:{3,}")
_MD_TAG_LINE = re.compile(r"^[ \t]*(?:</?[A-Za-z][\w.:-]*(?:[ \t][^<>]*|/)?>[ \t]*)+$")
_MD_OPEN_TAG_START = re.compile(r"^[ \t]*<[A-Za-z][\w.:-]*(?:[ \t][^<>]*)?$")
_MDX_STATEMENT = re.compile(r"^(?:import|export)\b")
_MD_COMMENT = re.compile(r"<!--.*?(?:-->|\Z)|\{/\*.*?(?:\*/\}|\Z)", re.S)
_MD_FRONT_TITLE = re.compile(r"^title[ \t]*[:=](.*)$")

_BACKTICKS = re.compile(r"`+")
_MD_URL = r"\([^()\[]*(?:\([^()\[]*\)[^()\[]*)*\)"  # one level of nested parentheses
_MD_IMAGE = re.compile(rf"!\[[^\[\]]*\](?:{_MD_URL}|\[[^\[\]]*\])")
_MD_LINK = re.compile(rf"\[([^\[\]]+)\]{_MD_URL}")
_MD_REFERENCE_LINK = re.compile(r"\[([^\[\]]+)\]\[[^\[\]]*\]")
_MD_AUTOLINK = re.compile(r"<((?:https?|mailto):[^<>\s]+)>")
_MD_MACRO = re.compile(
    r"""\{\{\s*([\w-]+)\s*(?:\((?:\s*(?:"([^"{}\n]*)"|'([^'{}\n]*)'))?[^{}]*?\)\s*)?\}\}"""
)
_MD_MACRO_KEEP = re.compile(r"(?i)(?:x?ref|element|header|method|status|glossary|attr)$")
_HTML_TAG = re.compile(r"</?(?:[a-z][\w.:-]*|[A-Z][\w.:-]+)(?:\s[^<>]*)?/?>")
_MD_ESCAPE = re.compile(r"\\([!-/:-@\[-`{-~])")
_PLACEHOLDER = re.compile(r"\x00(\d+)\x01")
_INNER_SPACES = re.compile(r"(?<=\S)[ \t]{2,}")


def _protect(pattern: re.Pattern[str], text: str, render: Callable[[re.Match[str]], str]):
    """Swap matches for placeholders so later rewrites cannot touch them."""
    saved: list[str] = []

    def stash(match: re.Match[str]) -> str:
        saved.append(render(match))
        return f"\x00{len(saved) - 1}\x01"

    return pattern.sub(stash, text), saved


def _restore(text: str, saved: list[str]) -> str:
    return _PLACEHOLDER.sub(lambda match: saved[int(match.group(1))], text)


def _protect_code_spans(text: str) -> tuple[str, list[str]]:
    """Placeholders for Markdown code spans: a backtick run up to the next run of the
    same length. One pass, where a regex would rescan to the end from each unclosed run."""
    runs = [match.span() for match in _BACKTICKS.finditer(text)]
    closers: list[int | None] = [None] * len(runs)
    next_of_length: dict[int, int] = {}
    for number in range(len(runs) - 1, -1, -1):
        length = runs[number][1] - runs[number][0]
        closers[number] = next_of_length.get(length)
        next_of_length[length] = number
    saved: list[str] = []
    parts: list[str] = []
    position = number = 0
    while number < len(runs):
        closer = closers[number]
        if closer is None:
            number += 1
            continue
        parts += [text[position : runs[number][0]], f"\x00{len(saved)}\x01"]
        saved.append(text[runs[number][0] : runs[closer][1]])
        position, number = runs[closer][1], closer + 1
    parts.append(text[position:])
    return "".join(parts), saved


def _tidy_lines(text: str) -> str:
    lines = (_INNER_SPACES.sub(" ", line).rstrip() for line in text.split("\n"))
    return "\n".join(line for line in lines if line.strip())


def _md_macro(match: re.Match[str]) -> str:
    """MDN-style `{{domxref("Window.fetch")}}` keeps its target; other macros vanish."""
    keep = _MD_MACRO_KEEP.search(match.group(1))
    return (match.group(2) or match.group(3) or "") if keep else ""


def _md_inline(text: str) -> str:
    text, spans = _protect_code_spans(text)
    text = _MD_IMAGE.sub("", text)
    text = _MD_LINK.sub(r"\1", text)
    text = _MD_REFERENCE_LINK.sub(r"\1", text)
    text = _MD_AUTOLINK.sub(r"\1", text)
    text = _MD_MACRO.sub(_md_macro, text)
    text = _HTML_TAG.sub("", text)
    text = html.unescape(_MD_ESCAPE.sub(r"\1", text))
    return _tidy_lines(_restore(text, spans))


def _front_matter(lines: list[str]) -> tuple[str, int]:
    """YAML (---) or TOML (+++) front matter: keep its title, skip the rest."""
    if not lines or lines[0].strip() not in ("---", "+++"):
        return "", 0
    marker = lines[0].strip()
    for end in range(1, min(len(lines), 200)):
        if lines[end].strip() == marker:
            for line in lines[1:end]:
                match = _MD_FRONT_TITLE.match(line.strip())
                if match:
                    title = match.group(1).strip()
                    if len(title) > 1 and title[0] == title[-1] and title[0] in "'\"":
                        title = title[1:-1].strip()
                    return title, end + 1
            return "", end + 1
    return "", 0


def _atx_text(text: str) -> str:
    """Heading text without its optional closing #s, which need a space before them."""
    text = text.rstrip(" \t")
    bare = text.rstrip("#")
    return bare.rstrip(" \t") if bare != text and (not bare or bare[-1] in " \t") else text


def _opens_fence(line: str) -> re.Match[str] | None:
    match = _MD_FENCE.match(line)
    if match and (match.group(2)[0] == "~" or "`" not in match.group(3)):
        return match
    return None


def _closes_fence(line: str, marker: str) -> bool:
    match = _MD_FENCE_CLOSE.match(line)
    return bool(match and match.group(1)[0] == marker[0] and len(match.group(1)) >= len(marker))


def _without_comments(lines: list[str]) -> list[str]:
    """Drop HTML and MDX comments, which may span lines, outside code fences."""
    out: list[str] = []
    pending: list[str] = []
    marker = ""

    def flush() -> None:
        if pending:
            out.extend(_MD_COMMENT.sub("", "\n".join(pending)).split("\n"))
            pending.clear()

    for line in lines:
        if marker:
            out.append(line)
            if _closes_fence(line, marker):
                marker = ""
            continue
        opened = _opens_fence(line)
        if opened:
            flush()
            out.append(line)
            marker = opened.group(2)
        else:
            pending.append(line)
    flush()
    return out


def _skip_statement(lines: list[str], start: int) -> int:
    """Skip an MDX import/export statement, following its brackets over lines."""
    depth = 0
    end = start
    while end < len(lines) and end - start < 200:
        line = lines[end]
        depth += sum(line.count(c) for c in "{([") - sum(line.count(c) for c in "})]")
        end += 1
        if depth <= 0:
            break
    return end


def markdown_blocks(text: str, *, mdx: bool = False) -> tuple[str, list[Block]]:
    lines = text.split("\n")
    title, start = _front_matter(lines)
    lines = _without_comments(lines[start:])
    blocks: list[Block] = []
    paragraph: list[str] = []
    last_paragraph_line = -2

    def flush() -> None:
        if paragraph:
            body = _md_inline("\n".join(paragraph))
            if body:
                blocks.append(Block("text", body))
            paragraph.clear()

    i = 0
    while i < len(lines):
        line = lines[i]
        opened = _opens_fence(line)
        if opened:
            flush()
            indent, marker, info = opened.groups()
            code: list[str] = []
            i += 1
            while i < len(lines) and not _closes_fence(lines[i], marker):
                code.append(_strip_indent(lines[i], _indent(indent)))
                i += 1
            blocks.append(Block("code", "\n".join(code), lang=code_lang(info)))
            i += 1
            continue
        stripped = line.strip()
        heading = _MD_ATX.match(line) if stripped else None
        if not stripped:
            flush()
        elif heading:
            flush()
            blocks.append(Block("heading", _md_inline(_atx_text(heading.group(2) or "")), len(heading.group(1))))
        elif (
            _MD_SETEXT.match(line)
            and paragraph
            and last_paragraph_line == i - 1
            and not _MD_LIST_OR_TABLE.match(paragraph[-1])
        ):
            heading_text = paragraph.pop()
            flush()
            blocks.append(Block("heading", _md_inline(heading_text), 1 if stripped[0] == "=" else 2))
        elif _MD_BREAK.match(line) or _MD_CONTAINER_FENCE.match(line):
            flush()
        elif _MD_TABLE_RULE.match(stripped) or _MD_LINK_DEFINITION.match(line):
            pass
        elif mdx and _MDX_STATEMENT.match(line):
            flush()
            i = _skip_statement(lines, i)
            continue
        elif _MD_TAG_LINE.match(line):
            flush()
        elif _MD_OPEN_TAG_START.match(line):
            flush()
            end = i + 1
            while end < len(lines) and end - i < 50 and ">" not in lines[end]:
                end += 1
            i = end + 1
            continue
        else:
            paragraph.append(line)
            last_paragraph_line = i
        i += 1
    flush()
    return title, blocks


# -- reStructuredText ------------------------------------------------------

_RST_ADORNMENT = re.compile(r"^([!-/:-@\[-`{-~])\1{2,}$")
_RST_DIRECTIVE = re.compile(r"^\.\.[ \t]+([A-Za-z0-9][\w.+-]*(?::[\w.+-]+)*)::(?:[ \t]+(.*))?$")
_RST_FIELD = re.compile(r"^:\w[\w .-]*:(?:[ \t]|$)")
_RST_GRID_BORDER = re.compile(r"^\+[-=+:]+\+$")
_RST_TABLE_BORDER = re.compile(r"^=+(?:[ \t]+=+)+$")

_RST_LITERAL = re.compile(r"``([\s\S]+?)``")
_RST_ROLE = re.compile(r":([A-Za-z][\w.+-]*(?::[\w.+-]+){0,3}):`([^`]+)`")
_RST_LINK = re.compile(r"`([^`<]*)<([^`<>]*)>`_{1,2}")
_RST_REFERENCE = re.compile(r"`([^`]+)`_{1,2}")
_RST_SUBSTITUTION = re.compile(r"\|([^|\s](?:[^|]*[^|\s])?)\|_{0,2}")
_RST_FOOTNOTE_REFERENCE = re.compile(r"\s?\[(?:#[\w-]*|\*|\d+)\]_")
_RST_ESCAPE = re.compile(r"\\([!-/:-@\[-`{-~])")

_RST_CODE = frozenset(
    {"code-block", "code", "sourcecode", "code-tab", "parsed-literal", "productionlist",
     "testcode", "testoutput", "doctest"}
)  # fmt: skip
_RST_LANG_FROM_ARGUMENT = frozenset({"code-block", "code", "sourcecode", "code-tab"})
_RST_DROP = frozenset(
    {"image", "figure", "toctree", "index", "meta", "raw", "contents", "video", "include",
     "literalinclude", "rst-class", "default-role", "role", "tabularcolumns", "graphviz",
     "inheritance-diagram", "sectionauthor", "moduleauthor", "codeauthor", "audio", "youtube",
     "vimeo", "currentmodule", "default-domain", "sectnum", "header", "footer", "target-notes",
     "title"}
)  # fmt: skip
_RST_ADMONITIONS = {
    "note": "Note", "warning": "Warning", "tip": "Tip", "important": "Important",
    "caution": "Caution", "danger": "Danger", "attention": "Attention", "hint": "Hint",
    "error": "Error", "seealso": "See also", "todo": "To do", "admonition": "",
    "versionadded": "New in version", "versionchanged": "Changed in version",
    "versionremoved": "Removed in version", "deprecated": "Deprecated since version",
}  # fmt: skip
_RST_OBJECTS = frozenset(
    {"function", "method", "class", "attribute", "data", "exception", "module", "decorator",
     "decoratormethod", "property", "describe", "object", "option", "envvar", "cmdoption",
     "type", "member", "macro", "struct", "union", "enum", "enumerator", "var",
     "staticmethod", "classmethod", "abstractmethod"}
)  # fmt: skip
_RST_LABELLED = frozenset(
    {"rubric", "topic", "sidebar", "centered", "table", "list-table", "csv-table", "tab",
     "group-tab", "math"}
)  # fmt: skip
_RST_CODE_ROLES = frozenset(
    {"func", "meth", "class", "attr", "data", "const", "obj", "mod", "exc", "code", "samp",
     "file", "envvar", "option", "program", "type", "macro", "member", "var", "struct",
     "union", "enum", "enumerator", "kbd", "command", "any", "token", "keyword", "expr",
     "texpr", "function", "concept"}
)  # fmt: skip


def _role_target(text: str) -> tuple[str, str] | None:
    """Role text `title <target>` as (title, target), or None when it has no target."""
    text = text.rstrip()
    start = text.rfind("<")
    target = text[start + 1 : -1]
    if start < 0 or not text.endswith(">") or not target or ">" in target:
        return None
    return text[:start].rstrip(), target


def _rst_role(match: re.Match[str]) -> str:
    role = match.group(1).rsplit(":", 1)[-1].lower()
    text = match.group(2)
    target = _role_target(text)
    if target:
        text = target[0] or target[1]
        if not any(char.isalnum() for char in text):
            return ""  # a permalink shown only as a symbol, as in Godot's class reference
    elif text.startswith("~"):
        text = text.rsplit(".", 1)[-1]
    text = text.lstrip("~!")
    return f"`{text}`" if role in _RST_CODE_ROLES else text


def _rst_inline(text: str) -> str:
    text, literals = _protect(_RST_LITERAL, text, lambda match: f"`{match.group(1)}`")
    text = _RST_ROLE.sub(_rst_role, text)
    text = _RST_LINK.sub(lambda match: match.group(1).rstrip() or match.group(2), text)
    text = _RST_REFERENCE.sub(r"\1", text)
    text = _RST_SUBSTITUTION.sub(r"\1", text)
    text = _RST_FOOTNOTE_REFERENCE.sub("", text)
    text = _RST_ESCAPE.sub(r"\1", text.replace("\\ ", ""))
    return _tidy_lines(_restore(text, literals))


@dataclass
class _RstState:
    """Section levels follow the order adornment styles first appear in."""

    levels: dict[tuple[str, bool], int] = field(default_factory=dict)
    title: str = ""
    highlight: str = ""

    def level(self, char: str, overline: bool) -> int:
        key = (char, overline)
        if key not in self.levels:
            self.levels[key] = min(len(self.levels) + 1, 6)
        return self.levels[key]


def _rst_block_end(lines: list[str], start: int) -> int:
    """End of the indented block from `start`: until a non-blank, unindented line."""
    end = start
    while end < len(lines) and (not lines[end].strip() or _indent(lines[end]) > 0):
        end += 1
    return end


def _rst_dedent(lines: list[str]) -> list[str]:
    widths = [_indent(line) for line in lines if line.strip()]
    if not widths:
        return []
    base = min(widths)
    return [_strip_indent(line, base) for line in lines]


def _rst_explicit(first: str, rest: list[str], out: list[Block], state: _RstState, depth: int) -> None:
    """Directives keep code and prose; comments, targets and substitutions are dropped."""
    match = _RST_DIRECTIVE.match(first)
    if not match:
        return
    name, argument = match.group(1).lower(), (match.group(2) or "").strip()
    base = name.rsplit(":", 1)[-1]
    body = _rst_dedent(rest)
    options = 0
    while options < len(body) and body[options].strip() and _RST_FIELD.match(body[options]):
        options += 1
    body = body[options:]
    if name == "highlight":
        state.highlight = code_lang(argument)
    elif base in _RST_CODE and ":" not in name:
        lang = code_lang(argument) if base in _RST_LANG_FROM_ARGUMENT and argument else ""
        code = _trim_lines("\n".join(body))
        if code.strip():
            out.append(Block("code", code, lang=lang or state.highlight))
    elif base in _RST_DROP or base.startswith("auto"):
        return
    elif base in _RST_ADMONITIONS and ":" not in name:
        label = _RST_ADMONITIONS[base]
        if base.startswith("version") or base == "deprecated":
            head = f"{label} {argument}:" if argument else f"{label}:"
        elif not label:
            head = f"{argument}:" if argument else ""
        else:
            head = f"{label}: {argument}" if argument else f"{label}:"
        if head:
            out.append(Block("text", _rst_inline(head)))
        _rst_body(body, out, state, depth + 1)
    elif ":" in name or base in _RST_OBJECTS:
        if argument:
            out.append(Block("text", f"`{argument}`"))
        _rst_body(body, out, state, depth + 1)
    else:
        if argument and base in _RST_LABELLED:
            out.append(Block("text", _rst_inline(argument)))
        _rst_body(body, out, state, depth + 1)


def _rst_title(lines: list[str], i: int) -> tuple[str, str, bool, int] | None:
    """Return (title, adornment char, has overline, lines used) for a section title."""
    stripped = lines[i].strip()
    if _RST_ADORNMENT.match(stripped) and i + 2 < len(lines):
        title, under = lines[i + 1].strip(), lines[i + 2].strip()
        if (
            title
            and not _RST_ADORNMENT.match(title)
            and under[:1] == stripped[0]
            and _RST_ADORNMENT.match(under)
        ):
            return title, stripped[0], True, 3
    if i + 1 < len(lines) and not _RST_ADORNMENT.match(stripped) and not stripped.startswith(".."):
        under = lines[i + 1].rstrip()
        if (
            under
            and not under[0].isspace()
            and _RST_ADORNMENT.match(under)
            and len(under) >= min(len(stripped), 4)
        ):
            return stripped, under[0], False, 2
    return None


def _rst_body(lines: list[str], out: list[Block], state: _RstState, depth: int = 0) -> None:
    if depth > MAX_RST_DEPTH:
        text = _rst_inline("\n".join(line.strip() for line in lines).strip())
        if text:
            out.append(Block("text", text))
        return
    paragraph: list[str] = []
    literal_next = False

    def flush() -> None:
        nonlocal literal_next
        if not paragraph:
            return
        text = "\n".join(paragraph).rstrip()
        paragraph.clear()
        if text.endswith("::"):
            literal_next = True
            head = text[:-2]  # "::" alone vanishes, "Text ::" keeps "Text", "Text::" keeps "Text:"
            text = head.rstrip() if not head or head.endswith((" ", "\t", "\n")) else text[:-1]
        text = _rst_inline(text)
        if text:
            out.append(Block("text", text))

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if not stripped:
            flush()
            i += 1
            continue
        if _indent(line):
            if paragraph:  # a definition, or a list item continued on the next line
                paragraph.append(stripped)
                i += 1
                continue
            end = _rst_block_end(lines, i)
            block = _rst_dedent(lines[i:end])
            if literal_next:
                code = _trim_lines("\n".join(block))
                if code.strip():
                    out.append(Block("code", code, lang=state.highlight))
            else:
                _rst_body(block, out, state, depth + 1)
            literal_next = False
            i = end
            continue
        literal_next = False
        title = _rst_title(lines, i)
        if title:
            flush()
            text, char, overline, used = title
            out.append(Block("heading", _rst_inline(text), state.level(char, overline)))
            state.title = state.title or text
            i += used
            continue
        if _RST_ADORNMENT.match(stripped):
            flush()
        elif _RST_GRID_BORDER.match(stripped) or _RST_TABLE_BORDER.match(stripped):
            pass
        elif depth == 0 and not state.title and _RST_FIELD.match(stripped):
            pass  # document metadata before the title, such as Godot's `:github_url:`
        elif stripped == ".." or stripped.startswith((".. ", "..\t")):
            flush()
            end = _rst_block_end(lines, i + 1)
            _rst_explicit(stripped, lines[i + 1 : end], out, state, depth)
            i = end
            continue
        else:
            paragraph.append(line)
        i += 1
    flush()


def rst_blocks(text: str) -> tuple[str, list[Block]]:
    state = _RstState()
    blocks: list[Block] = []
    _rst_body(text.split("\n"), blocks, state)
    return state.title, blocks


# -- HTML ------------------------------------------------------------------

_HTML_SKIP = frozenset(
    {"script", "style", "noscript", "template", "svg", "math", "nav", "footer", "iframe",
     "button", "form", "select", "textarea", "canvas", "object", "embed", "audio", "video",
     "picture", "head", "dialog", "map"}
)  # fmt: skip
_HTML_VOID = frozenset(
    {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param",
     "source", "track", "wbr"}
)  # fmt: skip
# Permalink markers, copy buttons, line numbers, screen-reader-only text and
# rustdoc's "Source" links and "Expand description" toggles.
# Not applied to headings: Docusaurus marks its headings themselves `anchor`.
_HTML_NOISE_CLASSES = frozenset(
    {"headerlink", "anchor", "doc-anchor", "hash-link", "rightside", "sr-only",
     "visually-hidden", "screen-reader-text", "skip-link", "copybtn", "copy-button", "linenos",
     "lineno", "line-numbers", "test-arrow", "src", "hideme"}
)  # fmt: skip
_HTML_PARAGRAPH = frozenset(
    {"p", "div", "section", "article", "main", "header", "aside", "blockquote", "ul", "ol",
     "dl", "table", "figure", "figcaption", "details", "summary", "address", "fieldset",
     "body", "center", "caption"}
)  # fmt: skip
_HTML_LINE = frozenset({"li", "dt", "dd", "tr", "thead", "tbody", "tfoot"})
_HTML_HEADINGS = {f"h{level}": level for level in range(1, 7)}
_HTML_INLINE_CODE = frozenset({"code", "kbd", "samp", "tt"})
_HTML_LANG_PREFIXES = ("language-", "lang-", "highlight-", "sourcecode-")
_HTML_KNOWN_LANGS = frozenset(
    {"rust", "python", "py", "javascript", "js", "typescript", "ts", "jsx", "tsx", "c", "cpp",
     "csharp", "cs", "java", "kotlin", "swift", "go", "gdscript", "glsl", "bash", "sh",
     "shell", "console", "powershell", "json", "toml", "yaml", "html", "css", "sql", "lua",
     "ruby", "php", "dart"}
)  # fmt: skip


def _class_lang(classes: Sequence[str]) -> str:
    for name in classes:
        for prefix in _HTML_LANG_PREFIXES:
            if name.startswith(prefix):
                return code_lang(name[len(prefix) :])
        if name in _HTML_KNOWN_LANGS:
            return name
    return ""


@dataclass
class _Element:
    tag: str
    skip: bool = False
    main: bool = False
    chrome: bool = False
    lang: str = ""  # from its classes, else inherited from the nearest ancestor with one


class _HtmlConverter(HTMLParser):
    """Collects blocks from page content, leaving out navigation and page chrome.

    When the page marks its content with `<main>` or `role="main"`, only that
    is kept. Otherwise prose in `<header>` and `<aside>` is dropped.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.stack: list[_Element] = []
        self.open: Counter[str] = Counter()  # open elements by tag, so stray end tags cost O(1)
        self.skip = self.main = self.chrome = 0
        self.saw_main = self.redirect = False
        self.title = ""
        self.title_parts: list[str] | None = None
        self.found: list[tuple[Block, bool, bool]] = []
        self.lines: list[str] = []
        self.line: list[str] = []
        self.line_has_text = False
        self.prefix = ""
        self.heading: tuple[int, list[str]] | None = None
        self.code: list[str] | None = None
        self.code_lang = ""

    # Output buffers.

    def emit(self, block: Block) -> None:
        self.found.append((block, self.main > 0, self.chrome > 0))

    def write(self, text: str) -> None:
        if self.heading is not None:
            self.heading[1].append(text)
            return
        has_text = bool(text.strip())
        if self.prefix and has_text:
            if not self.line_has_text:
                self.line = [self.prefix]
            self.prefix = ""
        self.line.append(text)
        self.line_has_text = self.line_has_text or has_text

    def newline(self) -> None:
        text = " ".join("".join(self.line).split())
        if text:
            self.lines.append(text)
        self.line = []
        self.line_has_text = False

    def paragraph(self) -> None:
        self.newline()
        if self.lines and self.lines[-1] and not self.prefix:
            self.lines.append("")

    def flush(self) -> None:
        self.newline()
        text = "\n".join(self.lines).strip()
        self.lines, self.prefix = [], ""
        if text:
            self.emit(Block("text", text))

    def end_code(self) -> None:
        if self.code is not None:
            code = "".join(self.code)
            self.code = None
            code = code[1:] if code.startswith("\n") else code
            if code.strip():
                self.emit(Block("code", code, lang=self.code_lang))

    def end_heading(self) -> None:
        if self.heading is not None:
            level, parts = self.heading
            self.heading = None
            text = " ".join("".join(parts).split())
            if text:
                self.emit(Block("heading", text, level))

    # Parser callbacks.

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = {name: value or "" for name, value in attrs}
        if tag == "meta" and values.get("http-equiv", "").lower() == "refresh":
            self.redirect = True
        if tag == "title" and not self.title:
            self.title_parts = []
        if tag in _HTML_VOID:
            if not self.skip and tag == "br":
                if self.code is not None:
                    self.code.append("\n")
                else:
                    self.newline()
            elif not self.skip and tag == "hr" and self.code is None:
                self.paragraph()
            return
        classes = values.get("class", "").split()
        lang = _class_lang(classes) if classes else ""
        element = _Element(
            tag,
            skip=tag in _HTML_SKIP
            or (tag not in _HTML_HEADINGS and not _HTML_NOISE_CLASSES.isdisjoint(classes))
            or values.get("aria-hidden") == "true"
            or "hidden" in values,
            main=tag == "main" or values.get("role") == "main",
            chrome=tag in ("aside", "header"),
            lang=lang or (self.stack[-1].lang if self.stack else ""),
        )
        if (element.main or element.chrome) and not self.skip:
            self.flush()
        self.stack.append(element)
        self.open[tag] += 1
        self.skip += element.skip
        self.main += element.main
        self.chrome += element.chrome
        if self.skip:
            return
        self.saw_main = self.saw_main or element.main
        if self.code is not None:
            if tag == "code" and lang and not self.code_lang:
                self.code_lang = lang
        elif tag in _HTML_HEADINGS:
            self.flush()
            self.heading = (_HTML_HEADINGS[tag], [])
        elif tag == "pre":
            self.flush()
            self.code = []
            self.code_lang = element.lang
        elif tag in _HTML_INLINE_CODE:
            self.write("`")
        elif tag == "li":
            self.newline()
            self.prefix = "- "
        elif tag in ("td", "th"):
            if self.line_has_text:
                self.write(" | ")
        elif tag in _HTML_LINE:
            self.newline()
        elif tag in _HTML_PARAGRAPH:
            self.paragraph()

    def handle_endtag(self, tag: str) -> None:
        if tag == "title" and self.title_parts is not None:
            self.title = " ".join("".join(self.title_parts).split())
            self.title_parts = None
        # Stray end tags are ignored.
        if not self.open[tag]:
            return
        while self.stack:
            element = self.stack.pop()
            self.open[element.tag] -= 1
            if not self.skip:
                self.close_element(element.tag)
                if element.main or element.chrome:
                    self.flush()  # attribute buffered text to the region it came from
            self.skip -= element.skip
            self.main -= element.main
            self.chrome -= element.chrome
            if element.tag == tag:
                break

    def close_element(self, tag: str) -> None:
        if self.code is not None:
            if tag == "pre":
                self.end_code()
        elif tag in _HTML_HEADINGS:
            self.end_heading()
        elif tag in _HTML_INLINE_CODE:
            self.write("`")
        elif tag in _HTML_LINE:
            self.newline()
        elif tag in _HTML_PARAGRAPH:
            self.paragraph()

    def handle_data(self, data: str) -> None:
        if self.title_parts is not None:
            self.title_parts.append(data)
        if self.skip:
            return
        if self.code is not None:
            self.code.append(data)
        else:
            self.write(data)

    def close(self) -> None:
        super().close()
        self.end_code()
        self.end_heading()
        self.flush()

    def blocks(self) -> list[Block]:
        if self.redirect:
            return []
        if self.saw_main:
            return [block for block, main, _ in self.found if main]
        return [block for block, _, chrome in self.found if not chrome or block.kind == "heading"]


# html.parser without the CVE-2025-6069 fix (2025) rescans the rest of the page from every
# "<" whose construct never closes, so a crafted page could take hours. Openers after the
# last closer of their kind are text anyway, so they are escaped first. MS Office's
# <![if ...]> sections end at "]>"; other marked sections, such as CDATA, end at "]]>".
_MS_SECTION = r"(?i:if|else|endif)(?![-_.a-zA-Z0-9])"
# html.parser raises on <![ before any other name; HTML5 reads it as a bogus comment, which
# "<! [" is to html.parser too.
_HTML_ODD_SECTION = re.compile(r"<!\[(?!(?i:cdata|temp|ignore|include|rcdata|if|else|endif)(?![-_.a-zA-Z0-9]))")
_HTML_UNCLOSED = (
    (re.compile("<!--"), re.compile(r"--\s*>")),
    (re.compile(rf"<!\[(?!{_MS_SECTION})"), re.compile(r"]\s*]\s*>")),
    (re.compile(rf"<!\[(?={_MS_SECTION})"), re.compile(r"]\s*>")),
    (re.compile("<"), re.compile(">")),
)
# A start tag can still run past the last ">" inside a quoted value, to the end of the
# page or into a quote that never closes. This comment ends it there, in every state the
# tag can be in, so the rest of the page is one broken tag, as HTML5 has it. On a
# well-formed page it is only a comment.
_HTML_END = "<!--\"'-->"


def _escape_unclosed(text: str) -> str:
    text = _HTML_ODD_SECTION.sub("<! [", text)
    for opener, closer in _HTML_UNCLOSED:
        end = max((match.end() for match in closer.finditer(text)), default=0)
        if opener.search(text, end):
            text = text[:end] + opener.sub(lambda match: "&lt;" + match.group(0)[1:], text[end:])
    return text


def html_blocks(text: str) -> tuple[str, list[Block]]:
    parser = _HtmlConverter()
    parser.feed(_escape_unclosed(text) + _HTML_END)
    parser.close()
    return parser.title, parser.blocks()


# -- plain text and dispatch -----------------------------------------------


def text_blocks(text: str) -> tuple[str, list[Block]]:
    paragraphs = re.split(r"\n[ \t]*\n", text)
    return "", [Block("text", part.strip("\n")) for part in paragraphs if part.strip()]


_CONVERTERS: dict[str, Callable[[str], tuple[str, list[Block]]]] = {
    ".md": markdown_blocks,
    ".markdown": markdown_blocks,
    ".mdx": partial(markdown_blocks, mdx=True),
    ".rst": rst_blocks,
    ".rest": rst_blocks,
    ".html": html_blocks,
    ".htm": html_blocks,
    ".txt": text_blocks,
    ".text": text_blocks,
}
SUPPORTED_SUFFIXES = frozenset(_CONVERTERS)


def document_chunks(text: str, path: str) -> list[Chunk]:
    """Convert one file, chosen by the suffix of its `path`, into passages."""
    name = PurePosixPath(path)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    title, blocks = _CONVERTERS[name.suffix.lower()](text)
    return chunk_blocks(blocks, title or name.stem)

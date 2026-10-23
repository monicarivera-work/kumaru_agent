"""MultiPL-E — the HumanEval problems in seven more programming languages.

HumanEval asks for Python only. MultiPL-E (Cassano et al., 2023) translated its
problems, docstrings and unit tests into other languages, so one model can be
scored per language on the same problems: a gap between its Python and its
Rust score then says something about its Rust, not about harder problems.
`kumaru bench run multipl_e` runs eight benchmarks, one per language:
`multipl_e_py` (HumanEval itself, for the comparison), `multipl_e_js`, `_ts`,
`_cs`, `_java`, `_cpp`, `_go` and `_rs`. All use the 152 problems every
language has (MultiPL-E could not translate twelve into some languages), in
the same order, so a `-n` sample asks every language the same problems.

Asking. The prompt shows the problem as MultiPL-E gives it (a signature and a
docstring comment in the target language) and asks for the complete function
in a fenced block. A reply that only continues the function body is pasted
after the prompt, as MultiPL-E does with base models.

Assembling. Chat models answer with more than the function: imports, helpers,
a `main`, example calls, tests of their own. Each language's assembler keeps
the imports and helpers, drops what would clash with the hidden tests (`main`,
test functions, `package` lines, top-level statements), and joins the rest
with the tests into one program. `codeparse` finds the top-level pieces.

Running. `codeexec` compiles and runs each program in a fresh temporary
directory with a time limit per step and a scrubbed environment, and kills the
whole process tree at the end. A program passes only if it exits with 0 *and*
prints a random marker placed after its last test. That guards against
accidents, not adversaries: `--sandbox docker` runs every program in a
throwaway container without network access (docker/sandbox/Dockerfile).

Toolchains. Python; Node.js for JavaScript, and for TypeScript `tsc` when it
is installed, else Node 22.7+'s built-in type stripping (which runs the code
without type-checking it); the .NET 10 SDK for C#; a JDK plus the javatuples
jar the Java tests import; g++ or clang++ with libstdc++; Go; rustc. A canary
problem checks each toolchain before any model loads. A language whose
toolchain is missing is skipped with a hint, never scored 0.

Differences from MultiPL-E's own harness (see docs/BENCHMARKS.md): C# builds
with .NET 10 instead of Mono; Rust compiles as edition 2021 instead of 2015,
with warnings allowed; programs get 15 s to run (MultiPL-E: 5-30 s by
language) and 120 s to compile; Windows builds of C++ and Rust get the 8 MB
stack Linux gives them; and whole-function answers may bring their own
imports, which completions cannot (Go keeps only those the code still uses,
because Go refuses to compile an unused import).

Fixed tests. The pinned dataset predates one fix MultiPL-E has made to its
translator since: its C# tests compared List and Dictionary results with
`.Equals`, which .NET answers by reference, so 43 of the 152 C# problems
could not be passed (nuprl/MultiPL-E#71). `fix_tests` rewrites those
assertions as the fixed translator writes them, and loading fails unless
exactly those problems change. Kumaru applies only fixes MultiPL-E has made.

Sources: Cassano et al., 2023, "MultiPL-E: A Scalable and Polyglot Approach to
Benchmarking Neural Code Generation" (IEEE TSE); dataset `nuprl/MultiPL-E`
(MIT licence), configs `humaneval-<language>`, split `test`. Python: Chen et
al., 2021; dataset `openai/openai_humaneval`.
"""

from __future__ import annotations

import hashlib
import os
import platform
import re
import subprocess
import sys
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

from kumaru.core.errors import KumaruError
from kumaru.eval.benchmarks import codeexec
from kumaru.eval.benchmarks.base import BenchItem, Benchmark, Grade, load_hf_rows
from kumaru.eval.benchmarks.codeexec import Program, Step, ToolchainMissing
from kumaru.eval.benchmarks.codeparse import (
    Chunk,
    chunks,
    defines,
    find_definition,
    mask,
    matching,
    uncomment,
)
from kumaru.eval.benchmarks.extract import defines_python, extract_code, fences, open_fence
from kumaru.eval.benchmarks.humaneval import assemble_program

DATASET = "nuprl/MultiPL-E"
DATASET_REVISION = "28441b6024e71d4a1c1c0f6bf171c935cd5a43f2"
HUMANEVAL = "openai/openai_humaneval"
HUMANEVAL_REVISION = "7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544"
# HumanEval problems missing from at least one language's translation. Leaving
# them out everywhere keeps every language on the same 152 problems.
EXCLUDED = frozenset({12, 22, 32, 38, 50, 90, 103, 125, 128, 137, 148, 162})
N_PROBLEMS = 152
# Defects in the pinned tests that MultiPL-E has fixed since, in its translator
# but not in its dataset: per language, the issue and the problems whose tests
# `fix_tests` rewrites. Loading fails unless exactly these problems change.
TEST_FIXES: dict[str, tuple[str, frozenset[int]]] = {
    # The C# problems whose function returns a List or a Dictionary.
    "cs": (
        "nuprl/MultiPL-E#71",
        frozenset({
            1, 5, 6, 7, 9, 14, 17, 21, 25, 26, 29, 30, 33, 34, 37, 42, 58, 62, 68, 70, 74, 81,
            87, 88, 96, 100, 101, 104, 105, 106, 111, 113, 116, 117, 120, 123, 129, 130, 145,
            149, 152, 159, 163,
        }),
    ),
}

RUN_TIMEOUT_S = 15.0
COMPILE_TIMEOUT_S = 120.0
# How much of a failed program's output a result keeps.
STDERR_TAIL = 800

# The Java tests use org.javatuples (Pair, Triplet...), which MultiPL-E puts on
# the classpath. Kumaru checks the jar but never downloads it by itself.
JAVATUPLES = "javatuples-1.2.jar"
JAVATUPLES_URL = "https://repo1.maven.org/maven2/org/javatuples/javatuples/1.2/javatuples-1.2.jar"
JAVATUPLES_SHA256 = "2eda5b19d9820e1cc2f69fcd01639a715a673c11f8507e3d1ed593cf765d5e0a"
# Where the sandbox image keeps what is not on its PATH.
DOCKER_JAVATUPLES = "/opt/kumaru/javatuples-1.2.jar"
DOCKER_TSC = "/opt/kumaru/typescript/node_modules/typescript/lib/tsc.js"


@dataclass(frozen=True)
class Language:
    # MultiPL-E's language code, which names the benchmark (multipl_e_rs).
    key: str
    # Shown to the model and in reports.
    name: str
    # The code-fence tag the prompt uses.
    fence: str
    # Fence tags that count as this language when picking the answer's block.
    aliases: frozenset[str]


LANGUAGES: dict[str, Language] = {
    lang.key: lang
    for lang in (
        Language("py", "Python", "python", frozenset({"python", "py", "python3"})),
        Language("js", "JavaScript", "javascript", frozenset({"javascript", "js", "node", "jsx", "mjs", "cjs"})),
        Language("ts", "TypeScript", "typescript", frozenset({"typescript", "ts", "tsx"})),
        Language("cs", "C#", "csharp", frozenset({"csharp", "cs", "c#"})),
        Language("java", "Java", "java", frozenset({"java"})),
        Language("cpp", "C++", "cpp", frozenset({"cpp", "c++", "cc", "cxx", "hpp"})),
        Language("go", "Go", "go", frozenset({"go", "golang"})),
        Language("rs", "Rust", "rust", frozenset({"rust", "rs"})),
    )
}

PROMPT = (
    "Complete the following {language} function. Keep its name and signature, and return "
    "the complete function, plus any helper functions or imports it needs, in a single "
    "```{fence} code block. Do not include a main function, tests or example usage.\n\n"
    "```{fence}\n{prompt}```"
)

# -- reading the answer ---------------------------------------------------------

_FENCE_TAG = re.compile(r"[ \t]*([^\s`]*)")
# The functions MultiPL-E prompts end with; the last one is the problem's.
_ENTRY = {
    "js": r"(?<![\w$.])function\s+([A-Za-z_$][\w$]*)\s*\(",
    "ts": r"(?<![\w$.])function\s+([A-Za-z_$][\w$]*)\s*[<(]",
    "go": r"(?m)^func\s+(\w+)\s*[\[(]",
    "rs": r"(?<![\w.])fn\s+(\w+)\s*[<(]",
}


def _tag(tag: str) -> str:
    """A fence's language tag, normalised: "```rust,ignore" and "```{rust}" are Rust."""
    return re.split(r"[,{]", tag.strip("{}"), maxsplit=1)[0].lower()


def defines_entry(code: str, entry: str, lang: str) -> bool:
    """Does `code` define the function the problem asks for?"""
    if lang == "py":
        return defines_python(code, entry)
    return defines(code, entry, lang)


def extract_answer(reply: str, lang: str, entry: str) -> str:
    """The code in a reply.

    The fenced block that defines `entry` (the longest, if several do), else
    the longest block tagged with this language or untagged, else the longest
    block. An unterminated fence (an answer cut off by the token limit) yields
    the code after it; a reply without fences is used whole.
    """
    reply = reply or ""
    blocks = [(_tag(tag), body) for tag, body in fences(reply, _FENCE_TAG)]
    if not blocks:
        code = open_fence(reply)
        return reply if code is None else code
    aliases = LANGUAGES[lang].aliases
    candidates = [body for tag, body in blocks if not tag or tag in aliases]
    candidates = candidates or [body for _, body in blocks]
    defining = [body for body in candidates if defines_entry(body, entry, lang)]
    return max(defining or candidates, key=len)


def solution_source(prompt: str, code: str, entry: str, lang: str) -> tuple[str, str]:
    """The model's solution as source code, and how it was read.

    "whole": the code defines the function itself. "continuation": it is a
    function body that completes the prompt's open function, the way
    MultiPL-E's completions do; the prompt goes first and any braces still
    open are closed.
    """
    if defines_entry(code, entry, lang):
        return code, "whole"
    source = prompt + code
    if not source.endswith("\n"):
        source += "\n"
    masked = mask(source, lang)
    missing = masked.count("{") - masked.count("}")
    return source + "}\n" * max(missing, 0), "continuation"


def entry_point(prompt: str, lang: str) -> str:
    """The name of the function a MultiPL-E prompt asks for: the last one it opens."""
    masked = mask(prompt, lang)
    if lang in _ENTRY:
        names = re.findall(_ENTRY[lang], masked)
    else:  # C++, Java, C#: the prompt ends with the signature
        last = next((line for line in reversed(masked.splitlines()) if line.strip()), "")
        names = re.findall(r"(\w+)\s*\(", last)[:1]
    if not names:
        raise KumaruError(f"Could not find the function a {LANGUAGES[lang].name} prompt asks for.")
    return names[-1]


# -- fixing the tests -----------------------------------------------------------


def fix_tests(lang: str, prompt: str, tests: str, entry: str) -> str:
    """`tests` with the defects in `TEST_FIXES` fixed, as MultiPL-E now writes them.

    nuprl/MultiPL-E#71: the C# tests check a List or Dictionary result with
    `.Equals`, which .NET answers by reference, so no answer can pass them. The
    translator has compared Lists with `SequenceEqual`, and Dictionaries by
    `Count` and `Except`, since January 2026 (commit 85833ac); the dataset on
    the Hub predates that. When the function returns a List or a Dictionary,
    each assertion `Debug.Assert(F(...).Equals(expected));` is rewritten as the
    fixed translator writes it, and one that is not of that form is an error.
    Any other problem's tests are kept as published.
    """
    if lang != "cs":
        return tests
    signature = next((line for line in reversed(mask(prompt, lang).splitlines()) if line.strip()), "")
    returns = re.search(rf"\bstatic\s+(List|Dictionary)<[^(]*\s{re.escape(entry)}\s*\(", signature)
    if not returns:
        return tests
    masked = mask(tests, lang)
    call = re.compile(rf"{re.escape(entry)}\s*\(")
    pieces: list[str] = []
    done = 0
    for match in re.finditer(r"\bDebug\.Assert\(", masked):
        opened = match.end() - 1
        closed = matching(masked, opened)
        found = call.match(masked, opened + 1)
        called = matching(masked, found.end() - 1) if found else -1
        compared = called + len(".Equals(")
        rewritable = (
            called >= 0
            and closed >= 0
            and masked.startswith(".Equals(", called + 1)
            and matching(masked, compared) == closed - 1
        )
        if not rewritable:
            assertion = tests[match.start() :].split("\n", 1)[0][:200]
            raise KumaruError(
                f"Could not rewrite a C# test as {TEST_FIXES[lang][0]} does: {assertion}",
                hint=f"Has the dataset changed? Kumaru pins {DATASET} at revision {DATASET_REVISION}.",
            )
        left, right = tests[opened + 1 : called + 1], tests[compared + 1 : closed - 1]
        if returns.group(1) == "List":
            check = f"{left}.SequenceEqual({right})"
        else:
            check = f"{left}.Count == {right}.Count && !{left}.Except({right}).Any()"
        pieces += [tests[done : opened + 1], check]
        done = closed
    return "".join(pieces) + tests[done:]


# -- shared assembly helpers ----------------------------------------------------


def _lead(chunk: Chunk) -> int:
    """Offset of a piece's first code character, after its comments and blank lines."""
    return len(chunk.code) - len(chunk.code.lstrip())


def _join(pieces: Iterable[str]) -> str:
    return "\n".join(piece.strip("\n") for piece in pieces if piece.strip())


def _drop_leading_brace(tests: str, lang: str) -> str:
    """MultiPL-E's C++, Rust, Java and C# tests start with the "}" that closes the
    prompt's function (its completions stop before it); a solution closes its own."""
    masked = mask(tests, lang)
    i = len(masked) - len(masked.lstrip())
    return tests[:i] + tests[i + 1 :] if masked.startswith("}", i) else tests


def _before_last_brace(text: str, lang: str, insertion: str) -> str:
    """`text` with `insertion` just before its last "}" (the end of the test function)."""
    i = mask(text, lang).rfind("}")
    if i < 0:
        raise KumaruError(f"These {LANGUAGES[lang].name} tests have no closing brace.")
    return text[:i] + insertion + text[i:]


def _is_main(chunk: Chunk, name: str) -> bool:
    """Is this piece a definition of `main`? The hidden tests bring their own."""
    if name not in chunk.code:  # most pieces; a reply can have one per character
        return False
    signature = chunk.code.split("{", 1)[0]
    return find_definition(signature + "{}", name) >= 0


# -- JavaScript and TypeScript --------------------------------------------------

# Top-level pieces worth keeping: declarations. Everything else at the top level
# (example calls, console.log, module.exports, "use strict") is dropped.
_JS_KEEP = re.compile(r"(?:async\s+)?function\b|class\b|const\b|let\b|var\b|@")
_TS_KEEP = re.compile(
    r"(?:async\s+)?function\b|(?:abstract\s+)?class\b|const\b|let\b|var\b|@|interface\b"
    r"|type\s+[A-Za-z_$]|enum\b|declare\b|namespace\b|module\s+[A-Za-z_$\"']"
)
_JS_EXPORT = re.compile(r"export\s+(?:default\s+)?(?=[\w@])")
_JS_IMPORT = re.compile(r"import\b(?!\s*[.(])")
_JS_MODULE = re.compile(r"\s*(['\"][^'\"]+['\"])")
_JS_ATTRIBUTES = re.compile(r"\s*(?:with|assert)\s*\{")
_TS_REQUIRE = "declare var require: any;"


def _js_from(s: str) -> tuple[str, str] | None:
    r"""`(clause, module)` for `import <clause> from '<module>'`, import
    attributes (`with {...}`) allowed; None for any other statement.

    What fullmatching
    `import\s*(.+?)\s*from\s*(['"][^'"]+['"])(?:\s*(?:with|assert)\s*\{[^}]*\})?`
    finds when the clause is not blank, by a scan: on a long statement that
    regex backtracks in quadratic time or worse. The clause ends at the first
    "from" after its first character that the rest of the statement fits. A
    blank clause (`import from 'm'`) imports nothing: None.
    """
    if not s.startswith("import"):
        return None
    clause_start = len(s) - len(s[6:].lstrip())
    inner_brace = s.rfind("}", 0, len(s) - 1)  # the attributes' {...} holds no "}"
    at = s.find("from", clause_start + 1)
    while at >= 0:
        module = _JS_MODULE.match(s, at + 4)
        if module:
            end = module.end()
            attributes = _JS_ATTRIBUTES.match(s, end) if s.endswith("}") else None
            if end == len(s) or (attributes is not None and attributes.end() - 1 > inner_brace):
                return s[6:at].strip(), module.group(1)
        at = s.find("from", at + 4)
    return None


def _js_import(statement: str) -> str:
    """An ES or TypeScript `import` as the CommonJS `require` it stands for ("" for types)."""
    s = statement.strip().rstrip(";").strip()
    match = re.fullmatch(r"import\s+([A-Za-z_$][\w$]*)\s*=\s*require\s*\((.*)\)", s, re.DOTALL)
    if match:
        return f"const {match.group(1)} = require({match.group(2).strip()});"
    if re.match(r"import\s+type\b", s):
        return ""
    match = re.fullmatch(r"import\s*(['\"][^'\"]+['\"])", s)
    if match:
        return f"require({match.group(1)});"
    found = _js_from(s)
    if found is None:
        return ""
    clause, module = found
    out = []
    # The named imports: from the first "{" to the last "}".
    first, last = clause.find("{"), clause.rfind("}")
    braces = clause[first + 1 : last] if 0 <= first < last else None
    for name in (clause[:first] if braces is not None else clause).split(","):
        name = name.strip()
        star = re.fullmatch(r"\*\s*as\s+([A-Za-z_$][\w$]*)", name)
        if star or re.fullmatch(r"[A-Za-z_$][\w$]*", name):
            out.append(f"const {star.group(1) if star else name} = require({module});")
    if braces is not None:
        names = []
        for spec in braces.split(","):
            spec = spec.strip()
            if not spec or re.match(r"type\s", spec):
                continue
            # (?<!\s): one try per run of spaces, not one per space.
            parts = re.split(r"(?<!\s)\s+as\s+", spec)
            names.append(parts[0] if len(parts) == 1 else f"{parts[0]}: {parts[1]}")
        if names:
            out.append(f"const {{ {', '.join(names)} }} = require({module});")
    return "\n".join(out)


def _assemble_js(lang: str, source: str, tests: str, marker: str) -> str:
    """JavaScript and TypeScript, run as CommonJS.

    Declarations are kept, imports become `require`s and other top-level
    statements are dropped. The tests run in a block of their own, so their
    `const assert` and `function test` cannot clash with the model's names.
    """
    keep = _TS_KEEP if lang == "ts" else _JS_KEEP
    kept: list[str] = []
    for chunk in chunks(source, lang):
        lead = _lead(chunk)
        text, head = chunk.text, chunk.code[lead:]
        export = _JS_EXPORT.match(head)
        if export:
            text = text[:lead] + text[lead + export.end() :]
            head = head[export.end() :]
        if _JS_IMPORT.match(head):
            kept.append(_js_import(uncomment(text, lang)))
        elif keep.match(head):
            kept.append(text)
    header = ""
    if lang == "ts":
        # `declare` is only allowed at the top level, not in the tests' block.
        header = _TS_REQUIRE + "\n"
        tests = tests.replace(_TS_REQUIRE, "", 1)
    finish = f"require('node:fs').writeSync(1, '\\n{marker}\\n');"
    return f"{header}{_join(kept)}\n\n{{\n{tests.strip(chr(10))}\n}}\n{finish}\n"


# -- Go -------------------------------------------------------------------------

_GO_PATH = re.compile(r"""["`]([^"`\n]+)["`]""")
_GO_ALIAS_START = re.compile(r"[A-Za-z_]")
_GO_TEST_FUNC = re.compile(r"func\s+(?:Test|Benchmark|Fuzz|Example)(?![a-z])\w*\s*\(")
# Names checked one by one for use; see _assemble_go.
_GO_MAX_ODD_NAMES = 64


def _go_specs(text: str) -> list[tuple[str, str]]:
    r"""(alias, path) for each quoted path in `text`, with the alias before it, if any.

    What findall with (?:([A-Za-z_]\w*|\.)\s+)?["`]([^"`\n]+)["`] finds, by a
    scan: that regex retries every letter of a long word, in quadratic time.
    """
    specs = []
    pos = 0
    while (path := _GO_PATH.search(text, pos)) is not None:
        start = space = path.start()
        while space > pos and text[space - 1].isspace():
            space -= 1
        alias = ""
        if pos < space < start:
            if text[space - 1] == ".":
                alias = "."
            else:
                word = space
                while word > pos and (text[word - 1].isalnum() or text[word - 1] == "_"):
                    word -= 1
                first = _GO_ALIAS_START.search(text, word, space)
                alias = text[first.start() : space] if first else ""
        specs.append((alias, path.group(1)))
        pos = path.end()
    return specs


def _go_imports(text: str) -> list[tuple[str, str]]:
    """(alias, path) for each package an `import` declaration names; alias "" if none."""
    return _go_specs(re.sub(r"^\s*import\b", "", uncomment(text, "go")))


def _go_name(path: str) -> str:
    """The name an import path binds by default: its last element, skipping a /vN suffix."""
    parts = [part for part in path.split("/") if part]
    if len(parts) > 1 and re.fullmatch(r"v\d+", parts[-1]):
        return parts[-2]
    return parts[-1] if parts else path


def _assemble_go(prompt: str, source: str, tests: str, marker: str) -> str:
    """Go: one `_test` package with the imports the code and the tests use, the
    model's code minus its `package` line and test functions, then the tests.

    Go refuses to compile an unused import, and a model's imports often served
    only the `main` it wrote, so its imports are kept only while still used.
    """
    package = re.search(r"(?m)^\s*package\s+(\w+)", mask(prompt, "go"))
    imports: dict[str, str] = {}
    for chunk in chunks(prompt, "go"):
        if re.match(r"import\b", chunk.head):
            for alias, path in _go_imports(chunk.text):
                imports.setdefault(path, alias)
    kept: list[str] = []
    wanted: list[tuple[str, str]] = []
    for chunk in chunks(source, "go"):
        if re.match(r"package\b", chunk.head):
            continue
        if re.match(r"import\b", chunk.head):
            wanted += _go_imports(chunk.text)
        elif not _GO_TEST_FUNC.match(chunk.head):
            kept.append(chunk.text)
    code = _join(kept)
    used = mask(code, "go") + "\n" + mask(tests, "go")
    # The names used as `name.X`, in one pass rather than one search per import.
    qualifiers = set(re.findall(r"(?<![\w.])(\w+)\s*\.", used))
    # Default names that are not identifiers ("yaml.v2") are searched for one
    # by one; past the first few (only a degenerate reply has more), they
    # count as unused.
    odd: dict[str, bool] = {}
    for alias, path in wanted:
        if path in imports:
            continue
        name = alias or _go_name(path)
        if re.fullmatch(r"\w+", name):
            is_used = name in qualifiers
        else:
            if name not in odd and len(odd) < _GO_MAX_ODD_NAMES:
                odd[name] = re.search(rf"(?<![\w.]){re.escape(name)}\s*\.", used) is not None
            is_used = odd.get(name, False)
        if alias in ("_", ".") or is_used:
            imports[path] = alias
    imports.setdefault("fmt", "")
    specs = "".join(f'\t{alias + " " if alias else ""}"{path}"\n' for path, alias in imports.items())
    tests = _before_last_brace(tests, "go", f'\tfmt.Print("\\n{marker}\\n")\n')
    name = package.group(1) if package else "main_test"
    return f"package {name}\n\nimport (\n{specs})\n\n{code}\n\n{tests.strip(chr(10))}\n"


# -- Rust -----------------------------------------------------------------------

_RS_VIS = r"(?:pub(?:\s*\([^)]*\))?\s+)?"
_RS_ATTRS = re.compile(r"(?:#\[[^\]]*\]\s*)*")
_RS_USE = re.compile(rf"{_RS_VIS}use\b")


def _rust_use_names(use: str) -> set[str]:
    """The names a `use` declaration binds (globs and `self` aside)."""
    body = re.sub(rf"^\s*{_RS_VIS}use\s+", "", use).strip().rstrip(";")
    names = set()
    for part in re.split(r"[{},]", body):
        part = part.strip()
        alias = re.search(r"\bas\s+(\w+)$", part)
        name = alias.group(1) if alias else part.split("::")[-1].strip()
        if name and name not in ("self", "*", "_"):
            names.add(name)
    return names


def _assemble_rs(prompt: str, source: str, tests: str, marker: str) -> str:
    """Rust: one crate with the model's items minus its `main` and its tests, the
    prompt's `use` lines that the model's own do not replace, then the tests' `main`."""
    inner: list[str] = []
    kept: list[str] = []
    names: set[str] = set()
    for chunk in chunks(source, "rs"):
        head = chunk.head
        if head.startswith("#!["):  # inner attributes must come first
            inner.append(chunk.text)
            continue
        attrs = _RS_ATTRS.match(head)
        assert attrs is not None  # the pattern matches the empty string
        rest = head[attrs.end() :]
        if "cfg(test" in attrs.group(0) or "#[test]" in attrs.group(0):
            continue
        if re.match(rf"{_RS_VIS}(?:mod\s+tests?\b|fn\s+main\s*\()", rest):
            continue
        if _RS_USE.match(rest):
            names |= _rust_use_names(rest)
        kept.append(chunk.text)
    uses = [
        chunk.text
        for chunk in chunks(prompt, "rs")
        if _RS_USE.match(chunk.head) and not _rust_use_names(chunk.head) & names
    ]
    finish = (
        "    { use std::io::Write; let mut kumaru_out = std::io::stdout(); "
        f'let _ = kumaru_out.write_all(b"\\n{marker}\\n"); let _ = kumaru_out.flush(); }}\n'
    )
    tests = _before_last_brace(_drop_leading_brace(tests, "rs"), "rs", finish)
    return _join([*inner, *uses, *kept, tests]) + "\n"


# -- C++ ------------------------------------------------------------------------

# Before the model's code. An abort (an uncaught exception, a failed check)
# exits with 134 at once, instead of reporting a crash to Windows.
_CPP_PRELUDE = """\
#include <cstdio>
#include <cstdlib>
#include <csignal>
static void kumaru_on_abort(int) { std::_Exit(134); }
static const int kumaru_abort_hook = (std::signal(SIGABRT, kumaru_on_abort), 0);
"""
# After the model's code, so that its NDEBUG cannot switch the tests' asserts off.
_CPP_ASSERT = """\
#undef assert
#define assert(...) ((__VA_ARGS__) ? (void)0 : (std::fputs("Assertion failed: " #__VA_ARGS__ "\\n", stderr), std::fflush(stderr), std::_Exit(1)))
"""


def _assemble_cpp(prompt: str, source: str, tests: str, marker: str) -> str:
    """C++: the prompt's includes, the model's code minus its `main`, then the tests' `main`."""
    includes = [chunk.text for chunk in chunks(prompt, "cpp") if chunk.head.startswith("#")]
    kept = [chunk.text for chunk in chunks(source, "cpp") if not _is_main(chunk, "main")]
    finish = f'    std::fputs("\\n{marker}\\n", stdout);\n    std::fflush(stdout);\n'
    tests = _before_last_brace(_drop_leading_brace(tests, "cpp"), "cpp", finish)
    return _join([*includes, _CPP_PRELUDE, *kept, _CPP_ASSERT, tests]) + "\n"


# -- Java and C# ----------------------------------------------------------------

_JAVA_TYPE = re.compile(
    r"(?:@\w+(?:\.\w+)*(?:\s*\([^)]*\))?\s*)*"
    r"(?:(?:public|protected|private|static|final|abstract|sealed|non-sealed|strictfp)\s+)*"
    r"(class|interface|enum|record|@interface)\s+(\w+)"
)
_CS_TYPE = re.compile(
    r"(?:\[[^\]]*\]\s*)*"
    r"(?:(?:public|private|protected|internal|static|sealed|abstract|partial|file|ref|readonly"
    r"|unsafe|new)\s+)*"
    r"(class|struct|interface|enum|record(?:\s+class|\s+struct)?|delegate)\s+(\w+)"
)
_CS_USING = re.compile(
    r"(?:global\s+)?using\s+(?!var\b)(?:static\s+)?[\w.]+\s*(?:=[^;(]+)?;|extern\s+alias\b"
)
# How deep _collect looks into nested C# namespaces.
_MAX_NAMESPACES = 32
# A loose C# piece that is a member (a method, a field), not a top-level statement.
_CS_MEMBER = re.compile(
    r"(?:\[[^\]]*\]\s*)*(?:public|private|protected|internal|static|const|readonly)\b"
)
# File-based C# apps (.NET 10) default to implicit usings, nullable warnings and
# native AOT; MultiPL-E's prompts expect none of them.
_CS_DIRECTIVES = (
    "#:property PublishAot=false\n#:property ImplicitUsings=disable\n#:property Nullable=disable\n"
)
# Debug.Assert is the C# tests' assert. Without this listener a failed one
# would crash the process (and ask Windows to report it) instead of failing.
_CS_LISTENER = """\
class KumaruAssertListener : System.Diagnostics.TraceListener {
    public override void Write(string message) { }
    public override void WriteLine(string message) { }
    public override void Fail(string message) { Fail(message, null); }
    public override void Fail(string message, string detailMessage) {
        System.Console.Out.Flush();
        System.Console.Error.WriteLine("Assertion failed. " + message);
        System.Console.Error.Flush();
        System.Environment.Exit(1);
    }
}"""
# First in the tests' main. Exit code 97 says the harness itself is wrong:
# without -ea (Java) or DEBUG (C#), every assertion would pass unchecked.
_JAVA_PRELUDE = """
        boolean kumaruAssertsOn = false;
        assert kumaruAssertsOn = true;
        if (!kumaruAssertsOn) { System.out.println("KUMARU_NO_ASSERT"); System.exit(97); }
"""
_CS_PRELUDE = """
#if !DEBUG
        System.Console.WriteLine("KUMARU_NO_DEBUG"); System.Environment.Exit(97);
#endif
        System.AppDomain.CurrentDomain.UnhandledException += (kumaruSender, kumaruArgs) => {
            System.Console.Out.Flush();
            var kumaruError = kumaruArgs.ExceptionObject as System.Exception;
            System.Console.Error.WriteLine("Unhandled exception. " + (kumaruError == null ? "" : kumaruError.GetType().FullName + ": " + kumaruError.Message));
            System.Console.Error.Flush();
            System.Environment.Exit(1);
        };
        System.Diagnostics.Trace.Listeners.Clear();
        System.Diagnostics.Trace.Listeners.Add(new KumaruAssertListener());
"""


def _java_finish(marker: str) -> str:
    return (
        "        System.out.flush();\n"
        "        try {\n"
        "            java.io.FileOutputStream kumaruOut = new java.io.FileOutputStream(java.io.FileDescriptor.out);\n"
        f'            kumaruOut.write("\\n{marker}\\n".getBytes());\n'
        "            kumaruOut.flush();\n"
        "        } catch (java.io.IOException kumaruError) {\n"
        "        }\n"
    )


def _cs_finish(marker: str) -> str:
    return (
        "        System.Console.Out.Flush();\n"
        "        using (var kumaruOut = System.Console.OpenStandardOutput()) {\n"
        f'            var kumaruBytes = System.Text.Encoding.UTF8.GetBytes("\\n{marker}\\n");\n'
        "            kumaruOut.Write(kumaruBytes, 0, kumaruBytes.Length);\n"
        "            kumaruOut.Flush();\n"
        "        }\n"
    )


@dataclass(frozen=True)
class _Type:
    """A top-level type declaration in a Java or C# answer."""

    chunk: Chunk
    name: str
    # Where its keyword ("class") starts, from the start of the chunk.
    keyword: int
    static: bool
    # Absolute positions of its body's braces; -1 without a body.
    open: int
    close: int


@dataclass
class _Pieces:
    header: dict[str, str] = field(default_factory=dict)
    types: list[_Type] = field(default_factory=list)
    loose: list[Chunk] = field(default_factory=list)

    def add_header(self, text: str, lang: str) -> None:
        self.header.setdefault(" ".join(uncomment(text, lang).split()), text)


def _is_import(head: str, lang: str) -> bool:
    return bool(re.match(r"import\b", head) if lang == "java" else _CS_USING.match(head))


def _collect(
    source: str,
    masked: str,
    lang: str,
    pieces: _Pieces,
    start: int = 0,
    end: int | None = None,
    depth: int = 0,
) -> None:
    """Sort a Java or C# answer's top-level pieces into imports, types and loose members.

    C# namespaces are looked into, down to `_MAX_NAMESPACES` deep: each level
    reads its whole body again, and only a degenerate reply nests further.
    """
    type_pattern = _JAVA_TYPE if lang == "java" else _CS_TYPE
    for chunk in chunks(source, lang, masked=masked, start=start, end=end):
        head = chunk.head
        if _is_import(head, lang):
            pieces.add_header(chunk.text, lang)
            continue
        if re.match(r"package\b", head) if lang == "java" else head.startswith("#"):
            continue
        namespace = re.match(r"namespace\s+[\w.]+\s*([{;])", head) if lang == "cs" else None
        if namespace:
            if namespace.group(1) == "{" and depth < _MAX_NAMESPACES:  # its body holds more pieces
                begin = chunk.start + _lead(chunk) + namespace.end() - 1
                close = matching(masked, begin)
                if close > begin:
                    _collect(source, masked, lang, pieces, begin + 1, close, depth + 1)
            continue
        match = type_pattern.match(head)
        if match:
            lead = _lead(chunk)
            begin = masked.find("{", chunk.start + lead + match.end(), chunk.end)
            close = matching(masked, begin) if begin >= 0 else -1
            static = re.search(r"\bstatic\b", head[: match.start(1)]) is not None
            pieces.types.append(_Type(chunk, match.group(2), lead + match.start(1), static, begin, close))
        elif lang == "java" or _CS_MEMBER.match(head):
            pieces.loose.append(chunk)


def _without_public(declared: _Type) -> str:
    """A Java type without `public`: a file may declare one public type, named
    after the file, and Problem.java's is Problem."""
    text = declared.chunk.text
    match = re.search(r"(?<![\w-])public\s+", declared.chunk.code[: declared.keyword])
    return text if match is None else text[: match.start()] + text[match.end() :]


def _rename(text: str, lang: str, old: str, new: str) -> str:
    """`text` with the identifier `old` replaced by `new` outside strings and comments."""
    masked = mask(text, lang)
    pieces, last = [], 0
    for match in re.finditer(rf"(?<![\w$.]){re.escape(old)}(?![\w$])", masked):
        pieces += [text[last : match.start()], new]
        last = match.end()
    return "".join(pieces) + text[last:]


def _class_main(tests: str, lang: str, marker: str) -> str:
    """The tests' main method, with the harness's checks first and the marker last.

    MultiPL-E's tests close the prompt's method, define main, then close the
    class; the caller closes the class itself.
    """
    tests = _drop_leading_brace(tests, lang)
    masked = mask(tests, lang)
    end = masked.rfind("}")  # the class's closing brace
    signature = re.search(r"\b(?:main|Main)\s*\([^)]*\)\s*\{", masked[:end]) if end > 0 else None
    close = masked.rfind("}", 0, end)  # main's
    if signature is None or close < signature.end():
        raise KumaruError(f"These {LANGUAGES[lang].name} tests have no main method.")
    prelude = _JAVA_PRELUDE if lang == "java" else _CS_PRELUDE
    finish = _java_finish(marker) if lang == "java" else _cs_finish(marker)
    return tests[: signature.end()] + prelude + tests[signature.end() : close] + finish + tests[close:end]


def _assemble_class(lang: str, prompt: str, source: str, tests: str, marker: str, entry: str) -> str:
    """Java and C#: a class Problem with the model's methods and the tests' main.

    The model's class may have any name (Solution, Main...): the type that
    defines the function becomes Problem, keeping its members, and references
    to its old name follow. Other types go before it. Its own main, if any,
    gives way to the tests'.
    """
    pieces = _Pieces()
    for chunk in chunks(prompt, lang):  # the prompt's imports come first
        if _is_import(chunk.head, lang):
            pieces.add_header(chunk.text, lang)
    masked = mask(source, lang)
    _collect(source, masked, lang, pieces)
    main = "main" if lang == "java" else "Main"
    bodies = [t for t in pieces.types if t.close > t.open >= 0]
    defining = [t for t in bodies if find_definition(masked[t.open : t.close], entry) >= 0]
    entry_type = next((t for t in defining if t.name == "Problem"), defining[0] if defining else None)
    entry_type = entry_type or next((t for t in bodies if t.name == "Problem"), None)
    members: list[str] = []
    if entry_type is not None:
        inside = chunks(source, lang, masked=masked, start=entry_type.open + 1, end=entry_type.close)
        members = [chunk.text for chunk in inside if not _is_main(chunk, main)]
    members += [chunk.text for chunk in pieces.loose if not _is_main(chunk, main)]
    others = [
        _without_public(t) if lang == "java" else t.chunk.text for t in pieces.types if t is not entry_type
    ]
    body, before = _join(members), _join(others)
    if entry_type is not None and entry_type.name != "Problem":
        body = _rename(body, lang, entry_type.name, "Problem")
        before = _rename(before, lang, entry_type.name, "Problem")
    static = "static " if lang == "cs" and entry_type is not None and entry_type.static else ""
    problem = f"{static}class Problem {{\n{body}\n{_class_main(tests, lang, marker).strip(chr(10))}\n}}"
    if lang == "java":
        return _join([*pieces.header.values(), before, problem]) + "\n"
    return _CS_DIRECTIVES + _join([*pieces.header.values(), before, problem, _CS_LISTENER]) + "\n"


def assemble(lang: str, prompt: str, source: str, tests: str, marker: str, entry: str) -> str:
    """One program: the solution (from `solution_source`), the hidden tests, then `marker`."""
    if lang in ("js", "ts"):
        return _assemble_js(lang, source, tests, marker)
    if lang == "go":
        return _assemble_go(prompt, source, tests, marker)
    if lang == "rs":
        return _assemble_rs(prompt, source, tests, marker)
    if lang == "cpp":
        return _assemble_cpp(prompt, source, tests, marker)
    if lang in ("java", "cs"):
        return _assemble_class(lang, prompt, source, tests, marker, entry)
    raise ValueError(f"no assembler for {lang!r}")


# -- toolchains -----------------------------------------------------------------

_WINDOWS = sys.platform == "win32"
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
# Linux gives a program's main thread 8 MB of stack, Windows 1 MB. Deep
# recursion that passed in MultiPL-E's Linux container must not crash here.
_STACK_BYTES = 8 * 1024 * 1024

_DOTNET_ENV = {
    "DOTNET_CLI_TELEMETRY_OPTOUT": "1",
    "DOTNET_NOLOGO": "1",
    "DOTNET_SKIP_FIRST_TIME_EXPERIENCE": "1",
    "DOTNET_CLI_UI_LANGUAGE": "en",
    # Culture-neutral (ordinal) string comparison and formatting, as the
    # tests, translated from Python, expect.
    "DOTNET_SYSTEM_GLOBALIZATION_INVARIANT": "1",
    "DOTNET_GENERATE_ASPNET_CERTIFICATE": "false",
    # No MSBuild worker processes left running after the build.
    "MSBUILDDISABLENODEREUSE": "1",
}
_DOTNET_KEEP = frozenset({
    "APPDATA", "LOCALAPPDATA", "PROGRAMDATA", "PROGRAMFILES", "PROGRAMFILES(X86)", "PROGRAMW6432",
    "COMMONPROGRAMFILES", "ALLUSERSPROFILE", "DOTNET_ROOT", "NUGET_PACKAGES",
})  # fmt: skip
_CPP_KEEP = frozenset({
    "INCLUDE", "LIB", "LIBPATH", "PROGRAMFILES", "PROGRAMFILES(X86)", "PROGRAMDATA",
    "CPATH", "LIBRARY_PATH", "CPLUS_INCLUDE_PATH", "SDKROOT", "DEVELOPER_DIR",
})  # fmt: skip
# Never download modules or a newer Go; ignore the user's `go env -w` settings.
_GO_ENV = {
    "GOTOOLCHAIN": "local",
    "GOPROXY": "off",
    "GOFLAGS": "-buildvcs=false",
    "CGO_ENABLED": "0",
    "GOENV": "off",
}
_GO_KEEP = frozenset({"GOCACHE", "GOPATH", "GOROOT", "GOMODCACHE", "LOCALAPPDATA", "XDG_CACHE_HOME"})
_RS_KEEP = frozenset({
    "RUSTUP_HOME", "CARGO_HOME", "RUSTUP_TOOLCHAIN", "PROGRAMFILES", "PROGRAMFILES(X86)",
    "PROGRAMDATA", "LOCALAPPDATA",
})  # fmt: skip


@dataclass(frozen=True)
class Toolchain:
    """The tools one language runs with, and how to call them."""

    # Role ("node", "javac", "cxx"...) -> executable: an absolute path, or a
    # plain name inside the sandbox image.
    tools: dict[str, str]
    # Part of the results' settings: another compiler makes another run.
    version: str
    env: dict[str, str] = field(default_factory=dict)
    keep_env: frozenset[str] = frozenset()
    # TypeScript only: "tsc" type-checks; "strip" runs the code with its types removed.
    checker: str = ""
    # Extra compiler flags (on Windows: the stack size, static linking).
    flags: tuple[str, ...] = ()
    # Java only: the javatuples jar.
    jar: str = ""
    # Steps every program runs first (in the sandbox: a writable Go build cache).
    setup: tuple[Step, ...] = ()


def _probe(argv: list[str], *, cwd: str | None = None, env: dict[str, str] | None = None) -> str:
    """The output of a toolchain's version command; ToolchainMissing if it fails."""
    try:
        proc = subprocess.run(
            argv,
            cwd=cwd,
            env=env,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=60,
            creationflags=_NO_WINDOW,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ToolchainMissing(f"Could not run {' '.join(argv)}: {exc}") from exc
    output = (proc.stdout or "") + (proc.stderr or "")
    if proc.returncode != 0:
        raise ToolchainMissing(
            f"{' '.join(argv)} failed with exit code {proc.returncode}: {output.strip()[-300:]}"
        )
    return output


def _numbers(version: str) -> tuple[int, ...]:
    return tuple(int(part) for part in re.findall(r"\d+", version)[:3])


def _detect_py() -> Toolchain:
    # The base interpreter, as for HumanEval: `-I` leaves the venv out anyway.
    python = getattr(sys, "_base_executable", None) or sys.executable
    return Toolchain({"python": python}, f"python {platform.python_version()}")


def _node(minimum: tuple[int, ...], what: str) -> tuple[str, str]:
    node = codeexec.find_program("node")
    if not node:
        raise ToolchainMissing(
            f"{what} needs Node.js, which is not on PATH.",
            hint="Install Node.js 22 or later: https://nodejs.org",
        )
    version = _probe([node, "--version"]).strip()
    if _numbers(version) < minimum:
        raise ToolchainMissing(
            f"{what} needs Node.js {'.'.join(map(str, minimum))} or later; found {version}.",
            hint="Install a current Node.js: https://nodejs.org",
        )
    return node, version


def _detect_js() -> Toolchain:
    node, version = _node((18,), "JavaScript")
    return Toolchain({"node": node}, f"node {version}")


def _find_tsc() -> str | None:
    """The tsc.js behind a `tsc` on PATH (npm installs a shim that runs it)."""
    shim = codeexec.find_program("tsc")
    if not shim:
        return None
    base = Path(shim).resolve().parent
    for candidate in (
        base / "node_modules" / "typescript" / "lib" / "tsc.js",  # npm on Windows
        base.parent / "lib" / "tsc.js",  # the package's own bin/, symlink resolved
        base.parent / "lib" / "node_modules" / "typescript" / "lib" / "tsc.js",  # npm's prefix/bin
    ):
        if candidate.is_file():
            return str(candidate)
    return None


def _detect_ts() -> Toolchain:
    tsc = _find_tsc()
    if tsc:
        node, version = _node((18,), "TypeScript")
        checker = _probe([node, tsc, "--version"]).strip().removeprefix("Version ")
        return Toolchain({"node": node, "tsc": tsc}, f"node {version}; tsc {checker}", checker="tsc")
    # Without tsc, Node 22.7+ runs TypeScript by removing its types, unchecked.
    node, version = _node((22, 7), "TypeScript without tsc")
    return Toolchain({"node": node}, f"node {version}; types stripped", checker="strip")


def _detect_cs() -> Toolchain:
    hint = "Install the .NET 10 SDK: https://dotnet.microsoft.com/download"
    dotnet = codeexec.find_program("dotnet")
    if not dotnet:
        raise ToolchainMissing("C# needs the .NET SDK, which is not on PATH.", hint=hint)
    # Asked outside any folder whose global.json might pin an older SDK.
    env = {**os.environ, **_DOTNET_ENV}
    lines = _probe([dotnet, "--version"], cwd=tempfile.gettempdir(), env=env).strip().splitlines()
    version = lines[-1].strip() if lines else ""
    if _numbers(version)[:1] < (10,):
        raise ToolchainMissing(
            f"C# needs the .NET 10 SDK or later, which builds single-file programs; found {version or 'none'}.",
            hint=hint,
        )
    return Toolchain({"dotnet": dotnet}, f".NET SDK {version}", env=dict(_DOTNET_ENV), keep_env=_DOTNET_KEEP)


def javatuples_jar() -> Path:
    """Where the javatuples jar is expected: KUMARU_JAVATUPLES_JAR, else Kumaru's cache."""
    configured = os.environ.get("KUMARU_JAVATUPLES_JAR", "").strip()
    return Path(configured).expanduser() if configured else Path.home() / ".cache" / "kumaru" / JAVATUPLES


def _javatuples() -> str:
    path = javatuples_jar()
    fetch = f'{"curl.exe" if _WINDOWS else "curl"} -L --create-dirs -o "{path}" {JAVATUPLES_URL}'
    if not path.is_file():
        raise ToolchainMissing(
            f"MultiPL-E's Java tests use org.javatuples, and {JAVATUPLES} is not at {path}.",
            hint=f"Download it once from Maven Central: {fetch} (or set KUMARU_JAVATUPLES_JAR to a copy).",
        )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != JAVATUPLES_SHA256:
        raise ToolchainMissing(
            f"{path} is not Maven Central's {JAVATUPLES}: its SHA-256 is {digest}, not {JAVATUPLES_SHA256}.",
            hint=f"Download it again: {fetch}",
        )
    return str(path)


def _detect_java() -> Toolchain:
    hint = "Install a JDK, 17 or later (for example Eclipse Temurin: https://adoptium.net)."
    exe = ".exe" if _WINDOWS else ""
    javac = codeexec.find_program("javac")
    home = os.environ.get("JAVA_HOME", "").strip()
    # Only an absolute JAVA_HOME: a relative one would name a folder under the current one.
    if not javac and home and Path(home).is_absolute() and (Path(home) / "bin" / f"javac{exe}").is_file():
        javac = str(Path(home) / "bin" / f"javac{exe}")
    if not javac:
        raise ToolchainMissing("Java needs a JDK, and javac is neither on PATH nor in JAVA_HOME.", hint=hint)
    sibling = Path(javac).resolve().with_name(f"java{exe}")
    java = str(sibling) if sibling.is_file() else codeexec.find_program("java")
    if not java:
        raise ToolchainMissing(f"Found {javac}, but no java to run what it compiles.", hint=hint)
    found = re.search(r"javac\s+\S+", _probe([javac, "-version"]))
    version = f"{found.group(0) if found else 'javac'}; {JAVATUPLES}"
    return Toolchain({"javac": javac, "java": java}, version, jar=_javatuples())


def _detect_cpp() -> Toolchain:
    cxx = codeexec.find_program("g++") or codeexec.find_program("clang++")
    if not cxx:
        raise ToolchainMissing(
            "C++ needs g++ or clang++, and neither is on PATH.",
            hint="Install GCC (Windows: WinLibs or MSYS2's MinGW-w64; Linux: build-essential). "
            "MultiPL-E's tests include <bits/stdc++.h>, which needs GCC's standard library.",
        )
    version = _probe([cxx, "--version"]).strip().splitlines()[0]
    flags: tuple[str, ...] = ()
    if _WINDOWS:
        target = _probe([cxx, "-dumpmachine"]).strip()
        # MinGW: static, so that no other program's libstdc++ DLL on PATH is loaded.
        msvc = "msvc" in target
        flags = (f"-Wl,/STACK:{_STACK_BYTES}",) if msvc else ("-static", f"-Wl,--stack,{_STACK_BYTES}")
    return Toolchain({"cxx": cxx}, version, keep_env=_CPP_KEEP, flags=flags)


def _detect_go() -> Toolchain:
    go = codeexec.find_program("go")
    if not go:
        raise ToolchainMissing("Go needs the go command, which is not on PATH.", hint="Install Go: https://go.dev/dl/")
    version = _probe([go, "version"], env={**os.environ, **_GO_ENV}).strip()
    return Toolchain({"go": go}, version, env=dict(_GO_ENV), keep_env=_GO_KEEP)


def _detect_rs() -> Toolchain:
    rustc = codeexec.find_program("rustc")
    if not rustc:
        raise ToolchainMissing("Rust needs rustc, which is not on PATH.", hint="Install Rust: https://rustup.rs")
    version = _probe([rustc, "--version"]).strip()
    host = re.search(r"^host:\s*(\S+)", _probe([rustc, "-vV"]), re.MULTILINE)
    target = host.group(1) if host else ""
    flags: tuple[str, ...] = ()
    if "windows-gnu" in target:
        flags = ("-C", f"link-arg=-Wl,--stack,{_STACK_BYTES}")
    elif "windows-msvc" in target:
        flags = ("-C", f"link-arg=/STACK:{_STACK_BYTES}")
    return Toolchain({"rustc": rustc}, version, keep_env=_RS_KEEP, flags=flags)


def _docker_toolchain(lang: str) -> Toolchain:
    """The sandbox image's tools, which are on its PATH (see docker/sandbox/Dockerfile)."""
    image = codeexec.check_docker()
    tools = {
        "py": {"python": "python3"},
        "js": {"node": "node"},
        "ts": {"node": "node", "tsc": DOCKER_TSC},
        "cs": {"dotnet": "dotnet"},
        "java": {"javac": "javac", "java": "java"},
        "cpp": {"cxx": "g++"},
        "go": {"go": "go"},
        "rs": {"rustc": "rustc"},
    }[lang]
    env = {"cs": _DOTNET_ENV, "go": {**_GO_ENV, "GOCACHE": "/tmp/gocache"}}.get(lang, {})
    # The image's prebuilt Go standard library, copied to where builds may write.
    setup = (Step(("cp", "-R", "/opt/gocache", "/tmp/gocache"), 60.0, "setup"),) if lang == "go" else ()
    return Toolchain(
        tools,
        f"docker image {image.removeprefix('sha256:')[:12]}",
        env=dict(env),
        checker="tsc" if lang == "ts" else "",
        jar=DOCKER_JAVATUPLES if lang == "java" else "",
        setup=setup,
    )


_DETECT = {
    "py": _detect_py,
    "js": _detect_js,
    "ts": _detect_ts,
    "cs": _detect_cs,
    "java": _detect_java,
    "cpp": _detect_cpp,
    "go": _detect_go,
    "rs": _detect_rs,
}


def detect_toolchain(lang: str, sandbox: str = "none") -> Toolchain:
    """The toolchain `lang` runs with: this machine's, or the sandbox image's."""
    return _docker_toolchain(lang) if sandbox == "docker" else _DETECT[lang]()


# -- programs -------------------------------------------------------------------


def build_program(lang: str, toolchain: Toolchain, text: str, marker: str, *, local: bool = True) -> Program:
    """The files and steps that compile and run one assembled program.

    `local` is False for the sandbox, which runs Linux whatever the host is.
    """
    tool = toolchain.tools
    exe = ".exe" if local and _WINDOWS else ""
    build, run = COMPILE_TIMEOUT_S, RUN_TIMEOUT_S
    if lang == "py":
        name, steps = "program.py", [Step((tool["python"], "-I", "-X", "utf8", "program.py"), run)]
    elif lang == "js":
        name, steps = "program.cjs", [Step((tool["node"], "program.cjs"), run)]
    elif lang == "ts" and toolchain.checker == "tsc":
        name = "program.ts"
        steps = [
            Step((tool["node"], tool["tsc"], "--target", "esnext", "program.ts"), build, "compile"),
            Step((tool["node"], "program.js"), run),
        ]
    elif lang == "ts":
        # .cts is CommonJS whatever a package.json says; "transform" rather than
        # "strip" so that enums and namespaces run too.
        name = "program.cts"
        flags = ("--experimental-transform-types", "--disable-warning=ExperimentalWarning")
        steps = [Step((tool["node"], *flags, "program.cts"), run)]
    elif lang == "cs":
        name = "program.cs"
        options = ("-c", "Debug", "--disable-build-servers", "-v", "q", "-nologo", "-o", "out")
        steps = [
            Step((tool["dotnet"], "build", "program.cs", *options), build, "compile"),
            Step((tool["dotnet"], "out/program.dll"), run),
        ]
    elif lang == "java":
        name = "Problem.java"
        classpath = f".{';' if local and _WINDOWS else ':'}{toolchain.jar}"
        # A JVM that starts fast: the tests are short, and the CPU may be shared.
        jvm = ("-XX:+UseSerialGC", "-XX:TieredStopAtLevel=1", "-XX:-UsePerfData")
        javac = (tool["javac"], *(f"-J{option}" for option in jvm), "-encoding", "UTF8")
        locale = ("-Duser.language=en", "-Duser.country=US")
        steps = [
            Step((*javac, "-cp", toolchain.jar, "-d", ".", "Problem.java"), build, "compile"),
            Step((tool["java"], "-ea", *jvm, *locale, "-cp", classpath, "Problem"), run),
        ]
    elif lang == "cpp":
        name = "program.cpp"
        steps = [
            Step((tool["cxx"], "-std=c++17", *toolchain.flags, "-o", f"program{exe}", "program.cpp"), build, "compile"),
            Step((f"./program{exe}",), run),
        ]
    elif lang == "go":
        name = "program_test.go"
        steps = [
            Step((tool["go"], "test", "-c", "-o", f"prog.test{exe}", "program_test.go"), build, "compile"),
            Step((f"./prog.test{exe}",), run),
        ]
    elif lang == "rs":
        name = "program.rs"
        rustc = (tool["rustc"], "--edition", "2021", "-A", "warnings", *toolchain.flags)
        steps = [
            Step((*rustc, "-o", f"program{exe}", "program.rs"), build, "compile"),
            Step((f"./program{exe}",), run),
        ]
    else:
        raise ValueError(f"unknown language {lang!r}")
    return Program(
        {name: text}, (*toolchain.setup, *steps), marker, env=dict(toolchain.env), keep_env=toolchain.keep_env
    )


def _python_finish(marker: str) -> str:
    # Straight to file descriptor 1, so a replaced sys.stdout cannot swallow it.
    return f"\n\nimport os as _kumaru_finish\n_kumaru_finish.write(1, b'\\n{marker}\\n')\n"


# -- self-test ------------------------------------------------------------------


@dataclass(frozen=True)
class _Canary:
    """A problem in the dataset's own format, with one right and one wrong answer."""

    prompt: str
    tests: str
    # A whole function in a fenced block, as a chat model answers. Must pass.
    good: str
    # A body that continues the prompt, as a completion model answers. Must fail.
    bad: str


_JS_TESTS = (
    "const assert = require('node:assert');\n\n\nfunction test() {\n  let candidate = add;\n"
    "  assert.deepEqual(candidate(2, 3),5);\n  assert.deepEqual(candidate(-1, 1),0);\n}\n\ntest();\n"
)
_GO_TESTS = """\
func TestAdd(t *testing.T) {
  candidate := add
	type test struct {
		actual   interface{}
		expected interface{}
	}
   tests := []test{
     { actual: candidate(2, 3), expected: 5 },
     { actual: candidate(-1, 1), expected: 0 },
   }

	for i, tc := range tests {
		t.Run(fmt.Sprintf("test num % d", i), func(t *testing.T) {
			if fmt.Sprintf("%v", tc.actual) != fmt.Sprintf("%v", tc.expected) {
				t.Errorf("expected '%s', got '%s'", tc.expected, tc.actual)
			}
		})
	}
}
"""
_CS_USINGS = (
    "using System;\nusing System.Numerics;\nusing System.Diagnostics;\nusing System.Collections.Generic;\n"
    "using System.Linq;\nusing System.Text;\nusing System.Security.Cryptography;\n"
)
_JAVA_IMPORTS = (
    "import java.util.*;\nimport java.lang.reflect.*;\nimport org.javatuples.*;\nimport java.security.*;\n"
    "import java.math.*;\nimport java.io.*;\nimport java.util.stream.*;\n"
)
_CANARIES = {
    "py": _Canary(
        'def add(x: int, y: int) -> int:\n    """Add two numbers.\n    >>> add(2, 3)\n    5\n    """\n',
        "\n\ndef check(candidate):\n    assert candidate(2, 3) == 5\n    assert candidate(-1, 1) == 0\n",
        "```python\ndef add(x: int, y: int) -> int:\n    return x + y\n```",
        "    return x - y\n",
    ),
    "js": _Canary(
        "//Add two numbers.\n// >>> add(2, 3)\n// 5\nfunction add(x, y){\n",
        _JS_TESTS,
        "```javascript\nfunction add(x, y) {\n  return x + y;\n}\n```",
        "  return x - y;\n}\n",
    ),
    "ts": _Canary(
        "//Add two numbers.\n// >>> add(2, 3)\n// 5\nfunction add(x: number, y: number): number {\n",
        "declare var require: any;\n" + _JS_TESTS,
        "```typescript\nfunction add(x: number, y: number): number {\n  return x + y;\n}\n```",
        "  return x - y;\n}\n",
    ),
    "cs": _Canary(
        _CS_USINGS + "class Problem {\n    // Add two numbers.\n    // >>> Add((2L), (3L))\n    // (5L)\n"
        "    public static long Add(long x, long y) {\n",
        "    }\n    public static void Main(string[] args) {\n    Debug.Assert(Add((2L), (3L)) == (5L));\n"
        "    Debug.Assert(Add((-1L), (1L)) == (0L));\n    }\n\n}\n",
        "```csharp\nclass Problem {\n    public static long Add(long x, long y) {\n        return x + y;\n    }\n}\n```",
        "        return x - y;\n",
    ),
    "java": _Canary(
        _JAVA_IMPORTS + "class Problem {\n    // Add two numbers.\n    // >>> add((2l), (3l))\n    // (5l)\n"
        "    public static long add(long x, long y) {\n",
        # Pair checks that the javatuples jar is on the class path.
        "    }\n    public static void main(String[] args) {\n"
        "    assert(add((2l), Pair.with(3l, 4l).getValue0()) == (5l));\n    assert(add((-1l), (1l)) == (0l));\n"
        "    }\n\n}\n",
        "```java\nclass Problem {\n    public static long add(long x, long y) {\n        return x + y;\n    }\n}\n```",
        "        return x - y;\n",
    ),
    "cpp": _Canary(
        "#include<assert.h>\n#include<bits/stdc++.h>\n// Add two numbers.\n// >>> add((2), (3))\n// (5)\n"
        "long add(long x, long y) {\n",
        "}\nint main() {\n    auto candidate = add;\n    assert(candidate((2), (3)) == (5));\n"
        "    assert(candidate((-1), (1)) == (0));\n}\n",
        "```cpp\nlong add(long x, long y) {\n    return x + y;\n}\n```",
        "    return x - y;\n",
    ),
    "go": _Canary(
        'package add_test\n\nimport (\n    "testing"\n    "fmt"\n)\n\n// Add two numbers.\n// >>> add(2, 3)\n'
        "// 5\nfunc add(x int, y int) int {\n",
        _GO_TESTS,
        "```go\nfunc add(x int, y int) int {\n\treturn x + y\n}\n```",
        "\treturn x - y\n}\n",
    ),
    "rs": _Canary(
        "/// Add two numbers.\n/// >>> add(2, 3)\n/// 5\nfn add(x: isize, y: isize) -> isize {\n",
        "}\n\nfn main() {\n    let candidate = add;\n    assert_eq!(candidate(2, 3), 5);\n"
        "    assert_eq!(candidate(-1, 1), 0);\n}\n",
        "```rust\nfn add(x: isize, y: isize) -> isize {\n    x + y\n}\n```",
        "    x - y\n",
    ),
}


# -- the benchmarks -------------------------------------------------------------


class MultiPLE(Benchmark):
    """One language of MultiPL-E's HumanEval. Subclasses set `lang`."""

    lang: ClassVar[str]
    needs_code_execution = True
    blocking_grader = True

    def __init__(self) -> None:
        self.sandbox = "none"
        self._toolchain: Toolchain | None = None

    def configure(self, *, sandbox: str = "none") -> None:
        """Choose where programs run (`codeexec.SANDBOXES`). Call before `preflight`."""
        if sandbox not in codeexec.SANDBOXES:
            raise KumaruError(
                f"Unknown sandbox '{sandbox}'.", hint=f"Use one of: {', '.join(codeexec.SANDBOXES)}."
            )
        if sandbox != self.sandbox:
            self.sandbox, self._toolchain = sandbox, None

    @property
    def toolchain(self) -> Toolchain:
        if self._toolchain is None:
            self._toolchain = detect_toolchain(self.lang, self.sandbox)
        return self._toolchain

    def preflight(self) -> None:
        """Find the toolchain, then grade a known right and a known wrong answer with it."""
        canary = _CANARIES[self.lang]
        entry = "add" if self.lang == "py" else entry_point(canary.prompt, self.lang)
        meta = {"prompt": canary.prompt, "tests": canary.tests, "entry_point": entry}
        item = BenchItem(id="canary", prompt="", meta=meta)
        for answer, should_pass in ((canary.good, True), (canary.bad, False)):
            grade = self.grade(item, answer)
            if grade.correct != should_pass:
                verdict = "a right answer failed" if should_pass else "a wrong answer passed"
                output = grade.detail.get("stderr", "")
                raise ToolchainMissing(
                    f"The {LANGUAGES[self.lang].name} toolchain ({self.toolchain.version}) failed "
                    f"Kumaru's self-test: {verdict} ({grade.extracted}). {output}".strip(),
                    hint="Check that the toolchain works by itself, or leave this language out.",
                )

    def fingerprint_extra(self) -> dict[str, Any]:
        extra = {"sandbox": self.sandbox, "toolchain": self.toolchain.version}
        if self.toolchain.checker:
            extra["checker"] = self.toolchain.checker
        if self.lang in TEST_FIXES:
            # Results graded on the tests as published are not resumed with fixed ones.
            extra["test_fixes"] = TEST_FIXES[self.lang][0]
        return extra

    def generation_overrides(self) -> dict[str, Any]:
        # As for HumanEval: code repeats tokens legitimately.
        return {"presence_penalty": 0.0, "repetition_penalty": 1.0}

    def load(self) -> list[BenchItem]:
        language = LANGUAGES[self.lang]
        problems: list[tuple[int, str, str, str, str]] = []
        if self.lang == "py":
            for row in load_hf_rows(HUMANEVAL, None, "test", revision=HUMANEVAL_REVISION):
                n, entry = int(row["task_id"].rsplit("/", 1)[1]), row["entry_point"]
                problems.append((n, f"HumanEval_{n}_{entry}", row["prompt"], row["test"], entry))
        else:
            rows = load_hf_rows(DATASET, f"humaneval-{self.lang}", "test", revision=DATASET_REVISION)
            for row in rows:
                n = int(row["name"].split("_")[1])
                problems.append((n, row["name"], row["prompt"], row["tests"], entry_point(row["prompt"], self.lang)))
        items = []
        fixed: set[int] = set()
        for n, name, prompt, published, entry in sorted(problems):
            if n in EXCLUDED:
                continue
            tests = fix_tests(self.lang, prompt, published, entry)
            if tests != published:
                fixed.add(n)
            shown = prompt if prompt.endswith("\n") else prompt + "\n"
            items.append(
                BenchItem(
                    id=name,
                    prompt=PROMPT.format(language=language.name, fence=language.fence, prompt=shown),
                    meta={"prompt": prompt, "tests": tests, "entry_point": entry, "language": self.lang, "problem": n},
                )
            )
        if len(items) != N_PROBLEMS:
            raise KumaruError(
                f"Expected {N_PROBLEMS} {language.name} problems, found {len(items)}.",
                hint=f"Has the dataset changed? Kumaru pins {DATASET} at revision {DATASET_REVISION}.",
            )
        issue, expected = TEST_FIXES.get(self.lang, ("", frozenset()))
        if fixed != expected:
            raise KumaruError(
                f"Fixing the {language.name} tests ({issue}) changed {len(fixed)} problems, "
                f"not the {len(expected)} expected: {sorted(fixed ^ expected)} differ.",
                hint=f"Has the dataset changed? Kumaru pins {DATASET} at revision {DATASET_REVISION}.",
            )
        return items

    def grade(self, item: BenchItem, output: str) -> Grade:
        meta, lang = item.meta, self.lang
        entry, marker = meta["entry_point"], codeexec.new_marker()
        if lang == "py":
            whole = defines_entry(extract_code(output, entry), entry, "py")
            mode = "whole" if whole else "continuation"
            text = assemble_program(meta["prompt"], output, meta["tests"], entry) + _python_finish(marker)
        else:
            source, mode = solution_source(meta["prompt"], extract_answer(output, lang, entry), entry, lang)
            text = assemble(lang, meta["prompt"], source, meta["tests"], marker, entry)
        program = build_program(lang, self.toolchain, text, marker, local=self.sandbox == "none")
        outcome = codeexec.run(program, sandbox=self.sandbox)
        detail: dict[str, Any] = {"status": outcome.status, "stage": outcome.stage, "mode": mode}
        if outcome.returncode not in (0, None):
            detail["exit_code"] = outcome.returncode
        if outcome.note:
            detail["note"] = outcome.note
        if not outcome.passed:
            # Compilers disagree on the stream: dotnet and tsc report errors on stdout.
            tail = (outcome.stderr.strip() or outcome.stdout.strip())[-STDERR_TAIL:]
            if tail:
                detail["stderr"] = tail
        return Grade(correct=outcome.passed, extracted=outcome.status, detail=detail)


class MultiPLEPython(MultiPLE):
    lang = "py"
    name = "multipl_e_py"
    description = "The Python baseline: HumanEval's 152 shared problems, asked like the other languages."
    source = "openai/openai_humaneval (152 of 164, as in MultiPL-E) - Chen et al. 2021"


class MultiPLEJavaScript(MultiPLE):
    lang = "js"
    name = "multipl_e_js"
    description = "HumanEval in JavaScript (MultiPL-E), run by its unit tests on Node.js."
    source = "nuprl/MultiPL-E humaneval-js (152 problems) - Cassano et al. 2023"


class MultiPLETypeScript(MultiPLE):
    lang = "ts"
    name = "multipl_e_ts"
    description = "HumanEval in TypeScript (MultiPL-E): type-checked by tsc when installed, run on Node.js."
    source = "nuprl/MultiPL-E humaneval-ts (152 problems) - Cassano et al. 2023"


class MultiPLECSharp(MultiPLE):
    lang = "cs"
    name = "multipl_e_cs"
    description = "HumanEval in C# (MultiPL-E), built and run with the .NET 10 SDK."
    source = "nuprl/MultiPL-E humaneval-cs (152 problems) - Cassano et al. 2023"


class MultiPLEJava(MultiPLE):
    lang = "java"
    name = "multipl_e_java"
    description = "HumanEval in Java (MultiPL-E), compiled with javac and run with assertions on."
    source = "nuprl/MultiPL-E humaneval-java (152 problems) - Cassano et al. 2023"


class MultiPLECpp(MultiPLE):
    lang = "cpp"
    name = "multipl_e_cpp"
    description = "HumanEval in C++17 (MultiPL-E), compiled with g++ or clang++."
    source = "nuprl/MultiPL-E humaneval-cpp (152 problems) - Cassano et al. 2023"


class MultiPLEGo(MultiPLE):
    lang = "go"
    name = "multipl_e_go"
    description = "HumanEval in Go (MultiPL-E), built as a go test binary."
    source = "nuprl/MultiPL-E humaneval-go (152 problems) - Cassano et al. 2023"


class MultiPLERust(MultiPLE):
    lang = "rs"
    name = "multipl_e_rs"
    description = "HumanEval in Rust (MultiPL-E), compiled with rustc (edition 2021)."
    source = "nuprl/MultiPL-E humaneval-rs (152 problems) - Cassano et al. 2023"


BY_LANGUAGE: dict[str, type[MultiPLE]] = {
    cls.lang: cls
    for cls in (
        MultiPLEPython,
        MultiPLEJavaScript,
        MultiPLETypeScript,
        MultiPLECSharp,
        MultiPLEJava,
        MultiPLECpp,
        MultiPLEGo,
        MultiPLERust,
    )
}

"""Protect the scanner MultiPL-E's assemblers cut model code with.

What must never regress: a brace or comment marker inside a string, character
or regular expression passing for code (the function would end in the wrong
place and a right answer would not compile); offsets drifting between the
masked and the original text; calls or prototypes passing for definitions; and
time linear in the text's length, as a reply can be thousands of repetitions
of one character and must still grade in moments.
"""

import functools
import string
import time
from collections.abc import Callable

import pytest

from kumaru.eval.benchmarks.codeparse import (
    LANGUAGES,
    chunks,
    defines,
    find_definition,
    mask,
    matching,
    uncomment,
)


def cpu_seconds(work: Callable[[], object], limit: float) -> float:
    """CPU seconds `work` takes; the faster of two tries if the first passes `limit`.

    CPU time, not wall time: other programs busy on the machine barely change it.
    """
    best = float("inf")
    for _ in range(2):
        start = time.process_time()
        work()
        best = min(best, time.process_time() - start)
        if best <= limit:
            break
    return best


def test_mask_blanks_comments_and_zeroes_literals_at_the_same_offsets() -> None:
    code = 'int f() { // } here\n  return "}"; /* } */ }\n'
    masked = mask(code, "cpp")
    assert len(masked) == len(code)
    assert masked == 'int f() {          \n  return 000;         }\n'
    # Comments are gone, strings stay: what import and `use` lines are read from.
    assert uncomment('x = "// kept"; // gone', "js") == 'x = "// kept";        '


@pytest.mark.parametrize(
    ("lang", "code"),
    [
        # A template literal whose interpolation holds an object and a string.
        ("js", "const s = `a ${ {x: '}'}.x } }`;\nfunction f() {}\n"),
        # A regular expression with "/" and "}" in it; "1 / 2" stays a division.
        ("js", "const re = /[/}]+/g;\nfunction f() { return 1 / 2; }\n"),
        ("ts", "const s: string = `${'}'}`;\nfunction f(): void {}\n"),
        ("go", "var s = `}\n// not a comment`\nfunc f() {}\n"),
        # Lifetimes are not characters; raw strings and chars hold braces.
        ("rs", "fn f<'a>(x: &'a str) -> char { let _ = r#\"}\"#; '}' }\n"),
        ("rs", "/* outer /* nested } */ still a comment */ fn f() {}\n"),
        # Raw strings and digit separators.
        ("cpp", 'auto s = R"x(})")x"; int n = 1\'000;\nint f() {}\n'),
        # A verbatim string ending in a backslash; escaped braces in an interpolation.
        ("cs", 'var a = @"C:\\"; var b = $"{x}}}{{";\nvoid F() {}\n'),
        ("cs", 'var r = """\n  }\n  """;\nvoid F() {}\n'),
        ("java", 'String s = """\n  }\n  """;\nvoid f() {}\n'),
        # A probe for a regular expression that fails remembers what it read;
        # a later one reading the same characters outside a [class] still closes.
        ("js", "x = /[ /}/;\nfunction f() {}\n"),
        # A run of "@" that starts no string does not hide a later one that does.
        ("cs", '@@@@x = 1; var s = @"}";\nvoid F() {}\n'),
    ],
)
def test_braces_inside_literals_and_comments_are_not_code(lang: str, code: str) -> None:
    masked = mask(code, lang)
    assert len(masked) == len(code)
    assert (masked.count("{"), masked.count("}")) == (1, 1)  # only the function's own


def test_matching_finds_the_bracket_that_closes_an_opener() -> None:
    masked = "f(a[1], {b: (2)}) + g()"
    assert matching(masked, 1) == masked.index(")", masked.index("}"))
    assert matching(masked, 3) == 5
    assert matching("f(", 1) == -1
    with pytest.raises(KeyError):
        matching(masked, 0)  # "f" opens nothing


def test_chunks_cut_top_level_pieces_and_give_the_text_back() -> None:
    code = (
        "#include <vector>\n"
        "// Adds.\n"
        "long add(long a, long b) {\n    return a + b; // }\n}\n"
        "struct P { int x; };\n"
        "int main() { return add(1, 2); }\n"
        "// trailing comment\n"
    )
    pieces = chunks(code, "cpp")
    assert [piece.head.split()[0] for piece in pieces] == ["#include", "long", "struct", "int"]
    # The comment above a function belongs to it.
    assert pieces[1].text.lstrip().startswith("// Adds.")
    assert "".join(piece.text for piece in pieces) == code.removesuffix("\n// trailing comment\n")


def test_javascript_statements_end_at_a_line_break_unless_the_next_line_continues() -> None:
    code = (
        "const a = 1\n"
        "const b = a\n  + 2\n"
        "if (a) {\n  x()\n} else {\n  y()\n}\n"
        "console.log(b)\n"
    )
    heads = [piece.head for piece in chunks(code, "js")]
    assert heads == [
        "const a = 1",
        "const b = a\n  + 2",
        "if (a) {\n  x()\n} else {\n  y()\n}",
        "console.log(b)",
    ]


def test_go_statements_end_at_a_line_break_after_a_complete_expression() -> None:
    code = 'import "fmt"\nvar x = 1 +\n  2\nfunc f() {\n}\n'
    heads = [piece.head for piece in chunks(code, "go")]
    assert heads == ["import 00000", "var x = 1 +\n  2", "func f() {\n}"]


def test_a_brace_that_ends_a_type_does_not_end_the_statement() -> None:
    # HumanEval_8 in Go returns []interface{}: cut there, the body's "{" would
    # start a line of its own, which Go rejects.
    go = "func f(n []int) []interface{} {\n\treturn nil\n}\nvar x interface{} = 1\nvar y = struct{}{}\n"
    assert [piece.head for piece in chunks(go, "go")] == [
        "func f(n []int) []interface{} {\n\treturn nil\n}",
        "var x interface{} = 1",
        "var y = struct{}{}",
    ]
    ts = "function f(): { a: number } {\n  return { a: 1 };\n}\n"
    assert [piece.head for piece in chunks(ts, "ts")] == ["function f(): { a: number } {\n  return { a: 1 };\n}"]


def test_an_import_or_export_list_does_not_end_at_its_brace() -> None:
    code = "import { a, b as c } from './m';\nimport type { T } from './t'\nexport { a } from './m';\nconst x = 1;\n"
    heads = [piece.head for piece in chunks(code, "ts")]
    assert [head.split()[0] for head in heads] == ["import", "import", "export", "const"]
    assert all(" from " in head for head in heads[:3])


def test_preprocessor_lines_and_inner_attributes_are_pieces_of_their_own() -> None:
    cpp = chunks("#include <x>\n#define TWICE(x) \\\n  ((x) * 2)\nint a;\n", "cpp")
    assert [piece.head.split()[0] for piece in cpp] == ["#include", "#define", "int"]
    rust = chunks("#![allow(dead_code)]\nfn f() {}\n", "rs")
    assert [piece.head for piece in rust] == ["#![allow(dead_code)]", "fn f() {}"]


@pytest.mark.parametrize(
    ("lang", "code", "expected"),
    [
        ("js", "function add(a, b) { return a + b }", True),
        ("js", "const add = (a, b) => a + b;", True),
        ("js", "let total = add(1, 2);", False),
        ("js", "// function add(a, b) {}\nconst x = 1;", False),
        ("ts", "function add<T>(a: T): T { return a }", True),
        ("go", "func add(a int) int { return a }", True),
        ("go", "func (s S) add() {}", False),  # a method, not the function
        ("rs", "pub fn add(a: i32) -> i32 { a }", True),
        ("rs", "let y = x.add(1);", False),
        ("cpp", "long add(long a, long b) { return a + b; }", True),
        ("cpp", "std::vector<long> add(std::vector<long> v) { return v; }", True),
        ("cpp", "long add(long a, long b);", False),  # a prototype
        ("cpp", "int f() { return add(1, 2); }", False),  # a call
        ("cpp", "int f() { (void)add(1, 2); return 0; }", False),  # a call, cast
        ("java", "public static long add(long a) {\n    return a;\n}", True),
        ("java", "static void add(long[] a) {\n    a[0]++;\n}", True),
        ("java", "x = add(1) + add(2);", False),
        ("cs", "public static long Add(long a) => a;", True),
        ("cs", "static void Add(List<long> a) { a.Add(1); }", True),
        ("cs", "var r = new Add(1);", False),
    ],
)
def test_defines_tells_definitions_from_calls(lang: str, code: str, expected: bool) -> None:
    name = "Add" if lang == "cs" else "add"
    assert defines(code, name, lang) is expected


def test_an_unknown_language_is_refused() -> None:
    with pytest.raises(ValueError, match="unsupported language"):
        mask("x", "kotlin")


# -- time linear in the text's length ---------------------------------------------

# What a reply that degenerates into repetition might repeat: every ASCII
# punctuation character, the prefixes the scanner probes for a literal or a
# comment, and pieces of definitions and statements.
PUMPS = (
    *string.punctuation,
    *("$@", '@"', '$"', "${", "`${", "/[", "\\/", "r#", 'r#"', "br#", "b'", 'R"(', 'u8R"'),
    *('"""', "'''", "/*", "//", "#["),
    *("\n", "+\n", "add(", "long add(", "const add:", "a = add(1);\n"),
)


def read_everything(text: str, lang: str, name: str) -> None:
    masked = mask(text, lang)
    uncomment(text, lang, masked=masked)
    chunks(text, lang, masked=masked)
    defines(text, name, lang, masked=masked)
    find_definition(masked, name)


@pytest.mark.parametrize("lang", LANGUAGES)
def test_a_repeated_piece_is_read_in_linear_time(lang: str) -> None:
    # A probe that reads again what a failed one read makes this quadratic:
    # 4,000 "@" then took C# 1 s, and 2,000 "/[" JavaScript 0.4 s.
    # The slowest piece now takes under 0.02 s.
    name = "Add" if lang == "cs" else "add"
    slow = {}
    for pump in PUMPS:
        pump = pump.replace("add", name)
        text = pump * (4000 // len(pump))
        seconds = cpu_seconds(functools.partial(read_everything, text, lang, name), 0.25)
        if seconds > 0.25:
            slow[pump] = round(seconds, 2)
    assert slow == {}


def test_a_long_run_of_a_binding_without_a_value_is_read_in_linear_time() -> None:
    # Searching on from each "const add" for its "=" made this quadratic: 8 s.
    text = "const add:" * 10_000
    assert cpu_seconds(functools.partial(defines, text, "add", "js"), 0.5) < 0.5


@pytest.mark.parametrize(("lang", "hole"), [("js", "`${"), ("ts", "`${"), ("cs", '$"{')])
def test_interpolations_nested_past_the_limit_run_to_the_end_of_the_text(lang: str, hole: str) -> None:
    # Python's recursion limit used to end this with a RecursionError.
    text = hole * 2000
    assert mask(text, lang) == "0" * len(text)

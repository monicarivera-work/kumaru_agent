"""Protect the scanner MultiPL-E's assemblers cut model code with.

What must never regress: a brace or comment marker inside a string, character
or regular expression passing for code (the function would end in the wrong
place and a right answer would not compile); offsets drifting between the
masked and the original text; and calls or prototypes passing for definitions.
"""

import pytest

from kumaru.eval.benchmarks.codeparse import chunks, defines, mask, matching, uncomment


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

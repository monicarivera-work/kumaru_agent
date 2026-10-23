"""Protect MultiPL-E: reading answers, assembling programs, toolchains and grading.

What must never regress: a chat model's whole answer (imports, helpers, its own
main and tests) and a completion model's bare function body must both become
one program in which the hidden tests run and print the marker last; a
toolchain that grades a known answer wrongly must never score a model; and the
grader self-check (scripts/check_graders.py) may excuse only the differences
from MultiPL-E it documents.

Assembly is checked as text, so most of these tests need no compiler. The
canary and chat-answer tests compile and run programs for real: Python,
JavaScript and TypeScript whenever they are installed, the compiled languages
only with KUMARU_TEST_TOOLCHAINS=1 (seconds per program).
"""

import dataclasses
import functools
import importlib.util
import os
import random
import re
import sys
import time
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import pytest

from kumaru.core.errors import KumaruError
from kumaru.core.registry import suite_members
from kumaru.eval.benchmarks import codeexec, multipl_e, runner
from kumaru.eval.benchmarks.base import BenchItem, Grade
from kumaru.eval.benchmarks.codeexec import Outcome, Step, ToolchainMissing
from kumaru.eval.benchmarks.extract import fences, open_fence
from kumaru.eval.benchmarks.multipl_e import Toolchain, assemble, extract_answer

MARK = "kumaru-finished-0123456789abcdef"
CANARIES = multipl_e._CANARIES
COMPILED = {"cs", "java", "cpp", "go", "rs"}
REPO = Path(__file__).resolve().parents[1]


def canary_item(lang: str) -> BenchItem:
    canary = CANARIES[lang]
    entry = "add" if lang == "py" else multipl_e.entry_point(canary.prompt, lang)
    return BenchItem(id="canary", prompt="", meta={"prompt": canary.prompt, "tests": canary.tests, "entry_point": entry})


def finished(status: str, stage: str = "run", returncode: int | None = 0, stdout: str = "", stderr: str = ""):
    """A stand-in for `codeexec.run` that ends every program the same way."""

    def run(program, *, sandbox="none"):
        return Outcome(status, stage, returncode, stdout, stderr, 0.1)

    return run


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


def installed(lang: str) -> multipl_e.MultiPLE:
    """The benchmark for `lang` with this machine's toolchain, or skip."""
    if lang in COMPILED and os.environ.get("KUMARU_TEST_TOOLCHAINS") != "1":
        pytest.skip("compiles; set KUMARU_TEST_TOOLCHAINS=1 to run")
    bench = multipl_e.BY_LANGUAGE[lang]()
    try:
        bench.toolchain  # noqa: B018 - finds the toolchain
    except ToolchainMissing as exc:
        pytest.skip(f"not installed: {exc}")
    return bench


# What chat models answer: a whole program with imports, helpers, a demo main
# and sometimes tests of their own, between explanations.
CHAT_ANSWERS = {
    "py": '''\
import math


def _same(x: int) -> int:
    return x


def add(x: int, y: int) -> int:
    """Add two numbers."""
    return _same(x) + y


if __name__ == "__main__":
    print(add(2, 3))
''',
    "js": """\
const assert = require('assert');

/** Add two numbers. */
function add(x, y) {
  return same(x) + y;
}

const same = (x) => x;

console.log(add(2, 3)); // 5
module.exports = { add };
""",
    "ts": """\
import { strict as assert } from 'node:assert';

type Num = number;

export function add(x: Num, y: Num): Num {
  return x + y;
}

assert.equal(add(2, 3), 5);
""",
    "cs": """\
using System;
using System.Collections.Generic;
using System.Linq;

public class Solution
{
    public static long Add(long x, long y)
    {
        return new List<long> { x, y }.Sum();
    }

    public static void Main(string[] args)
    {
        Console.WriteLine(Add(2, 3));
    }
}
""",
    "java": """\
import java.util.*;
import java.util.stream.*;

public class Main {
    public static long add(long x, long y) {
        return LongStream.of(x, y).sum();
    }

    public static void main(String[] args) {
        System.out.println(add(2, 3));
    }
}
""",
    "cpp": """\
#include <iostream>
#include <numeric>
#include <vector>
using namespace std;

long add(long x, long y) {
    vector<long> both = {x, y};
    return accumulate(both.begin(), both.end(), 0L);
}

int main() {
    cout << add(2, 3) << endl;
    return 0;
}
""",
    "go": """\
package main

import (
	"fmt"
	"strings"
)

func add(x int, y int) int {
	return x + y
}

func describe(x int) string {
	return strings.TrimSpace(fmt.Sprint(" ", x))
}

func main() {
	fmt.Println(describe(add(2, 3)))
}
""",
    "rs": """\
use std::collections::HashMap;

/// Adds two numbers.
pub fn add(x: isize, y: isize) -> isize {
    let mut memo: HashMap<(isize, isize), isize> = HashMap::new();
    *memo.entry((x, y)).or_insert(x + y)
}

fn main() {
    println!("{}", add(2, 3));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adds() {
        assert_eq!(add(2, 3), 5);
    }
}
""",
}


def chat_reply(lang: str) -> str:
    fence = multipl_e.LANGUAGES[lang].fence
    return f"Here is a solution.\n\n```{fence}\n{CHAT_ANSWERS[lang]}```\n\nIt runs in constant time.\n"


# -- reading the answer -----------------------------------------------------------


def test_the_block_that_defines_the_function_wins_over_a_longer_one() -> None:
    reply = (
        "Here it is:\n```rust\nfn add(x: isize, y: isize) -> isize {\n    x + y\n}\n```\n"
        'And how to call it:\n```rust\nfn main() {\n    println!("{}", add(2, 3));\n'
        '    println!("{}", add(-1, 1));\n    println!("{}", add(40, 2));\n}\n```\n'
    )
    assert extract_answer(reply, "rs", "add") == "fn add(x: isize, y: isize) -> isize {\n    x + y\n}\n"


@pytest.mark.parametrize("tag", ["rust", "rust,ignore", "{rust}", "Rust", "rs", ""])
def test_a_block_in_the_language_or_untagged_beats_a_longer_one_in_another(tag: str) -> None:
    reply = f"```text\n{'It adds its two arguments. ' * 5}\n```\nSo:\n```{tag}\n    x + y\n```\n"
    assert extract_answer(reply, "rs", "add") == "    x + y\n"


def test_an_answer_cut_off_inside_its_block_keeps_what_was_written() -> None:
    reply = "Sure:\n```go\nfunc add(x int, y int) int {\n\treturn x +"
    assert extract_answer(reply, "go", "add") == "func add(x int, y int) int {\n\treturn x +"


def test_a_reply_without_fences_is_used_whole() -> None:
    assert extract_answer("  return x + y;\n}\n", "js", "add") == "  return x + y;\n}\n"


def test_a_block_tagged_with_another_language_is_used_when_it_is_the_only_one() -> None:
    reply = "```c\nlong add(long x, long y) {\n    return x + y;\n}\n```"
    assert extract_answer(reply, "cpp", "add") == "long add(long x, long y) {\n    return x + y;\n}\n"


def test_a_whole_function_is_used_as_written_and_a_body_completes_the_prompt() -> None:
    cpp = CANARIES["cpp"]
    whole = extract_answer(cpp.good, "cpp", "add")
    assert multipl_e.solution_source(cpp.prompt, whole, "add", "cpp") == (whole, "whole")
    # MultiPL-E's completions stop before the closing brace: it is added.
    assert multipl_e.solution_source(cpp.prompt, cpp.bad, "add", "cpp") == (
        cpp.prompt + cpp.bad + "}\n",
        "continuation",
    )
    js = CANARIES["js"]  # this body closes the function itself
    assert multipl_e.solution_source(js.prompt, js.bad, "add", "js") == (js.prompt + js.bad, "continuation")
    # A body that names a local variable like its function is still a body.
    body = "  let add = x + y;\n  return add;\n}\n"
    assert multipl_e.solution_source(js.prompt, body, "add", "js") == (js.prompt + body, "continuation")


@pytest.mark.parametrize("lang", sorted(set(multipl_e.LANGUAGES) - {"py"}))
def test_the_entry_point_is_read_from_each_languages_prompt(lang: str) -> None:
    assert multipl_e.entry_point(CANARIES[lang].prompt, lang) == ("Add" if lang == "cs" else "add")


def test_the_entry_point_is_the_last_function_a_prompt_opens() -> None:
    prompt = "function helper(x){\n  return x;\n}\n// >>> sort_array([])\nfunction sort_array(arr){\n"
    assert multipl_e.entry_point(prompt, "js") == "sort_array"
    with pytest.raises(KumaruError, match="Could not find the function a Go prompt asks for"):
        multipl_e.entry_point("package x_test\n\n// func add(x int) int\n", "go")


# The regexes `fences` and `open_fence` replaced here: the same answers, in quadratic time.
OLD_FENCE = re.compile(r"```[ \t]*([^\s`]*)[^\n]*\n(.*?)```", re.DOTALL)
OLD_OPEN_FENCE = re.compile(r"```[ \t]*[^\s`]*[^\n]*\n(.*)$", re.DOTALL)


def test_the_fence_scans_find_what_the_regexes_they_replaced_found() -> None:
    rng = random.Random(9)
    pieces = ["`", "```", "\n", " ", "\t", "a", "rust", "{rust}", ",ignore", "x y"]
    for _ in range(3000):
        text = "".join(rng.choices(pieces, k=rng.randint(0, 14)))
        assert fences(text, multipl_e._FENCE_TAG) == OLD_FENCE.findall(text), text
        old = OLD_OPEN_FENCE.search(text)
        assert open_fence(text) == (old.group(1) if old else None), text


@pytest.mark.parametrize(
    "reply", ["`" * 100_000, "```" + "a" * 100_000, "```" * 33_333], ids=["backquotes", "open fence", "fences"]
)
def test_a_long_reply_without_a_closed_fence_is_read_in_linear_time(reply: str) -> None:
    # The regexes this replaced tried a fence from every ``` and read on to the
    # end of its line each time: 10 to 11 s each.
    assert cpu_seconds(functools.partial(extract_answer, reply, "rs", "add"), 0.5) < 0.5


# -- assembling one program ---------------------------------------------------------


def test_javascript_keeps_declarations_and_turns_imports_into_requires() -> None:
    js = CANARIES["js"]
    source = (
        '"use strict";\n'
        'import fs from "fs";\n'
        "const { strict: assert } = require('node:assert');\n"
        "export function add(x, y) {\n  return helper(x) + y;\n}\n"
        "const helper = (x) => x;\n"
        "console.log(add(2, 3));\n"
        "module.exports = { add };\n"
    )
    program = assemble("js", js.prompt, source, js.tests, MARK, "add")
    assert program.startswith('const fs = require("fs");\n')
    for kept in ("const { strict: assert } = require('node:assert');", "function add(x, y) {", "const helper"):
        assert kept in program
    for dropped in ("use strict", "export", "console.log", "module.exports"):
        assert dropped not in program
    # The tests run in a block of their own, so their `const assert` cannot clash.
    assert program.endswith(f"test();\n}}\nrequire('node:fs').writeSync(1, '\\n{MARK}\\n');\n")


def test_typescript_drops_type_only_imports_and_keeps_type_declarations() -> None:
    ts = CANARIES["ts"]
    source = (
        "import type { Pair } from './pair';\n"
        "import { readFileSync as read } from 'node:fs';\n"
        "interface Point { x: number; y: number }\n"
        "type Num = number;\n"
        "enum Sign { Minus = -1, Plus = 1 }\n"
        "function add(x: Num, y: Num): Num {\n  return x + y;\n}\n"
    )
    program = assemble("ts", ts.prompt, source, ts.tests, MARK, "add")
    # `declare` is only allowed at the top level: once, before everything.
    assert program.startswith("declare var require: any;\n")
    assert program.count("declare var require") == 1
    assert "import" not in program
    assert "const { readFileSync: read } = require('node:fs');" in program
    for kept in ("interface Point", "type Num = number;", "enum Sign", "function add(x: Num"):
        assert kept in program


def test_go_keeps_only_the_imports_that_the_code_and_tests_still_use() -> None:
    go = CANARIES["go"]
    source = (
        'package main\n\nimport (\n\t"fmt"\n\t"math"\n\t"strings"\n)\n\n'
        'func add(x int, y int) int {\n\treturn len(strings.Repeat("a", x)) + y\n}\n\n'
        "func main() {\n\tfmt.Println(add(2, 3))\n}\n\n"
        "func TestAdd(t *testing.T) {\n\tif math.Abs(float64(add(2, 3))) != 5 {\n\t\tt.Fail()\n\t}\n}\n"
    )
    program = assemble("go", go.prompt, source, go.tests, MARK, "add")
    # math served only the model's own test, which gives way to the hidden one;
    # Go refuses to compile an unused import.
    assert program.startswith('package add_test\n\nimport (\n\t"testing"\n\t"fmt"\n\t"strings"\n)\n\n')
    assert '"math"' not in program and "package main" not in program
    assert program.count("func TestAdd(") == 1
    assert program.count("func main()") == 1  # an ordinary function outside package main
    assert program.endswith(f'\tfmt.Print("\\n{MARK}\\n")\n}}\n')


def test_rust_keeps_the_models_items_and_drops_its_main_and_its_tests() -> None:
    rs = CANARIES["rs"]
    prompt = "use std::cmp::Ordering;\n\n" + rs.prompt
    source = (
        "#![allow(dead_code)]\n"
        "use std::collections::HashMap;\n\n"
        "fn add(x: isize, y: isize) -> isize {\n    let _seen: HashMap<isize, isize> = HashMap::new();\n"
        "    x + y\n}\n\n"
        'fn main() {\n    println!("{}", add(2, 3));\n}\n\n'
        "#[cfg(test)]\nmod tests {\n    use super::*;\n    #[test]\n    fn it_adds() {\n"
        "        assert_eq!(add(2, 3), 5);\n    }\n}\n"
    )
    program = assemble("rs", prompt, source, rs.tests, MARK, "add")
    order = ["#![allow(dead_code)]", "use std::cmp::Ordering;", "use std::collections::HashMap;", "fn add("]
    assert [program.index(text) for text in order] == sorted(program.index(text) for text in order)
    assert program.count("fn main()") == 1
    assert "mod tests" not in program and "it_adds" not in program
    assert program.rindex("assert_eq!") < program.index(MARK)
    # A `use` the model writes itself replaces the prompt's.
    own = source.replace("use std::collections", "use std::cmp::Ordering;\nuse std::collections")
    assert assemble("rs", prompt, own, rs.tests, MARK, "add").count("use std::cmp::Ordering;") == 1


def test_cpp_turns_the_tests_asserts_back_on_after_the_models_ndebug() -> None:
    cpp = CANARIES["cpp"]
    source = (
        "#define NDEBUG\n#include <vector>\n\n"
        "long add(long x, long y) {\n    return x + y;\n}\n\n"
        "int main() {\n    return add(2, 3) == 5 ? 0 : 1;\n}\n"
    )
    program = assemble("cpp", cpp.prompt, source, cpp.tests, MARK, "add")
    assert program.startswith("#include<assert.h>\n#include<bits/stdc++.h>\n")
    assert program.count("int main()") == 1
    assert program.index("#define NDEBUG") < program.index("#undef assert") < program.index("int main()")
    assert program.rindex("assert(") < program.index(f'std::fputs("\\n{MARK}\\n", stdout);')


def test_java_renames_the_answers_class_and_keeps_its_helpers() -> None:
    java = CANARIES["java"]
    source = (
        "import java.util.*;\nimport java.util.function.Function;\n\n"
        "public class Helper {\n    static long twice(long x) { return 2 * x; }\n}\n\n"
        "public class Solution {\n"
        "    static long half(long x) { return x / 2; }\n"
        "    public static long add(long x, long y) {\n        return Solution.half(Helper.twice(x)) + y;\n    }\n"
        "    public static void main(String[] args) {\n        System.out.println(add(2, 3));\n    }\n"
        "}\n"
    )
    program = assemble("java", java.prompt, source, java.tests, MARK, "add")
    assert program.count("import java.util.*;") == 1
    assert "import java.util.function.Function;" in program
    # Problem.java may declare one public type, Problem; references follow the rename.
    assert "public class" not in program and "Solution" not in program
    assert "Problem.half(Helper.twice(x))" in program
    assert program.index("class Helper") < program.index("class Problem {")
    assert program.count("void main(") == 1
    assert "KUMARU_NO_ASSERT" in program  # exits 97 if the JVM runs without -ea
    assert program.rindex("assert(") < program.index(MARK)


def test_csharp_unwraps_the_namespace_and_keeps_a_static_class_static() -> None:
    cs = CANARIES["cs"]
    source = (
        "using System;\nusing System.Linq;\n\n"
        "namespace Answers {\n"
        "    public static class Solution {\n"
        "        public static long Add(long x, long y) {\n            return new[] { x, y }.Sum();\n        }\n"
        "        public static void Main(string[] args) {\n            Console.WriteLine(Add(2, 3));\n        }\n"
        "    }\n"
        "}\n"
    )
    program = assemble("cs", cs.prompt, source, cs.tests, MARK, "Add")
    assert program.startswith(multipl_e._CS_DIRECTIVES)
    assert "namespace" not in program and "Solution" not in program
    assert "static class Problem {" in program
    assert program.count("void Main(") == 1
    assert program.count("using System;") == program.count("using System.Linq;") == 1
    assert "KUMARU_NO_DEBUG" in program  # exits 97 if built without DEBUG
    assert program.rindex("Debug.Assert(") < program.index(MARK)
    assert program.endswith(multipl_e._CS_LISTENER + "\n")


@pytest.mark.parametrize(
    ("statement", "expected"),
    [
        ("import fs from 'fs';", "const fs = require('fs');"),
        ('import * as path from "node:path";', 'const path = require("node:path");'),
        ("import { a, b as c } from './m';", "const { a, b: c } = require('./m');"),
        ("import d, { e } from 'm';", "const d = require('m');\nconst { e } = require('m');"),
        ("import './side-effect';", "require('./side-effect');"),
        ("import type { T } from './types';", ""),
        ("import { type T, u } from './m';", "const { u } = require('./m');"),
        ("import x = require('x');", "const x = require('x');"),
        ("import data from './data.json' with { type: 'json' };", "const data = require('./data.json');"),
    ],
)
def test_an_import_becomes_the_require_it_stands_for(statement: str, expected: str) -> None:
    assert multipl_e._js_import(statement) == expected


@pytest.mark.parametrize(
    ("path", "name"), [("fmt", "fmt"), ("math/rand", "rand"), ("math/rand/v2", "rand"), ("example.com/m/v3", "m")]
)
def test_a_go_import_binds_its_last_path_element(path: str, name: str) -> None:
    assert multipl_e._go_name(path) == name


@pytest.mark.parametrize(
    ("use", "names"),
    [
        ("use std::collections::{HashMap, HashSet as Set, self};", {"HashMap", "Set"}),
        ("pub use std::cmp::Ordering;", {"Ordering"}),
        ("use std::io::*;", set()),
    ],
)
def test_a_rust_use_binds_its_last_names(use: str, names: set[str]) -> None:
    assert multipl_e._rust_use_names(use) == names


# The regexes the scans in _js_from and _go_specs replaced: the same answers,
# in quadratic time or worse.
OLD_JS_FROM = re.compile(
    r"import\s*(.+?)\s*from\s*(['\"][^'\"]+['\"])(?:\s*(?:with|assert)\s*\{[^}]*\})?", re.DOTALL
)
OLD_GO_SPEC = re.compile(r"""(?:([A-Za-z_]\w*|\.)\s+)?["`]([^"`\n]+)["`]""")


def test_the_import_scan_reads_what_the_regex_it_replaced_read() -> None:
    rng = random.Random(9)
    tokens = ["import", "from", " ", "\n", "x", "{", "}", "'m'", '"m"', "'", '"', "with", "assert", ",", "*", "as"]
    ends = ["", " from 'm'", 'from"m"', " from 'm' with { x }", " from 'm'assert{}", " from 'm' with {}}"]
    for _ in range(3000):
        statement = "import" + "".join(rng.choices(tokens, k=rng.randint(0, 9))) + rng.choice(ends)
        old = OLD_JS_FROM.fullmatch(statement)
        # The regex also read a blank clause ("import from 'm'"): nothing is imported either way.
        expected = (old.group(1).strip(), old.group(2)) if old and old.group(1).strip() else None
        assert multipl_e._js_from(statement) == expected, statement


@pytest.mark.parametrize(
    ("text", "specs"),
    [
        ('"fmt"', [("", "fmt")]),
        ('m "math"\n. "strings"', [("m", "math"), (".", "strings")]),
        ('9abc "x"', [("abc", "x")]),  # an alias starts with a letter
        ('a.b "x"', [("b", "x")]),
        ('x. "p"', [(".", "p")]),
        ('"a" b "c"', [("", "a"), ("b", "c")]),
    ],
)
def test_a_go_import_is_its_path_and_the_alias_before_it(text: str, specs: list[tuple[str, str]]) -> None:
    assert multipl_e._go_specs(text) == specs


def test_the_go_import_scan_reads_what_the_regex_it_replaced_read() -> None:
    rng = random.Random(9)
    tokens = ['"', "`", "a", "ab", "9", "_", ".", " ", "\n", "\t", "é", "x/y"]
    for _ in range(3000):
        text = "".join(rng.choices(tokens, k=rng.randint(0, 14)))
        assert multipl_e._go_specs(text) == OLD_GO_SPEC.findall(text), text


def test_go_looks_up_only_so_many_import_names_that_are_not_identifiers(monkeypatch: pytest.MonkeyPatch) -> None:
    go = CANARIES["go"]
    source = (
        'import (\n\t"gopkg.in/yaml.v2"\n\t"x/a.b"\n)\n\n'
        "func add(x int, y int) int {\n\t_ = yaml.v2.X\n\t_ = a.b.Y\n\treturn x + y\n}\n"
    )
    program = assemble("go", go.prompt, source, go.tests, MARK, "add")
    assert '"gopkg.in/yaml.v2"' in program and '"x/a.b"' in program
    # Each such name is a search of the whole program. Past the limit, which
    # only a degenerate reply reaches, a name counts as unused.
    monkeypatch.setattr(multipl_e, "_GO_MAX_ODD_NAMES", 1)
    program = assemble("go", go.prompt, source, go.tests, MARK, "add")
    assert '"gopkg.in/yaml.v2"' in program and '"x/a.b"' not in program


def cs_assembled(source: str) -> str:
    cs = CANARIES["cs"]
    return assemble("cs", cs.prompt, source, cs.tests, MARK, "Add")


def test_csharp_looks_into_namespaces_only_so_deep(monkeypatch: pytest.MonkeyPatch) -> None:
    source = (
        "namespace A {\nnamespace B {\nnamespace C {\n"
        "class S { public static long Add(long x, long y) { return x + y; } }\n}\n}\n}\n"
    )
    assert "return x + y;" in cs_assembled(source)
    # Each level reads its whole body again; only a degenerate reply nests past the limit.
    monkeypatch.setattr(multipl_e, "_MAX_NAMESPACES", 2)
    assert "return x + y;" not in cs_assembled(source)


@pytest.mark.parametrize(
    "work",
    [
        # The regex this replaced backtracked over the spaces in cubic time: 5 s.
        functools.partial(multipl_e._js_import, "import" + " " * 2000 + "x"),
        # It tried an alias from every letter of the word, reading to its end: 11 s.
        functools.partial(multipl_e._go_imports, "import (\n" + "a" * 30_000 + "\n)"),
        # `=\s*[^;(]+` tried every split of the spaces between its two parts: 7 s.
        functools.partial(
            cs_assembled, "using a =" + " " * 30_000 + "\nclass S { static long Add(long x, long y) { return x; } }\n"
        ),
        # Each namespace read its body again, until Python's recursion limit
        # ended it: 2 s, then a RecursionError.
        functools.partial(cs_assembled, "namespace a {\n" * 1200 + "}\n" * 1200),
    ],
    ids=["js import", "go import", "cs using", "cs namespaces"],
)
def test_a_degenerate_declaration_is_assembled_in_linear_time(work: Callable[[], object]) -> None:
    assert cpu_seconds(work, 0.5) < 0.5


# -- the problems ---------------------------------------------------------------------


def hf_rows(dataset: str) -> list[dict[str, str]]:
    """All 164 problems, in reverse, in the given dataset's format."""
    if dataset == multipl_e.HUMANEVAL:
        return [
            {"task_id": f"HumanEval/{n}", "entry_point": f"f{n}", "prompt": f"def f{n}():\n", "test": "\n"}
            for n in reversed(range(164))
        ]
    return [
        {"name": f"HumanEval_{n}_f{n}", "prompt": f"fn f{n}() -> isize {{\n", "tests": "}\n\nfn main() {}\n"}
        for n in reversed(range(164))
    ]


def test_every_language_asks_the_same_152_problems_in_the_same_order(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    def load(dataset, config, split, revision=None):
        calls.append((dataset, config, split, revision))
        return hf_rows(dataset)

    monkeypatch.setattr(multipl_e, "load_hf_rows", load)
    rust, python = multipl_e.MultiPLERust().load(), multipl_e.MultiPLEPython().load()

    assert calls == [
        (multipl_e.DATASET, "humaneval-rs", "test", multipl_e.DATASET_REVISION),
        (multipl_e.HUMANEVAL, None, "test", multipl_e.HUMANEVAL_REVISION),
    ]
    assert len(rust) == multipl_e.N_PROBLEMS == 152
    assert [item.id for item in rust] == [item.id for item in python]
    assert [item.meta["problem"] for item in rust] == sorted(set(range(164)) - multipl_e.EXCLUDED)
    first = rust[0]
    assert (first.id, first.meta["entry_point"], first.meta["language"]) == ("HumanEval_0_f0", "f0", "rs")
    assert first.prompt.startswith("Complete the following Rust function.")
    assert first.prompt.endswith("```rust\nfn f0() -> isize {\n```")


def test_a_changed_dataset_is_an_error_not_a_smaller_benchmark(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(multipl_e, "load_hf_rows", lambda *args, **kwargs: hf_rows(multipl_e.DATASET)[:100])
    with pytest.raises(KumaruError, match="Expected 152 Rust problems, found 93"):
        multipl_e.MultiPLERust().load()


def cs_problem(signature: str, *asserts: str) -> tuple[str, str]:
    """A C# prompt and its tests, written the way MultiPL-E writes them."""
    prompt = f"{multipl_e._CS_USINGS}class Problem {{\n    // Does something.\n    public static {signature} {{\n"
    tests = "    }\n    public static void Main(string[] args) {\n"
    return prompt, tests + "".join(f"    {line}\n" for line in asserts) + "    }\n\n}\n"


LIST_TEST = 'Debug.Assert(Split(("a b")).Equals((new List<string>(new string[]{(string)"a", (string)"b"}))));'
# A string that looks like the end of the call and the comparison.
TRICKY_TEST = 'Debug.Assert(Split((")).Equals((")).Equals((new List<string>(new string[]{(string)")).Equals(("}))));'
DICT_TEST = 'Debug.Assert(Count(("a")).Equals((new Dictionary<string,long>(){{"a", 1L}})));'


def test_csharp_compares_list_and_dictionary_results_as_multipl_e_now_does() -> None:
    # nuprl/MultiPL-E#71: .Equals compares a List or a Dictionary by reference.
    prompt, tests = cs_problem("List<string> Split(string s)", LIST_TEST, TRICKY_TEST)
    assert multipl_e.fix_tests("cs", prompt, tests, "Split").splitlines()[2:4] == [
        '    Debug.Assert(Split(("a b")).SequenceEqual((new List<string>(new string[]{(string)"a", (string)"b"}))));',
        '    Debug.Assert(Split((")).Equals((")).SequenceEqual((new List<string>(new string[]{(string)")).Equals(("}))));',
    ]
    prompt, tests = cs_problem("Dictionary<string,long> Count(string s)", DICT_TEST)
    assert multipl_e.fix_tests("cs", prompt, tests, "Count").splitlines()[2] == (
        '    Debug.Assert(Count(("a")).Count == (new Dictionary<string,long>(){{"a", 1L}}).Count'
        ' && !Count(("a")).Except((new Dictionary<string,long>(){{"a", 1L}})).Any());'
    )


@pytest.mark.parametrize(
    "signature",
    ["Tuple<long, long> Split(string s)", "string Split(string s)", "long Split(List<long> xs)",
     "Tuple<List<string>, long> Split(string s)"],
)
def test_other_results_and_other_languages_keep_their_tests_as_published(signature: str) -> None:
    prompt, tests = cs_problem(signature, LIST_TEST)
    assert multipl_e.fix_tests("cs", prompt, tests, "Split") == tests
    prompt, tests = cs_problem("List<string> Split(string s)", LIST_TEST)
    assert multipl_e.fix_tests("java", prompt, tests, "Split") == tests


def test_a_list_test_that_cannot_be_rewritten_exactly_is_an_error() -> None:
    prompt, tests = cs_problem("List<string> Split(string s)", LIST_TEST, 'Debug.Assert(Split(("")).Count == 0);')
    with pytest.raises(KumaruError, match=re.escape('as nuprl/MultiPL-E#71 does: Debug.Assert(Split(("")).Count')):
        multipl_e.fix_tests("cs", prompt, tests, "Split")


def test_csharp_fixes_exactly_the_problems_it_expects_and_says_so(monkeypatch: pytest.MonkeyPatch) -> None:
    issue, expected = multipl_e.TEST_FIXES["cs"]
    lists = set(expected)

    def rows(*args, **kwargs):
        made = []
        for n in range(164):
            problem = cs_problem(f"{'List<string>' if n in lists else 'string'} Split(string s)", LIST_TEST)
            made.append({"name": f"HumanEval_{n}_split", "prompt": problem[0], "tests": problem[1]})
        return made

    monkeypatch.setattr(multipl_e, "load_hf_rows", rows)
    items = multipl_e.MultiPLECSharp().load()
    fixed = {item.meta["problem"] for item in items if "SequenceEqual" in item.meta["tests"]}
    assert fixed == expected and len(expected) == 43
    lists.add(max(set(range(164)) - expected - multipl_e.EXCLUDED))  # one more than expected
    with pytest.raises(KumaruError, match=r"Fixing the C# tests \(nuprl/MultiPL-E#71\) changed 44 problems"):
        multipl_e.MultiPLECSharp().load()

    cs, rs = multipl_e.MultiPLECSharp(), multipl_e.MultiPLERust()
    cs._toolchain = rs._toolchain = Toolchain({}, "fake")
    assert cs.fingerprint_extra()["test_fixes"] == issue  # runs graded on the old tests do not resume
    assert "test_fixes" not in rs.fingerprint_extra()


def test_dotnet_passes_a_right_list_or_dictionary_only_on_the_fixed_tests() -> None:
    bench = installed("cs")
    split = cs_problem("List<string> Split(string s)", LIST_TEST, TRICKY_TEST)
    count = cs_problem("Dictionary<string,long> Count(string s)", DICT_TEST)

    def graded(problem: tuple[str, str], entry: str, body: str, fixed: bool = True) -> str:
        prompt, tests = problem
        if fixed:
            tests = multipl_e.fix_tests("cs", prompt, tests, entry)
        signature = prompt.rstrip().splitlines()[-1].strip()
        answer = f"```csharp\nclass Problem {{\n    {signature}\n        {body}\n    }}\n}}\n```"
        item = BenchItem(id=entry, prompt="", meta={"prompt": prompt, "tests": tests, "entry_point": entry})
        return bench.grade(item, answer).extracted

    right = "return new List<string>(s.Split(' '));"
    assert graded(split, "Split", right) == "passed"
    assert graded(split, "Split", "return new List<string> { s };") == "failed"
    assert graded(split, "Split", right, fixed=False) == "failed"  # as published: by reference
    assert graded(count, "Count", "return new Dictionary<string,long> { { s, 1L } };") == "passed"
    # As many entries, one value wrong.
    assert graded(count, "Count", "return new Dictionary<string,long> { { s, 2L } };") == "failed"


def test_the_suite_runs_every_language_python_first() -> None:
    names = tuple(f"multipl_e_{lang}" for lang in multipl_e.LANGUAGES)
    assert names[0] == "multipl_e_py"
    assert suite_members("multipl_e") == suite_members("multipl-e") == names
    assert suite_members("humaneval") == ()
    assert isinstance(runner.resolve_benchmark("multipl-e-rs"), multipl_e.MultiPLERust)


# -- toolchains -----------------------------------------------------------------------


@pytest.mark.parametrize("lang", list(multipl_e.LANGUAGES))
def test_each_installed_toolchain_passes_its_canary(lang: str) -> None:
    bench = installed(lang)
    bench.preflight()  # a right answer passes, a wrong one fails
    assert bench.fingerprint_extra()["toolchain"] == bench.toolchain.version


@pytest.mark.parametrize("lang", list(multipl_e.LANGUAGES))
def test_a_typical_chat_answer_compiles_and_passes(lang: str) -> None:
    # Its imports, helpers, main and tests must each fit in or give way.
    bench = installed(lang)
    grade = bench.grade(canary_item(lang), chat_reply(lang))
    assert grade.correct, grade.detail
    assert grade.detail["mode"] == "whole"


@pytest.mark.parametrize(
    ("status", "verdict"), [("passed", "a wrong answer passed"), ("compile_error", "a right answer failed")]
)
def test_a_toolchain_that_grades_its_canary_wrongly_is_not_used(
    monkeypatch: pytest.MonkeyPatch, status: str, verdict: str
) -> None:
    monkeypatch.setitem(multipl_e._DETECT, "rs", lambda: Toolchain({"rustc": "rustc"}, "rustc 0.0-fake"))
    monkeypatch.setattr(codeexec, "run", finished(status, stderr="error: linker `cc` not found"))
    expected = f"The Rust toolchain (rustc 0.0-fake) failed Kumaru's self-test: {verdict} ({status})"
    with pytest.raises(ToolchainMissing, match=re.escape(expected)):
        multipl_e.MultiPLERust().preflight()


def test_the_sandbox_runs_the_images_own_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(codeexec, "check_docker", lambda image=None: "sha256:0123456789abcdef0123")
    go = multipl_e.MultiPLEGo()
    go.configure(sandbox="docker")
    assert go.fingerprint_extra() == {"sandbox": "docker", "toolchain": "docker image 0123456789ab"}
    program = multipl_e.build_program("go", go.toolchain, "", MARK, local=False)
    # The image's prebuilt standard library, copied where the read-only root lets builds write.
    assert program.steps[0] == Step(("cp", "-R", "/opt/gocache", "/tmp/gocache"), 60.0, "setup")
    assert program.env["GOCACHE"] == "/tmp/gocache"
    assert [step.argv for step in program.steps[1:]] == [
        ("go", "test", "-c", "-o", "prog.test", "program_test.go"),
        ("./prog.test",),
    ]
    java = multipl_e.MultiPLEJava()
    java.configure(sandbox="docker")
    run = multipl_e.build_program("java", java.toolchain, "", MARK, local=False).steps[-1]
    assert run.argv[run.argv.index("-cp") + 1] == ".:/opt/kumaru/javatuples-1.2.jar"
    ts = multipl_e.MultiPLETypeScript()
    ts.configure(sandbox="docker")
    assert ts.fingerprint_extra()["checker"] == "tsc"  # the image type-checks


def test_changing_the_sandbox_finds_the_toolchain_again_and_a_typo_is_refused() -> None:
    bench = multipl_e.MultiPLERust()
    fake = Toolchain({"rustc": "rustc"}, "fake")
    bench._toolchain = fake
    bench.configure(sandbox="none")
    assert bench._toolchain is fake
    bench.configure(sandbox="docker")
    assert bench._toolchain is None
    with pytest.raises(KumaruError, match="Unknown sandbox 'vm'"):
        bench.configure(sandbox="vm")


def test_each_language_compiles_and_runs_with_its_settings() -> None:
    def build(lang: str, **toolchain) -> codeexec.Program:
        return multipl_e.build_program(lang, Toolchain(**{"version": "fake", **toolchain}), "", MARK, local=False)

    compile_s, run_s = multipl_e.COMPILE_TIMEOUT_S, multipl_e.RUN_TIMEOUT_S
    assert build("py", tools={"python": "python3"}).steps == (
        Step(("python3", "-I", "-X", "utf8", "program.py"), run_s),
    )
    rs = build("rs", tools={"rustc": "rustc"}, flags=("-C", "x"))
    assert rs.steps == (
        Step(("rustc", "--edition", "2021", "-A", "warnings", "-C", "x", "-o", "program", "program.rs"), compile_s, "compile"),
        Step(("./program",), run_s),
    )
    cpp = build("cpp", tools={"cxx": "g++"}, flags=("-static",))
    assert cpp.steps[0].argv == ("g++", "-std=c++17", "-static", "-o", "program", "program.cpp")
    java = build("java", tools={"javac": "javac", "java": "java"}, jar="jt.jar")
    assert list(java.files) == ["Problem.java"]
    assert java.steps[1].argv[:2] == ("java", "-ea")  # Java ignores `assert` without -ea
    assert java.steps[1].argv[-3:] == ("-cp", ".:jt.jar", "Problem")
    cs = build("cs", tools={"dotnet": "dotnet"})
    assert cs.steps[0].argv[:5] == ("dotnet", "build", "program.cs", "-c", "Debug")  # Debug.Assert needs DEBUG
    assert cs.steps[1].argv == ("dotnet", "out/program.dll")
    strip = build("ts", tools={"node": "node"}, checker="strip")
    assert list(strip.files) == ["program.cts"]
    assert strip.steps == (
        Step(("node", "--experimental-transform-types", "--disable-warning=ExperimentalWarning", "program.cts"), run_s),
    )
    tsc = build("ts", tools={"node": "node", "tsc": "tsc.js"}, checker="tsc")
    assert [step.argv for step in tsc.steps] == [
        ("node", "tsc.js", "--target", "esnext", "program.ts"),
        ("node", "program.js"),
    ]


# -- grading ----------------------------------------------------------------------------


def test_a_failed_program_reports_its_stage_its_exit_code_and_the_end_of_its_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    programs = []

    def run(program, *, sandbox="none"):
        programs.append(program)
        return Outcome("failed", "run", 101, "", "x" * 2000 + "thread 'main' panicked", 0.4)

    monkeypatch.setattr(codeexec, "run", run)
    bench = multipl_e.MultiPLERust()
    bench._toolchain = Toolchain({"rustc": "rustc"}, "fake")
    grade = bench.grade(canary_item("rs"), CANARIES["rs"].good)

    assert (grade.correct, grade.extracted) == (False, "failed")
    detail = dict(grade.detail)
    stderr = detail.pop("stderr")
    assert detail == {"status": "failed", "stage": "run", "mode": "whole", "exit_code": 101}
    assert len(stderr) == multipl_e.STDERR_TAIL and stderr.endswith("panicked")
    [program] = programs
    assert program.marker in program.files["program.rs"]


def test_a_compiler_that_reports_errors_on_stdout_is_still_heard(monkeypatch: pytest.MonkeyPatch) -> None:
    error = "program.cs(3,5): error CS1002: ; expected"
    monkeypatch.setattr(codeexec, "run", finished("compile_error", "compile", 1, stdout=error))
    bench = multipl_e.MultiPLECSharp()
    bench._toolchain = Toolchain({"dotnet": "dotnet"}, "fake")
    grade = bench.grade(canary_item("cs"), CANARIES["cs"].bad)
    assert grade.extracted == "compile_error"
    assert (grade.detail["mode"], grade.detail["stderr"]) == ("continuation", error)


def test_a_passing_program_records_only_how_it_passed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(codeexec, "run", finished("passed"))
    bench = multipl_e.MultiPLEGo()
    bench._toolchain = Toolchain({"go": "go"}, "fake")
    grade = bench.grade(canary_item("go"), CANARIES["go"].good)
    assert grade.correct
    assert grade.detail == {"status": "passed", "stage": "run", "mode": "whole"}


# Tools that never run: these tests replace codeexec.run.
FAKE_TOOLS = {
    "py": {"python": "python"},
    "js": {"node": "node"},
    "ts": {"node": "node"},
    "cs": {"dotnet": "dotnet"},
    "java": {"javac": "javac", "java": "java"},
    "cpp": {"cxx": "c++"},
    "go": {"go": "go"},
    "rs": {"rustc": "rustc"},
}


@pytest.mark.parametrize("lang", list(multipl_e.LANGUAGES))
def test_a_degenerate_reply_is_graded_in_moments(lang: str, monkeypatch: pytest.MonkeyPatch) -> None:
    # A reply that repeats a piece until the token limit (some 30,000 to
    # 100,000 characters) is read before anything runs, with no time limit.
    # A probe that read again what a failed one had read made these pieces
    # quadratic: at a fifth of this length, C# took 13 s and JavaScript 2 s.
    monkeypatch.setattr(codeexec, "run", finished("compile_error", "compile", 1))
    bench = multipl_e.BY_LANGUAGE[lang]()
    bench._toolchain = Toolchain(FAKE_TOOLS[lang], "fake")
    body = "\n".join(piece * (20_000 // len(piece)) for piece in ("@", "$", "$@", "/[", "\\/"))
    reply = f"```{multipl_e.LANGUAGES[lang].fence}\n{body}\n```\n"
    item = canary_item(lang)
    grades: list[Grade] = []
    assert cpu_seconds(lambda: grades.append(bench.grade(item, reply)), 2.0) < 2.0
    assert grades[0].extracted == "compile_error"


# -- the grader self-check (scripts/check_graders.py) -----------------------------------


@pytest.fixture
def check_graders(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    spec = importlib.util.spec_from_file_location("check_graders", REPO / "scripts" / "check_graders.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "check_graders", module)  # dataclasses look it up
    spec.loader.exec_module(module)
    return module


def test_the_replayed_sample_is_half_passes_and_shares_the_rest_between_failures(check_graders) -> None:
    counts = {"OK": 500, "Exception": 300, "SyntaxError": 40, "Timeout": 90}
    # Timeouts cost 15 s each, so at most 10.
    assert check_graders.quotas(counts, 200) == {"OK": 100, "Exception": 50, "SyntaxError": 40, "Timeout": 10}
    # Too few failures: more passes instead.
    assert check_graders.quotas({"OK": 500, "Exception": 20}, 200) == {"OK": 180, "Exception": 20}


def test_each_verdict_in_the_sample_is_spread_across_problems(check_graders) -> None:
    rows = [
        {
            "problem": f"HumanEval_{p}_f",
            "completions": [f"{p}-{i}" for i in range(10)],
            "statuses": ["OK"] * 8 + ["Exception", "Timeout"],
            "stderrs": [None] * 10,
        }
        for p in range(6)
    ]
    sample = check_graders.sample_replays(rows, 20)
    assert sample == check_graders.sample_replays(rows, 20)  # the same every time
    assert Counter(replay.status for replay in sample) == {"OK": 10, "Exception": 5, "Timeout": 5}
    passes = Counter(replay.problem for replay in sample if replay.status == "OK")
    assert len(passes) == 6 and max(passes.values()) == 2
    assert all(replay.completion == f"{replay.problem.split('_')[1]}-{replay.index}" for replay in sample)
    assert {replay.stderr for replay in sample} == {""}


def test_only_documented_differences_from_multipl_e_are_excused(check_graders) -> None:
    replay = check_graders.Replay
    passed, failed = Grade(correct=True, extracted="passed"), Grade(correct=False, extracted="failed")
    go, rs, js = multipl_e.MultiPLEGo(), multipl_e.MultiPLERust(), multipl_e.MultiPLEJavaScript()
    excuse = check_graders.expected_difference

    newer_go = replay("HumanEval_114_f", 0, "SyntaxError", "", "./x_test.go:9:9: undefined: min")
    assert "built into Go since 1.21" in excuse(go, newer_go, passed)
    assert excuse(rs, newer_go, passed) == ""
    timeout = replay("HumanEval_0_f", 0, "Timeout", "", "")
    assert "MultiPL-E's limit (15 s to compile, 5 s to run)" in excuse(rs, timeout, passed)
    assert excuse(rs, timeout, failed) == ""
    assert excuse(js, replay("HumanEval_0_f", 0, "Exception", "", ""), passed) == ""

    type_error = replay("HumanEval_0_f", 0, "SyntaxError", "", "error TS2322")
    ts = multipl_e.MultiPLETypeScript()
    ts._toolchain = Toolchain({"node": "node"}, "fake", checker="strip")
    assert "types are not checked" in excuse(ts, type_error, passed)
    ts._toolchain = dataclasses.replace(ts._toolchain, checker="tsc")
    assert excuse(ts, type_error, passed) == ""

    # tsc failed on a file path written after the function, which Kumaru drops.
    prompt = "function f(x: number): number {\n"
    completion = "    return x;\n}\n/src/kata/7kyu/f.ts\n"
    tests = "declare var require: any;\nconst assert = require('node:assert');\nassert.deepEqual(f(1), 1);\n"
    item = BenchItem(id="HumanEval_0_f", prompt="", meta={"prompt": prompt, "tests": tests, "entry_point": "f"})
    tsc_says = "x.ts(4,6): error TS1499: Unknown regular expression flag.\n"
    stray = replay("HumanEval_0_f", 0, "SyntaxError", completion, "", prompt + completion + tests, tsc_says)
    assert "Kumaru drops" in excuse(ts, stray, passed, item)
    assert excuse(ts, stray, failed, item) == ""
    assert excuse(js, stray, passed, item) == ""
    # An error on a line Kumaru keeps is not excused, unless an earlier one led to it.
    kept_error = "x.ts(2,5): error TS2322: Type 'string' is not assignable.\n"
    assert excuse(ts, dataclasses.replace(stray, stdout=kept_error), passed, item) == ""
    assert excuse(ts, dataclasses.replace(stray, stdout=kept_error + tsc_says), passed, item) == ""
    assert excuse(ts, dataclasses.replace(stray, stdout=""), passed, item) == ""
    cut_off = "    return x;\n}\nconsole.log(f(\"1.txt.\n"
    runs_on = "x.ts(4,15): error TS1002: Unterminated string literal.\nx.ts(5,1): error TS1005: ',' expected.\n"
    call = replay("HumanEval_0_f", 0, "SyntaxError", cut_off, "", prompt + cut_off + tests, runs_on)
    assert "declare var require" in call.program.splitlines()[4]
    assert "Kumaru drops" in excuse(ts, call, passed, item)


def test_the_self_check_grades_in_the_sandbox_it_is_given(
    check_graders, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def no_docker(image=None):
        raise ToolchainMissing("no docker here")

    monkeypatch.setattr(codeexec, "check_docker", no_docker)
    monkeypatch.setattr(check_graders, "SETTINGS", check_graders.Settings(workers=1, sandbox="docker"))
    # Rust is checked in the image, not on this machine, so no Docker means not checked.
    assert check_graders.check_multipl_e_language("rs", skip_missing=True) == []
    assert "multipl_e_rs: not checked, no docker here" in capsys.readouterr().out
    assert check_graders.check_multipl_e_language("rs") == ["multipl_e_rs: no docker here"]
    with pytest.raises(SystemExit):
        check_graders.main(["multipl_e_rs", "--sandbox", "vm"])


def test_the_self_check_grades_on_the_tests_multipl_e_graded_with(
    check_graders, monkeypatch: pytest.MonkeyPatch
) -> None:
    # MultiPL-E's verdicts come from the tests as published, so the self-check
    # does not fix them (multipl_e.TEST_FIXES).
    prompt, tests = cs_problem("List<string> Split(string s)", LIST_TEST)
    row = {
        "problem": "HumanEval_1_split",
        "prompt": prompt,
        "tests": tests,
        "completions": ["        return new List<string>(s.Split(' '));\n"],
        "statuses": ["Exception"],
        "stderrs": [None],
    }
    graded: list[str] = []

    def run(program, *, sandbox="none"):
        graded.append(program.files["program.cs"])
        return Outcome("failed", "run", 1, "", "Assertion failed.", 0.1)

    monkeypatch.setattr(check_graders, "completion_rows", lambda lang: [row])
    monkeypatch.setattr(check_graders, "SETTINGS", check_graders.Settings(workers=1, sample=1))
    monkeypatch.setitem(multipl_e._DETECT, "cs", lambda: Toolchain({"dotnet": "dotnet"}, "fake"))
    monkeypatch.setattr(multipl_e.MultiPLECSharp, "preflight", lambda self: None)
    monkeypatch.setattr(codeexec, "run", run)
    assert check_graders.check_multipl_e_language("cs") == []
    [text] = graded
    assert LIST_TEST in text and "SequenceEqual" not in text

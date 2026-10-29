# `src\kumaru\eval\benchmarks\codeparse.py`

> Finds the top-level pieces of JavaScript, TypeScript, Java, C#, C++, Go and Rust source (imports, functions, classes, statements) without a parser per language, so MultiPL-E can rebuild a chat model's answer into a program that compiles beside the hidden tests.

**Read this when:** a MultiPL-E answer that looks right fails with `compile_error`, an assembled program has lost or duplicated a piece of the model's code, or you are adding a language to MultiPL-E.

---

## What it does

Two functions do the work; the rest are helpers built on them.

1. **`mask(text, lang)`** returns the text with every comment turned into spaces and every string or character literal turned into `0`s, character for character. A `}` inside a string or a comment can no longer pass for the end of a function, and every offset in the masked text is also an offset in the original, so code finds positions in the masked text and cuts the original. Comments keep their line breaks (lines still end where they did); literals lose theirs, so a multi-line string still reads as one complete expression.
2. **`chunks(text, lang)`** cuts the text into top-level pieces (`Chunk`s). A piece ends:
   - at a `;` outside any bracket;
   - at the `}` that closes its top-level block, unless what follows continues the statement: `;` (taken into the piece), a token such as `,` `.` `=` (and `{` `|` `&` `?` `:` `>` where the language allows), in JavaScript and TypeScript a `(` or `[` on the `}`'s own line (a call or a lookup, `{ a: 1 }["a"]`, or an array or indexed-access type, `{ x: number }[]`), or a word such as `else`, `catch`, `finally`, `while`, `as`, `satisfies`, and in JavaScript and TypeScript `from` (`import { a } from "m"`). A `(` or `[` on a later line starts a piece of its own, as an example call does after a function whose model left out the `;`;
   - in JavaScript, TypeScript and Go, which insert semicolons at line ends, at a line break after a complete expression (one ending in a word, a literal, `)`, `]`, `}`, `++` or `--`), unless (JavaScript, TypeScript) the next line starts with an operator or a continuing word such as `else` or `extends`;
   - a C++ or C# preprocessor line (`#include`, `#if`, following `\` continuations) and a Rust inner attribute (`#![...]`) are pieces of their own, and so is a stray closing bracket.

   The comments and blank lines before a piece belong to it. Joined, the pieces give back the text, less any comments after the last one.

The helpers:

- `uncomment` blanks comments but keeps literals, for reading an import's path or a `use` line.
- `matching` finds the bracket that closes an opener, in masked text.
- `defines(code, name, lang)` says whether `code` defines a function called `name`: `function name(`, `const name = …` (JavaScript, TypeScript), `func name(` (Go), `fn name(` (Rust), or a typed definition `T name(…) {` (C++, Java, C#, via `find_definition`), so a call or a prototype doesn't count.

### What the masking knows

| Language | Literals and comments it recognizes |
|---|---|
| all | `//` and `/* */` comments; `"…"` and `'…'` with backslash escapes; an unterminated string ends at its line |
| JavaScript, TypeScript | template literals, including `${…}` holes with strings, templates and braces inside; regular expressions (a `/` after an operator, `(`, or a word like `return` starts one, and it must close on the same line) |
| Java | text blocks `"""…"""` |
| C# | verbatim `@"…"` (`""` escapes a quote), interpolated `$"…{x}…"` with nested holes, raw `"""…"""` and `$"""…"""` |
| C++ | raw strings `R"delim(…)delim"` with prefixes (`u8R`, `LR`, …), prefixed strings (`u8"…"`, `L'x'`), digit separators (`1'000'000` is a number, not a character) |
| Go | raw strings `` `…` `` |
| Rust | nested block comments, raw strings `r#"…"#` (and `br`, `cr`), byte and C strings, characters (`'a'`, `'\n'`) versus lifetimes and labels (`'a`, `'outer:`) |

### Time linear in the text's length

A reply can degenerate into tens of thousands of characters of repetition (`@@@@…`, `/[/[/[…`, `` `${`${… ``), and grading it must still take moments. Everything here takes time linear in the text's length:

- **A failed probe for a literal is not read again.**
  - At each C# `@` or `$`, `scan` probes for a string (`@"…"`, `$"…"`). In a run of them with no `"` after it, the probe from each later character of the run would fail too, so `_cs_fail` remembers the run.
  - A JavaScript `/` probe that finds no closing `/` on its line marks every position it read as dead (`_dead`, inside and outside a `[class]` separately), and a later probe stops on reaching one.
- **Interpolation holes nest at most 32 deep** (`_MAX_NESTING`). A deeper hole runs to the end of the text. Real code never nests that deep; without the limit, Python's recursion limit ended the scan with a `RecursionError`.
- **Each lookup is indexed once.**
  - `matching` pairs every bracket of a text in one pass, and remembers the last 8 texts.
  - `chunks` measures each run of whitespace once.
  - `defines` finds the `=` after JavaScript's `const name: Type` in a sorted list of the `=` and `;` positions, and a binding's brace depth in a sorted list of the braces.
  - `find_definition` reads back from each `name(` for the type before it.

Before these, 4,000 `@` took 1 s to read as C#, and a reply of 20,000 took 148 s to grade; 10,000 `const add:` took 8 s as JavaScript. The tests read every ASCII punctuation character and every prefix the scanner probes, repeated to 4,000 characters, in each language, and fail if one takes more than 0.25 s. The slowest now takes under 0.02 s.

## Why it exists

MultiPL-E's own grader pastes a base model's completion of a function body between the prompt and the tests. A chat model answers differently: it rewrites the whole function and adds imports, helpers, a `main`, example calls, sometimes its own tests or a namespace. Pasting that next to the hidden tests gives two `main`s, a namespace the tests can't see, duplicate imports, or a class the tests don't know. To keep the model's helpers and imports and drop the rest, `multipl_e.py` must know where each top-level piece starts and ends in seven languages.

A real parser per language (tree-sitter grammars, or each compiler's own) would mean seven dependencies with native code, for a job that needs only the outline. Masking plus bracket depth gets far enough: chat answers are ordinary, conventionally formatted code, and code that confuses this module (unbalanced brackets, say) fails to compile anyway.

## Mental model

```
text    = 'f("}") // }\nint g() { return 1; }'
masked  = 'f(000)     \nint g() { return 1; }'
chunks  -> ['f("}")', ' // }\nint g() { return 1; }']    (cut at the ";" of a statement? no ";": cut at the line break? not in C++)
```

In practice, for C++:

```cpp
#include <vector>              // piece 1: a preprocessor line
using namespace std;           // piece 2: ends at ";"
// Doubles each value.         // \
vector<int> twice(vector<int> v) {   //  piece 3: the comment belongs to it, and it ends at the "}" that closes the body
    for (auto& x : v) x *= 2;  //  /
    return v;
}
int main() { twice({1}); }     // piece 4: multipl_e.py drops it, since the tests have their own main
```

`Chunk.head` is the piece's masked text, stripped: what callers classify (`import …`, `func Test…`, `#[cfg(test)]`, `int main(`).

## Public API

| Name | Signature | What it's for |
|---|---|---|
| `mask` | `mask(text: str, lang: str) -> str` | Comments to spaces, literals to `0`s, same length. Raises `ValueError` for a language not in `LANGUAGES`. |
| `uncomment` | `uncomment(text: str, lang: str, *, masked: str \| None = None) -> str` | Comments to spaces, literals kept. |
| `matching` | `matching(masked: str, i: int) -> int` | The index of the bracket closing `masked[i]` (`(`, `[` or `{`), or -1. Raises `KeyError` for any other character. |
| `Chunk` | `Chunk(start, end, text, code)`, `.head` | One piece: its span, its original text and its masked text. |
| `chunks` | `chunks(text, lang, *, masked=None, start=0, end=None) -> list[Chunk]` | The top-level pieces of `text[start:end]`. Pass `start` and `end` to cut the inside of a class body. |
| `defines` | `defines(code: str, name: str, lang: str, *, masked: str \| None = None) -> bool` | Whether `code` defines function `name`. In JavaScript and TypeScript that is `function name(`, or a `const`, `let` or `var` binding at the code's top level, its shallowest brace depth. Code that closes a `}` it never opened is a function body, and a binding in it is a local variable, even one named after the function (`let intersection = …` in `intersection`). |
| `find_definition` | `find_definition(masked: str, name: str) -> int` | C++, Java, C#: where `T name(...) {` (or `=> …`) starts, or -1. The word before `name` must be a type, not a keyword such as `return` or `new`. |
| `LANGUAGES` | `("js", "ts", "java", "cs", "cpp", "go", "rs")` | |

## How to use it

```python
from kumaru.eval.benchmarks.codeparse import chunks, defines

code = 'import { a } from "m";\nfunction f() { return "}"; }\nconsole.log(f());\n'
[c.head for c in chunks(code, "js")]
# ['import { a } from "m";', 'function f() { return "000"; }', 'console.log(f());']
defines(code, "f", "js")   # True
```

Callers in `multipl_e.py`:

- JavaScript and TypeScript keep declarations and imports, and drop other top-level statements (example calls);
- Go drops `package` lines and `Test…` functions and decides which imports are still used;
- Rust drops `fn main`, `mod tests` and `#[test]` items;
- C++ drops `main`;
- Java and C# find the type that defines the entry point and walk its members.

See [`multipl_e.md`](multipl_e.md).

## How to extend it

- **A new literal form** (a language adds a string syntax): handle it in `_Scanner._literal`, return the index just past the literal, and add a case to `test_mask_blanks_comments_and_zeroes_literals_at_the_same_offsets` in `tests\test_bench_codeparse.py`.
- **Keep it linear.** A probe that can fail must not be retried from each following character: remember what it read, as `_cs_fail` and `_dead` do. Recursion must be bounded, as holes are by `_MAX_NESTING`. Add each new prefix to `PUMPS` in `tests\test_bench_codeparse.py`, which times a long run of it in every language.
- **A token that continues a statement after `}`**: add it to `_AFTER_BRACE` for the language, or to `_AFTER_BRACE_WORDS`. One that continues it only on the `}`'s own line (JavaScript, TypeScript) goes in `_JS_SAME_LINE_AFTER_BRACE`.
- **A new language**: add it to `LANGUAGES`, then to the language checks in `_Scanner` (`_literal`, comments) and `chunks` (statement ends), and to `defines`. Then test `mask` and `chunks` on what models write in it: imports, generics, lambdas, nested types, attributes.
- **Bugs show up as compile errors.** When a MultiPL-E answer fails to compile, run `scripts\check_graders.py multipl_e_<lang>`: a disagreement on an answer MultiPL-E passed often means a piece was cut in the wrong place. Three such bugs are pinned as tests:
  - `import { a } from "m"` was cut at its `}`, which is why `from` now continues a statement;
  - `void` was taken for a keyword, so `void main(` was missed and a model's own `main` was kept beside the tests' `main`;
  - a function body that names a local variable after its function (`let intersection = …` in `intersection`) was taken for a whole answer, so the program never defined the function, and `tsc` reported `TS2349` (`… has no call signatures`) at the tests' calls. A binding now counts only at the top level. This one was found in the Docker sandbox, which type-checks TypeScript; 52 completions that MultiPL-E passed (19 JavaScript, 33 TypeScript) had failed this way.

  A code review found a fourth: a declaration whose `}` is followed on the same line by a call, a lookup or an array type (`const one = {value: 1}["value"];`, `type Points = { x: number }[];`) was cut at that `}`, and the rest dropped as an example statement. No MultiPL-E completion and no stored Qwen3.6 answer has that shape, so no verdict changed; `test_a_call_lookup_or_array_type_on_a_braces_line_continues_the_statement` pins it.

Pinned behaviour: `tests\test_bench_codeparse.py`.

## Failure modes

| Symptom | Likely cause | Fix |
|---|---|---|
| A MultiPL-E answer fails with `compile_error`, and the assembled program has half a function, or one split across pieces | A literal form or continuation this module doesn't know: the masked text has a stray bracket, or a piece ended early. | Reproduce with `chunks(answer, lang)`; fix the scanner or the continuation tokens, and add a test. |
| `mode: whole` on an answer that is a function body, and the tests fail with `TypeError: candidate is not a function`, or `ReferenceError` for one of the function's parameters (`tsc`: `TS2349`) | `defines` took something in the body for the function's definition. | Reproduce with `defines(answer, name, lang)`, and add the case to `test_defines_tells_definitions_from_calls`. |
| A model's helper function is missing from the assembled program | The helper's piece was classified as something to drop (an example call, a test). | Check `Chunk.head` for the piece and the caller's rules in `multipl_e.py`. |
| `ValueError: unsupported language 'py'` | Python isn't handled here; it doesn't need to be (`humaneval.py` assembles Python). | Use one of `LANGUAGES`. |
| Grading one answer takes seconds or more, the CPU busy in this module | A scan went quadratic on a long, repetitive reply: a probe or a regex that reads again what it read. | Profile `mask` and `chunks` on the item's `output` (`python -m cProfile`); fix the probe, and add the repeated piece to `PUMPS`. |

## Related files

- `src\kumaru\eval\benchmarks\multipl_e.py`: the only caller ([`multipl_e.md`](multipl_e.md)).
- `scripts\check_graders.py`: replays MultiPL-E's graded completions, which exercises this module ([`check_graders.py.md`](../../../../scripts/check_graders.py.md)).
- `tests\test_bench_codeparse.py`: pinned behaviour, including `test_a_repeated_piece_is_read_in_linear_time`, `test_a_long_run_of_a_binding_without_a_value_is_read_in_linear_time`, `test_interpolations_nested_past_the_limit_run_to_the_end_of_the_text` and `test_a_call_lookup_or_array_type_on_a_braces_line_continues_the_statement`.

# `src\kumaru\eval\benchmarks\multipl_e.py`

> MultiPL-E: HumanEval's problems translated into seven more programming languages, graded by compiling and running each language's own unit tests. `kumaru bench run multipl_e` scores a model per language (Python, JavaScript, TypeScript, C#, Java, C++, Go, Rust) on the same 152 problems, so the gaps show which languages it is weak in.

**Read this when:** you want to know how well a model writes a language other than Python, a language is skipped or fails its self-test, a MultiPL-E answer that looks right fails, or you are adding a language.

---

> **This runs model-written code on your machine,** like HumanEval: hence `--allow-code-exec`. Each program runs in a fresh temporary folder with time and memory limits, and everything it starts is killed afterwards, but it runs as you, with access to your files and the network. `--sandbox docker` runs each one in a throwaway container with no network instead. See [`codeexec.md`](codeexec.md).

## What it does

- **Eight benchmarks, one per language,** so each language gets its own score, confidence interval and paired comparison in `kumaru bench report` and `compare`:

  | Benchmark | Language | Toolchain it needs | Built and run with |
  |---|---|---|---|
  | `multipl_e_py` | Python | Python, the one Kumaru runs on | `python -I -X utf8 program.py` |
  | `multipl_e_js` | JavaScript | Node.js 18 or later | `node program.cjs` |
  | `multipl_e_ts` | TypeScript | `tsc` on `PATH` plus Node.js 18+; or, without `tsc`, Node.js 22.7+ (types stripped, not checked) | `node tsc.js --target esnext program.ts`, then `node program.js`; or `node --experimental-transform-types program.cts` |
  | `multipl_e_cs` | C# | the .NET 10 SDK (single-file apps) | `dotnet build program.cs -c Debug`, then `dotnet out/program.dll` |
  | `multipl_e_java` | Java | a JDK (javac and java) plus `javatuples-1.2.jar` | `javac -cp javatuples Problem.java`, then `java -ea -cp .;javatuples Problem` |
  | `multipl_e_cpp` | C++17 | g++ (or clang++ with GCC's libstdc++) | `g++ -std=c++17 -o program program.cpp`, then `./program` |
  | `multipl_e_go` | Go | Go | `go test -c -o prog.test program_test.go`, then `./prog.test` |
  | `multipl_e_rs` | Rust | rustc | `rustc --edition 2021 -A warnings -o program program.rs`, then `./program` |

  Each name also answers to `multipl-e-<lang>`. **`multipl_e`** (or `multipl-e`) is a *suite*: it runs all eight, Python first.
- **The same problems in every language.** MultiPL-E couldn't translate 12 of HumanEval's 164 problems into every language. Kumaru leaves those 12 (`EXCLUDED`) out everywhere and keeps the other 152 in HumanEval's order. Item ids are MultiPL-E's names (`HumanEval_116_sort_array`), and a `-n` sample with the same seed asks every language the same problems.
- **C#'s List and Dictionary tests are fixed as MultiPL-E fixed them** (`fix_tests`). The pinned dataset's C# tests compare a List or Dictionary result with `.Equals`, which .NET answers by reference, so no answer could pass 43 of the 152 C# problems ([nuprl/MultiPL-E#71](https://github.com/nuprl/MultiPL-E/issues/71)). MultiPL-E fixed its translator in January 2026 (commit 85833ac) but hasn't regenerated the dataset. `load` rewrites each of those assertions as the fixed translator writes it: `Debug.Assert(F(x).SequenceEqual(expected));` for a List, `Debug.Assert(F(x).Count == expected.Count && !F(x).Except(expected).Any());` for a Dictionary. Loading fails unless exactly the 43 problems in `TEST_FIXES` change, and an assertion in another form is an error, not a test left as it was. Kumaru applies only fixes MultiPL-E has made; the defects it hasn't fixed are listed under [Problems no answer can pass](#problems-no-answer-can-pass).
- **A language whose toolchain is missing is skipped, not scored 0.** Before any model loads, each language finds its toolchain and grades a canary problem with it: a right answer must pass and a wrong one must fail. In the suite, a language that fails this check is left out with a yellow note (`skipped multipl_e_go: Go needs the go command, which is not on PATH. Install Go: https://go.dev/dl/`), recorded under `skipped` in `run.json`, and listed again at the end as `not run`. A language named directly fails the run instead. If no language is left, the run fails with the first language's error.
- **For each answer** (`grade`):
  1. **Reads the code** (`extract_answer`): the fenced block that defines the function; else the longest block tagged with this language or untagged; else the longest block. An answer cut off inside its block keeps what it wrote, and a reply without a fence is used whole.
  2. **Decides how to read it** (`solution_source`): if the code defines the function itself, it is used as written (`mode: whole`). Otherwise it is taken as a function body that continues the prompt, as MultiPL-E's base-model completions are (`mode: continuation`): the prompt goes first, and any braces still open are closed. A definition counts only at the code's top level, so a body that names a local variable after its function (`let intersection = …` in `intersection`) is still a body.
  3. **Assembles one program** (`assemble`): the language's assembler keeps the answer's imports and helpers, drops what would clash with the hidden tests, and adds the tests (table below).
  4. **Compiles and runs it** through `codeexec.run`, with 120 s to compile and 15 s to run. It passes only if every step exits with 0 and the program prints a random marker placed after its last test.
- **Grading takes time linear in the reply's length.** Steps 1 to 3 run before step 4's time limits, and a reply that degenerates into repetition can reach 30,000 to 100,000 characters within the 8,192-token budget. Reading and assembling 100,000 characters of repetition takes at most 0.7 s of CPU in any language, and twice the length takes twice the time. The slowest piece found is a run of stray `)` or `]` in Java, since each stray closer is a piece of its own; C# comes next, at 0.5 s. Before, a fifth of that length took C# 13 s and JavaScript and TypeScript 2 s, and a reply of 20,000 `@` took C# 148 s. Three limits that only a degenerate reply reaches keep it linear:
  - C# namespaces are looked into down to 32 deep (`_MAX_NAMESPACES`). A namespace nested deeper is dropped, with what it holds.
  - Go checks the use of at most 64 import names that aren't identifiers, such as `gopkg.in/yaml.v2`'s `yaml.v2` (`_GO_MAX_ODD_NAMES`). Further ones count as unused.
  - Interpolation holes nest at most 32 deep ([`codeparse.md`](codeparse.md#time-linear-in-the-texts-length)).

### How each language's program is assembled

Chat models answer with more than the function: imports, helpers, a `main`, example calls, sometimes tests of their own or a namespace. [`codeparse.md`](codeparse.md) finds the top-level pieces; each assembler then keeps or drops them:

| Language | Kept from the answer | Dropped | The tests and the marker |
|---|---|---|---|
| Python | everything, as in [`humaneval.md`](humaneval.md) | | `check(<entry>)`, then the marker written to file descriptor 1 |
| JavaScript | declarations (`function`, `class`, `const`, `let`, `var`, decorators); imports, rewritten as `require` calls; `export` keywords removed | other top-level statements: example calls, `console.log`, `module.exports` | the tests in a `{ }` block of their own, so their names can't clash with the model's; then the marker via `require('node:fs').writeSync(1, …)` |
| TypeScript | as JavaScript, plus `interface`, `type`, `enum`, `namespace` and `declare`; type-only imports dropped | as JavaScript | as JavaScript; `declare var require: any;` moves to the top |
| C# | the type that defines the function becomes `class Problem` (a static class stays static), and references to its old name follow; other types go before it; `using`s; loose methods and fields go into `Problem`; namespaces are unwrapped | the model's `Main`; top-level statements; `#` directives | the tests' `Main` first checks that `Debug` is on (exit 97 if not), turns a failed `Debug.Assert` into "Assertion failed." and exit 1, and an unhandled exception into exit 1; then the marker |
| Java | the type that defines the function becomes `class Problem`; other types go before it, without `public`; imports, less those the prompt already has | the `package` line; the model's `main` | the tests' `main` first checks that `-ea` took effect (exit 97 with `KUMARU_NO_ASSERT` if not); the marker goes straight to `FileDescriptor.out` |
| C++ | everything; the prompt's `#include`s go first | the model's `main` | a prelude makes an abort exit with 134 (no crash dialog); after the model's code, `assert` is redefined, so the model's `#define NDEBUG` can't switch the tests' asserts off; the marker is printed before the tests' last `}` |
| Go | the code; the model's imports while the code or tests still use them (Go refuses an unused import). An import binds a name to a package, so an answer's `f "fmt"` is kept beside the tests' `"fmt"`; only the same name for the same package, or a second `_` import of one, is a duplicate. `fmt` is imported unless a name `fmt` already is | `package` lines; `Test…`, `Benchmark…`, `Fuzz…` and `Example…` functions | one `_test` package; the marker is printed before the tests' last `}` |
| Rust | the model's items and `use`s; `#![…]` attributes first; the prompt's `use`s unless the model's own bind the same names | `fn main`, `mod tests`, and items marked `#[cfg(test)]` or `#[test]` | the tests' `main`; the marker is written before its last `}` |

MultiPL-E's C++, Java, C# and Rust tests begin with the `}` that closes the prompt's open function; a whole answer closes its own, so that brace is dropped.

For example, this Rust answer:

````markdown
```rust
use std::cmp::max;

/// Adds x and y.
fn add(x: isize, y: isize) -> isize {
    helper(x) + y
}

fn helper(v: isize) -> isize { max(v, v) }

fn main() {
    println!("{}", add(2, 3));
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn it_adds() { assert_eq!(add(1, 1), 2); }
}
```
````

becomes this program (`mode: whole`):

```rust
use std::cmp::max;
/// Adds x and y.
fn add(x: isize, y: isize) -> isize {
    helper(x) + y
}
fn helper(v: isize) -> isize { max(v, v) }
fn main() {
    let candidate = add;
    assert_eq!(candidate(2, 3), 5);
    assert_eq!(candidate(-1, 1), 0);
    { use std::io::Write; let mut kumaru_out = std::io::stdout(); let _ = kumaru_out.write_all(b"\nkumaru-finished-0123456789abcdef\n"); let _ = kumaru_out.flush(); }
}
```

## Why it exists

- **HumanEval measures only Python.** To know whether a model can build an app, a website or a game in C#, TypeScript or Rust, it has to be measured in those languages. MultiPL-E (Cassano et al., 2023) translated HumanEval's problems, docstrings and tests into many languages. With the same problems everywhere, a gap between a model's Python and Rust scores is about its Rust, not about harder problems.
- **Graded by running tests,** like HumanEval: an answer counts only if it compiles and passes the hidden unit tests.
- **The Python baseline is asked the same way.** `multipl_e_py` uses HumanEval's own prompts and tests, but only the 152 shared problems, and the same instruction as the other languages. Compare it with them, not with `humaneval`, which asks all 164 problems with a different instruction.
- **Chat answers, not completions.** MultiPL-E's own harness pastes a base model's completion of the function body between the prompt and the tests. Kumaru asks a chat model for the whole function, so it has to rebuild the answer into a program that fits beside the tests. A completion-style answer still works (`mode: continuation`).
- **A missing toolchain must not look like a weak language.** Scoring an uninstalled Go 0 would read as "the model can't write Go". Hence the canary check and the skip.

## Mental model

```
prompt (MultiPL-E's signature + docstring, in the language)
  -> PROMPT: "Complete the following Rust function ... in a single ```rust code block. ..."
  -> model reply
  -> extract_answer -> solution_source (whole | continuation)
  -> assemble(lang): the answer's imports and helpers + the hidden tests + marker
  -> build_program: files + steps (compile 120 s, run 15 s) for this toolchain
  -> codeexec.run (local, or --sandbox docker)
  -> Grade(correct = exit 0 and marker printed, extracted = status, detail = {...})
```

**The prompt** (`PROMPT`) shows MultiPL-E's prompt in a fence of the language and says:

> Complete the following {language} function. Keep its name and signature, and return the complete function, plus any helper functions or imports it needs, in a single \`\`\`{fence} code block. Do not include a main function, tests or example usage.

As with HumanEval, presence and repetition penalties are turned off (`generation_overrides`), since code repeats tokens legitimately.

**What a result records** (`Grade.detail`), shown by `kumaru bench report`:

| Field | Meaning |
|---|---|
| `status` | `passed`, `failed`, `timeout` or `compile_error`. It is also the answer's `extracted` value, so the summary line reads `tests: passed 120, failed 20, compile_error 9, timeout 3`. |
| `stage` | Where it ended: `compile` or `run`. |
| `mode` | `whole` (the answer defined the function) or `continuation` (pasted after the prompt). |
| `exit_code` | The step's exit code, when it wasn't 0 and the step didn't time out. |
| `note` | E.g. `exited with code 0 before the tests finished (an early exit?)`. |
| `stderr` | For a failure: the last 800 characters of the step's error output, or of its standard output when that is empty (`dotnet` and `tsc` report compile errors on standard output). |

**What makes a run reproducible.** Besides the usual settings, the results fingerprint holds each language's `sandbox` and toolchain version (`node v24.17.0`, `.NET SDK 10.0.301`, `go version go1.27.3 windows/amd64`, `docker image 1a2b3c4d5e6f`…), for TypeScript whether types were checked (`checker`: `tsc` or `strip`), for C# the test fix (`test_fixes`: `nuprl/MultiPL-E#71`), and, for a local run on Windows, a memory cap other than the default 4096 MB (`memory_mb`, from `KUMARU_CODE_MEMORY_MB`; runs at the default record nothing, so their fingerprints are unchanged). Resuming a run with another compiler is refused, because the scores could differ, and so is resuming C# results graded on the tests as published. The datasets are pinned to fixed revisions.

## Public API

| Name | Signature | What it's for |
|---|---|---|
| `MultiPLE` | `Benchmark` subclass; `lang` set by each subclass | One language. `needs_code_execution = True`, `blocking_grader = True`. |
| `MultiPLEPython`, `MultiPLEJavaScript`, `MultiPLETypeScript`, `MultiPLECSharp`, `MultiPLEJava`, `MultiPLECpp`, `MultiPLEGo`, `MultiPLERust` | subclasses | Registered as `multipl_e_<lang>` (`kumaru.core.registry.MULTIPL_E_LANGUAGES`). |
| `BY_LANGUAGE` | `dict[str, type[MultiPLE]]` | Language key → class. |
| `MultiPLE.configure` | `configure(*, sandbox: str = "none") -> None` | Choose `none` or `docker`; forgets the toolchain found for the other. An unknown sandbox raises `KumaruError`. |
| `MultiPLE.toolchain` | property → `Toolchain` | Found on first use (`detect_toolchain`). Raises `ToolchainMissing` with an install hint. |
| `MultiPLE.preflight` | `preflight() -> None` | Finds the toolchain and grades the language's canary: a fenced whole answer must pass and a wrong continuation must fail, else `ToolchainMissing` ("... failed Kumaru's self-test: ..."). |
| `MultiPLE.fingerprint_extra` | `fingerprint_extra() -> dict` | `{"sandbox", "toolchain"}`, plus `"checker"` for TypeScript, `"test_fixes"` for C#, and `"memory_mb"` when `codeexec.fingerprint_limits` gives one. |
| `MultiPLE.load` | `load() -> list[BenchItem]` | The 152 problems, their tests fixed by `fix_tests`; `meta` = `prompt`, `tests`, `entry_point`, `language`, `problem` (HumanEval's number). Raises if the dataset doesn't give exactly 152, or if the fixes don't change exactly the problems in `TEST_FIXES`. |
| `MultiPLE.grade` | `grade(item, output) -> Grade` | Steps 1 to 4 above. Blocks for up to 135 s (compile plus run). |
| `LANGUAGES` | `dict[str, Language]` | `Language(key, name, fence, aliases)`: `name` is shown to the model, `fence` is the prompt's code-fence tag, `aliases` are the fence tags that count as this language. |
| `PROMPT` | `str` | The instruction (above). |
| `extract_answer` | `extract_answer(reply: str, lang: str, entry: str) -> str` | The code in a reply (step 1). |
| `solution_source` | `solution_source(prompt, code, entry, lang) -> tuple[str, str]` | The source, and `"whole"` or `"continuation"` (step 2). |
| `defines_entry` | `defines_entry(code, entry, lang) -> bool` | Whether `code` defines the function. |
| `entry_point` | `entry_point(prompt: str, lang: str) -> str` | The function a MultiPL-E prompt asks for: the last one it opens. |
| `fix_tests` | `fix_tests(lang, prompt, tests, entry) -> str` | The tests with the defects in `TEST_FIXES` fixed, as MultiPL-E's translator now writes them; other tests unchanged. Raises `KumaruError` ("Could not rewrite a C# test as nuprl/MultiPL-E#71 does: …") on an assertion it should fix but can't. |
| `assemble` | `assemble(lang, prompt, source, tests, marker, entry) -> str` | The program's text (step 3). |
| `build_program` | `build_program(lang, toolchain, text, marker, *, local=True) -> Program` | The files and steps for `codeexec.run`. `local=False` builds for the Linux sandbox. |
| `Toolchain` | `Toolchain(tools, version, env={}, keep_env=frozenset(), checker="", flags=(), jar="", setup=())` | The executables (absolute paths, or plain names inside the image), the version recorded in the fingerprint, the variables a step gets, extra compiler flags, the javatuples jar, and setup steps. |
| `detect_toolchain` | `detect_toolchain(lang: str, sandbox: str = "none") -> Toolchain` | This machine's tools, or the sandbox image's. |
| `javatuples_jar` | `javatuples_jar() -> Path` | `KUMARU_JAVATUPLES_JAR`, else `~/.cache/kumaru/javatuples-1.2.jar`. |
| Constants | `DATASET = "nuprl/MultiPL-E"`, `DATASET_REVISION`, `HUMANEVAL`, `HUMANEVAL_REVISION`, `EXCLUDED`, `N_PROBLEMS = 152`, `TEST_FIXES` (language → the upstream issue and the problems its fix changes), `RUN_TIMEOUT_S = 15.0`, `COMPILE_TIMEOUT_S = 120.0`, `STDERR_TAIL = 800`, `JAVATUPLES_URL`, `JAVATUPLES_SHA256` | |

## How to use it

### Run it

```powershell
# Every language whose toolchain is installed; 100 problems each (the default -n)
kumaru bench run multipl_e -c qwen36 --no-think --allow-code-exec

# All 152 problems, or one language
kumaru bench run multipl_e -c qwen36 --no-think --allow-code-exec --full
kumaru bench run multipl_e_rs multipl_e_ts -c qwen36 --no-think --allow-code-exec

# Each program in a container without network (build the image first: see below)
kumaru bench run multipl_e -c qwen36 --no-think --allow-code-exec --sandbox docker

kumaru bench report runs\bench\<run>
kumaru bench compare runs\bench\<baseline> runs\bench\<run>
```

`kumaru info` lists the suites under `benchmark suites`, and `kumaru bench list` shows each language with its description.

### Install the toolchains

Kumaru finds tools in `PATH`'s absolute folders (`codeexec.find_program`), never in the current folder or a relative `PATH` entry, and runs them by absolute path. It never installs them itself. Install the ones for the languages you want; the others are skipped.

| Language | What to install | Notes |
|---|---|---|
| JavaScript | [Node.js](https://nodejs.org) 18+ | |
| TypeScript | Node.js 22.7+; for type checking, also `npm install -g typescript` | Without `tsc`, types are stripped and not checked, so an answer with a type error can still pass. The fingerprint records which (`checker`), and runs with and without `tsc` don't mix. |
| C# | [.NET 10 SDK](https://dotnet.microsoft.com/download) | Builds single-file apps (`dotnet build program.cs`). Each answer takes a few seconds to build. |
| Java | a JDK, 17+ (e.g. [Eclipse Temurin](https://adoptium.net)), and the javatuples jar the tests import | Kumaru checks the jar's SHA-256 but never downloads it. Get it once: `curl.exe -L --create-dirs -o "$HOME\.cache\kumaru\javatuples-1.2.jar" https://repo1.maven.org/maven2/org/javatuples/javatuples/1.2/javatuples-1.2.jar`, or point `KUMARU_JAVATUPLES_JAR` at a copy. `javac` is found on `PATH` or in `JAVA_HOME` (an absolute one). |
| C++ | GCC: on Windows [WinLibs](https://winlibs.com) or MSYS2's MinGW-w64; on Linux `build-essential` | The tests include `<bits/stdc++.h>`, which only GCC's standard library has. On Windows, programs are linked statically with Linux's 8 MB stack, but `long` stays 32 bits, so two problems can't pass there as they do on Linux ([below](#problems-no-answer-can-pass)). |
| Go | [Go](https://go.dev/dl/) | Runs offline (`GOPROXY=off`, `GOTOOLCHAIN=local`). The first programs are slower while Go compiles the standard library into its cache. |
| Rust | [rustup](https://rustup.rs) (`rustc`) | Windows builds get Linux's 8 MB stack too. |

### Read the results

- A low score with many `compile_error`s means the model writes code the compiler rejects: wrong types, missing imports, syntax from another language. Many `failed` means code that compiles but computes the wrong answer. `kumaru bench report` shows each answer's `stderr`.
- Compare languages within one run: they answer the same problems. Compare a language across models with `kumaru bench compare`.
- Before trusting a language's grader, `python scripts\check_graders.py multipl_e_<lang>` replays MultiPL-E's own graded answers through it and reports the agreement ([`check_graders.py.md`](../../../../scripts/check_graders.py.md); the measured agreement is in [`BENCHMARKS.md`](../../../../../BENCHMARKS.md)). It grades them on the tests as published, which gave MultiPL-E's verdicts, not on the fixed C# tests.

### Differences from MultiPL-E's own harness

| | MultiPL-E | Kumaru |
|---|---|---|
| Answers | base-model completions of the body | whole functions from a chat model (completions still work) |
| Statements after the function | compiled and run with the tests, so a stray one (an example call, or a file path StarCoder2 wrote after the function) can fail the answer | JavaScript, TypeScript and C# keep only declarations and imports ([above](#how-each-languages-program-is-assembled)) |
| Time limits | 5 to 30 s by language | 120 s to compile, 15 s to run |
| C# | Mono (`csc`, then `mono`) | the .NET 10 SDK, `Debug` build |
| Rust | edition 2015 | edition 2021, warnings allowed |
| TypeScript | always type-checked with `tsc` | `tsc` when installed, else types stripped |
| Windows | (Linux only) | C++ and Rust get Linux's 8 MB stack; C++ links statically. C++'s `long` is 32 bits, not 64 ([below](#problems-no-answer-can-pass)). |
| Go | the Go in MultiPL-E's container, older than 1.21 | the installed Go, offline. Newer Go accepts code that older Go rejected, such as the built-in `min` and `max` (Go 1.21). |
| C# tests of a List or Dictionary result | `.Equals`, by reference: no answer passes ([#71](https://github.com/nuprl/MultiPL-E/issues/71); fixed in the translator since, not in the dataset) | rewritten as the fixed translator writes them (`SequenceEqual`; `Count` and `Except`) |

### Problems no answer can pass

Kumaru applies only fixes MultiPL-E has made. These defects of the pinned dataset remain, plus one that Windows adds, so some languages can't score every problem:

| Problem | Languages | Why |
|---|---|---|
| `HumanEval_92` (`any_int`) | all 7 translations | The answer turns on telling `3` from `3.0`, which Python can and the translations can't: each asks for the same call to be both true and false. JavaScript and TypeScript have one number type; C#, Java, C++ and Go pass floating-point numbers; Rust asserts `candidate(3.0, 4.0, 7.0)` both ways. MultiPL-E's fix ([#142](https://github.com/nuprl/MultiPL-E/issues/142)) drops contradictory assertions only when they are written alike, which only Rust's are. Kumaru doesn't apply it: Rust alone would gain a problem the other six can't pass, whereas leaving 92 unsolvable in all seven favours none. (#71, by contrast, puts C# back on a par with the others.) |
| `HumanEval_136` (`largest_smallest_integers`) | C#, Java | The function returns a pair of optional numbers, and the tests compare it with values of other types, which are never equal. C# returns `Tuple<Nullable<long>,Nullable<long>>`, but the tests expect values such as `Tuple.Create((Nullable<long>)null, 1L)`, a `Tuple<Nullable<long>,long>`, so only the 2 of its 11 assertions that expect two nulls can pass. Java's wrap the expected values once too often: `Optional.of(Optional.empty())` for a missing number, and `Optional.of(Pair.with(-2l, 1l))` for a whole pair. MultiPL-E fixed Java's other `Optional` tests ([#47](https://github.com/nuprl/MultiPL-E/issues/47)), but its translator still writes these ([reported on #47](https://github.com/nuprl/MultiPL-E/issues/47#issuecomment-6357018663)); it hasn't fixed C#'s either. Both are reported as [#195](https://github.com/nuprl/MultiPL-E/issues/195). |
| `HumanEval_51` (`remove_vowels`) | JavaScript | A test string holds a raw line break, a syntax error in JavaScript. Reported as [#196](https://github.com/nuprl/MultiPL-E/issues/196). |
| `HumanEval_87` (`get_row`) | Go | The function returns pairs, which Go's translation writes as `[]interface{}`. But some expected values put those pairs inside a `[][]int` literal, and one test passes a `[]interface{}` where the function takes `[][]int`. The test file never compiles, so no answer passes. Reported as [#193](https://github.com/nuprl/MultiPL-E/issues/193). |
| `HumanEval_114` (`minSubArraySum`), `HumanEval_139` (`special_factorial`) | C++, on Windows only | The C++ translations write Python's integers as `long`, which is 64 bits on Linux, where MultiPL-E grades, but 32 bits on Windows, with MinGW as with MSVC. 114's test passes −9999999999999999 in a `std::vector<long>`, which cuts it to 32 bits before the call, so no answer passes on Windows. 139's expects 125411328000, so an answer passes only if it widens the prompt's `long` return type to `long long`. On Linux both can pass: in the Docker sandbox, all 50 of MultiPL-E's published answers to each get its verdict, and right answers that keep `long` pass. Reported as [#194](https://github.com/nuprl/MultiPL-E/issues/194). |

So with all 152 problems, the best score is 152 in Python; 151 in TypeScript and Rust, and in C++ on Linux; and 150 in JavaScript, C#, Java and Go, and in C++ on Windows, where 139 also needs that wider type. The fix for #71 and the scan for C++ numbers beyond 32 bits covered all 152 problems. The other defects were found by reading every failure of the first run, Qwen3.6's ([`BENCHMARKS.md`](../../../../../BENCHMARKS.md#115-results)), which asked 100 of them; the other 52 haven't been audited, so a language's best score may be lower.

## How to extend it

**Add a language.** MultiPL-E has more (`humaneval-rb`, `-php`, `-swift`, `-scala`, `-lua`, `-jl` and others).

1. Add it to `LANGUAGES`, with its fence tag and aliases.
2. Write `_detect_<lang>` (raise `ToolchainMissing` with an install hint) and add it to `_DETECT`, plus a case in `_docker_toolchain`.
3. Add its steps to `build_program`.
4. Write `_assemble_<lang>` and dispatch to it from `assemble`. If the language is new to [`codeparse`](codeparse.md), teach that module its literals and statement ends first.
5. Add a canary to `_CANARIES` (a whole answer that passes, a body that fails).
6. Add a subclass and an entry in `BY_LANGUAGE`, then register it in `MULTIPL_E_LANGUAGES` in `src\kumaru\core\registry.py`, which also adds it to the suite.
7. Install the tool in `docker\sandbox\Dockerfile`.
8. Add the language to `scripts\check_graders.py`'s replay and check the agreement.

If a new language lacks a problem the others have, `load` fails with `Expected 152 … problems`. Then exclude that problem for every language (`EXCLUDED`, `N_PROBLEMS`), which changes the problem set and so needs a fresh baseline.

**Apply a fix MultiPL-E has made.** When MultiPL-E fixes a translator but not the dataset on the Hub, teach `fix_tests` to rewrite the tests exactly as the fixed translator writes them, and list the issue and the problems it changes in `TEST_FIXES`. `load` then fails if the fix changes any other problem, and the fingerprint keeps results graded on the old tests from being resumed with the new ones. Fix only what MultiPL-E has fixed, and only when the fix doesn't favour one language: see [Problems no answer can pass](#problems-no-answer-can-pass). When the Hub's dataset is regenerated, pin the new revision and drop the fix.

**MBPP.** MultiPL-E also translates MBPP (`mbpp-<lang>` configs). A second suite could reuse the assemblers and toolchains.

Pinned behaviour: `tests\test_bench_multipl_e.py`: answer reading, entry points, each assembler (including a JavaScript or TypeScript declaration that goes on after its `}`, `test_a_declaration_keeps_the_lookup_call_or_array_type_after_its_brace`, and a Go answer's own name for a package the tests import too, `test_go_keeps_an_answers_own_name_for_a_package_the_tests_import_too`), the problem set, the C# test fix (the exact rewrite, the problems it changes, and a real .NET run), the suite, the sandbox, the agreement sampling (graded on the tests as published), and the time a long, repetitive reply takes to grade (`test_a_degenerate_reply_is_graded_in_moments`). Twelve tests compile and run real programs; they need `KUMARU_TEST_TOOLCHAINS=1` and skip a language whose toolchain is missing.

## Failure modes

| Symptom | Likely cause | Fix |
|---|---|---|
| `skipped multipl_e_<lang>: … needs …, which is not on PATH.` | That language's toolchain isn't installed, or isn't on `PATH` for the shell that runs Kumaru, or only through a relative `PATH` entry or the current folder, which Kumaru doesn't search. | Install it (table above), or ignore it: the other languages still run. |
| `MultiPL-E's Java tests use org.javatuples, and javatuples-1.2.jar is not at …` | The jar isn't downloaded. | Run the `curl` command in the hint, or set `KUMARU_JAVATUPLES_JAR`. |
| `… is not Maven Central's javatuples-1.2.jar: its SHA-256 is …` | A different or damaged file. | Download it again. |
| `The <language> toolchain (…) failed Kumaru's self-test: a right answer failed` | The toolchain is broken or too unlike what the assembler expects: a compiler without GCC's `<bits/stdc++.h>`, an SDK older than .NET 10, a damaged install. The error shows the canary's output. | Check that the toolchain works by itself; or leave the language out. |
| `… failed Kumaru's self-test: a wrong answer passed` | Asserts are off or the pass marker is reached without the tests (a harness problem, not a model one). | Report it, with the toolchain version. |
| Exit code 97, with `KUMARU_NO_ASSERT` or `KUMARU_NO_DEBUG` | Java ran without assertions, or C# built without `DEBUG`: every assert would pass unchecked, so the program refuses to run its tests. | Shouldn't happen with `build_program`'s steps; report it, with the toolchain version. |
| Exit code 134 (C++) | The program aborted: an uncaught exception, or a library check. | A failure; read `stderr`. |
| `compile_error` on nearly every answer in one language | The toolchain differs from the ones the grader was checked with, rather than the model being weak. | Run `scripts\check_graders.py multipl_e_<lang>`: a low agreement points at the grader. |
| `Expected 152 <language> problems, found N.` | The dataset changed, or a cached copy is incomplete. | Kumaru pins the revisions; clear the cached dataset (`%USERPROFILE%\.cache\huggingface\hub\datasets--nuprl--MultiPL-E`) and retry. |
| `Fixing the C# tests (nuprl/MultiPL-E#71) changed N problems, not the 43 expected: [...] differ.` or `Could not rewrite a C# test as nuprl/MultiPL-E#71 does: …` | The C# tests aren't the pinned revision's, so the fix might rewrite them wrongly. | As above: clear the cached dataset. If a new revision is pinned on purpose, check whether it still needs the fix (`TEST_FIXES`). |
| `Could not find the function a <language> prompt asks for.` | A prompt in an unexpected shape. | A dataset problem; report the item. |
| `timeout` at the `compile` stage | A shared CPU: compilers are slowed by a model running on the CPU, a game, or too many programs at once. | Lower `KUMARU_CODE_WORKERS` ([`codeexec.md`](codeexec.md)). |
| Grading one answer takes seconds or more, the CPU busy in Kumaru rather than in a compiler | Reading or assembling went quadratic on a long, repetitive reply. | Profile `grade` on the item's `output` (`python -m cProfile`). Fix the scan ([`codeparse.md`](codeparse.md)) or regex, and add the reply to `test_a_degenerate_reply_is_graded_in_moments`. |
| `<run>\<bench> holds results from different settings (…)` when resuming | A compiler was updated, TypeScript gained or lost `tsc`, the sandbox changed, C# results were graded before its tests were fixed, or (local runs on Windows) `KUMARU_CODE_MEMORY_MB` changed: each is part of the fingerprint. | Start a new run (`--out` elsewhere), or set the variable back. |

## Related files

- `src\kumaru\eval\benchmarks\codeexec.py`: runs the programs ([`codeexec.md`](codeexec.md)).
- `src\kumaru\eval\benchmarks\codeparse.py`: finds the top-level pieces of an answer ([`codeparse.md`](codeparse.md)).
- `src\kumaru\eval\benchmarks\humaneval.py`: the Python assembly `multipl_e_py` reuses ([`humaneval.md`](humaneval.md)).
- `src\kumaru\core\registry.py`: `MULTIPL_E_LANGUAGES`, the `multipl_e` suite and its aliases ([`registry.md`](../../core/registry.md)).
- `src\kumaru\eval\benchmarks\runner.py`: the consent gate, `--sandbox`, suites and skipped languages ([`runner.md`](runner.md)).
- `docker\sandbox\Dockerfile`: the sandbox image, with every toolchain ([`Dockerfile.md`](../../../../docker/sandbox/Dockerfile.md)).
- `scripts\check_graders.py`: checks each language's grader against MultiPL-E's own verdicts ([`check_graders.py.md`](../../../../scripts/check_graders.py.md)).
- `docs\BENCHMARKS.md`: the method, the agreement check and the results ([`BENCHMARKS.md`](../../../../../BENCHMARKS.md)).
- `tests\test_bench_multipl_e.py`: pinned behaviour.

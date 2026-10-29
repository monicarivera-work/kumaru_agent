# `src\kumaru\agent\doc_chunks.py`

> Converts one documentation file into passages. It reads Markdown, MDX, reStructuredText, HTML or plain text and returns passages of at most 1,500 characters. Each passage is labelled with its section path, such as `json — JSON encoder and decoder > Basic Usage`. `doc_index.build_index` calls `document_chunks` on every file it indexes. The module holds pure functions over strings: no file access, network or execution.

**Read this when:** an indexed page gives wrong passages (missing text, leftover markup, a wrong section path or code language); you add a documentation format or change how one is converted; or `kumaru learn` reports `could not convert`, or is slow on a file.

---

## What it does

| Suffixes | Converter | Typical sources |
|---|---|---|
| `.md`, `.markdown` | `markdown_blocks` | The Rust Book, react.dev and npm sources, READMEs |
| `.mdx` | `markdown_blocks(text, mdx=True)`, which also drops `import` and `export` statements | Docusaurus and other MDX sites |
| `.rst`, `.rest` | `rst_blocks` | Sphinx sources, such as Godot's and Python's |
| `.html`, `.htm` | `html_blocks` | Rendered sites: Sphinx, rustdoc, mdBook, Docusaurus |
| `.txt`, `.text` | `text_blocks` | Plain text, split at blank lines |

`document_chunks(text, path)` does two things:
1. **Converts.** It chooses the converter by the suffix of `path`, ignoring case. The converter returns the document's title, if the format has one, and a flat list of `Block`s. Each block is a heading with a level from 1 to 6, a paragraph of prose, or code with its language. Markup is stripped and code is kept verbatim.
2. **Packs.** `chunk_blocks` packs the blocks into `Chunk(section, body)` passages.

For example, this `signals.md`:

````markdown
---
title: Signals
---

Signals let a node announce that something happened, without knowing who listens. See [the scripting tutorial](scripting.md).

## Connecting a signal {#connecting}

Call `connect()` on the signal, passing the method to run:

```gdscript
func _ready():
    $Button.pressed.connect(_on_button_pressed)
```

![The node dock](img/node_dock.png)

## Custom signals

Declare one with the `signal` keyword.
````

becomes one passage, because its sections are short. The section is `Signals`, and the body is:

````markdown
Signals let a node announce that something happened, without knowing who listens. See the scripting tutorial.

## Connecting a signal

Call `connect()` on the signal, passing the method to run:

```gdscript
func _ready():
    $Button.pressed.connect(_on_button_pressed)
```

## Custom signals

Declare one with the `signal` keyword.
````

## Why it exists
- **Search matches words, not syntax.** Without conversion, link targets, role names and class attributes would fill the index and the model's context.
- **Code stays exact.** Indentation and punctuation carry meaning in code, so it is kept verbatim in a fence labelled with its language.
- **Every passage can be cited.** The section path tells the model, and you, where a passage came from. It also helps ranking: `doc_index` weighs a match in the section path twice as much as one in the body.
- **Passages fit the budget.** At most 1,500 characters, several passages fit the default 6,000-character `agent.max_retrieved_chars`. Short sections are merged, so a passage carries enough context to be useful.
- **Inert.** Includes, links and code are never followed or run: a passage holds only what its own file says.
- **Bounded time.** Every converter runs in time proportional to the file's size, even on crafted files. One hostile page cannot stall a build.

## Mental model

### From blocks to passages
`chunk_blocks(blocks, title)` walks the blocks in order:
- **Section paths.** Headings form a stack, and a passage's section is the stack joined with ` > `. A document without a level-1 heading of its own is rooted at its title: the front matter's `title`, the HTML `<title>`, or else the file's name without its suffix. When there is none of these, the root is `Untitled`.
- **Headings are cleaned.** `{#id}` anchors, backticks and `*` are removed. A heading over 120 characters (`MAX_HEADING_CHARS`) is cut and ends with `…`. A path over 300 characters (`MAX_SECTION_CHARS`) becomes `first > … > last`. A heading with no text is dropped, and the text after it stays in the previous section.
- **Short sections are merged.** A passage is closed at the next heading once it holds 400 characters (`MIN_CHUNK_CHARS`). A shorter passage continues, with the heading kept inline as `## Heading`, and keeps the section it started in.
- **Long blocks are split.** A passage holds at most 1,500 characters (`MAX_CHUNK_CHARS`).
  - Prose is split at line ends, then at sentence ends, then after `,`, `;` and `:`, then at spaces. Pieces are rejoined greedily up to the limit, and no word is lost.
  - Code is split at line ends, and each piece gets a complete fence with the language. A fence is longer than any run of backticks that starts a line inside the code.

### Markdown and MDX
- **Front matter**, YAML between `---` lines or TOML between `+++` lines, within the first 200 lines: its `title` names the document, and the rest is dropped.
- **Code fences** of backticks or tildes are kept verbatim, dedented by the fence's own indent. The language is the first word of the info string: `rust,ignore` becomes `rust`, and `text`, `none` and `default` mean none. A fence closes only at a line of the same character, at least as long, so a fence of four backticks can show lines of three.
- **Indented code**, four columns past the margin, is kept verbatim too, in a fence without a language, as in CommonMark. In a list item, the margin is where the item's text starts. In a MkDocs admonition (`!!!` or `???`), content tab (`=== "Tab"`) or footnote, it is four columns in. Each block is dedented by its least indentation.
  - Indented code cannot interrupt a paragraph, so an indented line that continues one stays prose, as do a list item's later paragraphs.
  - `.mdx` files have no indented code. Their indentation is only layout.
- **Headings.** Both ATX (`## Title`) and setext (underlined with `===` or `---`) headings work. Closing `#`s need a space before them, so `# C#` keeps its `#`.
- **Dropped:**
  - images, HTML tags and comments (`<!-- -->`, and MDX's `{/* */}`);
  - link definitions, table rules (`|---|`), thematic breaks and `:::` container fences;
  - lines holding only tags, such as JSX components;
  - in `.mdx` files, `import` and `export` statements, with the lines that continue their brackets, up to 200 lines.
- **Links keep their text.** This covers `[text](url)` with one level of parentheses in the URL, `[text][ref]`, and `<https://...>` autolinks, which keep the URL. A link inside brackets, `[a [b](u) c]`, gives `[a b c]`.
- **MDN macros.** `{{domxref("Window.fetch")}}` keeps its target, `Window.fetch`, for macros named `*ref`, `element`, `header`, `method`, `status`, `glossary` and `attr`. Other macros, such as `{{Deprecated_Inline}}`, are removed.
- **Code spans**, such as `` `Vec<T>` ``, are kept verbatim, as are backticks that never close. Backslash escapes and HTML entities are decoded.
- **Comments and code spans** are told apart as in CommonMark: whichever opens first wins. So `` `<!--` `` is code and hides nothing, and backticks inside a comment belong to the comment.
  - A comment that starts its line may span blank lines. If it never closes, it hides the text up to the next code fence.
  - A comment inside a paragraph must close before the paragraph ends, at a blank line, a heading or a line that starts a comment. Otherwise its `<!--` stays as text.
- **Includes are not followed.** mdBook's `{{#include file.rs}}` is kept as written.

### reStructuredText
- **Section titles** are underlined, or over- and underlined. Levels follow the order in which adornment styles first appear, as in docutils, and the first title names the document. Field lists before it, such as Godot's `:github_url:`, are dropped.
- **Directives** are handled by kind:

| Kind | Examples | Result |
|---|---|---|
| Code | `code-block`, `code`, `sourcecode`, `code-tab`, `testcode`, `doctest`, `parsed-literal` | A fence. Its language is the argument of `code-block`, `code`, `sourcecode` or `code-tab`, else that of the last `.. highlight::` |
| Literal block | A paragraph ending in `::`, then an indented block | A fence in the `highlight` language. `Text::` keeps `Text:`. |
| Dropped | `image`, `figure`, `toctree`, `index`, `include`, `literalinclude`, `raw`, `meta`, `contents`, `auto*` | Nothing |
| Admonition | `note`, `warning`, `seealso`, `versionadded`, `deprecated`, `admonition` | A label such as `Note:` or `New in version 3.12:`, then the body |
| API object | `function`, `class`, `method`, `attribute`, and any `domain:name` | The signature in backticks, then the body |
| Labelled | `rubric`, `topic`, `tab`, `list-table`, `math` | The argument as text, then the body |
| Other | Any unknown directive | The body |

- **Roles keep their text.** `` :meth:`~collections.OrderedDict.move_to_end` `` gives `` `move_to_end` ``. `` :ref:`the signals tutorial <doc_signals>` `` gives `the signals tutorial`, or the target if the title is empty. A title made only of symbols, such as the 🔗 permalinks in Godot's class reference, is removed. Roles naming code, such as `func`, `meth`, `class`, `kbd` and `code`, put their text in backticks.
- **Links and references.** `` `Godot <https://godotengine.org>`_ `` gives `Godot`, and `` `<https://example.com>`_ `` the URL. `` `PEP 8`_ `` gives `PEP 8`, `|version|` gives `version`, and ``` ``literal`` ``` gives `` `literal` ``. Footnote references are removed.
- **Dropped:** comments, link targets (`.. _label:`), substitution definitions, and the borders of grid and simple tables. Cell text stays.
- **Nesting** deeper than 16 levels (`MAX_RST_DEPTH`) is kept as plain text, without structure. Real documentation nests a few levels deep; only crafted files go deeper.

### HTML
- **Content, not chrome.** If the page marks its content with `<main>` or `role="main"`, only that is kept. Otherwise, the prose in `<header>` and `<aside>` is dropped, and their headings stay.
- **Always dropped:**
  - `<nav>`, `<footer>`, `<head>`, `<script>`, `<style>`, `<template>`, `<svg>`, forms, media and `<dialog>`;
  - anything with the `hidden` attribute or `aria-hidden="true"`;
  - elements with page-furniture classes: permalinks (`headerlink`, `anchor`, `hash-link`), copy buttons, line numbers, screen-reader-only text, and rustdoc's `src` links and toggles. Headings are kept whatever their class, because Docusaurus marks its headings `anchor`.
- **Redirect stubs**, with `<meta http-equiv="refresh">`, give no passages.
- **Code.** `<pre>` is a code block. Its language comes from the nearest element, the `<pre>` itself or one around it, with a language class: `language-`, `lang-`, `highlight-` or `sourcecode-` followed by a name, as in `language-rust`, or a bare name such as `gdscript`, `rust` or `python`. If there is none, a `<code class="language-x">` inside the `<pre>` names it. Elsewhere, `<code>`, `<kbd>`, `<samp>` and `<tt>` become backticks.
- **Lists and tables.** List items start with `- `. Table rows go on their own lines, and cells are joined with ` | `.
- **Broken HTML** is read as browsers read it, where that matters:
  - A quote that never closes inside a tag makes the rest of the page part of that tag, so the rest is lost.
  - A `<script>` or `<style>` that never closes hides the rest of the page.
  - `<![CDATA[...]]>` is a bogus comment and is hidden. An MS Office `<![if !IE]>` section keeps its contents.
  - A `<!--` or `<![CDATA[` that never closes stays as text.

### Plain text
Paragraphs are split at blank lines and kept as written.

### Linear time
`doc_index` reads files of up to 4 MiB. A converter that is quadratic on some input, even input no real page contains, could stall a build for hours on one crafted file. So every converter takes time proportional to the file's size:
- **Regular expressions never rescan.** A repeat stops at the next opener: `[^\[\]]*`, not `.*?`.
- **No two parts of a pattern can match the same character.** Otherwise a failed match retries every way of sharing it, which doubles the work with each repeat. That is exponential, not just slow: `_MD_TAG_LINE` once let both the attribute text and an optional `/` take the `/` of `<a b="c" />`, so 20 such tags before some text took a second, and 40 would have taken weeks. Now the `/` belongs to one part only, and 5,000 tags take 10 ms. This is the pattern to look for when you add a regular expression: an optional part or a repeat next to another that accepts the same character.
- **Code spans and comments** are paired in one pass (`_protect_code_spans`, `_strip_comments`), because a regular expression would rescan the rest of the text from each unclosed run of backticks or `<!--`. Closers and paragraph ends are found by binary search in lists of their positions.
- **Indented code** is found in one pass over the lines (`_fence_indented_code`), which keeps the open list items on a stack.
- **HTML is guarded.** `html.parser` without the CVE-2025-6069 fix rescans the rest of the page from every `<` whose construct never closes.
  - `_escape_unclosed` escapes openers that come after the last closer of their kind, since they are text anyway.
  - `_HTML_ODD_SECTION` stops `html.parser` from raising on `<![` before an unexpected name.
  - `_HTML_END`, a comment, ends a tag that runs past the last `>` inside a quote.
- **No rescans per element.** The HTML converter counts open elements by tag and keeps the code language and the state of the current line as it goes. So stray end tags, deep nesting and long table rows cost the same per tag.
- **Bounded recursion.** reStructuredText rescans each nested block once per level, so nesting is capped at 16 levels.

Measured on a busy desktop PC, with files of 4 MiB, the largest `doc_index` reads:

| Input | Time |
|---|---|
| Crafted openers that never close, such as `[`, `<a `, `<!--`, `` `<!--` `` or `:ref:` | 0.75–5 s |
| One line of 220,000 `<Icon name="ok" />` tags and then text, or 10,500 lines of 20 | 0.65 s, 0.5 s |
| `<div>` nested 247,000 deep, around as many `<pre>` blocks | 7 s |
| reStructuredText nested 2,900 levels deep | 0.7 s |
| One-letter paragraphs, the most blocks a file can hold | Markdown 20 s, reStructuredText 13 s, HTML 7 s, plain text 4 s |

Real pages are far quicker: all 528 pages of Python 3.12's HTML documentation, 47 million characters, convert in about 22 s.

## Public API

| Name | Signature | What it's for |
|---|---|---|
| `MIN_CHUNK_CHARS`, `MAX_CHUNK_CHARS` | `400`, `1500` | A passage is closed at a heading once it holds 400 characters, and never holds more than 1,500. |
| `MAX_HEADING_CHARS`, `MAX_SECTION_CHARS` | `120`, `300` | Longer headings are cut and end with `…`. Longer paths become `first > … > last`. |
| `SECTION_SEPARATOR` | `" > "` | Joins the headings of a section path. |
| `MAX_RST_DEPTH` | `16` | reStructuredText nested deeper is kept as plain text. |
| `Block` | `@dataclass(frozen=True) Block(kind: Literal["heading", "text", "code"], text: str, level: int = 0, lang: str = "")` | One heading, paragraph or code block. `level` is set for headings and `lang` for code. |
| `Chunk` | `@dataclass(frozen=True) Chunk(section: str, body: str)` | One passage and its section path. |
| `code_lang` | `code_lang(word: str) -> str` | The language in a fence's info string or a class, lowercased: `rust,ignore` gives `rust`. `text`, `none` and `default` give `""`. |
| `fence` | `fence(code: str, lang: str = "") -> str` | Code in a Markdown fence longer than any backtick fence inside it. |
| `clean_heading` | `clean_heading(text: str) -> str` | A heading without its `{#id}`, backticks, `*` and extra spaces, cut to `MAX_HEADING_CHARS`. |
| `section_path` | `section_path(headings: Sequence[str]) -> str` | The headings joined with `SECTION_SEPARATOR`, at most `MAX_SECTION_CHARS`. |
| `text_pieces` | `text_pieces(text: str, limit: int = MAX_CHUNK_CHARS) -> list[str]` | Prose split into pieces of at most `limit` characters; see "From blocks to passages". |
| `code_pieces` | `code_pieces(code: str, lang: str = "", limit: int = MAX_CHUNK_CHARS) -> list[str]` | Code split at line ends into complete fences of at most `limit` characters. |
| `chunk_blocks` | `chunk_blocks(blocks: Sequence[Block], title: str) -> list[Chunk]` | Packs blocks into passages. `title` roots the section path when no level-1 heading has text. |
| `markdown_blocks` | `markdown_blocks(text: str, *, mdx: bool = False) -> tuple[str, list[Block]]` | The title from the front matter, or `""`, and the blocks. |
| `split_fenced_code` | `split_fenced_code(text: str) -> tuple[str, str]` | `(prose, code)`: the lines outside and inside the code fences of Markdown, such as a chat message, by the fence rules above. Fence lines are dropped, and a fence left open runs to the end. `doc_index` searches a question's prose first. |
| `rst_blocks` | `rst_blocks(text: str) -> tuple[str, list[Block]]` | The first section title and the blocks. |
| `html_blocks` | `html_blocks(text: str) -> tuple[str, list[Block]]` | The `<title>` and the blocks. |
| `text_blocks` | `text_blocks(text: str) -> tuple[str, list[Block]]` | `""` and one block per paragraph. |
| `SUPPORTED_SUFFIXES` | `frozenset[str]` | The lowercase suffixes that have a converter. `doc_index` indexes only these. |
| `document_chunks` | `document_chunks(text: str, path: str) -> list[Chunk]` | Converts one file's text into passages. `path` is `/`-separated; only its suffix and name are used. Windows and old Mac line ends become `\n`. Raises `KeyError` for a suffix not in `SUPPORTED_SUFFIXES`. |

## How to use it

### Preview a file's passages
To see what a file will contribute to an index, from the repository root:

```powershell
@'
import sys
from pathlib import Path
from kumaru.agent.doc_chunks import document_chunks

path = Path(sys.argv[1])
for chunk in document_chunks(path.read_text(encoding="utf-8-sig"), path.name):
    print(f"--- {chunk.section} ({len(chunk.body)} characters)\n{chunk.body}\n")
'@ | .venv\Scripts\python.exe - C:\docs\godot-docs\tutorials\scripting\signals.rst
```

Converting one file takes milliseconds, so you can check a format's output before you index a whole folder. To see which passages a question retrieves from a built index, see "See which passages match" in [doc_index.md](doc_index.md).

### Choose what to index
Index the documentation's source files when you have them. They hold just the content, with headings and code languages marked. Rendered HTML also works: the converter keeps the main content and drops the navigation. The Rust Book is the exception, because its Markdown pulls most code listings in with `{{#rustdoc_include}}` and `{{#include}}` lines, which are not followed. Index its HTML instead.

## How to extend it

### Support another format
1. Write a converter `xxx_blocks(text: str) -> tuple[str, list[Block]]`. It returns the document's title, or `""`, and its blocks: headings with a level from 1 to 6, prose with the markup stripped, and code with `lang` from `code_lang`. It must not read files, follow includes or links, or run anything.
2. Add its suffixes, in lower case, to `_CONVERTERS`. `SUPPORTED_SUFFIXES` follows, so `kumaru learn` picks up the new files.
3. Keep it linear; see "Keep converters linear".
4. Add tests to `tests\test_doc_chunks.py`: a realistic sample, the edge cases of the syntax, and crafted inputs for the linear-time tests.
5. Existing indexes keep working. Rebuild them to include the new files.

### Change a converter
- **Compare on real documentation.** Convert a whole folder with the old code and the new, and compare the passages file by file. Python's HTML documentation, which the python.org installer for Windows puts in `Doc\html`, makes a good test set. Every difference should be one you intended.
- **Rebuild to apply.** An index stores the passages it was built with, so a change reaches only indexes built after it. No `INDEX_FORMAT_VERSION` bump is needed: the format describes the tables, not how the passages were made.

### Keep converters linear
- **A repeat stops at the next opener.** In a regular expression, use `[^\[\]]*` rather than `.*?` when the closer may be missing. Two repeats must never be able to match the same run of characters, as `\s*` beside `[^>]*` can.
- **No possessive quantifiers or atomic groups.** They need Python 3.11, and Kumaru supports 3.10.
- **Keep state, don't rescan.** Per tag or per line, never search the open elements, the current line or the rest of the text. Keep a counter or a flag instead.
- **Cap recursion.** A converter that recurses per nesting level needs a depth limit, like `MAX_RST_DEPTH`. Python stops at about 1,000 nested calls.
- **Test with crafted input.** Add a case to `test_crafted_files_convert_in_linear_time` or `test_deeply_nested_files_convert_in_linear_time`. Each converts about 100,000 characters or more and must finish in 2 s; a quadratic converter takes far longer. Check that the test fails when you undo your fix.

## Failure modes

| Symptom | Likely cause | Fix |
|---|---|---|
| `KeyError: '.adoc'` from `document_chunks` | The suffix has no converter. | Check `SUPPORTED_SUFFIXES` first, as `doc_index` does, or add a converter. |
| `kumaru learn` reports `Skipped N (could not convert (ValueError)): ...` | A converter failed on these files. The others were indexed. | Set `$env:KUMARU_LOG_LEVEL = 'DEBUG'` and rebuild to log the traceback, then report it with the file. |
| A passage holds `{{#include ...}}` or other unconverted markup | Includes are never followed, and syntax a converter does not know is kept as text. | Index the rendered HTML instead. |
| An HTML page gives no passages, or only some | The page is a redirect stub, its text is outside `<main>`, or it is hidden or marked as page furniture. | Preview the file. Index the source files instead, if there are any. |
| An HTML page loses everything after some point | A quote, `<script>` or `<style>` that never closes. Browsers lose the same text. | Fix the page, or index its source. |
| The section path is the file's name, or `Untitled` | The file has no level-1 heading and no title. | None needed; the file's path is still cited. |
| Code has no language | Neither the fence, the directive nor the classes name one. | None needed; the code is still fenced. |
| Deeply nested reStructuredText loses its structure | It is nested more than 16 levels deep. | None needed for real documentation. |
| Indented Markdown prose appears in a code fence | As in CommonMark, text indented four columns past its margin, after a blank line, is code. Some site generators have containers the converter does not know, whose text is indented. | None needed: the text is still indexed and searchable. |
| One file takes many seconds to convert | The file is large and made of many small blocks. At 4 MiB, one-letter Markdown paragraphs take about 20 s. | None needed: time grows only in proportion to size, and `doc_index` skips files over 4 MiB. |

## Related files
- `src\kumaru\agent\doc_index.py` — builds and searches documentation indexes, calling `document_chunks` on each file; see [doc_index.md](doc_index.md).
- `src\kumaru\agent\knowledge.py` — single-file knowledge packs, which are sent whole and not converted; see [knowledge.md](knowledge.md).
- `tests\test_doc_chunks.py` — 60 offline tests: each format on realistic samples, the inline markup rules, comments next to code spans, indented code, broken HTML, passage limits, and linear time on crafted and deeply nested files.

"""Offline tests for turning documentation files into cited passages."""

import time

import pytest

from kumaru.agent.doc_chunks import (
    MAX_CHUNK_CHARS,
    MAX_RST_DEPTH,
    SUPPORTED_SUFFIXES,
    code_lang,
    document_chunks,
    fence,
)

# Each section is long enough to be a passage of its own (MIN_CHUNK_CHARS).
FILLER = "This sentence pads the section so that it closes before the next heading. " * 6


def bodies(chunks) -> str:
    return "\n\n".join(chunk.body for chunk in chunks)


def test_markdown_and_mdx_keep_sections_and_code_and_drop_markup() -> None:
    text = f"""---
title: Using State
---

import {{Sandpack}} from 'components';

# useState

<Intro>

`useState` is a React Hook that lets you add a [state variable](/learn/state) to your component.
{FILLER}

</Intro>

## Reference {{/*reference*/}}

Call `useState` at the top level of your component. <!-- editors: keep this short -->

```js {{2}} title="App.js"
const [count, setCount] = useState(0);
```
"""
    chunks = document_chunks(text, "reference/react/useState.mdx")
    assert [chunk.section for chunk in chunks] == ["useState", "useState > Reference"]
    text_out = bodies(chunks)
    assert "```js\nconst [count, setCount] = useState(0);\n```" in text_out
    assert "add a state variable to your component" in text_out
    for markup in ("title: Using State", "import", "<Intro>", "/learn/state", "/*", "editors"):
        assert markup not in text_out


def test_rst_titles_roles_directives_and_code_tabs() -> None:
    text = f""":article_outdated: True

.. _doc_using_character_body_2d:

Using CharacterBody2D/3D
========================

Introduction
------------

:ref:`CharacterBody2D <class_CharacterBody2D>` is for bodies controlled by code.
{FILLER}

.. note:: Movement code goes in ``_physics_process()``.

.. tabs::
 .. code-tab:: gdscript GDScript

    func _physics_process(delta):
        move_and_slide()

 .. code-tab:: csharp

    public override void _PhysicsProcess(double delta)
    {{
        MoveAndSlide();
    }}

Movement and collision
----------------------

Vector2 **get_real_velocity**\\ (\\ ) |const| :ref:`\U0001f517<class_CharacterBody2D_method_get_real_velocity>`

Call it like this::

    move_and_collide(velocity * delta)

Or with a velocity of its own:

::

    move_and_collide(Vector2(0, 50) * delta)
"""
    chunks = document_chunks(text, "tutorials/physics/using_character_body_2d.rst")
    assert [chunk.section for chunk in chunks] == [
        "Using CharacterBody2D/3D > Introduction",
        "Using CharacterBody2D/3D > Movement and collision",
    ]
    text_out = bodies(chunks)
    assert "CharacterBody2D is for bodies controlled by code." in text_out
    assert "`_physics_process()`" in text_out
    assert "```gdscript\nfunc _physics_process(delta):\n    move_and_slide()\n```" in text_out
    assert "```csharp\npublic override void _PhysicsProcess(double delta)\n{" in text_out
    assert "Call it like this:\n\n```\nmove_and_collide(velocity * delta)\n```" in text_out
    assert "own:\n\n```\nmove_and_collide(Vector2(0, 50) * delta)\n```" in text_out
    assert "Vector2 **get_real_velocity**() const\n" in text_out
    for markup in ("class_CharacterBody2D", ":ref:", ".. ", "article_outdated", "====", "----",
                   "|const|", "\U0001f517"):
        assert markup not in text_out


def test_html_keeps_the_main_content_and_drops_navigation_and_scripts() -> None:
    text = f"""<!DOCTYPE html><html><head><title>Using signals - Godot Engine</title>
<style>p {{ color: red }}</style><script>var leaked = "script text";</script></head>
<body><nav class="wy-nav-side"><a href="/">Home</a> Navigation text</nav>
<div role="main">
<h1>Using signals<a class="headerlink" href="#using-signals">¶</a></h1>
<p>Signals are Godot's version of the <em>observer</em> pattern. {FILLER}</p>
<h2>Connecting a signal in code</h2>
<div class="highlight-gdscript notranslate"><div class="highlight"><pre><span class="k">func</span> _ready():
    $Button.pressed.connect(_on_button_pressed)
</pre></div></div>
<p>Use <code>Callable</code> &amp; friends.</p>
</div>
<footer>Copyright Juan Linietsky</footer></body></html>"""
    chunks = document_chunks(text, "tutorials/scripting/signals.html")
    assert [chunk.section for chunk in chunks] == [
        "Using signals",
        "Using signals > Connecting a signal in code",
    ]
    text_out = bodies(chunks)
    assert "Signals are Godot's version of the observer pattern." in text_out
    assert "```gdscript\nfunc _ready():\n    $Button.pressed.connect(_on_button_pressed)\n```" in text_out
    assert "Use `Callable` & friends." in text_out
    for noise in ("Navigation text", "script text", "color: red", "\u00b6", "Copyright"):
        assert noise not in text_out


def test_rustdoc_html_keeps_signatures_docs_and_examples_without_page_chrome() -> None:
    text = """<html><head><title>Vec in std::vec - Rust</title></head><body>
<nav class="sidebar"><a href="#method.push">Sidebar link</a></nav>
<main><section id="main-content" class="content"><div class="main-heading">
<h1>Struct <span class="struct">Vec</span><button id="copy-path">Copy item path</button></h1>
<span class="sub-heading"><a class="src" href="../../src/alloc/vec/mod.rs.html">Source</a></span></div>
<details class="toggle top-doc" open><summary class="hideme"><span>Expand description</span></summary>
<div class="docblock"><p>A contiguous growable array type, written as <code>Vec&lt;T&gt;</code>.</p></div></details>
<section id="method.push" class="method"><span class="rightside"><span class="since">1.0.0</span>
<a class="src" href="#">Source</a></span><a href="#method.push" class="anchor">\u00a7</a>
<h4 class="code-header">pub fn <a href="#method.push" class="fn">push</a>(&amp;mut self, value: T)</h4></section>
<div class="docblock"><p>Appends an element to the back of a collection.</p>
<div class="example-wrap"><pre class="rust rust-example-rendered"><code>let mut vec = vec![1, 2];
vec.push(3);</code></pre></div></div>
</section></main></body></html>"""
    chunks = document_chunks(text, "std/vec/struct.Vec.html")
    assert [chunk.section for chunk in chunks] == ["Struct Vec"]
    text_out = bodies(chunks)
    assert "A contiguous growable array type, written as `Vec<T>`." in text_out
    assert "#### pub fn push(&mut self, value: T)\n\nAppends an element" in text_out
    assert "```rust\nlet mut vec = vec![1, 2];\nvec.push(3);\n```" in text_out
    for noise in ("Sidebar link", "Copy item path", "Source", "Expand description", "1.0.0", "\u00a7"):
        assert noise not in text_out


def test_plain_text_is_split_into_bounded_passages_without_losing_words() -> None:
    sentences = [f"Sentence number {n} explains one more thing about ownership." for n in range(300)]
    text = " ".join(sentences)
    chunks = document_chunks(text, "notes/ownership.txt")
    assert len(chunks) > 1
    assert all(len(chunk.body) <= MAX_CHUNK_CHARS for chunk in chunks)
    assert {chunk.section for chunk in chunks} == {"ownership"}  # untitled: named after the file
    assert bodies(chunks).split() == text.split()


def test_long_code_is_split_at_line_ends_into_complete_fences() -> None:
    code = "\n".join(f"    let value_{n} = compute({n});" for n in range(200))
    chunks = document_chunks(f"# Big example\n\n```rust\nfn main() {{\n{code}\n}}\n```\n", "big.md")
    assert len(chunks) > 2
    for chunk in chunks:
        assert len(chunk.body) <= MAX_CHUNK_CHARS
        assert chunk.body.startswith("```rust\n")
        assert chunk.body.endswith("\n```")
    lines = [line for chunk in chunks for line in chunk.body.split("\n") if "let value_" in line]
    assert len(lines) == 200


def test_fences_outlast_backticks_inside_the_code_and_languages_are_normalized() -> None:
    assert fence("print(1)", "python") == "```python\nprint(1)\n```"
    assert fence("```\nnested\n```", "md").startswith("````md\n")
    assert code_lang("rust,ignore") == "rust"
    assert code_lang("C#") == "c#"
    assert code_lang("text") == ""


@pytest.mark.parametrize(
    ("path", "text"),
    [
        ("empty.md", " \n\n\t\n"),
        ("redirect.html", '<html><head><meta http-equiv="refresh" content="0; url=new.html">'
                          "</head><body></body></html>"),
        ("only-front-matter.mdx", "---\ntitle: Nothing here\n---\n"),
    ],
    ids=["blank", "redirect", "front-matter"],
)
def test_files_without_text_yield_no_passages(path: str, text: str) -> None:
    assert document_chunks(text, path) == []


def test_every_supported_suffix_converts_and_windows_newlines_are_normalized() -> None:
    for suffix in SUPPORTED_SUFFIXES:
        chunks = document_chunks("Hello\r\nworld\r\n", f"page{suffix}")
        assert "\r" not in bodies(chunks)
        assert "Hello" in bodies(chunks), suffix


def test_markdown_inline_markup_keeps_link_text_macro_targets_and_code() -> None:
    text = """# Fetching data

Call {{domxref("Window.fetch", "fetch()")}} or {{ HTMLElement( 'a' ) }} here.{{Deprecated_Inline}}
![diagram](img/flow(1).png "Flow") See [the guide](https://example.com/a_(b)) and [MDN][mdn] or <https://example.org/x>.
Nested [a [b](u) c] stays readable. Code `[not](a link)` and ``{{x}} `y` `` stay, as does a lone ` backtick.
FILLER

## Using fetch() ##

FILLER

### Options {#options}

FILLER

### ###

After an empty heading.

# Title#

FILLER
""".replace("FILLER", FILLER)
    chunks = document_chunks(text, "web/api/fetch.md")
    assert [chunk.section for chunk in chunks] == [
        "Fetching data",
        "Fetching data > Using fetch()",
        "Fetching data > Using fetch() > Options",  # an empty heading names no section
        "Title#",  # a closing # needs a space before it
    ]
    text_out = bodies(chunks)
    assert "Call Window.fetch or a here.\n" in text_out
    assert "See the guide and MDN or https://example.org/x." in text_out
    assert "Nested [a b c] stays readable." in text_out
    assert "Code `[not](a link)` and ``{{x}} `y` `` stay, as does a lone ` backtick." in text_out
    assert "After an empty heading." in chunks[2].body
    for markup in ("Deprecated", "diagram", "flow(1)", "example.com", "[mdn]", "#options", "##"):
        assert markup not in text_out


def test_comment_openers_in_code_spans_hide_nothing() -> None:
    # `<!--` in backticks once hid everything up to the next `-->`, fence or end of file.
    text = f"""# Comments

Write `<!--` to start an HTML comment.
{FILLER}

## Examples

A whole comment in backticks, `<!-- note -->`, shows as written. MDX writes `{{/* note */}}`.
{FILLER}

## Later section

Text after the examples survives. <!-- an editor's note --> Inline comments go.

<!--
A block comment

with a blank line inside
-->

The end.
"""
    chunks = document_chunks(text, "comments.md")
    assert [chunk.section for chunk in chunks] == [
        "Comments",
        "Comments > Examples",
        "Comments > Later section",
    ]
    text_out = bodies(chunks)
    assert "Write `<!--` to start an HTML comment.\n" in text_out
    assert "A whole comment in backticks, `<!-- note -->`, shows as written. MDX writes `{/* note */}`.\n" in text_out
    assert chunks[-1].body == "Text after the examples survives. Inline comments go.\n\nThe end."


@pytest.mark.parametrize(
    ("text", "body"),
    [
        ("Text <!-- a ` b --> more ` end.", "Text more ` end."),  # the comment opens first
        ("A lone ` backtick, then <!-- a note --> text.", "A lone ` backtick, then text."),
        # Inside a paragraph, a comment cannot cross a blank line: its opener is text.
        ("a <!-- b\n\nc --> d", "a <!-- b\n\nc --> d"),
        ("<!-- block\n\nwith a blank line -->\nAfter.", "After."),
        ("Before.\n\n<!-- never closed\n\nHidden.", "Before."),
    ],
    ids=["backtick-in-comment", "lone-backtick", "inline-across-blank", "block-across-blank", "unclosed-block"],
)
def test_markdown_comments_end_where_commonmark_ends_them(text: str, body: str) -> None:
    assert bodies(document_chunks(f"# T\n\n{text}", "x.md")) == body


def test_indented_code_is_kept_verbatim_and_indented_prose_stays_prose() -> None:
    # Indented code once went through the inline rewrites: "<span>OK</span>" became "OK".
    text = """# Setting HTML

Assign the markup as a string:

    element.innerHTML = "<span>OK</span>";
    // <!-- not a comment --> [not](a link)

A paragraph
    continued by an indented line.

1. Install the package.

    A continuation paragraph, indented four.

        npm install kumaru

!!! note
    An admonition paragraph is prose.

<!--
    Indented text in a comment is not code.
-->
"""
    text_out = bodies(document_chunks(text, "x.md"))
    assert (
        '```\nelement.innerHTML = "<span>OK</span>";\n// <!-- not a comment --> [not](a link)\n```'
        in text_out
    )
    assert "```\nnpm install kumaru\n```" in text_out
    assert text_out.count("```") == 4
    for prose in ("continued by an indented line.", "A continuation paragraph", "An admonition paragraph"):
        assert prose in text_out
    assert "in a comment" not in text_out
    # MDX has no indented code: indentation there is component layout.
    mdx = document_chunks("# T\n\n<Note>\n    Indented text in a component.\n</Note>\n", "x.mdx")
    assert bodies(mdx) == "Indented text in a component."


def test_rst_roles_and_links_keep_their_titles() -> None:
    text = """Signals
=======

See :doc:`spaced title   <a/b>` and `Godot <https://godotengine.org>`_ or `<https://bare.example>`_.
Method :ref:`\U0001f517<class_Node_method_x>` then :not a role: and :class:`~godot.Node2D`.
Also :py:meth:`Path.read_text() <pathlib.Path.read_text>` and :term:`signal`.
"""
    assert [chunk.body for chunk in document_chunks(text, "signals.rst")] == [
        "See spaced title and Godot or https://bare.example.\n"
        "Method then :not a role: and `Node2D`.\n"
        "Also `Path.read_text()` and signal."
    ]


@pytest.mark.parametrize(
    ("page", "body"),
    [
        ("<p>Before</p><![ CDATA[hidden]]><p>After</p>", "Before\n\nAfter"),
        ("<p>Before</p><![if !IE]><p>Shown</p><![endif]><p>After</p>", "Before\n\nShown\n\nAfter"),
        # A quote that never closes makes the rest of the page one broken tag, as in HTML5.
        ("<p>Before</p><p>Text <a title='>' href=\"x>y</a> lost</p><p>Lost</p>", "Before\n\nText"),
        ("<p>Before</p><p>a < b and c <!-- never closed", "Before\n\na < b and c <!-- never closed"),
    ],
    ids=["odd-marked-section", "office-conditional", "unclosed-quote", "unclosed-comment"],
)
def test_broken_html_keeps_the_text_before_the_break(page: str, body: str) -> None:
    assert [chunk.body for chunk in document_chunks(f"<h1>Page</h1>{page}", "page.html")] == [body]


# Repeated openers that never close, each of which once made a converter rescan the rest
# of the file: quadratic, so a crafted 4 MiB page would take hours. Linear takes ~0.1 s.
@pytest.mark.parametrize(
    ("path", "unit", "tail"),
    [
        ("x.md", "[", ""),
        ("x.md", "[x ", ""),
        ("x.md", "![x](", ""),
        ("x.md", '{{x("', ""),
        ("x.md", "` `` ", ""),
        ("x.md", "<http:", ""),
        ("x.md", "{#", ""),
        ("x.md", "|--", "x"),
        ("x.md", " ", "#"),
        ("x.md", "<a ", ""),
        ("x.md", '<a b="c" /> ', "x"),
        ("x.md", "a <!-- `", ""),
        ("x.md", "` {/* ", ""),
        ("x.md", "- a\n\n    b\n", ""),
        ("x.md", "\n    <!--", ""),
        ("x.rst", ":a", ""),
        ("x.rst", ":ref:`a <", "`"),
        ("x.rst", "`a <", ""),
        ("x.rst", "|a ", ""),
        ("x.html", "<a ", ""),
        ("x.html", "<!--x>", ""),
        ("x.html", "</a", ""),
        ("x.html", "<?", ""),
        ("x.html", "<![CDATA[>", "]>"),
        ("x.html", "<![if x>", ""),
        ("x.html", "<![ ", ""),
        ("x.html", "<a x='>' ", ' y="'),
        ("x.html", "<a x='>' ", ""),
        ("x.txt", "a", ""),
    ],
)
def test_crafted_files_convert_in_linear_time(path: str, unit: str, tail: str) -> None:
    text = unit * (100_000 // len(unit)) + tail
    start = time.perf_counter()
    document_chunks(text, path)
    assert time.perf_counter() - start < 2


def test_code_spans_that_never_close_convert_in_linear_time() -> None:
    text = " ".join("`" * length for length in range(1, 440))  # ~100k characters
    start = time.perf_counter()
    assert bodies(document_chunks(text, "x.md")).split() == text.split()
    assert time.perf_counter() - start < 2


# Nesting thousands deep, or end tags that match nothing: each once made a converter
# rescan everything still open (quadratic) or recurse past Python's recursion limit.
@pytest.mark.parametrize(
    ("path", "text"),
    [
        ("x.rst", "\n\n".join(" " * level + "word" for level in range(1200))),
        ("x.rst", "\n\n".join(" " * 3 * level + ".. note::" for level in range(600)) + "\n\n" + " " * 1800 + "word"),
        ("x.html", "<div>" * 10_000 + "</span>" * 10_000 + "word"),
        ("x.html", "<table><tr>" + "<td>word" * 20_000),
        ("x.html", "<div>" * 10_000 + "<pre>word</pre>" * 10_000),
    ],
    ids=["rst-indents", "rst-directives", "html-stray-end-tags", "html-table-cells", "html-pre-in-divs"],
)
def test_deeply_nested_files_convert_in_linear_time(path: str, text: str) -> None:
    start = time.perf_counter()
    chunks = document_chunks(text, path)
    assert time.perf_counter() - start < 2
    assert "word" in bodies(chunks)


def test_rst_nested_past_the_depth_limit_keeps_its_words_as_text() -> None:
    levels = range(MAX_RST_DEPTH + 4)
    text = "\n\n".join(" " * level + f"word{level}" for level in levels)
    assert bodies(document_chunks(text, "x.rst")).split() == [f"word{level}" for level in levels]

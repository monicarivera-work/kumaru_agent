# `ui/academic.html`

Static shell of the academic review dashboard, served at `/academic` when both
`academic.enabled` and UI serving are on. It holds only structure: the header
(navigation, editorial-check chip, connection pill), the Library workbench
(`aside#subjects`, `section#library`, `section#inspector`), the Sources of Sources
and Review queue pages, a toast region and `<script type="module" src="/static/academic.js">`.
There is no inline script, style, handler, catalogue data or remote resource; the
favicon is the same inline SVG data URI as the chat page's, so the browser does not
request `/favicon.ico`.

The Library toolbar holds search, sort, `#discover-button` ("Discover sources") and
`#free-copies-button` ("Find free copies", whose `title` explains the batch); the
modules wire both, and `academic.css` orders them per breakpoint.

Landmarks, a skip link and one `h1` are static; everything dynamic is built by the
modules with DOM nodes and text. The page keeps the words "Academic review" because
a test asserts them. Add a new region here, then give it a module and a CSS block.
[Behavior](academic.js.md), [styles](academic.css.md), [workflow](../../ACADEMIC.md#review-dashboard).

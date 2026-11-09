# `ui/academic.js`

Entry module of the dashboard (native ES modules, no build step). It wires the page
shell to the modules in [`ui/academic/`](academic/): bootstrap, the hash router and
view switching, the header (queue badge, editorial-check button, tools menu),
the subjects drawer and the unsaved-changes guard.

Routes: `#/library`, `#/containers`, `#/queue`, `#/source/<id>[/<tab>[/<claimId>]]`.
Back/Forward work; if a dialog has unsaved edits, a navigation asks first and restores
the hash when the user cancels; `beforeunload` is guarded the same way. Bootstrap loads
topics, the scoped and catalogue-wide overview and monitor status, then applies the
route. A 401 shows "API key needed", a network failure "Server unreachable". Each
overview refresh copies the provider flags into state: `brave_configured`,
`openalex_key_configured` and `unpaywall_configured`.

Safety invariants live in the modules: DOM is built with `el()`/text nodes (never
`innerHTML`), links are http(s) only, the API key stays in `api.js` memory, and
stale responses are dropped with generation counters. Add a view by creating a
section in the HTML, a module and a branch in `applyRoute`.
[HTML](academic.html.md), [styles](academic.css.md), [workflow](../../ACADEMIC.md#review-dashboard).

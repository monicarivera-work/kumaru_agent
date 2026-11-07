# `ui/academic/dom.js`

DOM construction, vocabulary and small components. `el(tag, attrs, ...children)` makes
nodes and text nodes only (no markup parsing); `link()` allows http(s) and renders
anything else as inert text; `copyText()` falls back when the clipboard API is not
available. The label tables (access, provenance, permission, kinds, claim status,
notice states, evidence order) keep wording in one place.

`badge()` always pairs text with an icon and a tone, so colour is never the only signal.
`meter()` draws a 0–max bar and hatches null values with a dash. Formatters use `Intl`
(dates, relative time, tabular scores). Do not add `innerHTML` here or anywhere in the
dashboard.

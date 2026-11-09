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

Open-access wording lives here too: `OA_STATUS` (gold, diamond, hybrid, bronze, green,
closed; a plain label such as "Green — free copy in a repository" and a short chip
label such as "Green OA"), `OA_VERSIONS` (published version, accepted manuscript,
submitted preprint), `OA_PROVIDERS`, and `licenseLabel()`, which turns a license code
into what it allows: none stated is "free to read, not to reuse", CC0/public domain
allows reuse, CC BY and CC BY-SA need attribution, NC/ND variants are restricted and
anything else is "Other license (…) — check its terms". `RELATION_LABELS` includes
`open_copy` ("Free copy").

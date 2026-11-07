# `ui/academic/actions.js`

The three network actions on a source: inspect the page, collect permitted text and
check for editorial updates. `collectBlockers()` and `checkBlockers()` list what is
missing (human authorship, text permission, reviewed topics, no pending notice or
retraction, open or unknown access; a DOI on an individual work), so a disabled
button can say why. The server enforces the same rules.

`runSourceAction()` shows a busy label, calls the endpoint, toasts the result and emits
`data-changed`. An HTTP 503 (DNS or provider outage) adds that the recorded access
category was not changed. Never add an action that approves anything.

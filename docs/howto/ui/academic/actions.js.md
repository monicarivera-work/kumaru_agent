# `ui/academic/actions.js`

The four network actions on a source: inspect the page, collect permitted text,
check for editorial updates and find free copies. `collectBlockers()` and
`checkBlockers()` list what is missing (human authorship, text permission, reviewed
topics, no pending notice or retraction, open or unknown access; a DOI on an
individual work), so a disabled button can say why. The server enforces the same
rules. `openAccessBlockers()` is the same DOI rule, for free-copy lookups.

`runSourceAction()` shows a busy label, calls the endpoint, toasts the result and emits
`data-changed`. An HTTP 503 (DNS or provider outage) adds that the recorded access
category was not changed (an action can override that line; free-copy lookups drop
it, since they never touch access). Find free copies posts `/sources/{id}/open-access`
(OpenAlex by default) and toasts the plain-language status, how many free copies were
catalogued for review and the server's warning, or only the warning when the index
has no record of the DOI. Never add an action that approves anything.

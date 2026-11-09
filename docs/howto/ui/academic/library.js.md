# `ui/academic/library.js`

The middle pane: filter groups in the rail (Access, Type, Review, as `aria-pressed`
toggles with counts from `GET /overview`, shown as a dash until it has loaded), debounced search, sort, the context line
(subject path, server count, removable "In: container" chip), rows and the
"1–30 of N" pager. Rows show kind, title, authors/publisher/year, access, provenance,
"Notice pending" or "Retracted", and two mini meters for legitimacy and usefulness
(null is a hatched dash, containers say "Not scored"). One quiet open-access chip
follows: "Free copy" on a catalogued copy, "1 free copy"/"N free copies" on a work with
`open_copy` links, otherwise the short status ("Gold OA", "No free copy known") with
the plain label as its tooltip. The chip stays neutral; a green unlock icon marks
"free to read". A copy has the work's title, so this chip and the host in the meta
line tell them apart.

"Find free copies" in the toolbar posts `/open-access?topic_id=<subject>&limit=20`
(DOI works never checked first, then the least recently checked), shows "Searching…"
while it runs, then toasts "Checked N works; M free copies catalogued." with works the
index did not know, failures and the reason a batch stopped early (a rate limit, for
example); per-work errors are under "Show details". If nothing could be checked the
toast is an error; an empty subject says no work has a DOI. It emits `data-changed`.

It calls `GET /library` with a generation counter so stale pages are dropped, keeps
filters and paging in memory, supports arrow-key movement between rows, and renders
honest loading, error, "no sources yet" (with discover/add actions) and "no match"
(with Clear filters) states.

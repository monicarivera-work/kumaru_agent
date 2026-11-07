# `ui/academic/library.js`

The middle pane: filter groups in the rail (Access, Type, Review, as `aria-pressed`
toggles with counts from `GET /overview`, shown as a dash until it has loaded), debounced search, sort, the context line
(subject path, server count, removable "In: container" chip), rows and the
"1–30 of N" pager. Rows show kind, title, authors/publisher/year, access, provenance,
"Notice pending" or "Retracted", and two mini meters for legitimacy and usefulness
(null is a hatched dash, containers say "Not scored").

It calls `GET /library` with a generation counter so stale pages are dropped, keeps
filters and paging in memory, supports arrow-key movement between rows, and renders
honest loading, error, "no sources yet" (with discover/add actions) and "no match"
(with Clear filters) states.

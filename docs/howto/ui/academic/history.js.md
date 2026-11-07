# `ui/academic/history.js`

`historyTimeline()` turns append-only review records into a newest-first timeline: a
date, a type (Source review, Notice review, Claim review) and a summary built from the
payload, such as "Origin → Human verified · Permission → Text permitted · Access →
Paywall · 2 findings". The raw record is one "Audit record" disclosure away, wrapped so
it never overflows.

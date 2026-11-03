# `src/kumaru/academic/catalog.py`

`Catalog(path)` owns SQLite topic/source membership, directed source graph edges
and an append-only human-review trail. Connections are per operation; foreign keys
and transactions enforce membership, serialized writes, and review/content
consistency. Discovery de-duplicates URLs/DOIs while preserving human decisions.
Topic and containment cycles are refused. Graph navigation derives ancestors,
descendants and siblings rather than duplicating those edges.

Use `discover`, `review`, `put_topic`, `link`, `list_sources`, `scores`, `graph` and
`history`. Metadata inspection cannot approve origin/rights or store text.

Additive tables hold immutable assertions, referenced findings and review history,
editorial notices and last-check states. `review_claim` scopes and audits findings;
`review_notice` atomically updates report state, assessed evidence, audit records
and stale content. Pending notices/retractions gate trusted text. Score references
with unresolved notices or confirmed factual flaws are not independent evidence.
Summary counts/quarantine state scan the catalogue, not just a display page.
Database failures raise `ConfigError`, invalid operations raise `RequestError`.
Back up the SQLite file before schema changes. This is local storage, not a
distributed crawler. [Workflow and schema concepts](../../../../ACADEMIC.md).

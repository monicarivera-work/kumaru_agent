# `src/kumaru/academic/catalog.py`

`Catalog(path)` owns SQLite topic/source membership, directed source graph edges
and an append-only human-review trail. Connections are per operation; foreign keys
and transactions enforce membership, serialized writes, and review/content
consistency. Discovery de-duplicates URLs/DOIs while preserving human decisions.
Topic and containment cycles are refused. Graph navigation derives ancestors,
descendants and siblings rather than duplicating those edges.

Use `discover`, `review`, `put_topic`, `link`, `list_sources`, `scores`, `graph` and
`history`. Metadata inspection cannot approve origin/rights or store text.
Database failures raise `ConfigError`, invalid operations raise `RequestError`.
Back up the SQLite file before schema changes. This is local storage, not a
distributed crawler. [Workflow and schema concepts](../../../../ACADEMIC.md).

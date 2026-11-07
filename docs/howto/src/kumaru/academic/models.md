# `src/kumaru/academic/models.py`

Validated Pydantic records for topics, sources, evidence, human reviews,
relationships and scores. Unknown fields fail. URLs are canonicalized; source IDs
derive from URLs. Text requires human-verified provenance and permission.
Historical acceptance, proofs and factual criticism require topic scopes.
`mathematics_topics()` supplies an editable starter hierarchy, not a universal
curriculum.

Add new fields here before wiring catalogue, CLI/API, scoring and tests. Evidence
is data, never executable instructions; unknown provenance is never auto-approved.
See [the academic guide](../../../../ACADEMIC.md).

`Review.access_note` (up to 2000 characters) lets a human record why a source is
walled off; it is applied with `access` and kept in the review history. `SourceKind`
names the allowed source kinds so catalogue counts and the `Source` model agree.

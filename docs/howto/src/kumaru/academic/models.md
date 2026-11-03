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

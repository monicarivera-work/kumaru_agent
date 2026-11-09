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

`Source.open_access` is an `OpenAccess` record: what OpenAlex or Unpaywall reported
(`OAStatus`, `CopyVersion`, license, host, PDF link, provider, request URL and check
time). It is metadata, never a review, and older catalogues load without it.
`RelationKind` includes `open_copy`, directed from a work to a free copy of it.
`normalized_doi` (shared by editorial monitoring and open-access lookups) strips DOI
URL prefixes, lowercases, and raises `RequestError` for anything not shaped
`10.prefix/suffix`.

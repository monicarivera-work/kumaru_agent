# `src/kumaru/academic/discovery.py`

`discover` builds a contextual academic query from area/parent/topic, validates
provider metadata and persists unverified candidate sources. Crossref supplies
scholarly metadata by default, including references/citations where reported.
Optional Brave Search supplies broader web URLs and requires a configured key.
Crossref index membership becomes `contains` links, not peer-review evidence.

Add providers with typed response models, explicit errors, provenance request
URLs and offline fixtures. Never fabricate authors, counts, licenses, human
authorship or credentials from a search rank/domain. Queries are sent externally,
not stored text/reviews. [Provider setup](../../../../ACADEMIC.md#start-with-mathematics).

Crossref works that lack a title or any URL/DOI (components, peer-review records)
are skipped and counted: `discover` returns `skipped` next to `source_ids`. The
response envelope and records with unsafe values are still validated strictly, so a
malformed response writes nothing. Provider outages raise `TransientFetchError`.

`provider="openalex"` searches OpenAlex with the same contextual terms and a `select`
list of only the fields used, links results to an "OpenAlex scholarly index"
container, and passes each work's free locations to
[`record_open_access`](openaccess.md), so the response also lists `copy_ids`.
`open_access_only=True` adds OpenAlex's `is_oa:true` filter and is refused for other
providers. Works without a title, or without both a DOI and a landing page, and works
whose fields fail validation are skipped and counted. `openalex_api_key` goes in a
bearer header, never the stored URL.

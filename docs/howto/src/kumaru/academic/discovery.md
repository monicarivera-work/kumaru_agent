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

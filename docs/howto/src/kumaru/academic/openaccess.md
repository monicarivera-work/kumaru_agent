# `src/kumaru/academic/openaccess.py`

`find_open_access(catalog, source_id, provider=...)` asks OpenAlex (default) or
Unpaywall where a DOI work can be read free and under which license, then records the
answer as metadata. `find_open_access_batch` does the same for up to 20 works chosen by
`Catalog.open_access_due`. `record_open_access` is the shared writer, also used by
OpenAlex discovery for the locations a search already returned.

What it writes:

- The work's `open_access` (`OpenAccess` in [models](models.md)): the index's status
  for the work, and the license, version, host and PDF link of the publisher's own copy
  when that copy is free. A repository record of the work's own URL counts as the
  publisher's copy too.
- One unreviewed `Source` per free copy hosted elsewhere, linked from the work by an
  `open_copy` relationship whose note names the host, version and license. A copy's
  `open_access.status` is empty: the status belongs to the work.

Invariants:

- **Metadata, never approval.** Nothing here touches origin, permission, topic review
  or text. A license only informs the reviewer, and an unlicensed copy is free to read,
  not to reuse.
- **Copies carry no DOI.** `Catalog.discover` merges sources that share a DOI, which
  would fold a copy back into the work. An existing source at a copy's URL keeps its
  title, reviews and reviewed topics; only its metadata is refreshed.
- **Secrets stay out of storage.** Sources keep their request `metadata_url`, so the
  OpenAlex key travels in an `Authorization: Bearer` header and Unpaywall's required
  `email` parameter is left out of the stored URL.
- **Validate, then write.** Provider payloads and every new record are validated
  before the first write, so a malformed answer changes nothing.

Locations are normalised to `Location` and de-duplicated by `_distinct`: indexes list
one arXiv preprint as its abstract page, its bare PDF and its DataCite DOI. A location
that is only another one's PDF is dropped, one location per host and version is kept
(the one with a PDF, a license and a non-DOI URL wins), and a host's version-less
entries are dropped once it has a versioned one. URLs are made HTTPS-first with
`web.https_upgrade`. OpenAlex host types `journal`, `conference`, `ebook platform` and
`book series` count as the publisher; `repository` and unknown hosts become copies.

Failures: `ProviderMissing` (HTTP 404) means "not indexed" and is recorded as a check
with no status; `RateLimited` gains a hint about `academic.openalex_api_key` when no key
is set; other `TransientFetchError`s propagate as HTTP 503 and stop a batch, so an
outage or a spent budget does not fail every remaining work. Unpaywall without
`academic.contact_email` is a `RequestError` before any request is sent.

Tests: `tests/test_academic_open_access.py` uses payloads shaped like live OpenAlex and
Unpaywall answers. [User guide](../../../../ACADEMIC.md#find-legal-open-access-copies).

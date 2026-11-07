# `src/kumaru/academic/web.py`

`PublicWeb` provides bounded synchronous public-network requests. DNS answers are
validated and the transport connects to a pinned IP with original-host TLS
verification. Environment proxies and redirects carrying API keys are disabled.
Robots rules, pacing, response limits and explicit access errors apply equally to
CLI and API. `analyze_page` extracts landing-page metadata and text using existing
`doc_chunks.html_blocks`; no scripts or page links are executed/followed.

`inspect_source` stores metadata only. `collect_source` gates on reviewed human
provenance, rights and topic coverage, then records text/hash/time/final URL.
AI disclosure revokes trust. Gated, unsupported and unavailable sources remain
catalogued with access notes; actual failures are surfaced.

Test both mocked DNS/transport and live HTTPS when the environment permits.
Do not replace DNS pinning with a check followed by a second hostname resolution,
enable proxy forwarding, or assume a public abstract licenses full text.
[Limitations](../../../../ACADEMIC.md#collect-and-integrate-with-kumaru-learning).

`TransientFetchError` (a `FetchError` with HTTP 503) marks outages that say nothing
about the source: DNS failure, `httpx` transport errors and timeouts, robots
`429`/`5xx`, provider `429`/`5xx`, and source pages answering `429`/`5xx`. `inspect_source` and `collect_source` re-raise
it without touching the stored access category; every other `FetchError` still
records `inaccessible`. Raise it only for conditions that could pass on retry.

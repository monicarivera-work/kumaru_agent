# `src/kumaru/academic/web.py`

`PublicWeb` provides bounded synchronous public-network requests. DNS answers are
validated and the transport connects to a pinned IP with original-host TLS
verification. Environment proxies and redirects carrying API keys are disabled.
Robots rules, pacing, response limits and explicit access errors apply equally to
CLI and API. `analyze_page` extracts landing-page metadata and text using existing
`doc_chunks.html_blocks`; no scripts or page links are executed/followed.

Redirects are followed manually so each hop gets DNS, robots and scheme checks.
An HTTPS hop that redirects to plain HTTP (as `doi.org` does for many publishers)
is retried as the HTTPS form of that target via `_https_upgrade`; plain HTTP is never
requested after HTTPS. A redirect back to the same URL, or to a nonstandard port,
still raises `AccessBlocked("HTTPS downgrades are not followed")`.

Robots rules come from [`robots.py`](robots.md), which follows RFC 9309 (longest
match) rather than `urllib.robotparser` (first match, no wildcards). `Page.challenge`
names Cloudflare (`cf-mitigated: challenge`) or AWS WAF (`x-amzn-waf-action:
challenge`/`captcha`) when either answers with a browser challenge. `analyze_page`
turns that into `AccessBlocked` (`inaccessible`) whatever the status, so a `202`
challenge is classified instead of failing as an unexpected status, and a challenge
on robots.txt fails closed. Other services' challenges are still caught by the page-text
phrases when they answer `200`.

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

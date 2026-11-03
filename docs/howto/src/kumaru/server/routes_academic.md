# `src/kumaru/server/routes_academic.py`

Opt-in `/v1/academic` router registered by `create_app` only when
`academic.enabled` is true. Every route uses existing bearer authentication.
Operations are synchronous so SQLite/network work stays off the event loop;
network actions require one of two per-process slots or return HTTP 429.
No inference readiness check is needed. Listings omit text; retrieval of
reviewed text is explicit.

Use the `academic` section in `/docs`. Request models reject unknown fields and
invalid bounds. Domain errors use the existing JSON handler. Extend this router
alongside the standalone CLI and offline endpoint tests.
[Endpoint table](../../../../ACADEMIC.md#http-api).

Claim and notice endpoints expose independent findings, audit trails, dependency
impact and human review. `/sources/{id}/check-updates` and `/monitor` share network
admission; `/monitor/status` reports optional scheduler configuration/results
without secrets. Source details include counts/quarantine state across all pages.
Text retrieval refuses pending editorial notices or reviewed retractions.

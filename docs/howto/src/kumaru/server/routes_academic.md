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

The dashboard's read endpoints are `GET /overview` (counts, per-topic totals, scoring
formula, `brave_configured`) and `GET /library` (a filtered, sorted, paged summary
with a server total; no text or evidence). Their parameters are plain strings and
integers validated by the catalogue, so every invalid value is HTTP 400 like other
domain errors (the older `GET /sources` keeps FastAPI's 422). Source, claim and
notice details add titles (`related`, `source`, `references`, `dependent_sources`,
`affected_claims`), and `GET /claims` accepts `reviewed`. A `TransientFetchError`
from an action route is HTTP 503.

Open access: `POST /sources/{id}/open-access?provider=` and the batch
`POST /open-access?topic_id=&limit=&provider=` share network admission with the other
action routes. They read `academic.openalex_api_key` and `academic.contact_email` from
configuration; `GET /overview` reports only whether each is set
(`openalex_key_configured`, `unpaywall_configured`), never the values. `POST /search`
accepts `open_access_only`.

# `ui/academic/api.js`

`api(path, {method, body, params})` calls `/v1/academic`. The optional API key lives
only in this module's memory (`setApiKey`), is sent as a bearer header and is never
written to storage, cookies or the URL; a reload forgets it.

Errors are mapped before the body is trusted: a failed `fetch` becomes "Kumaru server
unreachable", a non-JSON response "HTTP <status> from server", 401 the key message and
404 "Academic API unavailable"; otherwise the server's own message (and hint) is used.
`ApiError` carries `status`, `type`, `hint` and `unreachable` for callers.

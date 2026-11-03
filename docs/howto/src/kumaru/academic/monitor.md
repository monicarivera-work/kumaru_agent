# `src/kumaru/academic/monitor.py`

`check_updates` uses Crossref's documented `updates:DOI` filter to retrieve
editorial notices for a work. It validates target identity, preserves metadata
provenance, maps update types and records explicit success/error/partial state.
Queries are capped at 100 records and incomplete scans are labeled partial.
Provider absence of notices is never proof of no corrections.

`check_batch` selects never/least-recently checked DOI works, reports per-source
failures, and limits batches to 20. `UpdateMonitor` schedules those batches only
when enabled by `academic.monitor_interval_s`; its default is zero/manual-only.
It shares the server's two network slots and joins an in-progress thread on
shutdown. Only one server process should enable periodic monitoring.

Do not automatically generate reviewed criticism, credentials or copying
permission from a provider report. Human notice confirmation is handled
transactionally by `Catalog.review_notice`.
[Configuration and limits](../../../../ACADEMIC.md#corrections-concerns-and-retractions).

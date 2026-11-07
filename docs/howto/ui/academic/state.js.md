# `ui/academic/state.js`

Shared state (connection, topics, selected subject, overview counts, formula, route,
filters), a tiny event bus (`emit`/`on`, for example `data-changed`), generation gates
that drop stale responses, the unsaved-changes check dialogs register, and hash helpers
(`navigate`, `replaceHash`, `sourceHash`). Keep it small: state that only one module
needs stays in that module.

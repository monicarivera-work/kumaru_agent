# `ui/academic/inspector.js`

The right-hand pane. For a selected source it renders the header (kind icon, title,
URL, access/provenance/permission/topic/editorial badges, authors, DOI, provider counts,
the source ID with Copy), the actions with visible reasons when disabled, and the tabs
(Overview, Evidence, Claims, Notices, Relations, History; ARIA tab pattern, sticky
below the header). With nothing selected it shows the library at a glance.

Tabs are rendered by their own modules from one `GET /sources/{id}` payload; Claims
and Notices fetch their pages lazily. `data-changed` reloads the source in place, keeping
the tab. A generation counter drops stale responses, and loading or failure states say
what happened with a way back. History is the append-only review timeline.

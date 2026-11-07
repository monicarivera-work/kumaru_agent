# `ui/academic/dialogs.js`

Native `<dialog>` helpers. `openDialog()` builds a modal form with a heading label, an
inline `role=alert` error region, a busy Save state and a dirty guard: Esc, Cancel and
the close button ask before discarding changes (a snapshot of the form data decides
what "changed" means). A failed save keeps every input; a successful one closes, and
focus returns to the control that opened it (`rememberFocus`/`applyPendingFocus` re-find
it after a re-render). Below 760px dialogs are full screen.

`confirmDialog()` is the stacked confirmation; `toast()` shows `role=status` messages
that dismiss after 5 seconds and persistent `role=alert` errors with Dismiss and
expandable details. The module registers the dirty check the router and
`beforeunload` use.

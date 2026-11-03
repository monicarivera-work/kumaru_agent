# `ui/academic.js`

Plain-JavaScript authenticated catalogue/review client. DOM nodes and text are
created directly; untrusted source titles, evidence notes and histories are never
inserted as HTML. External links require HTTP(S). The optional API key remains
in page memory only, never browser storage.

Fetches real topic, source, claim, notice and monitor APIs. Human approvals require
explicit form decisions and supporting evidence. Claim findings do not silently
approve claim attribution. Notice confirmations explain their content/score
effects. Read generations reject stale responses; unsaved edits are protected.
Errors preserve entered form data.

Sources, claims and notices are paginated. Source-wide counts/quarantine flags
come from the server, not just currently visible rows. All screenshots/demo data
used during development are synthetic fixtures, not production discoveries.
Validate actual form saves and keyboard/mobile behavior, not merely static markup.
[API/workflow](../../ACADEMIC.md#review-dashboard).

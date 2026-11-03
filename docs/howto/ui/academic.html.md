# `ui/academic.html`

Static academic review workspace served at `/academic` when both
`academic.enabled` and UI serving are enabled. It extends Kumaru's incumbent
operator interface without replacing chat. Contains connection, subject/search,
source-list and evidence-review regions, with semantic headings and a skip link.
The static shell includes no private catalogue content or API keys.

Styles and behavior are in `academic.css` and `academic.js`.
All data operations require the existing academic API authorization.
[User workflow](../../ACADEMIC.md#review-dashboard).

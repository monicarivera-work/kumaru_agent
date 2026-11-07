# `ui/academic.css`

All dashboard styles, scoped under `body.academic`, built from the tokens in
`styles.css` (`--bg*`, `--border`, `--text*`, `--accent`, `--ok/--warn/--err`,
`--radius`, `--sans`, `--mono`). It adds only academic custom properties: a spacing
and type scale, hatching, and a few text tints. No fonts, images or frameworks.

Design rules it encodes: colour carries meaning only (access, provenance, status);
numbers use tabular figures; unknown scores are hatched, never empty; the focal
element is the score anatomy. Layout is three panes from 1280px, a subjects drawer
below, a 340px list from 1023px and one column below 760px, with 40px touch targets
and full-screen dialogs. Motion is limited to ~150ms transitions and is disabled for
`prefers-reduced-motion`.

`styles.css` styles bare elements (`button`, `input`) and some classes (`.chip`,
`.check`, `.slider`, `.pill`). The resets at the top use `:where()` to stay at that
specificity, and academic components use their own class names (`tag`, `checkbox`,
`range`, `status-pill`) so chat styles are never edited or inherited by accident.
Inspect desktop and mobile in one bounded round after changes.
[HTML](academic.html.md), [behavior](academic.js.md).

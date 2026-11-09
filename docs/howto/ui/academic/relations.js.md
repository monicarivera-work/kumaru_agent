# `ui/academic/relations.js`

The Relations tab: the containment family (ancestors minus parents, parents, this
source, children, descendants minus children, and siblings that share a parent) as
buttons with kind icon, title and access badge that navigate; long lists collapse
behind "Show more". Other links are grouped by direction (Cites, Cited by, Corroborates,
Corroborated by, Criticizes, Criticized by, Free copies, Free copy of, Related) with
their notes; an `open_copy` link points from the work to its free copy.

"Add relationship" phrases the choice as a sentence ("This source is cited by…"), uses
the source picker and requires a note. Reverse phrasings swap source and target in
`POST /relationships`. Hints explain that a criticism link changes no score, what
corroboration requires, and that a free copy ("This source has a free copy at…" /
"This source is a free copy of…") can be a different version.

# `ui/academic/relations.js`

The Relations tab: the containment family (ancestors minus parents, parents, this
source, children, descendants minus children, and siblings that share a parent) as
buttons with kind icon, title and access badge that navigate; long lists collapse
behind "Show more". Other links are grouped by direction (Cites, Cited by, Corroborates,
Corroborated by, Criticizes, Criticized by, Related) with their notes.

"Add relationship" phrases the choice as a sentence ("This source is cited by…"), uses
the source picker and requires a note. Reverse phrasings swap source and target in
`POST /relationships`. Hints explain that a criticism link changes no score and what
corroboration requires.

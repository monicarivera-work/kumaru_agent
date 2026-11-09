# `ui/academic/anatomy.js`

The score anatomy, the dashboard's focal element. `scoreCard()` draws, per subject,
the server's headline legitimacy, usefulness and evidence coverage, then two weighted
bars. Segment widths come from `GET /overview`'s `formula` (legitimacy weights and
usefulness points), never from constants in the UI. A dimension with reviewed
evidence is filled by its value; one without is hatched and labelled; criticism and
retraction penalties are named with their effect.

The headline is always the server's number (never recomputed). When legitimacy is
null everything is hatched, with the first warning as the reason: unknown is not zero.
Provider counts that were not reported are hatched; reported zeros are an empty bar.
The legend repeats every value as text, so the bar is never the only channel.

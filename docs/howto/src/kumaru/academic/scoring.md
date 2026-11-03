# `src/kumaru/academic/scoring.py`

`score_source(source, topic_id, independent_sources=...)` is a pure, conservative,
versioned heuristic. It uses only reviewed evidence applicable to that topic and
returns components, evidence IDs, coverage and warnings. Unknown legitimacy stays
unknown. Containers never inherit accuracy from their contents or vice versa.
Criticism/retraction penalize credit; popularity and text volume cannot overwhelm
legitimacy. Repeated corroborating IDs do not accumulate points.

Changing weights requires updating formula version, [documented
weights](../../../../ACADEMIC.md#scores-and-their-limitations) and numeric regression
tests. Never describe the output as a scientifically calibrated probability.

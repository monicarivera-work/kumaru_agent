# `src/kumaru/academic/reliability.py`

Pydantic claim/finding and editorial-notice records, plus pure `claim_score`.
Assertions, exact locations and assumptions are immutable after creation.
Claims receive only their explicit reviewed findings, never another assertion's
proof or a whole source's reputation. Referenced works must have reviewed human
origin and topic coverage. Contextual findings do not score; criticisms must
identify factual flaws. Claim assessments expose `claim-v1`, base score
components, coverage, evidence IDs and limitations.

Editorial reports are distinct from human-confirmed impacts. Notice identity
combines source, type, URL and event date; rediscovery cannot overwrite review
decisions. Confirming a retraction requires an explicit impact of 1 in the
catalogue service. New schema fields must be wired through storage, CLI, API,
dashboard and tests.

See [claims and notices](../../../../ACADEMIC.md#claim-level-evidence).

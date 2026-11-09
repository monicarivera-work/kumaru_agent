# `ui/academic/evidence.js`

The Evidence tab and the finding dialog. Findings are grouped by dimension
(historical acceptance, credentials, proof, peer review, replication, corroboration,
factual criticism, retraction, origin, permission) with a value meter, a reviewed or
unreviewed tag, subject tags, the note, the link and the recorded date. Findings set
by a notice review show "Set by a notice review" and have no Edit, because the notice
decision owns them.

The dialog records or edits a finding: kind, value (slider plus number), subject, URL,
note and "I personally reviewed this finding" (off by default, so nothing counts until
you say so). Proof, historical acceptance and factual criticism must name a subject.
Editing re-sends the stable evidence ID.

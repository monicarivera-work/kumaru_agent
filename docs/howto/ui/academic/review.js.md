# `ui/academic/review.js`

The Review source dialog: authorship, copying permission, access (category plus note)
and topic coverage in one audited save. Approving authorship or permission requires an
evidence URL and note (the server also requires reviewed evidence at value 1); revoking
warns that collected text is deleted. Only changed fields are sent, so the history
stays truthful, and when subjects are sent `topic_reviewed` is always explicit.
Changing the subjects un-ticks "I reviewed topic coverage" so approval is never carried
over to new tags silently.

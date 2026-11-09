# `ui/academic/overview.js`

The Overview tab and "Library at a glance". Overview lists "Before relying on this
source" (authorship, permission, access, open access, topic coverage, editorial
status, trusted text), each with a text status and an action that opens the right
dialog or tab, then one score card per subject (containers get an explanation
instead, because a catalogue's reputation does not establish its contents).

The open-access row is metadata, never a review, and is skipped for containers and for
sources with no DOI, no recorded lookup and no `open_copy` link. A work never checked
says so and offers "Find free copies". A checked work shows the plain status label, the
publisher copy's version, license and PDF when the index listed one, the number of free
copies (`open_copy` links from it, a button to the Relations tab) and when it was
checked, with "Check again". A free copy (an `open_copy` link to it) names the work it
copies as a link, its version and host, its license in plain words, the PDF via
`link()`, and for a submitted preprint the caution that it can differ from the
peer-reviewed publication.

The glance panel is a ledger rather than tiles: sources, human verified, needs review,
pending notices and claims for the selected subject, each a button that applies the
filter, plus a labelled access distribution bar and next actions.

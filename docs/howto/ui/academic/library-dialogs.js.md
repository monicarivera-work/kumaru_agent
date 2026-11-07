# `ui/academic/library-dialogs.js`

Discover and Add-source dialogs. Discover shows the subject path and a query, offers
Crossref or Brave (disabled with a reason when no key is configured), says it contacts
an external provider and that results are unreviewed, and reports the `skipped` count.
Add source collects URL, title, kind, optional DOI, an access category, subjects (a
checkbox tree) and an optional container with a required note; the container link is
a second `POST /relationships`, and if it fails the source stays catalogued and the
error says so. Neither dialog approves anything.

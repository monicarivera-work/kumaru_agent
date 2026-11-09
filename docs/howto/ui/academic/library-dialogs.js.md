# `ui/academic/library-dialogs.js`

Discover and Add-source dialogs. Discover shows the subject path and a query, offers
Crossref, OpenAlex or Brave (disabled with a reason when no key is configured), says it
contacts an external provider and that results are unreviewed, and reports the
`skipped` count. OpenAlex adds open-access status and licenses and catalogues the free
copies it lists; without `academic.openalex_api_key` its hint says it uses a small
keyless budget shared by the network. "Only works with a free copy" sends
`open_access_only` and is enabled only while OpenAlex is selected (it is unchecked
otherwise, because the server rejects it for other providers). The toast adds how many
free copies were catalogued (`copy_ids`).
Add source collects URL, title, kind, optional DOI, an access category, subjects (a
checkbox tree) and an optional container with a required note; the container link is
a second `POST /relationships`, and if it fails the source stays catalogued and the
error says so. Neither dialog approves anything.

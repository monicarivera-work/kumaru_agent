# `ui/academic/notices.js`

The Notices tab and its dialogs. It shows the last-check line (relative and absolute
time, partial scan, failure), a paginated list of notices (kind, state, title, provider,
event date, link, affected claims, dependent sources by title) and "Report notice".

"Review notice" records Reported, Confirmed or Dismissed with a required note. Confirming
needs an impact; a retraction is fixed at 1 and applies to the whole source; a positive
non-retraction impact needs affected subjects. The dialog warns that confirming clears
collected text and can change scores. Reported notices keep trusted text blocked.

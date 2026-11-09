# `ui/academic/claims.js`

The Claims tab and its dialogs. Claims are paginated (30 per page, server counts);
selecting one expands it in place and updates the address without a re-render. The
detail shows location, assumptions, the claim's own status and score anatomy, its
findings (stance, referenced source title, locator) and its review history.

Dialogs: record a claim (statement, location and assumptions cannot change later),
review attribution (checkbox plus required note) and attach or edit a finding (stance
constrains the evidence kind; the referenced source is chosen with the picker).
Attribution review stays separate from findings: attaching a finding re-sends the
claim's current `reviewed` value, so it never approves attribution. Containers show an
explanation instead, since claims belong to individual works.

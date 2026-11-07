# `ui/academic/topics.js`

The subject index and its UI. The left rail is an ARIA tree (`tree`/`treeitem`, levels,
expanded, selected) with roving focus and Up/Down/Left/Right/Home/End/Enter/Space;
counts are distinct sources per subtree from `GET /overview`'s `topic_totals`. The topic
dialog adds a child (next level down), a root area or edits a name and aliases through
`PUT /topics/{id}`; because that call upserts, the dialog refuses an ID that already
exists. `topicCheckboxTree()` is the checkbox tree other dialogs use, and
`breadcrumb()` renders a subject path.

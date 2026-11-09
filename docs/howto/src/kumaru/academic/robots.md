# `src/kumaru/academic/robots.py`

`parse_robots(text, user_agent)` returns the `RobotsRules` one crawler must obey for
one origin. [`PublicWeb`](web.md) caches one per origin and calls `allowed(url)`
before every request and redirect hop. Matching follows RFC 9309:

- A group starts at a `User-agent` line and runs until a `User-agent` line that
  follows a rule; blank lines and `Sitemap` lines do not end it. Every group naming
  the product token (`KumaruAcademic`, compared case-insensitively and exactly, so
  `KumaruAcademic/2` names it and `Kumaru` does not) is combined and replaces the
  `*` groups. Rules before any `User-agent` line are ignored.
- The longest matching rule wins, and `Allow` wins a tie. `*` matches any run of
  characters and a final `$` anchors the end. The path includes the query.
- Paths and rules are compared after percent-encoding normalisation: `%7E` equals
  `~`, `%2f` equals `%2F` (but not `/`), and `café` equals `caf%C3%A9`.

It replaces `urllib.robotparser`, which obeys the first matching rule and reads `*`
and `$` literally. Publishers write `Disallow: /` followed by narrower `Allow` lines
(one large publisher allows `/10*`, `/chapter/` and `/book*` this way), so that parser
blocked every page they permit while ignoring wildcard `Disallow` lines entirely.

`Crawl-delay` (any finite, non-negative number) and `Request-rate` (`n/seconds`) are
not part of the RFC. A named group without them inherits the `*` group's, so pacing
never gets faster than before; `PublicWeb` refuses delays over 60 seconds.

Matching never backtracks. With `*` as the only wildcard, taking each literal
segment's leftmost match is optimal, so a crafted rule costs a few linear
`str.find` scans (`test_hostile_wildcard_patterns_match_in_linear_time`). Keep it that
way: compiling rules into regular expressions would let one robots.txt stall a
worker. Tests live in `tests/test_academic_robots.py`.

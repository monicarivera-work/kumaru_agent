# Academic source discovery and permitted text collection

`kumaru academic` discovers potential information carriers, records provenance,
organizes their academic subjects and relationships, and collects **reviewed,
permitted human-authored text**. It does not use an LLM to generate references,
detect AI, review mathematics, or approve sources. Its HTTP API is opt-in.
The existing chat workflow is preserved. Enable the academic API to use the
separate `/academic` review dashboard, linked from the chat sidebar.

## Start with Mathematics

```powershell
kumaru academic --database C:\references\academic.sqlite topics
kumaru academic --database C:\references\academic.sqlite search calculus --query "fundamental theorem" --limit 10
kumaru academic --database C:\references\academic.sqlite list --topic mathematics
kumaru academic --database C:\references\academic.sqlite show SOURCE_ID
```

The database defaults to `~/.local/share/kumaru/academic.sqlite`. It is created
with its parent folder as needed, preserved across commands, and not overwritten.
Group flags (`--database`, `--config`) must precede the subcommand.
No model, GPU or optional package is required.

Crossref is the default scholarly discovery provider. It searches scholarly
metadata, not every web page. Titles, authors, publisher, DOI, publication year,
citation count and reference count are retained when supplied, with the exact
metadata request URL. Missing counts remain unknown. Search combines the area,
immediate parent and selected topic to disambiguate terms. `--query` adds a
narrower question. Search returns candidate IDs; it never approves authorship,
peer review, accuracy, access or permission.

For broader web discovery, configure a Brave Search API subscription:

```powershell
$env:KUMARU_ACADEMIC__BRAVE_API_KEY = "YOUR_KEY"
kumaru academic search number-theory --provider brave --query "textbook university"
```

Only query terms go to search providers. Stored text, review evidence and API keys
are not sent to an inference backend by this subsystem. Do not place secrets in
search terms. Keys stay in request headers and are never forwarded on redirects.
`kumaru config` prints configuration secrets, including this key; do not share its
output unredacted.

OpenAlex (`--provider openalex`) searches a CC0 scholarly index and also records each
work's open-access status and catalogues its free copies. `--open-access-only` keeps
just the works that have one. See [Find legal open-access
copies](#find-legal-open-access-copies) for its budget and optional key.

## Subject hierarchy

Topics form an editable tree: **area > subfield > unit > chapter > subchapter**.
The Mathematics seed includes algebra, analysis, geometry, number theory,
topology, discrete mathematics, probability, statistics, applied mathematics and
logic, with example calculus and linear-algebra descendants. This is a starting
taxonomy, not a claim that every curriculum organizes chapters identically.
Subchapters are optional and should be used sparingly.

You can search any seed topic and add arbitrary missing subfields or other areas.
Save a `Topic` JSON file and run `kumaru academic topic FILE.json`:

```json
{
  "id": "measure-theory",
  "name": "Measure theory",
  "level": "unit",
  "parent_id": "analysis",
  "aliases": ["Lebesgue measure"]
}
```

Root topics must be areas; children must be more specific than their parents.
Cycles and updates that would invalidate existing children are refused.
Aliases are recorded for taxonomy consumers; provider queries currently use topic
names and explicit `--query` terms, not automatic synonym expansion.
Listing an area includes sources tagged with any descendant.

Discovery tags are **potential coverage**, not an assertion that the work covers
the topic. A human must confirm coverage before scoring or collecting text.
Scores are calculated per tagged topic; proof, historical acceptance and factual
criticism evidence must identify their exact topic(s). Evidence on integration
cannot silently credit all of Mathematics.

## Sources of Sources and access

Add an individual work, website or database using `kumaru academic add FILE.json`.
For a journal index, library, collection or repository that cannot be analyzed
work by work, use:

```json
{
  "url": "https://www.crossref.org/",
  "title": "Crossref scholarly metadata index",
  "kind": "sources_of_sources",
  "topic_ids": ["mathematics"],
  "access": "open",
  "access_note": "Public metadata; child works may require payment."
}
```

Kinds are `article`, `book`, `study`, `website`, `sources_of_sources`.
Discovery automatically links Crossref results to its metadata index.
Canonical URLs and DOI matches de-duplicate discovery without destroying reviews
or stored text. A source's ID derives from its canonical initial URL.

Every source independently records one access category:

| Category | Meaning |
|---|---|
| `unknown` | Access has not been established. |
| `open` | The landing page is immediately reachable; not a reuse license. |
| `account` | Account/login wall. |
| `paywall` | Paid gate. |
| `inaccessible` | Application cannot retrieve/analyze it: robots refusal, browser challenges, unsupported formats, oversized responses or redirect loops. |

A DNS failure, timeout, connection error, or a `429`/`5xx` answer from robots, a provider
or the source page itself is not evidence about the source's access. Those requests fail with **HTTP 503** and leave the
stored access category and note untouched, so a local network outage cannot relabel
good sources.

A bot-mitigation challenge is different: it is the site's answer, and this client can
never pass it. Responses that Cloudflare (`cf-mitigated: challenge`) or AWS WAF
(`x-amzn-waf-action: challenge` or `captcha`) mark as challenges record `inaccessible`
with the service named, whatever their status code (usually `403` or `202`).

```powershell
kumaru academic list --containers --access paywall
kumaru academic inspect SOURCE_ID
```

Inspection stores metadata and access observations, **not page text**. Public
abstracts/landing pages may still conceal full-text restrictions, so access
classification is conservative evidence for review, not a definitive paywall
detector. No login, payment, cookie import, JavaScript execution or bypass is
attempted. Many mainstream publishers' landing pages answer with a bot challenge, a
paywall marker or a dead DOI target; those works stay catalogued with the reason, and
their Crossref metadata remains available for review and scoring. Containers are
never full-text collected or scored as if their
reputation established the truth of all their contents.

Link reviewed relationships with `kumaru academic link FILE.json`:

```json
{
  "source_id": "PARENT_ID",
  "target_id": "CHILD_ID",
  "kind": "contains",
  "note": "This textbook is catalogued by this library."
}
```

`contains` is parent-to-child membership; the graph derives grandparents/all
ancestors, descendants and siblings sharing an immediate parent. Multiple parents
are allowed, but cycles are not. Other directed links are `cites`, `corroborates`,
`criticizes`, `related` and `open_copy` (from a work to a free copy of it; see
[Find legal open-access copies](#find-legal-open-access-copies)). A criticism edge
alone is not factual-flaw evidence:
record a reviewed finding to change a legitimacy score.

## Find legal open-access copies

Many works that are paywalled at their publisher can be read legally elsewhere: an
accepted manuscript in a university repository, a preprint on arXiv, or an open
edition. `open-access` asks an index where a DOI work can be read free, and under
which license:

```powershell
kumaru academic open-access SOURCE_ID                       # OpenAlex, the default
kumaru academic open-access SOURCE_ID --provider unpaywall  # needs academic.contact_email
kumaru academic open-access-batch --topic calculus --limit 20
kumaru academic search calculus --provider openalex --open-access-only --query "fundamental theorem"
```

The work's `open_access` records the index's status for it, plus the license,
version and PDF link of the publisher's copy when that copy is free:

| Status | Meaning |
|---|---|
| `gold` | Open at the publisher. |
| `diamond` | Open at the publisher, with no author fees. |
| `hybrid` | An open article in a subscription journal. |
| `bronze` | Free to read at the publisher, but with no license. |
| `green` | Closed at the publisher; a free copy exists in a repository. |
| `closed` | No free copy known. |

Each free copy hosted elsewhere becomes its own unreviewed source, linked from the
work by `open_copy`. The link's note names the host, version and license. A copy
starts with `access: unknown` and no DOI of its own (so it is never merged into the
work); inspect and review it like any other source.

| Version | Meaning |
|---|---|
| `publishedVersion` | The publisher's final version. |
| `acceptedVersion` | The peer-reviewed manuscript before typesetting. |
| `submittedVersion` | A preprint, not yet peer reviewed. It can differ from the publication. |

A license is evidence for your review, never an approval: collection still needs
reviewed `permission` evidence. `cc0`, public domain, `cc-by` and `cc-by-sa` allow
reuse on their conditions; `-nc` and `-nd` variants restrict it; a copy with no
license, including every `bronze` one, is free to read but grants no reuse. Confirm
the license on the page itself, because indexes can be wrong or out of date, and
"not indexed" does not mean that no free copy exists. PDF links are kept as
`pdf_url`; like every PDF they remain metadata-only.

Providers:

- **OpenAlex** (default) needs no account, and DOI lookups are free. Searches draw on
  a daily budget: without a key, everyone on your network shares a small one, and
  OpenAlex may briefly refuse anonymous searches under load. Kumaru waits once when
  told to retry within 20 seconds. A free key (openalex.org, Settings, API key) gives
  you your own budget. It is sent only in an `Authorization` header, so it never
  appears in the metadata URLs that sources store.
- **Unpaywall** (the same team and data as OpenAlex; its search was retired) answers
  DOI lookups and requires your own email address. Kumaru sends the address only to
  Unpaywall and stores the request URL without it.

```powershell
$env:KUMARU_ACADEMIC__OPENALEX_API_KEY = "YOUR_KEY"
$env:KUMARU_ACADEMIC__CONTACT_EMAIL = "you@your-university.edu"
```

Batches check DOI works never checked first, then the least recently checked, up to
20 per run. They skip containers and sources that are themselves free copies. A rate
limit or outage stops the batch and reports it in `stopped`, rather than failing
every remaining work. `kumaru config` prints both settings; redact them before
sharing its output.

## Human review and permission

There is no reliable universal automated detector of human versus AI-authored
text. Default provenance is `unknown`, which excludes text from the trusted
collection. Explicitly AI-disclosed pages are marked `ai`. `human_verified` is
a human decision supported by provenance evidence, not a machine guarantee.
Peer review or publication date alone does not prove human authorship.

Likewise, free accessibility is not permission to copy. Do not approve copyrighted
text unless its license, public-domain status or your actual permission allows
collection. A subscription or personal account alone is not necessarily reuse
permission. Paid sources can be catalogued and scored from authorized evidence
even when their text remains unavailable to this application.

Save a `Review` JSON and run `kumaru academic review SOURCE_ID FILE.json`.
The following is a **shape example**, not evidence or permission for a real work.
Replace URLs and notes with your actual verified findings:

```json
{
  "origin": "human_verified",
  "permission": "text_permitted",
  "topic_ids": ["calculus"],
  "evidence": [
    {
      "id": "authorship-review",
      "kind": "origin",
      "value": 1,
      "url": "https://example.org/author-provenance",
      "note": "Describe the human-authorship evidence you actually checked.",
      "reviewed": true
    },
    {
      "id": "rights-review",
      "kind": "permission",
      "value": 1,
      "url": "https://example.org/license",
      "note": "Identify the applicable license/permission and its collection scope.",
      "reviewed": true
    },
    {
      "id": "calculus-proof-review",
      "kind": "proof",
      "value": 0.9,
      "url": "https://example.org/proof",
      "note": "Explain the checked proof, assumptions and any remaining limitations.",
      "topic_ids": ["calculus"],
      "reviewed": true
    }
  ]
}
```

Evidence values are normalized reviewer assessments from 0 to 1, not invented
provider facts. Each record includes URL, explanation, timestamp and review state.
Retain stable evidence IDs to update or revoke a previous finding; new IDs append
findings. All reviews are retained in an audit trail. Only reviewed evidence
affects scores. Origin/permission approvals require reviewed supporting evidence
of the respective kind at value 1. Revoking the sole supporting approval also
requires changing its decision to `unknown`/`metadata_only`/`ai`; it cannot leave
an unsupported approval behind.

Each source and claim stores up to 500 findings, with up to 100 per review.
Updates to existing IDs do not consume another slot. Capacity failures leave the
record unchanged; a notice impact that would exceed source capacity is not saved
and remains pending.

An explicit `"topic_reviewed": false` revokes topic-coverage approval and clears
cached text. Passing `topic_ids` without that field still confirms coverage, for
compatibility with existing review files.

Changing origin away from `human_verified` or permission away from
`text_permitted` removes stored text and its content metadata.
Explicit AI disclosure discovered on recollection also revokes trust.

A review can also set `access` and a free-text `access_note` (up to 2000 characters)
so a human can record why a source is walled off or how it was reached. Omitting
`access_note` keeps the recorded note; sending an empty string clears it. Both are
part of the review audit history.

## Scores and their limitations

Scores are **explainable heuristics, not calibrated probabilities of truth**.
`show` returns components, reviewed evidence IDs, evidence coverage, warnings and
formula version `academic-v1`. Unreviewed topics, unverified human provenance,
containers and absence of academic evidence produce `null`, not a fabricated
numeric rating. Do not rank `null` as zero or treat a source score as proof that
every assertion in that source is correct.

Legitimacy credit uses these weights:

| Dimension | Maximum credit |
|---|---|
| Historical acceptance, scoped to the mathematical topic | 20 |
| Relevant author credentials | 15 |
| Actual proof/evidence and checked assumptions | 30 |
| Peer review evidence | 20 |
| Independent replication/confirmation | 15 |

The weight times the average reviewed value contributes credit. Reports sharing
an evidence URL and topic scope count once, using their lowest assessment.
Missing dimensions
give no credit and lower evidence coverage; a credential alone cannot score
100/100. Repeating equivalent claims does not accumulate weight. Credited
legitimacy is multiplied by `1 - 0.8 * strongest factual-criticism value`.
Retraction evidence caps legitimacy at `10 * (1 - strongest retraction value)`.
Criticism must concern factual flaws, not mere disagreement or popularity.
Human reviewers decide what historical acceptance, proof and replication mean
for their discipline. This first formula is intentionally not learned/calibrated;
changing discipline-specific weights requires code and a new formula version.

Usefulness combines **65% legitimacy**, up to **15 points for citations**,
**5 for outgoing references**, **13 for independent corroborating sources**, and
only **2 for topic text length**. These points live in `scoring.USEFULNESS_POINTS`
next to the legitimacy `WEIGHTS`, and `GET /overview` returns both, so clients never
hard-code the formula. Counts/length have logarithmic saturation
(1000 citations, 100 references, 5000 topic words); corroboration saturates at five
unique works. Retractions cap usefulness at 10. Reported citation/reference counts
are not proof of correctness and include criticism/self-citation where the
provider does not distinguish them.

Independent corroboration requires an explicit `corroborates` edge directed from
the peer to the evaluated source, reviewed human origin and the same reviewed
topic. Different known publishers and disjoint known authors are required, and
matching DOIs are excluded. This conservative proxy does not prove institutional
independence or eliminate every mirror/self-citation; unknown independence gives
no credit. Shared container membership alone gives none.

Length is assigned only when a reviewed page has **one** tagged topic. Multi-topic
pages keep total words but get no per-topic length credit until a future
section-level implementation can measure it honestly. Bulk text never adds more
than two usefulness points. Scores must not be used as automatic training approval.

## Collect and integrate with Kumaru learning

```powershell
kumaru academic collect SOURCE_ID
kumaru academic export SOURCE_ID --output C:\references\calculus.txt
kumaru learn Calculus --version reviewed --source C:\references\calculus.txt -o C:\references\calculus.json
```

Only reviewed topic coverage, verified human origin and explicit text permission
permit collection. Export creates a **new** UTF-8 file and never overwrites.
It contains the text, source URL, final retrieval URL, collection time,
content hash, topic tags, scores and an untrusted-reference notice.
Export multiple sources into a documentation folder and use `kumaru learn`
on that folder for existing BM25 retrieval. Import/export is explicit; unknown
discoveries never automatically enter chat context or training data.

Retrieval supports HTML and plain text, strips markup using Kumaru's existing
documentation converter, and executes no scripts. PDF, book files, scanned pages,
JavaScript-only sites and gated full text remain metadata-only. Each response is
bounded to 2 MB, with 20-second HTTP timeouts, public HTTP(S) only, DNS validation
and IP-pinned TLS connections using the system's trusted CA certificates.
Non-public addresses, mixed public/private DNS
answers, custom ports and credential URLs are refused. A redirect from HTTPS to a
plain-HTTP URL (common for DOI resolvers) is followed only as its HTTPS form; nothing
is sent over plain HTTP, and a site that insists on downgrading is refused.
Every redirect is checked. Cross-host full-text redirects require separately
cataloguing/reviewing the destination so permissions are not transferred silently.

Robots rules are honored at each page/redirect, matched as RFC 9309 specifies: the
longest matching rule wins (`Allow` on a tie), `*` and a final `$` are patterns, and
a group naming `KumaruAcademic` replaces the `*` group. Requests are paced by host (at
least one second, including declared crawl delays/rates). Missing robots (404)
allows access; policy failures, including a challenge on robots.txt itself, fail
closed. Retrieval never recursively follows
page links or bulk-downloads a repository. For large catalogues, paginate `list`
with `--limit` (up to 500) and `--offset`.

Provider, network, policy and parsing errors are explicit. Inaccessible inspection/
collection updates its source access note; a transient network failure (HTTP 503)
raises an error without changing the access category.
Internet DNS/HTTPS access is required for real search. Offline tests use fixtures,
not fictional references masquerading as discovered works.

## HTTP API

```powershell
$env:KUMARU_ACADEMIC__ENABLED = "true"
$env:KUMARU_ACADEMIC__DATABASE = "C:\references\academic.sqlite"
kumaru serve -c test
```

Visit `/docs` and select the `academic` endpoints. All endpoints use the existing
`server.api_key` bearer-token gate. Bind to localhost unless you have configured
authentication and deployment controls. The API works independently of model
readiness; the example uses echo to avoid loading a model.

| Method/path under `/v1/academic` | Operation |
|---|---|
| `GET /topics`, `PUT /topics/{id}` | Select/edit the hierarchy. `PUT` upserts, so check the ID is new before adding. |
| `POST /search` | `{topic_id, query, provider, limit, open_access_only}` -> `{source_ids, skipped, ...}`. `provider` is `crossref`, `openalex` or `brave`; OpenAlex also returns the `copy_ids` it catalogued, and only OpenAlex accepts `open_access_only`. Crossref works without a title or DOI/URL are skipped and counted in `skipped`; a malformed response envelope still fails. |
| `GET /sources`, `POST /sources` | Filter/paginate full metadata (arbitrary order, no total) or add a source. |
| `GET /overview?topic_id=` | Aggregate counts for a subject subtree (`by_access`, `by_kind`, `by_origin`, `needs_review`, `pending_notices`, `claims`, `unreviewed_claims`), direct `topic_counts`, distinct rolled-up `topic_totals`, the scoring `formula`, `brave_configured`, `openalex_key_configured` and `unpaywall_configured`. |
| `GET /library` | One page of source summaries with a server `total`. Filters: `topic_id` (subtree), `access`, `kind` (`works`/`containers`), `status` (`needs_review`/`human_verified`/`ai`), `q` (literal, case-insensitive substring of title, DOI, publisher and authors), `container_id`. `sort` is `updated`, `title`, `cited` or `year`; `limit` 1..100. Items include counts, `open_access`, `free_copies`, `is_free_copy`, the best in-scope `score` and never text or evidence. Invalid parameters are HTTP 400. |
| `GET /sources/{id}` | Metadata, topic scores, graph, reviews, notices, and `related`: id, title, kind, access and origin for every source in the graph. |
| `POST /sources/{id}/review` | Apply an evidence-backed human review (origin, permission, access and `access_note`, topics, findings). |
| `POST /relationships` | Add a directed graph edge. |
| `POST /sources/{id}/inspect` | Inspect page metadata/access without retaining text. |
| `POST /sources/{id}/collect` | Collect permitted text. |
| `POST /sources/{id}/open-access?provider=` | Find free copies of one DOI work (`openalex` by default, or `unpaywall`) -> `{indexed, status, license, copy_ids, warning}`. |
| `POST /open-access?topic_id=&limit=&provider=` | The same for up to 20 never/least-recently checked DOI works -> `{results, errors, stopped}`. |
| `GET /sources/{id}/text` | Explicit untrusted-reference text response. |

Listings and source details omit text. Network actions run in the server's
thread pool with two admitted jobs per process; extra jobs get HTTP 429 instead
of blocking inference. Robots/bytes/time limits apply to CLI and API alike.
Disabled academic mode registers no routes and creates no catalogue on startup.
SQLite is suitable for a local application, not an unbounded multi-worker crawler.

## Review dashboard

With `academic.enabled: true` and UI serving enabled, open
**http://127.0.0.1:8080/academic**, or follow **Academic source review** from the
chat sidebar. Use the existing configured `server.api_key` if required: the
connection pill in the header opens a dialog for it. The key is held in page
memory only, never in local/session storage, cookies or the address bar, so a reload
forgets it. The static page contains no catalogue data and loads nothing from other
sites; all data and mutations use the authenticated academic API.

**Library** (`#/library`) is a three-pane workbench: a *Subjects* tree with rolled-up
source counts and Access/Type/Review filters, the source list (search, sort, 30 rows
per page, score meters), and the *Inspector*. With nothing selected the inspector
shows the library at a glance for the selected subject. Add a topic under the
selected node, add an area, or edit a topic from the rail; topics tag sources and
never approve them. **Discover sources** asks Crossref (or Brave, when configured)
for unreviewed candidates and reports how many incomplete records were skipped.

The inspector for a source has **Overview**, **Evidence**, **Claims**, **Notices**,
**Relations** and **History** tabs. Overview lists what to check before relying on
the source (authorship, permission, access, topic coverage, editorial status, trusted
text) and shows one *score card* per subject. The score anatomy shows the legitimacy
and usefulness bars split by the formula's weights: reviewed dimensions are filled,
missing ones are hatched, and factual-criticism or retraction penalties are named.
The headline numbers are always the server's; unknown scores read "Unknown" or "—",
never 0. Relations shows ancestors, parents, children, descendants and siblings by
title, plus citation, corroboration and criticism links, and adds a relationship in a
plain sentence ("this source is cited by…") with a source picker instead of IDs.
History is a timeline of the append-only review records with the raw record one
disclosure away.

**Sources of Sources** (`#/containers`) groups catalogues by access and opens a
container's contents in the library. **Review queue** (`#/queue`) pages through
sources awaiting human review, pending editorial notices and unreviewed claims; a
quiet queue is not proof that every source is current.

Every decision is a native dialog: review source, add finding, record claim,
claim attribution, claim finding, report/review notice, add relationship, add source,
discover, topic and connection. Dialogs keep the user's input when a save fails,
show the error inline, ask before discarding changes, and return focus to the control
that opened them. Provenance and permission approvals need an evidence URL and note;
topic coverage needs its own checkbox (and is un-ticked when the subjects change);
confirming a notice needs an explicit impact (a retraction requires 1). Nothing is
approved automatically.

Routes are hash-based, so Back/Forward and links work:
`#/library`, `#/containers`, `#/queue` and `#/source/<id>[/<tab>[/<claimId>]]`.
Unsaved dialog edits are protected on navigation and page exit. The layout uses
three panes from 1280px, a subjects drawer below that, and one column below 760px
(Library and Inspector screens). Trees, tabs, comboboxes and dialogs have keyboard
support, text meets contrast guidelines, and motion respects reduced-motion.

The dashboard needs no build step or frontend dependency and does not use an AI
detector or let a discovered URL silently become approved training material.

## Claim-level evidence

A source score cannot establish every theorem or assertion inside it. Record
individual claims and keep their evidence separate:

```powershell
kumaru academic claim-add C:\references\claim.json
kumaru academic claims --source SOURCE_ID --limit 100 --offset 0
kumaru academic claim-show CLAIM_ID
kumaru academic claim-review CLAIM_ID C:\references\claim-review.json
```

A `ClaimCreate` file looks like:

```json
{
  "source_id": "SOURCE_ID",
  "topic_id": "calculus",
  "statement": "The assertion you actually located and are assessing.",
  "locator": "Chapter 2, theorem 1, page 42",
  "assumptions": ["The domain and conditions under which the assertion holds."]
}
```

The source must be an individual work/website tagged with the claim topic, not a
Sources of Sources container. The assertion, location and assumptions remain
immutable: a different or revised assertion is a new claim, not a silent rewrite
with an old proof attached.

Review attribution separately from proof. `ClaimReview` includes `reviewed`,
an audit `note` and optional findings:

```json
{
  "reviewed": true,
  "note": "Describe your actual attribution and evidence review.",
  "evidence": [
    {
      "id": "stable-finding-id",
      "source_id": "EVIDENCE_SOURCE_ID",
      "stance": "supports",
      "kind": "proof",
      "value": 0.9,
      "url": "https://example.org/actual-evidence",
      "locator": "Proof of theorem 1, paragraph 3",
      "note": "The checked reasoning and its limitations.",
      "topic_ids": ["calculus"],
      "reviewed": true
    }
  ]
}
```

This is a record-shape example, not evidence for a real source. Replace the URLs,
IDs and findings with your actual reviewed evidence. Finding source IDs must
exist in the catalogue. Their human provenance and topic coverage must be
reviewed before they contribute credit. Findings from containers, unverified
sources, sources with pending notices or retracted sources are excluded. External
references with reviewed factual flaws on the topic are also excluded until the
source evidence is resolved or a different reliable reference is supplied.

Stances are `supports`, `challenges`, and `context`; contextual findings are
visible but do not score. Challenges must be `factual_criticism` or `retraction`,
not a vague disagreement. Every scoped finding must match this claim's exact
topic. Origin/permission findings belong on sources, never claims.
Stable IDs update findings; setting their `reviewed` flag false withdraws credit.
Every claim review is retained in its audit trail.

Assessments expose the existing dimension weights and `claim-v1` eligibility
rules, with states `unreviewed`, `insufficient_evidence`, `evidence_supported`,
`disputed` and `withdrawn`. Whole-source proof, credentials and peer-review scores
are **not inherited**. Credentials/process evidence alone or zero-strength proof
does not earn an `evidence_supported` state. Positive, applicable reviewed proof,
historical acceptance or replication is required. These states still do not
certify truth. Source-wide confirmed withdrawals and topic-specific reviewed
factual flaws can limit claims attributed to that source.

## Corrections, concerns and retractions

```powershell
kumaru academic check-updates SOURCE_ID
kumaru academic monitor --limit 10
kumaru academic notices --state reported --limit 100 --offset 0
kumaru academic notice-add C:\references\notice.json
kumaru academic notice-review NOTICE_ID C:\references\notice-review.json
```

For DOI works, monitoring uses Crossref's documented `updates:DOI` filter to find
**editorial notices for that work**, not merely the work's own record.
Notice DOI, target match, type, date, provider and request URL are retained.
Errata/corrigenda become corrections; withdrawals become retractions; concerns
and unrecognized editorial update types remain distinct. A record that does not
identify the requested DOI fails visibly rather than being attributed to the
wrong source.

Automatically discovered notices start as **reported**, with no invented
numeric impact. They warn about unresolved updates and quarantine trusted text
collection, export and API retrieval until human review. Claim evidence using
the affected source is held back; affected direct citations/corroborating
references and primary/evidence-linked claims are listed. Container membership
alone does not make a database's other contents flawed. These dependency lists
are direct, not a claim to have proved every transitive impact.

Manual reports use a `NoticeCreate` record with `source_id`, `kind`, `url`,
`title` and optional `event_date`; kinds are `correction`, `retraction`,
`expression_of_concern`, `update`. Non-DOI websites can therefore still be
reviewed. They are not automatically monitored by Crossref.

A `NoticeReview` contains:

```json
{
  "state": "confirmed",
  "note": "Your checked factual impact and evidence.",
  "impact": 0.5,
  "affected_topic_ids": ["calculus"]
}
```

States are `confirmed`, `dismissed` and `reported` (reopen). Confirmation requires
an explicit impact from 0 to 1. Non-retraction factual impacts above zero require
specific source topic IDs. A confirmed retraction requires impact **1**; dismiss
a mistaken identity instead of confirming it with a zero penalty.
Zero can acknowledge a typographical/non-factual correction without inventing
a flaw. Confirming clears cached text so an obsolete capture is not reused.
Positive confirmed impacts become reviewed topic-scoped factual criticism or
source-wide retraction evidence and affect scores. Confirmed retractions remain
excluded from trusted text, including manually entered reviewed retraction
evidence. Dismissing/reopening removes that notice's assessed penalty but retains
the audit history; rediscovery preserves the human decision. Revised claims
should be recorded and reviewed as new assertions.

Each check persists its time, error and partial-scan state. Provider/DNS/schema
failures are not recorded as clean results. A bounded batch reports errors per
source and scans never/least-recently checked works first. Each source query is
limited to the first 100 editorial records; a larger result count is explicitly
reported as partial. Crossref metadata is not comprehensive; no reported notice
does not prove that no correction exists.

Periodic checks are optional and run only while the Kumaru server is alive:

```yaml
academic:
  enabled: true
  monitor_interval_s: 3600
  monitor_batch_size: 10
```

`monitor_interval_s: 0` (default) means **manual only**. Positive values wait that
interval between batches; the configured batch does not mean every catalogue
source is checked each hour. Each process has one scheduler using the same
two-slot network limit as manual actions. Shutdown stops further sources and
joins an in-progress worker. Multi-process deployments should enable the
scheduler in only one process to avoid duplicate polling.

Additional authenticated API endpoints:

| Method/path under `/v1/academic` | Operation |
|---|---|
| `GET /claims`, `POST /claims` | Filter (`source_id`, `topic_id`, `reviewed`) and paginate, or record located assertions. |
| `GET /claims/{id}`, `POST /claims/{id}/review` | Findings, assessments and human review. Details add `source` and `references` (id, title, kind, access, origin) for the claim's source and each finding's source. |
| `GET /notices`, `POST /notices` | Editorial queue or manual report. Details add `source`, `dependent_sources` and `affected_claims` (id, source_id, topic_id, statement) so clients show titles, not IDs. |
| `POST /notices/{id}/review` | Confirm, dismiss or reopen with an audit note. |
| `POST /sources/{id}/check-updates` | Check one DOI work. A DNS/provider outage is HTTP 503 and is recorded as a failed check, never as a clean one. |
| `POST /monitor?limit=10`, `GET /monitor/status` | Bounded batch and optional scheduler status. |

New tables are added with `CREATE TABLE IF NOT EXISTS`; existing catalogue
sources, reviews and collected content are preserved. Back up the SQLite file
before upgrading. No new optional packages are required.

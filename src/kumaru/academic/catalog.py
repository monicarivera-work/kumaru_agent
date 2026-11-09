"""SQLite source graph with transactional updates and an append-only review trail."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Collection, Iterable, Iterator
from contextlib import closing, contextmanager
from pathlib import Path
from typing import get_args

from kumaru.academic.models import (
    Access,
    Origin,
    Relationship,
    Review,
    Score,
    Source,
    SourceKind,
    Topic,
    mathematics_topics,
    now,
)
from kumaru.academic.reliability import (
    Claim,
    ClaimCreate,
    ClaimReview,
    Notice,
    NoticeCreate,
    NoticeReview,
    claim_score,
)
from kumaru.academic.scoring import USEFULNESS_POINTS, WEIGHTS, score_source
from kumaru.core.errors import ConfigError, RequestError

LIBRARY_SORTS = {
    "updated": "json_extract(payload, '$.updated_at') DESC, id",
    "title": "casefold(json_extract(payload, '$.title')), id",
    "cited": (
        "(json_extract(payload, '$.citation_count') IS NULL), "
        "json_extract(payload, '$.citation_count') DESC, id"
    ),
    "year": (
        "(json_extract(payload, '$.published_year') IS NULL), "
        "json_extract(payload, '$.published_year') DESC, id"
    ),
}
LIBRARY_STATUSES = ("needs_review", "human_verified", "ai")
NEEDS_REVIEW = (
    "(json_extract(payload, '$.origin') = 'unknown' "
    "OR COALESCE(json_extract(payload, '$.topic_reviewed'), 0) = 0)"
)


def _in(values: Collection[object]) -> str:
    return "(" + ",".join("?" for _ in values) + ")"


def _casefold(value: object) -> str:
    return value.casefold() if isinstance(value, str) else ""


def _like(text: str) -> str:
    escaped = text.casefold().replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"%{escaped}%"


class Catalog:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser().resolve()
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise ConfigError(f"Cannot create academic catalogue directory: {exc}") from exc
        with self.connection() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS topics (
                    id TEXT PRIMARY KEY, parent_id TEXT REFERENCES topics(id), payload TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS sources (
                    id TEXT PRIMARY KEY, url TEXT UNIQUE NOT NULL, payload TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS source_topics (
                    source_id TEXT REFERENCES sources(id), topic_id TEXT REFERENCES topics(id),
                    PRIMARY KEY(source_id, topic_id)
                );
                CREATE TABLE IF NOT EXISTS links (
                    source_id TEXT REFERENCES sources(id), target_id TEXT REFERENCES sources(id),
                    kind TEXT NOT NULL, note TEXT NOT NULL,
                    PRIMARY KEY(source_id, target_id, kind), CHECK(source_id != target_id)
                );
                CREATE TABLE IF NOT EXISTS reviews (
                    id INTEGER PRIMARY KEY, source_id TEXT REFERENCES sources(id),
                    recorded_at TEXT NOT NULL, payload TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS links_target ON links(target_id, kind);
                CREATE INDEX IF NOT EXISTS topics_sources ON source_topics(topic_id);
                CREATE INDEX IF NOT EXISTS source_doi ON sources(json_extract(payload, '$.doi'));
                CREATE TABLE IF NOT EXISTS claims (
                    id TEXT PRIMARY KEY, source_id TEXT NOT NULL REFERENCES sources(id),
                    topic_id TEXT NOT NULL REFERENCES topics(id), payload TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS claim_refs (
                    claim_id TEXT REFERENCES claims(id), source_id TEXT REFERENCES sources(id),
                    PRIMARY KEY(claim_id, source_id)
                );
                CREATE TABLE IF NOT EXISTS claim_reviews (
                    id INTEGER PRIMARY KEY, claim_id TEXT REFERENCES claims(id),
                    recorded_at TEXT NOT NULL, payload TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS notices (
                    id TEXT PRIMARY KEY, source_id TEXT NOT NULL REFERENCES sources(id),
                    payload TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS update_checks (
                    source_id TEXT PRIMARY KEY REFERENCES sources(id), checked_at TEXT NOT NULL,
                    error TEXT, partial INTEGER NOT NULL DEFAULT 0
                );
                CREATE INDEX IF NOT EXISTS claims_source ON claims(source_id);
                CREATE INDEX IF NOT EXISTS claim_refs_source ON claim_refs(source_id);
                CREATE INDEX IF NOT EXISTS notices_source ON notices(source_id);
            """)
            for topic in mathematics_topics():
                db.execute(
                    "INSERT OR IGNORE INTO topics VALUES (?, ?, ?)",
                    (topic.id, topic.parent_id, topic.model_dump_json()),
                )

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        try:
            with closing(sqlite3.connect(self.path, timeout=15)) as db, db:
                db.row_factory = sqlite3.Row
                db.execute("PRAGMA foreign_keys = ON")
                yield db
        except sqlite3.Error as exc:
            raise ConfigError(f"Academic catalogue database error: {exc}") from exc

    def topics(self) -> list[Topic]:
        with self.connection() as db:
            return [
                Topic.model_validate_json(row["payload"])
                for row in db.execute("SELECT payload FROM topics ORDER BY id")
            ]

    def topic(self, id_: str) -> Topic:
        for topic in self.topics():
            if topic.id == id_:
                return topic
        raise RequestError(f"Unknown academic topic: {id_}")

    def topic_path(self, id_: str) -> list[Topic]:
        topics = {topic.id: topic for topic in self.topics()}
        self.topic(id_)
        path = [topics[id_]]
        while path[-1].parent_id:
            path.append(topics[path[-1].parent_id])
        return list(reversed(path))

    def descendants(self, id_: str) -> set[str]:
        self.topic(id_)
        with self.connection() as db:
            return {
                row[0] for row in db.execute(
                    """WITH RECURSIVE tree(id) AS (
                        SELECT id FROM topics WHERE id = ?
                        UNION SELECT t.id FROM topics t JOIN tree ON t.parent_id = tree.id
                    ) SELECT id FROM tree""", (id_,),
                )
            }

    def put_topic(self, topic: Topic) -> Topic:
        if topic.parent_id == topic.id:
            raise RequestError("A topic cannot parent itself")
        existing = {item.id: item for item in self.topics()}
        levels = {"area": 0, "subfield": 1, "unit": 2, "chapter": 3, "subchapter": 4}
        if topic.parent_id:
            parent = self.topic(topic.parent_id)
            if levels[parent.level] >= levels[topic.level]:
                raise RequestError("Topic parents must be at a broader academic level")
        elif topic.level != "area":
            raise RequestError("Only an academic area can be a root topic")
        for child in existing.values():
            if child.parent_id == topic.id and levels[child.level] <= levels[topic.level]:
                raise RequestError("Changing the level would invalidate existing children")
        if topic.id in existing and topic.parent_id in self.descendants(topic.id):
            raise RequestError("Topic hierarchy cannot contain cycles")
        with self.connection() as db:
            db.execute(
                "INSERT INTO topics VALUES (?, ?, ?) ON CONFLICT(id) DO UPDATE SET "
                "parent_id=excluded.parent_id, payload=excluded.payload",
                (topic.id, topic.parent_id, topic.model_dump_json()),
            )
        return topic

    @staticmethod
    def _get(db: sqlite3.Connection, id_: str) -> Source:
        row = db.execute("SELECT payload FROM sources WHERE id = ?", (id_,)).fetchone()
        if row is None:
            raise RequestError(f"Unknown academic source: {id_}")
        return Source.model_validate_json(row["payload"])

    def get(self, id_: str) -> Source:
        with self.connection() as db:
            return self._get(db, id_)

    @staticmethod
    def _save(db: sqlite3.Connection, source: Source) -> None:
        source = Source.model_validate(source.model_dump())
        if not source.topic_ids:
            raise RequestError("Catalogue sources must identify at least one academic topic")
        for topic_id in source.topic_ids:
            if not db.execute("SELECT 1 FROM topics WHERE id = ?", (topic_id,)).fetchone():
                raise RequestError(f"Unknown academic topic: {topic_id}")
        for evidence in source.evidence:
            if not set(evidence.topic_ids) <= set(source.topic_ids):
                raise RequestError("Evidence topics must be tagged on the source")
        source.updated_at = now()
        db.execute(
            "INSERT INTO sources VALUES (?, ?, ?) ON CONFLICT(id) DO UPDATE SET "
            "payload=excluded.payload", (source.id, source.url, source.model_dump_json()),
        )
        db.execute("DELETE FROM source_topics WHERE source_id = ?", (source.id,))
        db.executemany(
            "INSERT INTO source_topics VALUES (?, ?)",
            [(source.id, topic_id) for topic_id in dict.fromkeys(source.topic_ids)],
        )

    def discover(self, candidate: Source) -> Source:
        """Refresh metadata, but never overwrite a human review or collected text."""
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT payload FROM sources WHERE id = ? OR "
                "(? IS NOT NULL AND lower(json_extract(payload, '$.doi')) = lower(?)) LIMIT 1",
                (candidate.id, candidate.doi, candidate.doi),
            ).fetchone()
            if row:
                current = Source.model_validate_json(row["payload"])
                updates = {
                    name: getattr(candidate, name)
                    for name in (
                        "citation_count", "reference_count", "metadata_url", "doi",
                        "publisher", "published_year", "open_access",
                    )
                    if getattr(candidate, name) is not None
                }
                if not current.topic_reviewed:
                    updates["topic_ids"] = sorted(set(current.topic_ids + candidate.topic_ids))
                candidate = Source.model_validate({**current.model_dump(), **updates})
            self._save(db, candidate)
        return self.get(candidate.id)

    def review(self, id_: str, review: Review) -> Source:
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            source = self._get(db, id_)
            evidence = {item.id: item for item in source.evidence}
            evidence.update({item.id: item for item in review.evidence})
            if len(evidence) > 500:
                raise RequestError("Sources support at most 500 findings; update an existing evidence ID")
            updates = review.model_dump(exclude_none=True, exclude={"evidence"})
            updates["evidence"] = [item.model_dump() for item in evidence.values()]
            if review.topic_ids is not None:
                updates["topic_reviewed"] = review.topic_reviewed if review.topic_reviewed is not None else True
            for field, kind, affirmative in (
                ("origin", "origin", "human_verified"),
                ("permission", "permission", "text_permitted"),
            ):
                if updates.get(field, getattr(source, field)) == affirmative and not any(
                    item.kind == kind and item.reviewed and item.value == 1
                    for item in evidence.values()
                ):
                    raise RequestError(f"{field} approval requires reviewed {kind} evidence at value 1")
            if (
                updates.get("origin", source.origin) != "human_verified"
                or updates.get("permission", source.permission) != "text_permitted"
                or not updates.get("topic_reviewed", source.topic_reviewed)
            ):
                updates.update(
                    text=None, text_url=None, text_sha256=None, collected_at=None,
                    content_type=None, word_count=0, truncated=False,
                    topic_word_counts={},
                )
            elif review.topic_ids is not None and set(review.topic_ids) != set(source.topic_ids):
                updates["topic_word_counts"] = {}
            source = Source.model_validate({**source.model_dump(), **updates})
            self._save(db, source)
            db.execute(
                "INSERT INTO reviews(source_id, recorded_at, payload) VALUES (?, ?, ?)",
                (id_, now(), review.model_dump_json()),
            )
        return self.get(id_)

    def save_metadata(self, id_: str, updates: dict) -> Source:
        allowed = {
            "title", "authors", "publisher", "doi", "access", "access_note", "content_type",
            "open_access",
        }
        if not set(updates) <= allowed:
            raise RequestError("Metadata updates cannot approve provenance, permissions or content")
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            source = self._get(db, id_)
            self._save(db, Source.model_validate({**source.model_dump(), **updates}))
        return self.get(id_)

    def save_collection(self, id_: str, updates: dict) -> Source:
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            source = self._get(db, id_)
            if source.kind == "sources_of_sources":
                raise RequestError("Catalogue individual works before collecting their text")
            source = Source.model_validate({**source.model_dump(), **updates})
            self._save(db, source)
        return self.get(id_)

    def list_sources(
        self, *, topic_id: str | None = None, access: Access | None = None,
        containers: bool | None = None, limit: int = 100, offset: int = 0,
    ) -> list[Source]:
        if not 1 <= limit <= 500 or offset < 0:
            raise RequestError("Source page requires limit 1..500 and nonnegative offset")
        ids = self.descendants(topic_id) if topic_id else None
        with self.connection() as db:
            clauses = []
            args: list[str | int] = []
            if ids is not None:
                clauses.append(
                    "id IN (SELECT source_id FROM source_topics WHERE topic_id IN ("
                    + ",".join("?" for _ in ids) + "))"
                )
                args.extend(sorted(ids))
            if access:
                clauses.append("json_extract(payload, '$.access') = ?")
                args.append(access)
            if containers is not None:
                clauses.append(
                    "json_extract(payload, '$.kind') "
                    + ("=" if containers else "!=") + " 'sources_of_sources'"
                )
            where = " WHERE " + " AND ".join(clauses) if clauses else ""
            rows = db.execute(
                "SELECT payload FROM sources" + where + " ORDER BY id LIMIT ? OFFSET ?",
                [*args, limit, offset],
            )
            return [Source.model_validate_json(row[0]) for row in rows]

    def open_access_due(self, *, topic_id: str | None = None, limit: int = 10) -> list[Source]:
        """DOI works never checked for free copies first, then the least recently checked.

        Sources catalogued as someone's free copy are skipped: they are the answer.
        """
        if not 1 <= limit <= 20:
            raise RequestError("Open-access batches check 1..20 works")
        clauses = [
            "COALESCE(json_extract(payload, '$.doi'), '') != ''",
            "json_extract(payload, '$.kind') != 'sources_of_sources'",
            "id NOT IN (SELECT target_id FROM links WHERE kind = 'open_copy')",
        ]
        args: list[str | int] = []
        if topic_id:
            scope = sorted(self.descendants(topic_id))
            clauses.append(
                "id IN (SELECT source_id FROM source_topics WHERE topic_id IN " + _in(scope) + ")"
            )
            args.extend(scope)
        with self.connection() as db:
            rows = db.execute(
                "SELECT json_remove(payload, '$.text') FROM sources WHERE " + " AND ".join(clauses)
                + " ORDER BY json_extract(payload, '$.open_access.checked_at') IS NOT NULL, "
                "json_extract(payload, '$.open_access.checked_at'), id LIMIT ?",
                [*args, limit],
            ).fetchall()
        return [Source.model_validate_json(row[0]) for row in rows]

    def source_cards(self, ids: Iterable[str]) -> dict[str, dict]:
        """Titles for source IDs, so clients never have to show raw identifiers."""
        unique = sorted(set(ids))
        cards: dict[str, dict] = {}
        with self.connection() as db:
            for start in range(0, len(unique), 500):
                chunk = unique[start:start + 500]
                rows = db.execute(
                    "SELECT id, json_extract(payload, '$.title'), json_extract(payload, '$.kind'), "
                    "json_extract(payload, '$.access'), json_extract(payload, '$.origin') "
                    "FROM sources WHERE id IN " + _in(chunk), chunk,
                )
                for row in rows:
                    cards[row[0]] = dict(
                        id=row[0], title=row[1], kind=row[2], access=row[3], origin=row[4],
                    )
        return cards

    def overview(self, topic_id: str | None = None) -> dict:
        """Aggregate counts for the sources tagged with a topic or any descendant."""
        ids = sorted(self.descendants(topic_id)) if topic_id else None

        def scoped(column: str) -> str:
            if ids is None:
                return "1"
            return (
                f"{column} IN (SELECT source_id FROM source_topics WHERE topic_id IN "
                + _in(ids) + ")"
            )

        args = ids or []
        by_access = dict.fromkeys(get_args(Access), 0)
        by_kind = dict.fromkeys(get_args(SourceKind), 0)
        by_origin = dict.fromkeys(get_args(Origin), 0)
        total = needs_review = 0
        with self.connection() as db:
            groups = db.execute(
                "SELECT json_extract(payload, '$.access'), json_extract(payload, '$.kind'), "
                "json_extract(payload, '$.origin'), "
                "COALESCE(json_extract(payload, '$.topic_reviewed'), 0), count(*) "
                "FROM sources WHERE " + scoped("id") + " GROUP BY 1, 2, 3, 4", args,
            )
            for access, kind, origin, topics_reviewed, count in groups:
                total += count
                by_access[access] += count
                by_kind[kind] += count
                by_origin[origin] += count
                if origin == "unknown" or not topics_reviewed:
                    needs_review += count
            pending = db.execute(
                "SELECT count(*) FROM notices WHERE json_extract(payload, '$.state') = 'reported' "
                "AND " + scoped("source_id"), args,
            ).fetchone()[0]
            claims, unreviewed = db.execute(
                "SELECT count(*), COALESCE(sum(CASE WHEN json_extract(payload, '$.reviewed') "
                "THEN 0 ELSE 1 END), 0) FROM claims WHERE " + scoped("source_id"), args,
            ).fetchone()
            topic_counts = {row[0]: 0 for row in db.execute("SELECT id FROM topics ORDER BY id")}
            for topic, count in db.execute(
                "SELECT topic_id, count(*) FROM source_topics GROUP BY topic_id",
            ):
                topic_counts[topic] = count
            # A source tagged with several topics of one subtree still counts once there.
            topic_totals = dict.fromkeys(topic_counts, 0)
            for topic, count in db.execute(
                """WITH RECURSIVE closure(ancestor, descendant) AS (
                    SELECT id, id FROM topics
                    UNION SELECT c.ancestor, t.id FROM closure c JOIN topics t ON t.parent_id = c.descendant
                ) SELECT c.ancestor, count(DISTINCT st.source_id) FROM closure c
                  JOIN source_topics st ON st.topic_id = c.descendant GROUP BY c.ancestor""",
            ):
                topic_totals[topic] = count
        return dict(
            topic_id=topic_id, total=total, by_access=by_access, by_kind=by_kind,
            by_origin=by_origin, needs_review=needs_review, pending_notices=pending,
            claims=claims, unreviewed_claims=unreviewed, topic_counts=topic_counts,
            topic_totals=topic_totals,
            formula=dict(
                version=Score.model_fields["version"].default,
                legitimacy_weights=dict(WEIGHTS), usefulness_points=dict(USEFULNESS_POINTS),
            ),
        )

    def library(
        self, *, topic_id: str | None = None, access: str | None = None, kind: str | None = None,
        status: str | None = None, q: str | None = None, container_id: str | None = None,
        sort: str = "updated", limit: int = 30, offset: int = 0,
    ) -> dict:
        """One filtered, sorted page of source summaries; never text or evidence."""
        if not 1 <= limit <= 100 or offset < 0:
            raise RequestError("Library page requires limit 1..100 and nonnegative offset")
        topic_id, access, kind, status, container_id = (
            value or None for value in (topic_id, access, kind, status, container_id)
        )
        if access is not None and access not in get_args(Access):
            raise RequestError("Access must be unknown, open, account, paywall or inaccessible")
        if kind not in {None, "works", "containers"}:
            raise RequestError("Library kind must be works or containers")
        if status is not None and status not in LIBRARY_STATUSES:
            raise RequestError("Library status must be needs_review, human_verified or ai")
        if sort not in LIBRARY_SORTS:
            raise RequestError("Library sort must be updated, title, cited or year")
        if q is not None and len(q) > 200:
            raise RequestError("Library search is limited to 200 characters")
        scope = self.descendants(topic_id) if topic_id else None
        clauses: list[str] = []
        args: list[str | int] = []
        if scope is not None:
            ids = sorted(scope)
            clauses.append(
                "id IN (SELECT source_id FROM source_topics WHERE topic_id IN " + _in(ids) + ")"
            )
            args.extend(ids)
        if access:
            clauses.append("json_extract(payload, '$.access') = ?")
            args.append(access)
        if kind:
            clauses.append(
                "json_extract(payload, '$.kind') "
                + ("!=" if kind == "works" else "=") + " 'sources_of_sources'"
            )
        if status == "needs_review":
            clauses.append(NEEDS_REVIEW)
        elif status:
            clauses.append("json_extract(payload, '$.origin') = ?")
            args.append(status)
        if q and q.strip():
            clauses.append(
                "(casefold(json_extract(payload, '$.title')) LIKE ? ESCAPE '\\' "
                "OR casefold(json_extract(payload, '$.doi')) LIKE ? ESCAPE '\\' "
                "OR casefold(json_extract(payload, '$.publisher')) LIKE ? ESCAPE '\\' "
                "OR EXISTS (SELECT 1 FROM json_each(payload, '$.authors') "
                "WHERE casefold(value) LIKE ? ESCAPE '\\'))"
            )
            args.extend([_like(q.strip())] * 4)
        if container_id:
            clauses.append(
                "id IN (SELECT target_id FROM links WHERE source_id = ? AND kind = 'contains')"
            )
            args.append(container_id)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        with self.connection() as db:
            db.create_function("casefold", 1, _casefold, deterministic=True)
            if container_id:
                self._get(db, container_id)
            total = db.execute("SELECT count(*) FROM sources" + where, args).fetchone()[0]
            # The stored text can be megabytes; a listing never needs it.
            rows = db.execute(
                "SELECT json_remove(payload, '$.text') FROM sources" + where
                + " ORDER BY " + LIBRARY_SORTS[sort] + " LIMIT ? OFFSET ?",
                [*args, limit, offset],
            ).fetchall()
            sources = [Source.model_validate_json(row[0]) for row in rows]
            page = [source.id for source in sources]
            counts: dict[str, dict[str, int]] = {
                "claims": {}, "children": {}, "pending": {}, "copies": {}, "copy_of": {},
            }
            if page:
                for name, query in (
                    ("claims", "SELECT source_id, count(*) FROM claims WHERE source_id IN "),
                    ("children", "SELECT source_id, count(*) FROM links "
                                 "WHERE kind = 'contains' AND source_id IN "),
                    ("pending", "SELECT source_id, count(*) FROM notices WHERE "
                                "json_extract(payload, '$.state') = 'reported' AND source_id IN "),
                    ("copies", "SELECT source_id, count(*) FROM links "
                               "WHERE kind = 'open_copy' AND source_id IN "),
                    ("copy_of", "SELECT target_id, count(*) FROM links "
                                "WHERE kind = 'open_copy' AND target_id IN "),
                ):
                    counts[name] = {
                        row[0]: row[1]
                        for row in db.execute(query + _in(page) + " GROUP BY 1", page)
                    }
        items = [
            dict(
                id=source.id, url=source.url, title=source.title, kind=source.kind,
                access=source.access, access_note=source.access_note, origin=source.origin,
                permission=source.permission, topic_ids=source.topic_ids,
                topic_reviewed=source.topic_reviewed, doi=source.doi, publisher=source.publisher,
                published_year=source.published_year, citation_count=source.citation_count,
                reference_count=source.reference_count, authors=source.authors[:3],
                author_count=len(source.authors), discovered_at=source.discovered_at,
                updated_at=source.updated_at, collected_at=source.collected_at,
                word_count=source.word_count, evidence_count=len(source.evidence),
                claim_count=counts["claims"].get(source.id, 0),
                child_count=counts["children"].get(source.id, 0),
                pending_notices=counts["pending"].get(source.id, 0),
                free_copies=counts["copies"].get(source.id, 0),
                is_free_copy=counts["copy_of"].get(source.id, 0) > 0,
                open_access=source.open_access.model_dump() if source.open_access else None,
                withdrawn=any(
                    item.kind == "retraction" and item.reviewed and item.value > 0
                    for item in source.evidence
                ),
                score=self._listed_score(source.id, scope),
            )
            for source in sources
        ]
        return dict(items=items, total=total, limit=limit, offset=offset)

    def _listed_score(self, id_: str, scope: set[str] | None) -> dict | None:
        """The most useful in-scope topic score, so one row never hides a better one."""
        scores = [item for item in self.scores(id_) if scope is None or item["topic_id"] in scope]
        if not scores:
            return None
        scored = [item for item in scores if item["usefulness"] is not None]
        best = max(scored, key=lambda item: item["usefulness"]) if scored else scores[0]
        return {
            name: best[name]
            for name in ("topic_id", "legitimacy", "usefulness", "evidence_coverage")
        }

    def link(self, relation: Relationship) -> Relationship:
        if relation.source_id == relation.target_id:
            raise RequestError("A source cannot link to itself")
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            self._get(db, relation.source_id)
            self._get(db, relation.target_id)
            if relation.kind == "contains":
                cycle = db.execute(
                    """WITH RECURSIVE children(id) AS (
                        SELECT target_id FROM links WHERE source_id = ? AND kind = 'contains'
                        UNION SELECT l.target_id FROM links l JOIN children ON l.source_id = children.id
                        WHERE l.kind = 'contains'
                    ) SELECT 1 FROM children WHERE id = ?""",
                    (relation.target_id, relation.source_id),
                ).fetchone()
                if cycle:
                    raise RequestError("Source containment cannot contain cycles")
            db.execute(
                "INSERT INTO links VALUES (?, ?, ?, ?) ON CONFLICT(source_id,target_id,kind) "
                "DO UPDATE SET note=excluded.note",
                (relation.source_id, relation.target_id, relation.kind, relation.note),
            )
        return relation

    def graph(self, id_: str) -> dict:
        self.get(id_)
        with self.connection() as db:
            links = [
                dict(row) for row in db.execute(
                    "SELECT * FROM links WHERE source_id = ? OR target_id = ?", (id_, id_),
                )
            ]
            ancestors = [
                row[0] for row in db.execute(
                    """WITH RECURSIVE ancestors(id) AS (
                        SELECT source_id FROM links WHERE target_id = ? AND kind = 'contains'
                        UNION SELECT l.source_id FROM links l JOIN ancestors ON l.target_id = ancestors.id
                        WHERE l.kind = 'contains'
                    ) SELECT id FROM ancestors ORDER BY id""", (id_,),
                )
            ]
            descendants = [
                row[0] for row in db.execute(
                    """WITH RECURSIVE children(id) AS (
                        SELECT target_id FROM links WHERE source_id = ? AND kind = 'contains'
                        UNION SELECT l.target_id FROM links l JOIN children ON l.source_id = children.id
                        WHERE l.kind = 'contains'
                    ) SELECT id FROM children ORDER BY id""", (id_,),
                )
            ]
            siblings = [
                row[0] for row in db.execute(
                    """SELECT DISTINCT b.target_id FROM links a JOIN links b ON a.source_id=b.source_id
                        WHERE a.target_id=? AND a.kind='contains' AND b.kind='contains'
                        AND b.target_id != ? ORDER BY b.target_id""", (id_, id_),
                )
            ]
        return dict(links=links, ancestors=ancestors, descendants=descendants, siblings=siblings)

    def scores(self, id_: str) -> list[dict]:
        source = self.get(id_)
        with self.connection() as db:
            # A graph edge is a reviewer's explicit corroboration, not a shared
            # publisher, mirror, citation or unverified search result.
            ids = [
                row[0] for row in db.execute(
                    "SELECT source_id FROM links WHERE target_id = ? AND kind = 'corroborates'",
                    (id_,),
                )
            ]
            peers = [self._get(db, peer) for peer in ids]
        blocked_peers = self.pending_notice_sources({peer.id for peer in peers})
        results = [
            score_source(source, topic, independent_sources=[
                peer.id for peer in peers
                if peer.origin == "human_verified" and peer.topic_reviewed
                and peer.kind != "sources_of_sources"
                and topic in peer.topic_ids
                and peer.id not in blocked_peers
                and not any(
                    item.reviewed and item.value > 0
                    and (item.kind == "retraction" or (
                        item.kind == "factual_criticism" and (not item.topic_ids or topic in item.topic_ids)
                    )) for item in peer.evidence
                )
                and peer.publisher and source.publisher
                and peer.publisher.strip().casefold() != source.publisher.strip().casefold()
                and peer.authors and source.authors
                and not (peer.doi and source.doi and peer.doi.casefold() == source.doi.casefold())
                and not {author.casefold() for author in peer.authors}
                & {author.casefold() for author in source.authors}
            ]).model_dump()
            for topic in source.topic_ids
        ]
        if id_ in self.pending_notice_sources({id_}):
            for result in results:
                result["warnings"].append("Editorial updates have been reported and require human review.")
        return results

    def add_claim(self, record: ClaimCreate) -> Claim:
        claim = Claim(**record.model_dump())
        with self.connection() as db:
            source = self._get(db, claim.source_id)
            if source.kind == "sources_of_sources" or claim.topic_id not in source.topic_ids:
                raise RequestError("Claims require an individual source tagged with the claim topic")
            db.execute(
                "INSERT INTO claims VALUES (?, ?, ?, ?)",
                (claim.id, claim.source_id, claim.topic_id, claim.model_dump_json()),
            )
        return claim

    @staticmethod
    def _claim(db: sqlite3.Connection, id_: str) -> Claim:
        row = db.execute("SELECT payload FROM claims WHERE id=?", (id_,)).fetchone()
        if row is None:
            raise RequestError(f"Unknown academic claim: {id_}")
        return Claim.model_validate_json(row[0])

    def get_claim(self, id_: str) -> Claim:
        with self.connection() as db:
            return self._claim(db, id_)

    def claims(
        self, *, source_id: str | None = None, topic_id: str | None = None,
        reviewed: bool | None = None, limit: int = 100, offset: int = 0,
    ) -> list[Claim]:
        if not 1 <= limit <= 500 or offset < 0:
            raise RequestError("Claim page requires limit 1..500 and nonnegative offset")
        clauses: list[str] = []
        args: list[str | int] = []
        if source_id:
            self.get(source_id)
            clauses.append("source_id=?")
            args.append(source_id)
        if topic_id:
            ids = sorted(self.descendants(topic_id))
            clauses.append("topic_id IN (" + ",".join("?" for _ in ids) + ")")
            args.extend(ids)
        if reviewed is not None:
            clauses.append("json_extract(payload, '$.reviewed') = ?")
            args.append(int(reviewed))
        with self.connection() as db:
            where = " WHERE " + " AND ".join(clauses) if clauses else ""
            rows = db.execute(
                "SELECT payload FROM claims" + where + " ORDER BY id LIMIT ? OFFSET ?",
                [*args, limit, offset],
            )
            return [Claim.model_validate_json(row[0]) for row in rows]

    def review_claim(self, id_: str, review: ClaimReview) -> Claim:
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            claim = self._claim(db, id_)
            findings = {item.id: item for item in claim.evidence}
            findings.update({item.id: item for item in review.evidence})
            if len(findings) > 500:
                raise RequestError("Claims support at most 500 findings; update an existing evidence ID")
            for item in findings.values():
                self._get(db, item.source_id)
                if item.topic_ids and set(item.topic_ids) != {claim.topic_id}:
                    raise RequestError("Claim findings must address only this claim's topic")
            claim = Claim.model_validate({
                **claim.model_dump(), "reviewed": review.reviewed,
                "evidence": [item.model_dump() for item in findings.values()], "updated_at": now(),
            })
            db.execute("UPDATE claims SET payload=? WHERE id=?", (claim.model_dump_json(), id_))
            db.execute("DELETE FROM claim_refs WHERE claim_id=?", (id_,))
            db.executemany(
                "INSERT INTO claim_refs VALUES (?, ?)",
                [(id_, source_id) for source_id in {item.source_id for item in claim.evidence}],
            )
            db.execute(
                "INSERT INTO claim_reviews(claim_id,recorded_at,payload) VALUES(?,?,?)",
                (id_, now(), review.model_dump_json()),
            )
        return claim

    def claim_details(self, id_: str) -> dict:
        claim = self.get_claim(id_)
        source = self.get(claim.source_id)
        references = {item.source_id: self.get(item.source_id) for item in claim.evidence}
        blocked = self.pending_notice_sources({source.id, *references})
        with self.connection() as db:
            history = [
                {"recorded_at": row[0], "review": json.loads(row[1])}
                for row in db.execute(
                    "SELECT recorded_at,payload FROM claim_reviews WHERE claim_id=? ORDER BY id",
                    (id_,),
                )
            ]
        cards = self.source_cards({claim.source_id, *references})
        return dict(
            claim=claim.model_dump(), assessment=claim_score(
                claim, source, references, blocked_sources=blocked,
            ), reviews=history, source=cards.get(claim.source_id),
            references={ref: cards[ref] for ref in references if ref in cards},
        )

    @staticmethod
    def _notice(db: sqlite3.Connection, id_: str) -> Notice:
        row = db.execute("SELECT payload FROM notices WHERE id=?", (id_,)).fetchone()
        if row is None:
            raise RequestError(f"Unknown editorial notice: {id_}")
        return Notice.model_validate_json(row[0])

    def report_notice(self, record: NoticeCreate) -> Notice:
        notice = Notice(**record.model_dump())
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            self._get(db, notice.source_id)
            row = db.execute("SELECT payload FROM notices WHERE id=?", (notice.id,)).fetchone()
            if row:
                current = Notice.model_validate_json(row[0])
                notice = Notice.model_validate({
                    **current.model_dump(), "last_seen_at": now(),
                    "title": notice.title, "metadata_url": notice.metadata_url,
                })
            db.execute(
                "INSERT INTO notices VALUES(?,?,?) ON CONFLICT(id) DO UPDATE SET payload=excluded.payload",
                (notice.id, notice.source_id, notice.model_dump_json()),
            )
        return notice

    def notices(
        self, *, source_id: str | None = None, state: str | None = None,
        limit: int = 100, offset: int = 0,
    ) -> list[Notice]:
        if not 1 <= limit <= 500 or offset < 0:
            raise RequestError("Notice page requires limit 1..500 and nonnegative offset")
        if state not in {None, "reported", "confirmed", "dismissed"}:
            raise RequestError("Unknown notice state")
        clauses, args = [], []
        if source_id:
            self.get(source_id)
            clauses.append("source_id=?")
            args.append(source_id)
        if state:
            clauses.append("json_extract(payload,'$.state')=?")
            args.append(state)
        with self.connection() as db:
            where = " WHERE " + " AND ".join(clauses) if clauses else ""
            rows = db.execute(
                "SELECT payload FROM notices" + where + " ORDER BY rowid DESC LIMIT ? OFFSET ?",
                [*args, limit, offset],
            )
            return [Notice.model_validate_json(row[0]) for row in rows]

    def notice_details(self, id_: str) -> dict:
        with self.connection() as db:
            notice = self._notice(db, id_)
            topics = list(notice.affected_topic_ids)
            claims = [
                dict(id=row[0], source_id=row[1], topic_id=row[2], statement=row[3])
                for row in db.execute(
                    "SELECT id, source_id, topic_id, json_extract(payload, '$.statement') "
                    "FROM claims WHERE (source_id=? OR id IN "
                    "(SELECT claim_id FROM claim_refs WHERE source_id=?))"
                    + (" AND topic_id IN " + _in(topics) if topics else "") + " ORDER BY id",
                    [notice.source_id, notice.source_id, *topics],
                )
            ]
            dependents = [
                row[0] for row in db.execute(
                    """SELECT source_id FROM links WHERE target_id=? AND kind='cites'
                       UNION SELECT target_id FROM links WHERE source_id=? AND kind='corroborates'""",
                    (notice.source_id, notice.source_id),
                )
            ]
        cards = self.source_cards({notice.source_id, *dependents})
        return dict(
            notice=notice.model_dump(), affected_claim_ids=[claim["id"] for claim in claims],
            dependent_source_ids=dependents, source=cards.get(notice.source_id),
            dependent_sources={dep: cards[dep] for dep in dependents if dep in cards},
            affected_claims=claims,
        )

    def review_notice(self, id_: str, review: NoticeReview) -> Notice:
        from kumaru.academic.models import Evidence

        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            notice = self._notice(db, id_)
            source = self._get(db, notice.source_id)
            if review.state == "confirmed" and notice.kind == "retraction" and review.impact != 1:
                raise RequestError("A confirmed retraction requires impact 1; dismiss an incorrect report instead")
            if not set(review.affected_topic_ids) <= set(source.topic_ids):
                raise RequestError("Notice impact topics must be tagged on the source")
            if (
                review.state == "confirmed" and review.impact and notice.kind != "retraction"
                and not review.affected_topic_ids
            ):
                raise RequestError("Factual impacts require specific affected topic IDs")
            notice = Notice.model_validate({
                **notice.model_dump(), "state": review.state, "impact": review.impact,
                "review_note": review.note, "affected_topic_ids": review.affected_topic_ids,
                "reviewed_at": now(),
            })
            evidence_id = f"notice:{id_}"
            findings = [item for item in source.evidence if item.id != evidence_id]
            if review.state == "confirmed" and review.impact:
                if len(findings) >= 500:
                    raise RequestError("Source evidence is full (500 findings); notice confirmation was not saved")
                findings.append(Evidence(
                    id=evidence_id, kind="retraction" if notice.kind == "retraction" else "factual_criticism",
                    value=review.impact, url=notice.url, note=review.note,
                    topic_ids=[] if notice.kind == "retraction" else review.affected_topic_ids,
                    reviewed=True,
                ))
            updates: dict = {"evidence": [item.model_dump() for item in findings]}
            if review.state == "confirmed":
                updates.update(
                    text=None, text_url=None, text_sha256=None, collected_at=None,
                    content_type=None, word_count=0, topic_word_counts={}, truncated=False,
                )
            self._save(db, Source.model_validate({**source.model_dump(), **updates}))
            db.execute("UPDATE notices SET payload=? WHERE id=?", (notice.model_dump_json(), id_))
            db.execute(
                "INSERT INTO reviews(source_id,recorded_at,payload) VALUES(?,?,?)",
                (source.id, now(), json.dumps({"notice_id": id_, **review.model_dump()})),
            )
        return notice

    def pending_notice_sources(self, ids: set[str]) -> set[str]:
        if not ids:
            return set()
        with self.connection() as db:
            rows = db.execute(
                "SELECT DISTINCT source_id FROM notices WHERE "
                "json_extract(payload,'$.state')='reported' AND source_id IN ("
                + ",".join("?" for _ in ids) + ")", sorted(ids),
            )
            return {row[0] for row in rows}

    def require_current_review(self, id_: str) -> None:
        source = self.get(id_)
        if any(item.kind == "retraction" and item.reviewed and item.value > 0 for item in source.evidence):
            raise RequestError("Retracted material is not eligible trusted text")
        with self.connection() as db:
            rows = db.execute("SELECT payload FROM notices WHERE source_id=?", (id_,))
            for row in rows:
                notice = Notice.model_validate_json(row[0])
                if notice.state == "reported" or (
                    notice.state == "confirmed" and notice.kind == "retraction" and notice.impact
                ):
                    raise RequestError("Editorial notices require review; retracted material is not trusted text")

    def review_summary(self, id_: str) -> dict:
        source = self.get(id_)
        with self.connection() as db:
            claim_count = db.execute("SELECT count(*) FROM claims WHERE source_id=?", (id_,)).fetchone()[0]
            notice_count = db.execute("SELECT count(*) FROM notices WHERE source_id=?", (id_,)).fetchone()[0]
        pending = id_ in self.pending_notice_sources({id_})
        withdrawn = any(
            item.kind == "retraction" and item.reviewed and item.value > 0 for item in source.evidence
        )
        return dict(
            claim_count=claim_count, notice_count=notice_count,
            pending_notices=pending, withdrawn=withdrawn, trusted_text_blocked=pending or withdrawn,
        )

    def record_check(self, id_: str, *, error: str | None = None, partial: bool = False) -> None:
        with self.connection() as db:
            self._get(db, id_)
            db.execute(
                "INSERT INTO update_checks VALUES(?,?,?,?) ON CONFLICT(source_id) DO UPDATE SET "
                "checked_at=excluded.checked_at,error=excluded.error,partial=excluded.partial",
                (id_, now(), error, int(partial)),
            )

    def check_status(self, id_: str) -> dict:
        with self.connection() as db:
            self._get(db, id_)
            row = db.execute("SELECT * FROM update_checks WHERE source_id=?", (id_,)).fetchone()
            return dict(row) if row else dict(source_id=id_, checked_at=None, error=None, partial=False)

    def due_sources(self, limit: int = 10) -> list[Source]:
        if not 1 <= limit <= 20:
            raise RequestError("Monitor batches are limited to 1..20 sources")
        with self.connection() as db:
            rows = db.execute(
                """SELECT s.payload FROM sources s LEFT JOIN update_checks c ON s.id=c.source_id
                   WHERE json_extract(s.payload,'$.doi') IS NOT NULL
                   AND json_extract(s.payload,'$.kind') != 'sources_of_sources'
                   ORDER BY c.checked_at, s.id LIMIT ?""", (limit,),
            )
            return [Source.model_validate_json(row[0]) for row in rows]

    def history(self, id_: str) -> list[dict]:
        self.get(id_)
        with self.connection() as db:
            return [
                {"recorded_at": row[0], "review": json.loads(row[1])}
                for row in db.execute(
                    "SELECT recorded_at,payload FROM reviews WHERE source_id=? ORDER BY id", (id_,),
                )
            ]

"""SQLite source graph with transactional updates and an append-only review trail."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import closing, contextmanager
from pathlib import Path

from kumaru.academic.models import (
    Access,
    Relationship,
    Review,
    Source,
    Topic,
    mathematics_topics,
    now,
)
from kumaru.academic.scoring import score_source
from kumaru.core.errors import ConfigError, RequestError


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
                        "publisher", "published_year",
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
            updates = review.model_dump(exclude_none=True, exclude={"evidence"})
            updates["evidence"] = [item.model_dump() for item in evidence.values()]
            if review.topic_ids is not None:
                updates["topic_reviewed"] = True
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
        allowed = {"title", "authors", "publisher", "doi", "access", "access_note", "content_type"}
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
        return [
            score_source(source, topic, independent_sources=[
                peer.id for peer in peers
                if peer.origin == "human_verified" and peer.topic_reviewed
                and topic in peer.topic_ids
                and peer.publisher and source.publisher
                and peer.publisher.strip().casefold() != source.publisher.strip().casefold()
                and peer.authors and source.authors
                and not (peer.doi and source.doi and peer.doi.casefold() == source.doi.casefold())
                and not {author.casefold() for author in peer.authors}
                & {author.casefold() for author in source.authors}
            ]).model_dump()
            for topic in source.topic_ids
        ]

    def history(self, id_: str) -> list[dict]:
        self.get(id_)
        with self.connection() as db:
            return [
                {"recorded_at": row[0], "review": json.loads(row[1])}
                for row in db.execute(
                    "SELECT recorded_at,payload FROM reviews WHERE source_id=? ORDER BY id", (id_,),
                )
            ]

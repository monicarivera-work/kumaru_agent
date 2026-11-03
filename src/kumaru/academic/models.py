"""Validated catalogue records. Discovery is not verification."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Literal
from urllib.parse import urlsplit, urlunsplit
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

TopicLevel = Literal["area", "subfield", "unit", "chapter", "subchapter"]
Access = Literal["unknown", "open", "account", "paywall", "inaccessible"]
Origin = Literal["unknown", "human_verified", "ai"]
EvidenceKind = Literal[
    "historical_acceptance", "credentials", "proof", "peer_review", "replication",
    "factual_criticism", "retraction", "corroboration", "origin", "permission",
]
RelationKind = Literal["contains", "cites", "corroborates", "criticizes", "related"]


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_url(value: str) -> str:
    try:
        parts = urlsplit(value.strip())
        port = parts.port
    except ValueError as exc:
        raise ValueError("Invalid URL") from exc
    if (
        parts.scheme not in {"https", "http"} or not parts.hostname
        or parts.username is not None or parts.password is not None
        or port not in {None, 80, 443}
    ):
        raise ValueError("Use a public HTTP(S) URL without credentials or nonstandard ports")
    host = parts.hostname.encode("idna").decode("ascii").lower()
    if ":" in host:
        host = f"[{host}]"
    default = 443 if parts.scheme == "https" else 80
    netloc = host if port in {None, default} else f"{host}:{port}"
    return urlunsplit((parts.scheme, netloc, parts.path or "/", parts.query, ""))


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class Topic(Record):
    id: str = Field(min_length=1, max_length=100, pattern=r"^[a-z0-9][a-z0-9._-]*$")
    name: str = Field(min_length=1, max_length=200)
    level: TopicLevel
    parent_id: str | None = None
    aliases: list[str] = Field(default_factory=list, max_length=30)


class Evidence(Record):
    id: str = Field(default_factory=lambda: uuid4().hex)
    kind: EvidenceKind
    value: float = Field(ge=0, le=1, allow_inf_nan=False)
    url: str
    note: str = Field(min_length=1, max_length=4000)
    # Empty means source-wide. Historical/proof/criticism claims require topics.
    topic_ids: list[str] = Field(default_factory=list, max_length=100)
    reviewed: bool = False
    recorded_at: str = Field(default_factory=now)

    _url = field_validator("url")(canonical_url)

    @model_validator(mode="after")
    def scope(self) -> Evidence:
        if self.kind in {"historical_acceptance", "proof", "factual_criticism"} and not self.topic_ids:
            raise ValueError(f"{self.kind} evidence must identify the specific topic(s)")
        return self


class Source(Record):
    id: str = ""
    url: str
    title: str = Field(min_length=1, max_length=2000)
    kind: Literal["article", "book", "study", "website", "sources_of_sources"] = "website"
    authors: list[str] = Field(default_factory=list, max_length=500)
    publisher: str | None = None
    doi: str | None = None
    published_year: int | None = Field(default=None, ge=1, le=9999)
    citation_count: int | None = Field(default=None, ge=0)
    reference_count: int | None = Field(default=None, ge=0)
    access: Access = "unknown"
    access_note: str = ""
    origin: Origin = "unknown"
    permission: Literal["unknown", "metadata_only", "text_permitted"] = "unknown"
    evidence: list[Evidence] = Field(default_factory=list, max_length=500)
    topic_ids: list[str] = Field(default_factory=list, max_length=100)
    # Assigned topics are candidates until reviewed against the material itself.
    topic_reviewed: bool = False
    metadata_url: str | None = None
    discovered_at: str = Field(default_factory=now)
    updated_at: str = Field(default_factory=now)
    text: str | None = Field(default=None, max_length=2_000_000)
    text_url: str | None = None
    text_sha256: str | None = None
    collected_at: str | None = None
    content_type: str | None = None
    word_count: int = Field(default=0, ge=0)
    topic_word_counts: dict[str, int] = Field(default_factory=dict)
    truncated: bool = False

    _url = field_validator("url", "metadata_url", "text_url")(
        lambda v: canonical_url(v) if v is not None else None
    )

    @field_validator("authors")
    @classmethod
    def authors_present(cls, value: list[str]) -> list[str]:
        return list(dict.fromkeys(" ".join(author.split()) for author in value if author.strip()))

    @model_validator(mode="after")
    def identity(self) -> Source:
        expected = hashlib.sha256(self.url.encode()).hexdigest()[:24]
        if self.id and self.id != expected:
            raise ValueError("Source ID must match its canonical URL")
        object.__setattr__(self, "id", expected)
        if self.text is not None and (
            self.origin != "human_verified" or self.permission != "text_permitted"
        ):
            raise ValueError("Text requires verified human provenance and permission")
        if any(count < 0 for count in self.topic_word_counts.values()):
            raise ValueError("Topic word counts cannot be negative")
        return self


class Relationship(Record):
    source_id: str
    target_id: str
    kind: RelationKind
    note: str = Field(min_length=1, max_length=4000)


class Review(Record):
    """An explicit human decision with linked, inspectable supporting evidence."""

    origin: Origin | None = None
    permission: Literal["unknown", "metadata_only", "text_permitted"] | None = None
    access: Access | None = None
    topic_ids: list[str] | None = Field(default=None, max_length=100)
    topic_reviewed: bool | None = None
    evidence: list[Evidence] = Field(default_factory=list, max_length=100)


class Score(Record):
    topic_id: str
    legitimacy: float | None
    usefulness: float | None
    evidence_coverage: float
    components: dict[str, float | None]
    evidence_ids: list[str]
    warnings: list[str]
    version: str = "academic-v1"


def mathematics_topics() -> list[Topic]:
    rows: list[tuple[str, str, TopicLevel, str | None, list[str]]] = [
        ("mathematics", "Mathematics", "area", None, ["math", "maths"]),
        ("algebra", "Algebra", "subfield", "mathematics", ["abstract algebra"]),
        ("analysis", "Analysis", "subfield", "mathematics", ["real analysis", "complex analysis"]),
        ("geometry", "Geometry", "subfield", "mathematics", ["euclidean geometry"]),
        ("number-theory", "Number theory", "subfield", "mathematics", []),
        ("topology", "Topology", "subfield", "mathematics", []),
        ("discrete-mathematics", "Discrete mathematics", "subfield", "mathematics", []),
        ("probability", "Probability", "subfield", "mathematics", []),
        ("statistics", "Statistics", "subfield", "mathematics", []),
        ("applied-mathematics", "Applied mathematics", "subfield", "mathematics", []),
        ("logic", "Mathematical logic", "subfield", "mathematics", ["set theory"]),
        ("linear-algebra", "Linear algebra", "unit", "algebra", []),
        ("groups", "Group theory", "unit", "algebra", []),
        ("calculus", "Calculus", "unit", "analysis", []),
        ("differentiation", "Differentiation", "chapter", "calculus", ["derivatives"]),
        ("integration", "Integration", "chapter", "calculus", ["integrals"]),
        ("chain-rule", "Chain rule", "subchapter", "differentiation", []),
        ("vector-spaces", "Vector spaces", "chapter", "linear-algebra", []),
        ("eigenvalues", "Eigenvalues and eigenvectors", "chapter", "linear-algebra", []),
    ]
    return [
        Topic(id=id_, name=name, level=level, parent_id=parent, aliases=aliases)
        for id_, name, level, parent, aliases in rows
    ]

"""Claim-scoped evidence and editorial notices; automated reports are not human verdicts."""

from __future__ import annotations

import hashlib
from typing import Literal
from uuid import uuid4

from pydantic import Field, field_validator, model_validator

from kumaru.academic.models import Evidence, Record, Source, canonical_url, now
from kumaru.academic.scoring import score_source

NoticeKind = Literal["retraction", "correction", "expression_of_concern", "update"]


class ClaimEvidence(Evidence):
    source_id: str
    stance: Literal["supports", "challenges", "context"]
    locator: str = Field(min_length=1, max_length=500)

    @model_validator(mode="after")
    def academic_finding(self) -> ClaimEvidence:
        if self.kind in {"origin", "permission"}:
            raise ValueError("Review authorship/permission on the source, not on a claim")
        if self.stance == "supports" and self.kind in {"factual_criticism", "retraction"}:
            raise ValueError("Criticism and retraction cannot be supporting evidence")
        if self.stance == "challenges" and self.kind not in {"factual_criticism", "retraction"}:
            raise ValueError("Challenges must identify factual criticism or retraction")
        return self


class ClaimCreate(Record):
    source_id: str
    topic_id: str
    statement: str = Field(min_length=1, max_length=8000)
    locator: str = Field(min_length=1, max_length=500)
    assumptions: list[str] = Field(default_factory=list, max_length=50)

    @field_validator("assumptions")
    @classmethod
    def bounded_assumptions(cls, values: list[str]) -> list[str]:
        if any(not value.strip() or len(value) > 2000 for value in values):
            raise ValueError("Each assumption must contain 1..2000 characters")
        return values


class Claim(ClaimCreate):
    id: str = Field(default_factory=lambda: uuid4().hex)
    reviewed: bool = False
    evidence: list[ClaimEvidence] = Field(default_factory=list, max_length=500)
    created_at: str = Field(default_factory=now)
    updated_at: str = Field(default_factory=now)


class ClaimReview(Record):
    reviewed: bool
    note: str = Field(min_length=1, max_length=4000)
    evidence: list[ClaimEvidence] = Field(default_factory=list, max_length=100)


class NoticeCreate(Record):
    source_id: str
    kind: NoticeKind
    url: str
    title: str = Field(min_length=1, max_length=2000)
    notice_doi: str | None = None
    event_date: str | None = None
    provider: Literal["crossref", "manual"] = "manual"
    metadata_url: str | None = None

    _url = field_validator("url", "metadata_url")(
        lambda value: canonical_url(value) if value is not None else None
    )


class Notice(NoticeCreate):
    id: str = ""
    state: Literal["reported", "confirmed", "dismissed"] = "reported"
    impact: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)
    affected_topic_ids: list[str] = Field(default_factory=list, max_length=100)
    review_note: str = ""
    discovered_at: str = Field(default_factory=now)
    last_seen_at: str = Field(default_factory=now)
    reviewed_at: str | None = None

    @model_validator(mode="after")
    def identity(self) -> Notice:
        identity = "\n".join([self.source_id, self.kind, self.url, self.event_date or ""])
        expected = hashlib.sha256(identity.encode()).hexdigest()[:24]
        if self.id and self.id != expected:
            raise ValueError("Notice ID must match its source, type, URL and event date")
        object.__setattr__(self, "id", expected)
        return self


class NoticeReview(Record):
    state: Literal["confirmed", "dismissed", "reported"]
    note: str = Field(min_length=1, max_length=4000)
    impact: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)
    affected_topic_ids: list[str] = Field(default_factory=list, max_length=100)

    @model_validator(mode="after")
    def confirmation(self) -> NoticeReview:
        if self.state == "confirmed" and self.impact is None:
            raise ValueError("Confirmation requires an explicit assessed impact from 0 to 1")
        return self


def claim_score(
    claim: Claim, source: Source, references: dict[str, Source], *, blocked_sources: set[str],
) -> dict:
    eligible: list[Evidence] = []
    warnings = []
    if source.id in blocked_sources:
        warnings.append("This source has an unresolved editorial notice; review it before relying on the claim.")
    for finding in claim.evidence:
        if not finding.reviewed or finding.stance == "context":
            continue
        reference = references[finding.source_id]
        if (
            reference.origin != "human_verified" or not reference.topic_reviewed
            or claim.topic_id not in reference.topic_ids
            or reference.kind == "sources_of_sources" or reference.id in blocked_sources
            or any(e.kind == "retraction" and e.reviewed and e.value > 0 for e in reference.evidence)
            or (reference.id != source.id and any(
                e.kind == "factual_criticism" and e.reviewed and e.value > 0
                and (not e.topic_ids or claim.topic_id in e.topic_ids)
                for e in reference.evidence
            ))
        ):
            warnings.append(f"Finding {finding.id}: referenced source is not eligible reviewed evidence.")
            continue
        # Only this claim's explicit findings establish proof, not a whole-book
        # reputation or another theorem's source-level evidence.
        eligible.append(Evidence.model_validate(finding.model_dump(exclude={"source_id", "stance", "locator"})))
    eligible.extend(
        item for item in source.evidence
        if item.reviewed and item.kind in {"retraction", "factual_criticism"}
        and (not item.topic_ids or claim.topic_id in item.topic_ids)
    )
    virtual = Source.model_validate({
        **source.model_dump(),
        "evidence": [],
        "topic_reviewed": source.topic_reviewed and claim.reviewed,
        "text": None, "word_count": 0, "topic_word_counts": {},
    })
    score = score_source(virtual, claim.topic_id, findings=eligible).model_dump()
    retracted = any(item.kind == "retraction" and item.value > 0 for item in eligible)
    criticized = any(item.kind == "factual_criticism" and item.value > 0 for item in eligible)
    if retracted:
        status = "withdrawn"
    elif not claim.reviewed or source.id in blocked_sources:
        status = "unreviewed"
        score["legitimacy"] = score["usefulness"] = None
    elif criticized:
        status = "disputed"
    elif score["legitimacy"] is None or not any(
        item.kind in {"proof", "replication", "historical_acceptance"} and item.value > 0
        for item in eligible
    ):
        status = "insufficient_evidence"
        warnings.append("Credentials or publication processes alone do not establish this assertion.")
    else:
        status = "evidence_supported"
    score["warnings"].extend(warnings)
    return dict(claim_id=claim.id, status=status, score=score, version="claim-v1")

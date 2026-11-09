"""Conservative, topic-scoped heuristics; missing evidence is not a negative finding."""

from __future__ import annotations

import math
from collections.abc import Sequence

from kumaru.academic.models import Evidence, Score, Source

WEIGHTS = {
    "historical_acceptance": 0.20,
    "credentials": 0.15,
    "proof": 0.30,
    "peer_review": 0.20,
    "replication": 0.15,
}
# Usefulness out of 100: legitimacy share plus capped, saturating add-ons.
USEFULNESS_POINTS = {
    "legitimacy": 65,
    "citations": 15,
    "references": 5,
    "independent_sources": 13,
    "content_length": 2,
}


def score_source(
    source: Source, topic_id: str, *, independent_sources: Sequence[str] = (),
    findings: Sequence[Evidence] | None = None,
) -> Score:
    evidence = [
        item for item in (source.evidence if findings is None else findings)
        if item.reviewed and (not item.topic_ids or topic_id in item.topic_ids)
    ]
    components: dict[str, float | None] = {}
    for kind in WEIGHTS:
        claims: dict[tuple[str, tuple[str, ...]], float] = {}
        for item in evidence:
            if item.kind == kind:
                key = (item.url, tuple(sorted(item.topic_ids)))
                claims[key] = min(item.value, claims.get(key, item.value))
        values = list(claims.values())
        # Same-provenance copies get one conservative assessment, not votes.
        components[kind] = sum(values) / len(values) if values else None
    coverage = sum(WEIGHTS[kind] for kind, value in components.items() if value is not None)
    warnings = []
    legitimacy = None
    usefulness = None
    if topic_id not in source.topic_ids:
        warnings.append("Source has not been tagged with this topic.")
    elif not source.topic_reviewed:
        warnings.append("Topic coverage is a discovery candidate, not reviewed coverage.")
    elif source.origin != "human_verified":
        warnings.append("Human authorship has not been verified; excluded from trusted text.")
    elif source.kind == "sources_of_sources":
        warnings.append("A container's reputation does not establish the accuracy of its contents.")
    elif coverage:
        # Unknown dimensions contribute no credit, rather than normalizing one
        # credential into a misleading 100/100. Coverage is exposed separately.
        legitimacy = 100 * sum(
            WEIGHTS[kind] * value for kind, value in components.items() if value is not None
        )
        criticism = max(
            (item.value for item in evidence if item.kind == "factual_criticism"), default=0,
        )
        retraction = max(
            (item.value for item in evidence if item.kind == "retraction"), default=0,
        )
        legitimacy *= 1 - 0.8 * criticism
        if retraction:
            legitimacy = min(legitimacy, 10 * (1 - retraction))
            warnings.append("Retraction evidence limits legitimacy.")
        components["factual_criticism"] = criticism
        components["retraction"] = retraction
        if coverage < 0.6:
            warnings.append("Sparse evidence: low coverage limits the score.")
        cited = min(1.0, math.log1p(source.citation_count or 0) / math.log1p(1000))
        references = min(1.0, math.log1p(source.reference_count or 0) / math.log1p(100))
        independent = min(1.0, len(set(independent_sources) - {source.id}) / 5)
        # Length gets at most two points, and is zero without reviewed topic text.
        topic_words = source.topic_word_counts.get(topic_id, 0)
        length = min(1.0, math.log1p(topic_words) / math.log1p(5000)) if source.text else 0
        points = USEFULNESS_POINTS
        usefulness = (
            points["legitimacy"] / 100 * legitimacy + points["citations"] * cited
            + points["references"] * references + points["independent_sources"] * independent
            + points["content_length"] * length
        )
        if retraction:
            usefulness = min(usefulness, 10)
        components.update(
            citations=cited, references=references, independent_sources=independent,
            content_length=length,
        )
        warnings.append("Citation/reference counts are provider reports, not proof of correctness.")
    else:
        warnings.append("No reviewed academic evidence; scores are unknown.")
    return Score(
        topic_id=topic_id,
        legitimacy=round(legitimacy, 2) if legitimacy is not None else None,
        usefulness=round(usefulness, 2) if usefulness is not None else None,
        evidence_coverage=round(coverage, 2),
        components=components,
        evidence_ids=[item.id for item in evidence],
        warnings=warnings,
    )

"""Claim isolation, editorial-update provenance, review gates and scheduler lifecycle."""

from __future__ import annotations

import asyncio
import json
from threading import BoundedSemaphore

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from typer.testing import CliRunner

from kumaru.academic.catalog import Catalog
from kumaru.academic.models import Evidence, Relationship, Review, Source
from kumaru.academic.monitor import UpdateMonitor, check_batch, check_updates
from kumaru.academic.reliability import (
    ClaimCreate,
    ClaimEvidence,
    ClaimReview,
    Notice,
    NoticeCreate,
    NoticeReview,
)
from kumaru.academic.web import FetchError, PublicWeb, collect_source
from kumaru.cli import app
from kumaru.core.config import AcademicConfig
from kumaru.core.errors import RequestError
from kumaru.server.app import create_app


@pytest.fixture
def catalog(tmp_path):
    return Catalog(tmp_path / "catalog.sqlite")


def add_source(catalog, path="paper", doi="10.1234/paper"):
    source = catalog.discover(Source(
        url=f"https://example.org/{path}", title=path, topic_ids=["calculus"],
        authors=["Scholar"], publisher="University", doi=doi,
    ))
    catalog.review(source.id, Review(
        origin="human_verified", permission="text_permitted", topic_ids=["calculus"],
        evidence=[
            Evidence(
                kind=kind, value=1, url=source.url, note="Checked human evidence", reviewed=True,
                topic_ids=["calculus"] if kind == "proof" else [],
            )
            for kind in ("origin", "permission", "proof")
        ],
    ))
    return catalog.get(source.id)


def add_claim(catalog, source):
    return catalog.add_claim(ClaimCreate(
        source_id=source.id, topic_id="calculus", statement="Derivative of x squared is 2x.",
        locator="Chapter 2, theorem 1", assumptions=["x is a real number"],
    ))


def finding(source, **updates):
    values = dict(
        kind="proof", value=1, url=source.url, note="Checked the actual proof and assumptions",
        topic_ids=["calculus"], reviewed=True, source_id=source.id,
        stance="supports", locator="Proof of theorem 1",
    )
    return ClaimEvidence.model_validate({**values, **updates})


def notice(catalog, source, kind="correction"):
    return catalog.report_notice(NoticeCreate(
        source_id=source.id, kind=kind, url="https://example.org/editorial",
        title="Editorial update", event_date="2025-01-01",
    ))


class EditorialWeb(PublicWeb):
    def __init__(self, payload):
        super().__init__()
        self.payload = payload
        self.urls = []

    def json(self, url, *, headers=None):
        self.urls.append(url)
        return self.payload


def editorial_payload(kind="retraction", target="10.1234/paper"):
    return {
        "status": "ok", "message": {
            "total-results": 1,
            "items": [{
                "DOI": "10.1234/editorial", "title": ["Editorial notice"],
                "update-to": [{
                    "DOI": target, "type": kind, "updated": {"date-parts": [[2025, 1, 2]]},
                }],
            }],
        },
    }


def test_claim_does_not_inherit_source_proof(catalog):
    source = add_source(catalog)
    claim = add_claim(catalog, source)
    catalog.review_claim(claim.id, ClaimReview(reviewed=True, note="Checked attribution"))
    result = catalog.claim_details(claim.id)
    assert catalog.scores(source.id)[0]["legitimacy"] == 30
    assert result["assessment"]["score"]["legitimacy"] is None
    assert result["assessment"]["status"] == "insufficient_evidence"
    assert result["claim"]["assumptions"] == ["x is a real number"]


@pytest.mark.parametrize("kind", ["website", "sources_of_sources"])
def test_corroboration_excludes_containers_pending_and_confirmed_flaws(catalog, kind):
    source = add_source(catalog)
    peer = catalog.discover(Source(
        url="https://example.org/independent", title="Independent reference",
        topic_ids=["calculus"], authors=["Independent scholar"],
        publisher="Another university", kind=kind,
    ))
    catalog.review(peer.id, Review(
        origin="human_verified", topic_ids=["calculus"],
        evidence=[Evidence(
            kind="origin", value=1, url=peer.url, note="Checked human origin", reviewed=True,
        )],
    ))
    catalog.link(Relationship(
        source_id=peer.id, target_id=source.id, kind="corroborates", note="Reviewed corroborating relation",
    ))
    expected = 0.2 if kind == "website" else 0
    assert catalog.scores(source.id)[0]["components"]["independent_sources"] == expected
    editorial = notice(catalog, peer)
    assert catalog.scores(source.id)[0]["components"]["independent_sources"] == 0
    catalog.review_notice(editorial.id, NoticeReview(state="dismissed", note="Wrong report"))
    assert catalog.scores(source.id)[0]["components"]["independent_sources"] == expected
    catalog.review_notice(editorial.id, NoticeReview(
        state="confirmed", impact=0.5, affected_topic_ids=["calculus"], note="Checked factual error",
    ))
    assert catalog.scores(source.id)[0]["components"]["independent_sources"] == 0


def test_evidence_limits_are_transactional_and_assessment_handles_combined_findings(catalog):
    source = add_source(catalog)
    criticisms = [
        Evidence(
            id=f"criticism-{i}", kind="factual_criticism", value=0.1,
            url=f"https://example.org/criticism/{i}", note="Checked factual flaw",
            reviewed=True, topic_ids=["calculus"],
        )
        for i in range(497)
    ]
    for offset in range(0, len(criticisms), 100):
        catalog.review(source.id, Review(evidence=criticisms[offset:offset + 100]))
    assert len(catalog.get(source.id).evidence) == 500
    with pytest.raises(RequestError, match="at most 500"):
        catalog.review(source.id, Review(evidence=[criticisms[0].model_copy(update={"id": "overflow"})]))
    assert len(catalog.get(source.id).evidence) == 500
    editorial = notice(catalog, source)
    with pytest.raises(RequestError, match="evidence is full"):
        catalog.review_notice(editorial.id, NoticeReview(
            state="confirmed", impact=0.5, affected_topic_ids=["calculus"], note="Checked correction",
        ))
    assert catalog.notice_details(editorial.id)["notice"]["state"] == "reported"
    catalog.review_notice(editorial.id, NoticeReview(state="dismissed", note="Duplicate report"))
    claim = add_claim(catalog, source)
    proofs = [finding(source, id=f"proof-{i}") for i in range(500)]
    for offset in range(0, len(proofs), 100):
        catalog.review_claim(claim.id, ClaimReview(
            reviewed=True, note="Checked assertion", evidence=proofs[offset:offset + 100],
        ))
    with pytest.raises(RequestError, match="at most 500"):
        catalog.review_claim(claim.id, ClaimReview(
            reviewed=True, note="Extra finding", evidence=[finding(source, id="overflow")],
        ))
    result = catalog.claim_details(claim.id)
    assert result["assessment"]["status"] == "disputed"
    assert len(result["assessment"]["score"]["evidence_ids"]) == 997


def test_claim_findings_are_scoped_and_audited(catalog):
    source = add_source(catalog)
    a = add_claim(catalog, source)
    b = add_claim(catalog, source)
    proof = finding(source)
    catalog.review_claim(a.id, ClaimReview(
        reviewed=True, note="Checked the assertion and proof", evidence=[proof],
    ))
    assert catalog.claim_details(a.id)["assessment"]["score"]["legitimacy"] == 30
    assert catalog.claim_details(b.id)["assessment"]["score"]["legitimacy"] is None
    catalog.review_claim(a.id, ClaimReview(
        reviewed=True, note="Withdrawing a finding", evidence=[
            ClaimEvidence.model_validate({**proof.model_dump(), "reviewed": False}),
        ],
    ))
    assert catalog.claim_details(a.id)["assessment"]["score"]["legitimacy"] is None
    assert len(catalog.claim_details(a.id)["reviews"]) == 2
    assert len(Catalog(catalog.path).claims(source_id=source.id)) == 2


def test_claim_evidence_requires_eligible_referenced_sources(catalog):
    source = add_source(catalog)
    peer = catalog.discover(Source(
        url="https://example.org/unreviewed", title="Unverified", topic_ids=["calculus"],
    ))
    claim = add_claim(catalog, source)
    catalog.review_claim(claim.id, ClaimReview(
        reviewed=True, note="Candidate reference", evidence=[finding(peer)],
    ))
    assert catalog.claim_details(claim.id)["assessment"]["score"]["legitimacy"] is None
    with pytest.raises(RequestError):
        catalog.review_claim(claim.id, ClaimReview(
            reviewed=True, note="Wrong subject", evidence=[
                finding(source, topic_ids=["algebra"]),
            ],
        ))
    with pytest.raises(RequestError):
        catalog.review_claim(claim.id, ClaimReview(
            reviewed=True, note="Missing reference", evidence=[finding(source, source_id="unknown")],
        ))
    with pytest.raises(ValidationError):
        finding(source, stance="challenges")


def test_challenging_findings_lower_only_target_claim(catalog):
    source = add_source(catalog)
    claim = add_claim(catalog, source)
    catalog.review_claim(claim.id, ClaimReview(
        reviewed=True, note="Specific contradiction", evidence=[
            finding(source),
            finding(source, kind="factual_criticism", stance="challenges", value=0.5),
        ],
    ))
    result = catalog.claim_details(claim.id)["assessment"]
    assert result["status"] == "disputed" and result["score"]["legitimacy"] == 18
    assert catalog.scores(source.id)[0]["legitimacy"] == 30


@pytest.mark.parametrize(("kind", "value"), [("credentials", 1), ("peer_review", 1), ("proof", 0)])
def test_credentials_or_zero_proof_are_not_claim_support(catalog, kind, value):
    source = add_source(catalog)
    claim = add_claim(catalog, source)
    catalog.review_claim(claim.id, ClaimReview(
        reviewed=True, note="Process evidence only", evidence=[finding(source, kind=kind, value=value)],
    ))
    assert catalog.claim_details(claim.id)["assessment"]["status"] == "insufficient_evidence"


def test_confirmed_reference_flaws_invalidate_its_claim_support(catalog):
    source = add_source(catalog)
    peer = add_source(catalog, "peer", "10.1234/peer")
    claim = add_claim(catalog, source)
    catalog.review_claim(claim.id, ClaimReview(
        reviewed=True, note="Independent proof", evidence=[finding(peer)],
    ))
    assert catalog.claim_details(claim.id)["assessment"]["score"]["legitimacy"] == 30
    item = notice(catalog, peer)
    catalog.review_notice(item.id, NoticeReview(
        state="confirmed", impact=0.5, affected_topic_ids=["calculus"], note="Flaw in that proof",
    ))
    assert catalog.claim_details(claim.id)["assessment"]["score"]["legitimacy"] is None


def test_editorial_check_uses_reverse_update_filter_and_preserves_human_reviews(catalog):
    source = add_source(catalog)
    web = EditorialWeb(editorial_payload())
    result = check_updates(catalog, source.id, web)
    assert "filter=updates%3A10.1234%2Fpaper" in web.urls[0]
    assert not result["partial"]
    item = catalog.notices()[0]
    assert item.kind == "retraction" and item.state == "reported"
    assert item.event_date == "2025-01-02"
    assert item.metadata_url == web.urls[0]
    assert catalog.scores(source.id)[0]["legitimacy"] == 30
    catalog.review_notice(item.id, NoticeReview(state="dismissed", note="Wrong identity"))
    check_updates(catalog, source.id, web)
    assert len(catalog.notices()) == 1 and catalog.notices()[0].state == "dismissed"
    assert catalog.check_status(source.id)["error"] is None


@pytest.mark.parametrize(("kind", "expected"), [
    ("erratum", "correction"), ("expression-of-concern", "expression_of_concern"),
    ("withdrawal", "retraction"), ("clarification", "update"),
])
def test_editorial_update_types(catalog, kind, expected):
    source = add_source(catalog)
    check_updates(catalog, source.id, EditorialWeb(editorial_payload(kind)))
    assert catalog.notices()[0].kind == expected


def test_monitor_errors_are_not_clean_scans(catalog):
    source = add_source(catalog)
    for payload in ({}, editorial_payload(target="10.9999/another")):
        with pytest.raises((FetchError, RequestError)):
            check_updates(catalog, source.id, EditorialWeb(payload))
        assert catalog.check_status(source.id)["error"]
        assert not catalog.notices()
    no_results = {"status": "ok", "message": {"total-results": 0, "items": []}}
    result = check_updates(catalog, source.id, EditorialWeb(no_results))
    assert "does not prove" in result["warning"]
    assert catalog.check_status(source.id)["error"] is None
    payload = editorial_payload()
    payload["message"]["total-results"] = 150
    assert check_updates(catalog, source.id, EditorialWeb(payload))["partial"]
    assert catalog.check_status(source.id)["partial"] == 1


def test_pending_notices_quarantine_text_and_affected_claims(catalog):
    source = add_source(catalog)
    claim = add_claim(catalog, source)
    catalog.review_claim(claim.id, ClaimReview(
        reviewed=True, note="Actual proof", evidence=[finding(source)],
    ))
    item = notice(catalog, source, "retraction")
    assert claim.id in catalog.notice_details(item.id)["affected_claim_ids"]
    assert catalog.claim_details(claim.id)["assessment"]["score"]["legitimacy"] is None
    with pytest.raises(RequestError):
        collect_source(catalog, source.id, EditorialWeb({}))
    catalog.review_notice(item.id, NoticeReview(state="dismissed", note="False report"))
    assert catalog.claim_details(claim.id)["assessment"]["score"]["legitimacy"] == 30


def test_confirming_correction_requires_topic_and_clears_cached_text(catalog):
    source = add_source(catalog)
    catalog.save_collection(source.id, dict(
        text="Reviewed old text", word_count=3, topic_word_counts={"calculus": 3},
    ))
    item = notice(catalog, source)
    with pytest.raises(RequestError):
        catalog.review_notice(item.id, NoticeReview(
            state="confirmed", impact=0.5, note="Identified factual flaw",
        ))
    assert catalog.get(source.id).text
    catalog.review_notice(item.id, NoticeReview(
        state="confirmed", impact=0.5, affected_topic_ids=["calculus"], note="Identified factual flaw",
    ))
    assert catalog.get(source.id).text is None
    assert catalog.scores(source.id)[0]["legitimacy"] == 18
    catalog.review_notice(item.id, NoticeReview(state="dismissed", note="Evidence refuted"))
    assert catalog.scores(source.id)[0]["legitimacy"] == 30
    assert len(catalog.history(source.id)) == 3


def test_retraction_confirmation_has_no_zero_impact_escape(catalog):
    source = add_source(catalog)
    claim = add_claim(catalog, source)
    item = notice(catalog, source, "retraction")
    with pytest.raises(RequestError):
        catalog.review_notice(item.id, NoticeReview(
            state="confirmed", impact=0, note="Invalid confirmation",
        ))
    catalog.review_notice(item.id, NoticeReview(state="confirmed", impact=1, note="Verified withdrawal"))
    assert catalog.scores(source.id)[0]["legitimacy"] == 0
    assert catalog.claim_details(claim.id)["assessment"]["status"] == "withdrawn"
    with pytest.raises(RequestError):
        catalog.require_current_review(source.id)


def test_notice_impact_follows_evidence_refs_not_container_membership(catalog):
    source = add_source(catalog)
    peer = add_source(catalog, "peer", "10.1234/peer")
    claim = add_claim(catalog, peer)
    catalog.review_claim(claim.id, ClaimReview(
        reviewed=True, note="Reference", evidence=[finding(source)],
    ))
    catalog.link(Relationship(
        source_id=peer.id, target_id=source.id, kind="cites", note="Cited this work",
    ))
    item = notice(catalog, source)
    result = catalog.notice_details(item.id)
    assert claim.id in result["affected_claim_ids"]
    assert peer.id in result["dependent_source_ids"]
    assert catalog.claim_details(claim.id)["assessment"]["score"]["legitimacy"] is None


def test_no_pagination_limit_hides_pending_notices(catalog):
    source = add_source(catalog)
    notice(catalog, source)
    with catalog.connection() as db:
        for number in range(510):
            entry = Notice(
                source_id=source.id, kind="correction", title="Older reviewed report",
                url=f"https://example.org/notices/{number}", state="dismissed",
            )
            db.execute(
                "INSERT INTO notices VALUES(?,?,?)", (entry.id, source.id, entry.model_dump_json()),
            )
    assert source.id in catalog.pending_notice_sources({source.id})
    with pytest.raises(RequestError):
        catalog.require_current_review(source.id)


def test_due_batch_prefers_unchecked_sources_and_reports_failures(catalog, monkeypatch):
    from kumaru.academic import monitor

    first = add_source(catalog)
    second = add_source(catalog, "second", "10.1234/second")
    catalog.record_check(first.id)
    assert catalog.due_sources(1)[0].id == second.id
    monkeypatch.setattr(monitor, "check_updates", lambda *args: (_ for _ in ()).throw(FetchError("offline")))
    result = check_batch(catalog, 2)
    assert not result["results"] and len(result["errors"]) == 2


@pytest.mark.asyncio
async def test_periodic_monitor_starts_stops_and_releases_admission(catalog, monkeypatch):
    from kumaru.academic import monitor

    calls = []
    monkeypatch.setattr(monitor, "check_batch", lambda *args: calls.append(1) or {"results": [], "errors": []})
    semaphore = BoundedSemaphore(2)
    worker = UpdateMonitor(catalog, semaphore, interval=0, batch_size=1)
    worker.start()
    await asyncio.sleep(0.03)
    await worker.close()
    assert calls and worker.task.done() and worker.stop.is_set()
    assert semaphore.acquire(blocking=False) and semaphore.acquire(blocking=False)
    semaphore.release()
    semaphore.release()


def test_claim_and_notice_api_and_dashboard_gate(tmp_path, echo_config):
    echo_config.academic = AcademicConfig(enabled=True, database=str(tmp_path / "api.sqlite"))
    application = create_app(echo_config)
    source = add_source(application.state.academic)
    with TestClient(application) as client:
        body = ClaimCreate(
            source_id=source.id, topic_id="calculus", statement="A theorem", locator="Theorem 1",
        )
        response = client.post("/v1/academic/claims", json=body.model_dump())
        assert response.status_code == 201
        id_ = response.json()["claim"]["id"]
        review = ClaimReview(reviewed=True, note="Verified", evidence=[finding(source)])
        assert client.post(
            f"/v1/academic/claims/{id_}/review", json=review.model_dump(),
        ).json()["assessment"]["score"]["legitimacy"] == 30
        assert len(client.get("/v1/academic/claims").json()) == 1
        assert client.get("/v1/academic/monitor/status").json()["periodic_enabled"] is False
        report = client.post("/v1/academic/notices", json=NoticeCreate(
            source_id=source.id, kind="correction", url="https://example.org/notice", title="Correction",
        ).model_dump())
        assert report.status_code == 201
        notice_id = report.json()["notice"]["id"]
        assert client.get(f"/v1/academic/sources/{source.id}/text").status_code == 400
        assert client.post(f"/v1/academic/notices/{notice_id}/review", json={
            "state": "dismissed", "note": "Reviewed and dismissed",
        }).status_code == 200
        echo_config.server.api_key = "test-only"
        assert client.get("/v1/academic/claims").status_code == 401
        assert client.get("/v1/academic/notices").status_code == 401


def test_cli_claim_and_notice_workflows(tmp_path, catalog):
    source = add_source(catalog)
    path = tmp_path / "claim.json"
    path.write_text(ClaimCreate(
        source_id=source.id, topic_id="calculus", statement="Theorem", locator="Page 1",
    ).model_dump_json())
    runner = CliRunner()
    prefix = ["academic", "-d", str(catalog.path)]
    result = runner.invoke(app, [*prefix, "claim-add", str(path)])
    assert result.exit_code == 0
    claim_id = json.loads(result.output)["claim"]["id"]
    assert runner.invoke(app, [*prefix, "claim-show", claim_id]).exit_code == 0
    assert runner.invoke(app, [*prefix, "notices", "--state", "reported"]).exit_code == 0


def test_explicit_topic_review_revocation_clears_text(catalog):
    source = add_source(catalog)
    catalog.save_collection(source.id, dict(text="Reviewed text", word_count=2))
    result = catalog.review(source.id, Review(topic_reviewed=False))
    assert not result.topic_reviewed and result.text is None
    assert catalog.scores(source.id)[0]["legitimacy"] is None
    catalog.review(source.id, Review(topic_reviewed=True))
    assert catalog.scores(source.id)[0]["legitimacy"] == 30


def test_manually_reviewed_retraction_also_blocks_trusted_text(catalog):
    source = add_source(catalog)
    catalog.review(source.id, Review(evidence=[Evidence(
        kind="retraction", value=1, url=source.url, note="Reviewed withdrawal", reviewed=True,
    )]))
    with pytest.raises(RequestError):
        catalog.require_current_review(source.id)


def test_dashboard_is_optional_and_linked_from_chat(tmp_path, echo_config):
    echo_config.academic = AcademicConfig(enabled=True, database=str(tmp_path / "ui.sqlite"))
    with TestClient(create_app(echo_config)) as client:
        assert client.get("/academic").status_code == 200
        assert "Academic review" in client.get("/academic").text
        assert client.get("/static/academic.js").status_code == 200
        assert client.get("/static/academic.css").headers["content-type"].startswith("text/css")
        assert client.get("/v1/info").json()["academic_enabled"] is True
        assert 'id="academic-link"' in client.get("/").text
    echo_config.academic.enabled = False
    with TestClient(create_app(echo_config)) as client:
        assert client.get("/academic").status_code == 404
        assert client.get("/v1/info").json()["academic_enabled"] is False


def test_schema_upgrade_preserves_sources_and_reviews(catalog):
    source = add_source(catalog)
    with catalog.connection() as db:
        for table in ("claim_refs", "claim_reviews", "claims", "notices", "update_checks"):
            db.execute(f"DROP TABLE {table}")
    upgraded = Catalog(catalog.path)
    assert upgraded.get(source.id).origin == "human_verified"
    assert upgraded.history(source.id)[0]["review"]["origin"] == "human_verified"
    assert add_claim(upgraded, source).id
    assert not upgraded.notices()

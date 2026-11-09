"""Dashboard read models, transient-failure handling and the review fixes behind them."""

from __future__ import annotations

import itertools
import math
import mimetypes
import socket
from contextlib import contextmanager
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from kumaru.academic.catalog import Catalog
from kumaru.academic.discovery import discover
from kumaru.academic.models import Evidence, Relationship, Review, Source
from kumaru.academic.reliability import (
    ClaimCreate,
    ClaimEvidence,
    ClaimReview,
    NoticeCreate,
    NoticeReview,
)
from kumaru.academic.scoring import USEFULNESS_POINTS, WEIGHTS, score_source
from kumaru.academic.web import (
    AccessBlocked,
    FetchError,
    Page,
    PublicWeb,
    TransientFetchError,
    collect_source,
    inspect_source,
    public_addresses,
)
from kumaru.core.config import AcademicConfig
from kumaru.core.errors import RequestError
from kumaru.server import routes_academic
from kumaru.server.app import create_app

UI_DIR = Path(__file__).resolve().parents[1] / "ui"
LIBRARY_ITEM_KEYS = {
    "id", "url", "title", "kind", "access", "access_note", "origin", "permission", "topic_ids",
    "topic_reviewed", "doi", "publisher", "published_year", "citation_count", "reference_count",
    "authors", "author_count", "discovered_at", "updated_at", "collected_at", "word_count",
    "evidence_count", "claim_count", "child_count", "pending_notices", "withdrawn", "score",
    "free_copies", "is_free_copy", "open_access",
}


@pytest.fixture
def catalog(tmp_path: Path) -> Catalog:
    return Catalog(tmp_path / "gui.sqlite")


def add(catalog: Catalog, name: str, **fields) -> Source:
    values = dict(url=f"https://example.org/{name}", title=name, topic_ids=["calculus"])
    values.update(fields)
    return catalog.discover(Source.model_validate(values))


def proof(topic: str = "calculus", value: float = 1, name: str = "proof") -> Evidence:
    return Evidence(
        kind="proof", value=value, url=f"https://example.org/{name}", note="Checked the proof",
        topic_ids=[topic], reviewed=True,
    )


def verify(catalog: Catalog, source: Source, *findings: Evidence, topics=("calculus",)) -> Source:
    return catalog.review(source.id, Review(
        origin="human_verified", permission="text_permitted", topic_ids=list(topics),
        evidence=[
            Evidence(kind=kind, value=1, url=source.url, note=f"Checked {kind}", reviewed=True)
            for kind in ("origin", "permission")
        ] + list(findings),
    ))


def contain(catalog: Catalog, parent: Source, child: Source) -> None:
    catalog.link(Relationship(
        source_id=parent.id, target_id=child.id, kind="contains", note="Indexed here",
    ))


def client_for(tmp_path: Path, echo_config, **academic) -> TestClient:
    echo_config.academic = AcademicConfig(
        enabled=True, database=str(tmp_path / "api.sqlite"), **academic,
    )
    return TestClient(create_app(echo_config))


# --- overview -------------------------------------------------------------------------


def test_empty_overview_has_every_key_at_zero(catalog):
    result = catalog.overview()
    assert result["topic_id"] is None and result["total"] == 0
    assert result["by_access"] == dict.fromkeys(
        ("unknown", "open", "account", "paywall", "inaccessible"), 0,
    )
    assert result["by_kind"] == dict.fromkeys(
        ("article", "book", "study", "website", "sources_of_sources"), 0,
    )
    assert result["by_origin"] == {"unknown": 0, "human_verified": 0, "ai": 0}
    for key in ("needs_review", "pending_notices", "claims", "unreviewed_claims"):
        assert result[key] == 0
    assert set(result["topic_counts"]) == {topic.id for topic in catalog.topics()}
    assert set(result["topic_counts"].values()) == {0}
    assert result["topic_totals"] == result["topic_counts"]


def test_overview_formula_matches_scoring_module(catalog):
    formula = catalog.overview()["formula"]
    assert formula == {
        "version": "academic-v1", "legitimacy_weights": WEIGHTS,
        "usefulness_points": USEFULNESS_POINTS,
    }
    assert sum(USEFULNESS_POINTS.values()) == 100
    assert math.isclose(sum(WEIGHTS.values()), 1)
    # The returned mappings are copies: callers cannot change the live formula.
    formula["usefulness_points"]["citations"] = 0
    assert USEFULNESS_POINTS["citations"] == 15


def test_overview_counts_are_scoped_to_a_topic_subtree(catalog):
    calculus = verify(catalog, add(catalog, "calculus-open", access="open", citation_count=3), proof())
    pending = add(catalog, "differentiation", topic_ids=["differentiation"], kind="book")
    container = add(
        catalog, "library", topic_ids=["linear-algebra"], kind="sources_of_sources",
        access="paywall",
    )
    ai = add(catalog, "ai-page", topic_ids=["integration"], access="account")
    catalog.review(ai.id, Review(origin="ai", topic_ids=["integration"]))

    def record(source: Source, statement: str, reviewed: bool = False) -> None:
        claim = catalog.add_claim(ClaimCreate(
            source_id=source.id, topic_id=source.topic_ids[0], statement=statement, locator="p. 1",
        ))
        if reviewed:
            catalog.review_claim(claim.id, ClaimReview(reviewed=True, note="Checked attribution"))

    record(calculus, "Reviewed", reviewed=True)
    record(calculus, "Unreviewed")
    record(pending, "Also unreviewed")
    for source, name in ((pending, "n1"), (container, "n2")):
        catalog.report_notice(NoticeCreate(
            source_id=source.id, kind="correction", url=f"https://example.org/{name}", title=name,
        ))
    dismissed = catalog.report_notice(NoticeCreate(
        source_id=calculus.id, kind="update", url="https://example.org/n3", title="Done",
    ))
    catalog.review_notice(dismissed.id, NoticeReview(state="dismissed", note="Not applicable"))

    everything = catalog.overview()
    assert everything["total"] == 4
    assert everything["by_kind"]["sources_of_sources"] == 1 and everything["by_kind"]["book"] == 1
    assert everything["by_access"] == {
        "unknown": 1, "open": 1, "account": 1, "paywall": 1, "inaccessible": 0,
    }
    assert everything["by_origin"] == {"unknown": 2, "human_verified": 1, "ai": 1}
    # Unknown origin or unreviewed coverage; AI-disclosed with reviewed topics is not queued.
    assert everything["needs_review"] == 2
    assert everything["pending_notices"] == 2
    assert everything["claims"] == 3 and everything["unreviewed_claims"] == 2
    direct = {"calculus": 1, "differentiation": 1, "linear-algebra": 1, "integration": 1}
    assert {key: value for key, value in everything["topic_counts"].items() if value} == direct
    assert everything["topic_counts"]["analysis"] == 0

    scoped = catalog.overview("analysis")
    assert scoped["topic_id"] == "analysis" and scoped["total"] == 3
    assert scoped["by_kind"]["sources_of_sources"] == 0
    assert scoped["pending_notices"] == 1
    assert scoped["claims"] == 3 and scoped["unreviewed_claims"] == 2
    # Direct tags cover the whole catalogue, whatever scope was requested.
    assert scoped["topic_counts"] == everything["topic_counts"]

    narrow = catalog.overview("differentiation")
    assert narrow["total"] == 1 and narrow["claims"] == 1 and narrow["unreviewed_claims"] == 1
    assert narrow["needs_review"] == 1 and narrow["pending_notices"] == 1
    empty = catalog.overview("topology")
    assert empty["total"] == 0 and empty["claims"] == 0 and empty["by_access"]["open"] == 0
    with pytest.raises(RequestError):
        catalog.overview("missing-topic")


def test_topic_totals_count_a_multi_tagged_source_once(catalog):
    add(catalog, "double", topic_ids=["calculus", "differentiation"])
    add(catalog, "single", topic_ids=["integration"])
    add(catalog, "elsewhere", topic_ids=["groups"])
    result = catalog.overview()
    # Direct tags are per tag; rolled-up totals are distinct sources per subtree.
    assert result["topic_counts"]["calculus"] == 1 and result["topic_counts"]["differentiation"] == 1
    assert result["topic_totals"]["calculus"] == 2
    assert result["topic_totals"]["differentiation"] == 1 and result["topic_totals"]["chain-rule"] == 0
    assert result["topic_totals"]["analysis"] == 2
    assert result["topic_totals"]["algebra"] == 1
    assert result["topic_totals"]["mathematics"] == 3
    assert result["topic_totals"]["topology"] == 0


def test_overview_endpoint_reports_brave_configuration(tmp_path, echo_config):
    with client_for(tmp_path, echo_config) as client:
        body = client.get("/v1/academic/overview").json()
        assert body["brave_configured"] is False and body["total"] == 0
        assert client.get("/v1/academic/overview?topic_id=calculus").json()["topic_id"] == "calculus"
        assert client.get("/v1/academic/overview?topic_id=missing").status_code == 400
    with client_for(tmp_path, echo_config, brave_api_key="test-only") as client:
        assert client.get("/v1/academic/overview").json()["brave_configured"] is True


# --- library --------------------------------------------------------------------------


def test_library_items_are_summaries_without_text_or_evidence(catalog):
    source = add(catalog, "full", authors=["A", "B", "C", "D"], doi="10.5555/full")
    verify(catalog, source, proof())
    catalog.save_collection(source.id, dict(text="Reviewed text " * 50, word_count=100))
    result = catalog.library()
    assert result["total"] == 1 and result["limit"] == 30 and result["offset"] == 0
    item = result["items"][0]
    assert set(item) == LIBRARY_ITEM_KEYS
    assert "text" not in item and "evidence" not in item
    assert item["authors"] == ["A", "B", "C"] and item["author_count"] == 4
    assert item["evidence_count"] == 3 and item["word_count"] == 100
    assert item["claim_count"] == item["child_count"] == item["pending_notices"] == 0
    assert item["withdrawn"] is False
    assert set(item["score"]) == {"topic_id", "legitimacy", "usefulness", "evidence_coverage"}


def test_library_filters(catalog):
    open_work = add(catalog, "open-work", access="open", kind="article")
    verify(catalog, open_work, proof())
    catalog.review(open_work.id, Review(topic_reviewed=True))
    paywalled = add(catalog, "paywalled", access="paywall", kind="book", topic_ids=["integration"])
    container = add(catalog, "container", access="open", kind="sources_of_sources")
    ai = add(catalog, "ai-work", topic_ids=["topology"])
    catalog.review(ai.id, Review(origin="ai"))
    catalog.review(ai.id, Review(topic_ids=["topology"], topic_reviewed=True))

    def ids(**filters) -> set[str]:
        return {item["id"] for item in catalog.library(**filters)["items"]}

    assert ids() == {open_work.id, paywalled.id, container.id, ai.id}
    assert ids(access="open") == {open_work.id, container.id}
    assert ids(access="paywall") == {paywalled.id}
    assert ids(kind="containers") == {container.id}
    assert ids(kind="works") == {open_work.id, paywalled.id, ai.id}
    assert ids(status="human_verified") == {open_work.id}
    assert ids(status="ai") == {ai.id}
    assert ids(status="needs_review") == {paywalled.id, container.id}
    assert ids(topic_id="analysis") == {open_work.id, paywalled.id, container.id}
    assert ids(topic_id="integration") == {paywalled.id}
    assert ids(topic_id="analysis", access="open", kind="works", status="human_verified") == {
        open_work.id,
    }
    # Empty strings from a form are "no filter", not an invalid value.
    assert ids(access="", kind="", status="", topic_id="", q="") == ids()


def test_library_search_is_a_literal_case_insensitive_substring(catalog):
    percent = add(catalog, "percent", title="100% reliable proofs")
    plain = add(catalog, "plain", title="100 reliable proofs")
    underscore = add(catalog, "under", title="Notes", doi="10.1/abc_def")
    other = add(catalog, "other", title="Notes", doi="10.1/abcxdef")
    author = add(
        catalog, "author", title="Real analysis", authors=["Walter Rudin"], publisher="McGraw-Hill",
    )
    accented = add(catalog, "accent", title="Möbius transformations", authors=["Paul Erdős"])
    backslash = add(catalog, "backslash", title=r"Path C:\temp notes")

    def ids(q: str) -> set[str]:
        return {item["id"] for item in catalog.library(q=q)["items"]}

    assert ids("%") == {percent.id}
    assert ids("100%") == {percent.id}
    assert ids("_") == {underscore.id}
    assert ids("abc_def") == {underscore.id}
    assert ids("abc%def") == set()
    assert ids("\\temp") == {backslash.id}
    assert ids("RUDIN") == {author.id}
    assert ids("mcgraw") == {author.id}
    assert ids("  real ANALYSIS  ") == {author.id}
    assert ids("MÖBIUS") == {accented.id}
    assert ids("erdős") == {accented.id}
    assert ids("10.1/abc") == {underscore.id, other.id}
    assert ids("reliable") == {percent.id, plain.id}
    assert ids("") == ids("   ") == {
        percent.id, plain.id, underscore.id, other.id, author.id, accented.id, backslash.id,
    }
    assert catalog.library(q="x" * 200)["total"] == 0
    with pytest.raises(RequestError):
        catalog.library(q="x" * 201)


def test_library_sort_orders_break_ties_by_id(catalog, monkeypatch):
    ticks = itertools.count(1)
    monkeypatch.setattr(
        "kumaru.academic.catalog.now", lambda: f"2026-01-01T00:00:00.{next(ticks):06d}+00:00",
    )
    banana = add(catalog, "banana", title="banana", citation_count=5, published_year=1900)
    apple = add(catalog, "apple", title="Apple", citation_count=None, published_year=None)
    cherry = add(catalog, "cherry", title="cherry", citation_count=100, published_year=2000)
    date = add(catalog, "date", title="Date", citation_count=5, published_year=1900)
    tied = sorted([banana.id, date.id])

    def order(sort: str) -> list[str]:
        return [item["id"] for item in catalog.library(sort=sort)["items"]]

    assert order("updated") == [date.id, cherry.id, apple.id, banana.id]
    assert order("title") == [apple.id, banana.id, cherry.id, date.id]
    assert order("cited") == [cherry.id, *tied, apple.id]
    assert order("year") == [cherry.id, *tied, apple.id]
    catalog.review(banana.id, Review(access="open"))
    assert order("updated")[0] == banana.id
    assert order("updated") == [banana.id, date.id, cherry.id, apple.id]
    assert [item["id"] for item in catalog.library()["items"]] == order("updated")


def test_library_pages_report_the_full_total(catalog):
    for number in range(5):
        add(catalog, f"paper-{number}", title=f"Paper {number}")
    first = catalog.library(sort="title", limit=2, offset=0)
    last = catalog.library(sort="title", limit=2, offset=4)
    beyond = catalog.library(sort="title", limit=2, offset=10)
    assert [len(page["items"]) for page in (first, last, beyond)] == [2, 1, 0]
    assert first["total"] == last["total"] == beyond["total"] == 5
    assert (last["limit"], last["offset"]) == (2, 4)
    titles = [
        item["title"] for offset in (0, 2, 4)
        for item in catalog.library(sort="title", limit=2, offset=offset)["items"]
    ]
    assert titles == [f"Paper {number}" for number in range(5)]


def test_library_container_filter_and_counts(catalog):
    parent = add(catalog, "parent", kind="sources_of_sources")
    first = add(catalog, "first")
    second = add(catalog, "second")
    add(catalog, "unrelated")
    contain(catalog, parent, first)
    contain(catalog, parent, second)
    catalog.link(Relationship(
        source_id=first.id, target_id=parent.id, kind="cites", note="Not membership",
    ))
    contents = catalog.library(container_id=parent.id)
    assert contents["total"] == 2
    assert {item["id"] for item in contents["items"]} == {first.id, second.id}
    assert catalog.library(container_id=first.id)["total"] == 0
    listing = {item["id"]: item for item in catalog.library()["items"]}
    assert listing[parent.id]["child_count"] == 2 and listing[first.id]["child_count"] == 0
    with pytest.raises(RequestError):
        catalog.library(container_id="0" * 24)


def test_library_counts_claims_pending_notices_and_withdrawals(catalog):
    reported = verify(catalog, add(catalog, "reported"), proof())
    retracted = verify(catalog, add(catalog, "retracted"), proof())
    catalog.add_claim(ClaimCreate(
        source_id=reported.id, topic_id="calculus", statement="A claim", locator="p. 1",
    ))
    catalog.report_notice(NoticeCreate(
        source_id=reported.id, kind="correction", url="https://example.org/c", title="Correction",
    ))
    catalog.report_notice(NoticeCreate(
        source_id=reported.id, kind="update", url="https://example.org/u", title="Update",
    ))
    withdrawal = catalog.report_notice(NoticeCreate(
        source_id=retracted.id, kind="retraction", url="https://example.org/r", title="Retraction",
    ))
    catalog.review_notice(withdrawal.id, NoticeReview(state="confirmed", note="Verified", impact=1))
    items = {item["id"]: item for item in catalog.library()["items"]}
    assert items[reported.id]["claim_count"] == 1 and items[reported.id]["pending_notices"] == 2
    assert items[reported.id]["withdrawn"] is False
    assert items[retracted.id]["pending_notices"] == 0 and items[retracted.id]["withdrawn"] is True


def test_library_score_uses_the_best_topic_inside_the_filter(catalog):
    source = add(catalog, "scored", topic_ids=["calculus", "differentiation"])
    verify(
        catalog, source, proof("calculus", 1, "p1"), proof("differentiation", 0.5, "p2"),
        topics=("calculus", "differentiation"),
    )
    catalog.review(source.id, Review(topic_reviewed=True))

    def score(**filters) -> dict:
        return catalog.library(**filters)["items"][0]["score"]

    assert (score()["topic_id"], score()["legitimacy"]) == ("calculus", 30)
    assert (score(topic_id="analysis")["topic_id"], score(topic_id="analysis")["usefulness"]) == (
        "calculus", 19.5,
    )
    narrowed = score(topic_id="differentiation")
    assert (narrowed["topic_id"], narrowed["legitimacy"], narrowed["usefulness"]) == (
        "differentiation", 15, 9.75,
    )
    assert narrowed["evidence_coverage"] == 0.3


def test_library_score_with_only_unknown_values_returns_the_first_in_scope_entry(catalog):
    add(catalog, "unreviewed", topic_ids=["calculus", "integration"])
    assert catalog.library()["items"][0]["score"] == {
        "topic_id": "calculus", "legitimacy": None, "usefulness": None, "evidence_coverage": 0,
    }
    scoped = catalog.library(topic_id="integration")["items"][0]["score"]
    assert scoped["topic_id"] == "integration" and scoped["legitimacy"] is None
    # Nothing in scope is a null score, not a fabricated zero.
    assert catalog._listed_score(catalog.library()["items"][0]["id"], {"topology"}) is None


@pytest.mark.parametrize("query", [
    "access=nope", "kind=nope", "status=nope", "sort=nope", "limit=0", "limit=101",
    "offset=-1", "q=" + "x" * 201, "topic_id=missing-topic", "container_id=" + "0" * 24,
])
def test_library_rejects_invalid_parameters_with_400(tmp_path, echo_config, query):
    with client_for(tmp_path, echo_config) as client:
        assert client.get("/v1/academic/library").status_code == 200
        assert client.get(f"/v1/academic/library?{query}").status_code == 400


def test_library_endpoint_uses_authentication_and_shape(tmp_path, echo_config):
    with client_for(tmp_path, echo_config) as client:
        source = add(client.app.state.academic, "api-work", access="open")
        echo_config.server.api_key = "test-only"
        assert client.get("/v1/academic/library").status_code == 401
        assert client.get("/v1/academic/overview").status_code == 401
        client.headers["Authorization"] = "Bearer test-only"
        body = client.get("/v1/academic/library?access=open&sort=title&limit=5").json()
        assert body["total"] == 1 and body["items"][0]["id"] == source.id
        assert set(body) == {"items", "total", "limit", "offset"}


# --- titles instead of identifiers ----------------------------------------------------


def test_source_detail_names_every_related_source(tmp_path, echo_config):
    with client_for(tmp_path, echo_config) as client:
        db: Catalog = client.app.state.academic
        grandparent = add(catalog=db, name="grandparent", title="Grand library", kind="sources_of_sources")
        parent = add(db, "parent", title="Parent journal", kind="sources_of_sources", access="paywall")
        child = add(db, "child", title="The child article")
        sibling = add(db, "sibling", title="A sibling article")
        peer = add(db, "peer", title="A corroborating paper")
        stranger = add(db, "stranger", title="Unlinked")
        contain(db, grandparent, parent)
        contain(db, parent, child)
        contain(db, parent, sibling)
        db.link(Relationship(
            source_id=peer.id, target_id=child.id, kind="corroborates", note="Independent check",
        ))
        body = client.get(f"/v1/academic/sources/{child.id}").json()
        assert {"source", "scores", "graph", "reviews", "notices", "update_check", "review_summary"} <= set(body)
        expected = {grandparent.id, parent.id, child.id, sibling.id, peer.id}
        assert set(body["related"]) == expected and stranger.id not in body["related"]
        assert body["related"][parent.id] == {
            "id": parent.id, "title": "Parent journal", "kind": "sources_of_sources",
            "access": "paywall", "origin": "unknown",
        }
        assert body["related"][sibling.id]["title"] == "A sibling article"
        top = client.get(f"/v1/academic/sources/{grandparent.id}").json()
        assert set(top["related"]) == {grandparent.id, parent.id, child.id, sibling.id}


def test_source_cards_are_one_titled_summary_per_known_id(catalog):
    first = add(catalog, "first", title="First", access="open")
    assert catalog.source_cards([]) == {}
    cards = catalog.source_cards([first.id, first.id, "0" * 24])
    assert cards == {first.id: {
        "id": first.id, "title": "First", "kind": "website", "access": "open", "origin": "unknown",
    }}
    many = [add(catalog, f"bulk-{number}").id for number in range(3)]
    assert set(catalog.source_cards(many)) == set(many)


def test_claim_details_name_their_source_and_referenced_sources(catalog):
    primary = verify(catalog, add(catalog, "primary", title="The primary paper"))
    reference = verify(catalog, add(catalog, "reference", title="A proof source"))
    claim = catalog.add_claim(ClaimCreate(
        source_id=primary.id, topic_id="calculus", statement="The theorem", locator="Theorem 1",
    ))
    assert catalog.claim_details(claim.id)["references"] == {}
    catalog.review_claim(claim.id, ClaimReview(reviewed=True, note="Checked", evidence=[
        ClaimEvidence(
            kind="proof", value=1, url=reference.url, note="The checked proof",
            topic_ids=["calculus"], reviewed=True, source_id=reference.id, stance="supports",
            locator="Section 2",
        ),
    ]))
    details = catalog.claim_details(claim.id)
    assert set(details) == {"claim", "assessment", "reviews", "source", "references"}
    assert details["source"]["title"] == "The primary paper"
    assert details["references"] == {reference.id: {
        "id": reference.id, "title": "A proof source", "kind": "website",
        "access": "unknown", "origin": "human_verified",
    }}


def test_notice_details_name_dependent_sources_and_claims(catalog):
    target = verify(catalog, add(catalog, "target", title="Target paper"))
    citing = add(catalog, "citing", title="Citing paper")
    corroborated = add(catalog, "corroborated", title="Corroborated paper")
    catalog.link(Relationship(source_id=citing.id, target_id=target.id, kind="cites", note="Cited"))
    catalog.link(Relationship(
        source_id=target.id, target_id=corroborated.id, kind="corroborates", note="Agrees",
    ))
    claim = catalog.add_claim(ClaimCreate(
        source_id=target.id, topic_id="calculus", statement="A located assertion", locator="p. 3",
    ))
    notice = catalog.report_notice(NoticeCreate(
        source_id=target.id, kind="correction", url="https://example.org/n", title="Erratum",
    ))
    details = catalog.notice_details(notice.id)
    assert set(details["dependent_source_ids"]) == {citing.id, corroborated.id}
    assert details["dependent_sources"] == {
        citing.id: catalog.source_cards([citing.id])[citing.id],
        corroborated.id: catalog.source_cards([corroborated.id])[corroborated.id],
    }
    assert details["dependent_sources"][citing.id]["title"] == "Citing paper"
    assert details["source"]["title"] == "Target paper"
    assert details["affected_claim_ids"] == [claim.id]
    assert details["affected_claims"] == [{
        "id": claim.id, "source_id": target.id, "topic_id": "calculus",
        "statement": "A located assertion",
    }]


def test_notice_affected_claims_are_filtered_by_reviewed_topics_in_sql(catalog):
    source = verify(catalog, add(catalog, "multi", topic_ids=["calculus", "integration"]), topics=(
        "calculus", "integration",
    ))
    other = verify(catalog, add(catalog, "other-source", title="Other"))
    own = catalog.add_claim(ClaimCreate(
        source_id=source.id, topic_id="integration", statement="Own", locator="p. 1",
    ))
    calc = catalog.add_claim(ClaimCreate(
        source_id=source.id, topic_id="calculus", statement="Calc", locator="p. 2",
    ))
    referencing = catalog.add_claim(ClaimCreate(
        source_id=other.id, topic_id="calculus", statement="Uses it", locator="p. 3",
    ))
    catalog.review_claim(referencing.id, ClaimReview(reviewed=True, note="Linked", evidence=[
        ClaimEvidence(
            kind="proof", value=1, url=source.url, note="Uses the multi-topic source",
            topic_ids=["calculus"], reviewed=True, source_id=source.id, stance="supports",
            locator="p. 3",
        ),
    ]))
    notice = catalog.report_notice(NoticeCreate(
        source_id=source.id, kind="correction", url="https://example.org/n", title="Erratum",
    ))
    assert catalog.notice_details(notice.id)["affected_claim_ids"] == sorted(
        [own.id, calc.id, referencing.id],
    )
    catalog.review_notice(notice.id, NoticeReview(
        state="confirmed", note="Only integration", impact=0.3, affected_topic_ids=["integration"],
    ))
    assert catalog.notice_details(notice.id)["affected_claim_ids"] == [own.id]
    catalog.review_notice(notice.id, NoticeReview(
        state="confirmed", note="Calculus too", impact=0.3, affected_topic_ids=["calculus"],
    ))
    assert catalog.notice_details(notice.id)["affected_claim_ids"] == sorted([calc.id, referencing.id])


# --- claims filter --------------------------------------------------------------------


def test_claim_review_filter_in_catalog_and_api(tmp_path, echo_config):
    with client_for(tmp_path, echo_config) as client:
        db: Catalog = client.app.state.academic
        source = verify(db, add(db, "claims"))
        first = db.add_claim(ClaimCreate(
            source_id=source.id, topic_id="calculus", statement="One", locator="p. 1",
        ))
        second = db.add_claim(ClaimCreate(
            source_id=source.id, topic_id="calculus", statement="Two", locator="p. 2",
        ))
        db.review_claim(first.id, ClaimReview(reviewed=True, note="Checked"))
        assert [claim.id for claim in db.claims(reviewed=False)] == [second.id]
        assert [claim.id for claim in db.claims(reviewed=True)] == [first.id]
        assert {claim.id for claim in db.claims()} == {first.id, second.id}
        pending = client.get("/v1/academic/claims?reviewed=false").json()
        assert [item["claim"]["id"] for item in pending] == [second.id]
        assert pending[0]["source"]["title"] == "claims"
        done = client.get(f"/v1/academic/claims?reviewed=true&source_id={source.id}").json()
        assert [item["claim"]["id"] for item in done] == [first.id]
        assert len(client.get("/v1/academic/claims").json()) == 2


# --- source review access note --------------------------------------------------------


def test_access_note_is_applied_and_audited_through_review_history(catalog):
    source = add(catalog, "walled", access="unknown")
    updated = catalog.review(source.id, Review(access="paywall", access_note="Publisher subscription"))
    assert (updated.access, updated.access_note) == ("paywall", "Publisher subscription")
    entry = catalog.history(source.id)[0]["review"]
    assert entry["access"] == "paywall" and entry["access_note"] == "Publisher subscription"
    # An omitted note keeps the recorded reason; an empty one clears it deliberately.
    assert catalog.review(source.id, Review(access="account")).access_note == "Publisher subscription"
    assert catalog.review(source.id, Review(access_note="")).access_note == ""
    assert len(catalog.history(source.id)) == 3
    with pytest.raises(ValidationError):
        Review(access_note="x" * 2001)
    assert Review(access_note="x" * 2000).access_note


def test_access_note_review_endpoint(tmp_path, echo_config):
    with client_for(tmp_path, echo_config) as client:
        source = add(client.app.state.academic, "endpoint-note")
        response = client.post(
            f"/v1/academic/sources/{source.id}/review",
            json={"access": "inaccessible", "access_note": "Browser challenge"},
        )
        assert response.status_code == 200 and response.json()["access_note"] == "Browser challenge"
        history = client.get(f"/v1/academic/sources/{source.id}").json()["reviews"]
        assert history[-1]["review"]["access_note"] == "Browser challenge"


# --- transient failures never reclassify access ---------------------------------------


class Offline(PublicWeb):
    def __init__(self, error: Exception) -> None:
        super().__init__()
        self.error = error

    def fetch(self, *args, **kwargs):
        raise self.error


def test_transient_failure_leaves_access_and_note_untouched_on_inspect(catalog):
    source = add(catalog, "stable", access="paywall", access_note="Publisher subscription")
    with pytest.raises(TransientFetchError) as caught:
        inspect_source(catalog, source.id, Offline(TransientFetchError("DNS unavailable")))
    assert caught.value.http_status == 503
    kept = catalog.get(source.id)
    assert (kept.access, kept.access_note) == ("paywall", "Publisher subscription")


def test_transient_failure_leaves_access_untouched_on_collect(catalog):
    source = add(catalog, "collect")
    verify(catalog, source, proof())
    catalog.review(source.id, Review(access="open", access_note="Reachable", topic_reviewed=True))
    with pytest.raises(TransientFetchError):
        collect_source(catalog, source.id, Offline(TransientFetchError("Timed out")))
    assert (catalog.get(source.id).access, catalog.get(source.id).access_note) == ("open", "Reachable")
    with pytest.raises(FetchError, match="exceeds"):
        collect_source(catalog, source.id, Offline(FetchError("Response exceeds 2000000 bytes")))
    assert catalog.get(source.id).access == "inaccessible"


def test_non_transient_failure_still_records_inaccessibility_on_inspect(catalog):
    source = add(catalog, "broken", access="open")
    with pytest.raises(FetchError) as caught:
        inspect_source(catalog, source.id, Offline(FetchError("Redirect has no Location header")))
    assert not isinstance(caught.value, TransientFetchError) and caught.value.http_status == 502
    assert catalog.get(source.id).access == "inaccessible"
    assert catalog.get(source.id).access_note == "Redirect has no Location header"


def test_access_blocked_is_still_classified_not_treated_as_transient(catalog):
    source = add(catalog, "blocked")
    inspected = inspect_source(catalog, source.id, Offline(AccessBlocked("robots.txt disallows", "account")))
    assert (inspected.access, inspected.access_note) == ("account", "robots.txt disallows")


class Serving(PublicWeb):
    def __init__(self, status: int) -> None:
        super().__init__()
        self.status = status

    def fetch(self, *args, **kwargs):
        return Page("https://example.org/page", self.status, "text/html", b"")


@pytest.mark.parametrize("status", [429, 500, 503])
def test_page_outages_are_transient_and_leave_access_untouched(catalog, status):
    source = add(catalog, f"outage-{status}", access="paywall", access_note="Publisher subscription")
    with pytest.raises(TransientFetchError, match=f"HTTP {status}"):
        inspect_source(catalog, source.id, Serving(status))
    kept = catalog.get(source.id)
    assert (kept.access, kept.access_note) == ("paywall", "Publisher subscription")


def test_missing_page_is_still_classified_inaccessible(catalog):
    source = add(catalog, "gone", access="open")
    assert inspect_source(catalog, source.id, Serving(404)).access == "inaccessible"


def test_dns_failure_is_transient(monkeypatch):
    def unresolved(*args, **kwargs):
        raise socket.gaierror(socket.EAI_NONAME, "Name or service not known")

    monkeypatch.setattr(socket, "getaddrinfo", unresolved)
    with pytest.raises(TransientFetchError, match="Cannot resolve"):
        public_addresses("example.org", 443)


@pytest.mark.parametrize(("error", "transient"), [
    (httpx.ConnectTimeout("slow"), True),
    (httpx.ConnectError("refused"), True),
    (httpx.ReadError("reset"), True),
    (httpx.TooManyRedirects("loop"), False),
    (httpx.DecodingError("bad gzip"), False),
])
def test_only_transport_errors_are_transient(monkeypatch, error, transient):
    from kumaru.academic import web as module

    monkeypatch.setattr(module, "public_addresses", lambda *args: ["93.184.216.34"])

    class Client:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        @contextmanager
        def stream(self, *args, **kwargs):
            raise error
            yield  # pragma: no cover

    monkeypatch.setattr(httpx, "Client", Client)
    with pytest.raises(FetchError) as caught:
        PublicWeb().fetch("https://example.org/a", robots=False)
    assert isinstance(caught.value, TransientFetchError) is transient


@pytest.mark.parametrize(("status", "transient"), [
    (429, True), (500, True), (503, True), (418, False), (410, False),
])
def test_robots_and_provider_outages_are_transient_but_other_statuses_are_not(
    monkeypatch, status, transient,
):
    robots = PublicWeb()
    monkeypatch.setattr(robots, "_request", lambda url, headers=None, delay=1: Page(
        url, status, "text/plain", b"",
    ))
    with pytest.raises(FetchError, match="robots policy") as caught:
        robots.fetch("https://example.org/page")
    assert isinstance(caught.value, TransientFetchError) is transient
    provider = PublicWeb()
    monkeypatch.setattr(provider, "fetch", lambda *a, **k: Page(
        "https://api.example.org/", status, "application/json", b"{}",
    ))
    with pytest.raises(FetchError, match="Search provider") as caught:
        provider.json("https://api.example.org/")
    assert isinstance(caught.value, TransientFetchError) is transient


def test_robots_404_is_not_an_error(monkeypatch):
    web = PublicWeb()

    def request(url, headers=None, delay=1):
        status = 404 if url.endswith("/robots.txt") else 200
        return Page(url, status, "text/plain", b"public")

    monkeypatch.setattr(web, "_request", request)
    assert web.fetch("https://example.org/page").body == b"public"


def test_transient_failure_is_a_503_that_keeps_the_catalogue_unchanged(tmp_path, echo_config, monkeypatch):
    with client_for(tmp_path, echo_config) as client:
        source = add(client.app.state.academic, "api-offline", access="account")
        monkeypatch.setattr(
            routes_academic, "PublicWeb", lambda: Offline(TransientFetchError("No route to host")),
        )
        response = client.post(f"/v1/academic/sources/{source.id}/inspect")
        assert response.status_code == 503
        assert "No route to host" in response.json()["error"]["message"]
        assert client.app.state.academic.get(source.id).access == "account"


# --- discovery ------------------------------------------------------------------------


class Provider(PublicWeb):
    def __init__(self, payload: dict) -> None:
        super().__init__()
        self.payload = payload

    def json(self, url, *, headers=None):
        return self.payload


def work(title, **fields) -> dict:
    return {"title": title, "type": "journal-article", **fields}


def test_discovery_skips_incomplete_crossref_records_and_counts_them(catalog):
    payload = {"status": "ok", "message": {"items": [
        work(["Complete work"], DOI="10.1/complete", URL="https://example.org/complete"),
        work(["DOI only"], DOI="10.1/doi-only"),
        work(["No identifier"]),
        {"type": "peer-review", "URL": "https://example.org/untitled"},
        work([], DOI="10.1/empty-title"),
        work(["   "], DOI="10.1/blank-title"),
        work(["", "Later title"], URL="https://example.org/later"),
    ]}}
    report = discover(catalog, "calculus", web=Provider(payload))
    assert report["skipped"] == 4 and len(report["source_ids"]) == 3
    titles = {catalog.get(id_).title for id_ in report["source_ids"]}
    assert titles == {"Complete work", "DOI only", "Later title"}
    assert catalog.get(report["source_ids"][1]).url == "https://doi.org/10.1/doi-only"
    again = discover(catalog, "calculus", web=Provider(payload))
    assert again["skipped"] == 4 and again["source_ids"] == report["source_ids"]


def test_discovery_with_only_incomplete_records_adds_no_works(catalog):
    report = discover(catalog, "calculus", web=Provider({
        "status": "ok", "message": {"items": [work(["Component"]), {"type": "component"}]},
    }))
    assert report["skipped"] == 2 and report["source_ids"] == []
    assert not catalog.list_sources(containers=False)


@pytest.mark.parametrize("payload", [
    {"status": "ok"},
    {"status": "ok", "message": {"items": "none"}},
    {"status": "ok", "message": {"items": [{"title": ["Unsafe"], "URL": "file:///etc/passwd"}]}},
    {"status": "ok", "message": {"items": [work(["Negative"], DOI="10.1/n", **{"reference-count": -1})]}},
])
def test_discovery_still_rejects_a_malformed_response_envelope_or_unsafe_record(catalog, payload):
    with pytest.raises(FetchError):
        discover(catalog, "calculus", web=Provider(payload))
    assert not catalog.list_sources()


def test_brave_discovery_reports_zero_skipped(catalog):
    report = discover(
        catalog, "algebra", provider="brave", brave_api_key="test-only",
        web=Provider({"web": {"results": [{"title": "Books", "url": "https://example.org/books"}]}}),
    )
    assert report["skipped"] == 0 and len(report["source_ids"]) == 1


# --- scoring constants and static files -----------------------------------------------


def test_usefulness_points_preserve_the_published_formula(catalog):
    source = add(
        catalog, "formula", citation_count=42, reference_count=12, authors=["Scholar"],
        publisher="University",
    )
    verify(catalog, source, proof())
    catalog.review(source.id, Review(topic_reviewed=True))
    score = score_source(catalog.get(source.id), "calculus")
    cited = min(1.0, math.log1p(42) / math.log1p(1000))
    references = min(1.0, math.log1p(12) / math.log1p(100))
    assert score.legitimacy == 30
    assert score.usefulness == round(0.65 * 30.0 + 15 * cited + 5 * references, 2)
    assert score.components["citations"] == cited and score.components["references"] == references


def test_javascript_modules_are_served_with_a_javascript_content_type(tmp_path, echo_config):
    assert mimetypes.guess_type("module.js")[0] == "text/javascript"
    assert mimetypes.guess_type("module.mjs")[0] == "text/javascript"
    modules = sorted((UI_DIR / "academic").glob("*.js"))
    assert modules, "the dashboard is split into ES modules under ui/academic/"
    with client_for(tmp_path, echo_config) as client:
        entry = client.get("/static/academic.js")
        assert entry.status_code == 200 and entry.headers["content-type"].startswith("text/javascript")
        for module in modules:
            response = client.get(f"/static/academic/{module.name}")
            assert response.status_code == 200, module.name
            assert response.headers["content-type"].startswith("text/javascript"), module.name
        page = client.get("/academic").text
        assert '<script type="module" src="/static/academic.js">' in page
        assert "onclick=" not in page and "<style" not in page

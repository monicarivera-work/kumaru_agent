"""Offline evidence, graph, discovery, access, transport and integration regressions."""

from __future__ import annotations

import socket
import ssl
from contextlib import contextmanager
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from typer.testing import CliRunner

from kumaru.academic.catalog import Catalog
from kumaru.academic.discovery import discover
from kumaru.academic.models import Evidence, Relationship, Review, Source, Topic, canonical_url
from kumaru.academic.scoring import WEIGHTS, score_source
from kumaru.academic.web import (
    AccessBlocked,
    FetchError,
    Page,
    PublicWeb,
    analyze_page,
    collect_source,
    inspect_source,
    public_addresses,
)
from kumaru.cli import app as cli
from kumaru.core.config import AcademicConfig, KumaruConfig
from kumaru.core.errors import RequestError
from kumaru.server.app import create_app


@pytest.fixture
def catalog(tmp_path: Path) -> Catalog:
    return Catalog(tmp_path / "academic.sqlite")


def candidate(url: str = "https://example.org/calculus", **kwargs) -> Source:
    fields = dict(url=url, title="A human calculus reference", topic_ids=["calculus"])
    fields.update(kwargs)
    return Source.model_validate(fields)


def evidence(kind, value=1, **kwargs) -> Evidence:
    fields = dict(
        kind=kind, value=value, url="https://example.org/evidence",
        note=f"Reviewed {kind} evidence", reviewed=True,
        topic_ids=["calculus"] if kind in {"proof", "historical_acceptance", "factual_criticism"} else [],
    )
    fields.update(kwargs)
    return Evidence.model_validate(fields)


def approve(catalog: Catalog, id_: str, **kwargs) -> Source:
    return catalog.review(id_, Review(
        origin="human_verified", permission="text_permitted", topic_ids=["calculus"],
        evidence=[evidence("origin"), evidence("permission"), evidence("proof")], **kwargs,
    ))


class FakeWeb(PublicWeb):
    def __init__(self, page: Page | None = None, payload: dict | None = None) -> None:
        super().__init__()
        self.page = page or Page(
            "https://example.org/calculus", 200, "text/html; charset=utf-8",
            b"<html><title>Calculus</title><h1>Calculus</h1><p>Reviewed original mathematics.</p></html>",
        )
        self.payload = payload or {}
        self.calls: list[str] = []

    def fetch(self, url, *, robots=True, headers=None):
        self.calls.append(url)
        return self.page

    def json(self, url, *, headers=None):
        self.calls.append(url)
        return self.payload


def test_canonical_source_identity_and_invalid_urls():
    assert canonical_url("https://EXAMPLE.org/a#section") == "https://example.org/a"
    assert Source(url="https://example.org/a#first", title="a").id == Source(
        url="https://example.org/a#second", title="a",
    ).id
    for url in ("file:///etc/passwd", "https://u:p@example.org/", "https://example.org:1234"):
        with pytest.raises((ValueError, ValidationError)):
            Source(url=url, title="unsafe")
    with pytest.raises(ValidationError):
        Source(url="https://example.org/", title="a", id="invented")


def test_topic_hierarchy_and_area_filter(catalog):
    assert [t.id for t in catalog.topic_path("chain-rule")] == [
        "mathematics", "analysis", "calculus", "differentiation", "chain-rule",
    ]
    source = catalog.discover(candidate())
    assert catalog.list_sources(topic_id="mathematics")[0].id == source.id
    assert not catalog.list_sources(topic_id="algebra")
    catalog.put_topic(Topic(id="physics", name="Physics", level="area"))
    catalog.put_topic(Topic(id="mechanics", name="Mechanics", level="subfield", parent_id="physics"))
    assert catalog.descendants("physics") == {"physics", "mechanics"}
    with pytest.raises(RequestError):
        catalog.put_topic(Topic(id="analysis", name="Analysis", level="area", parent_id="calculus"))
    with pytest.raises(RequestError):
        catalog.put_topic(Topic(id="orphan", name="Orphan", level="chapter"))
    with pytest.raises(RequestError):
        catalog.put_topic(Topic(id="analysis", name="Analysis", level="chapter", parent_id="mathematics"))


def test_unknown_topics_cannot_be_saved(catalog):
    with pytest.raises(RequestError):
        catalog.discover(candidate(topic_ids=["missing"]))
    with pytest.raises(RequestError):
        catalog.discover(candidate(topic_ids=[]))


def test_reviews_are_persistent_and_rediscovery_preserves_them(catalog):
    source = catalog.discover(candidate(doi="10.1234/test"))
    approve(catalog, source.id)
    collect_source(catalog, source.id, FakeWeb())
    # DOI identity also de-duplicates a second landing-page URL.
    updated = catalog.discover(candidate(
        url="https://publisher.org/paper", doi="10.1234/TEST", citation_count=24,
    ))
    assert updated.id == source.id
    assert updated.origin == "human_verified" and updated.text
    assert updated.citation_count == 24
    reopened = Catalog(catalog.path)
    assert reopened.get(source.id).text_sha256 == updated.text_sha256
    assert len(reopened.history(source.id)) == 1


def test_review_requires_provenance_and_rights_evidence(catalog):
    source = catalog.discover(candidate())
    for review in (Review(origin="human_verified"), Review(permission="text_permitted")):
        with pytest.raises(RequestError):
            catalog.review(source.id, review)
    assert catalog.get(source.id).origin == "unknown"
    assert not catalog.history(source.id)
    with pytest.raises(ValidationError):
        Evidence(kind="proof", value=1, url=source.url, note="no topic")
    with pytest.raises(ValidationError):
        Evidence(kind="credentials", value=float("nan"), url=source.url, note="invalid")
    with pytest.raises(RequestError):
        catalog.review(source.id, Review(evidence=[
            Evidence(kind="proof", value=1, url=source.url, note="wrong topic", topic_ids=["algebra"]),
        ]))


def test_revocation_removes_text_and_cannot_leave_unverified_approval(catalog):
    source = catalog.discover(candidate())
    approve(catalog, source.id)
    collect_source(catalog, source.id, FakeWeb())
    origin = next(item for item in catalog.get(source.id).evidence if item.kind == "origin")
    with pytest.raises(RequestError):
        catalog.review(source.id, Review(evidence=[
            Evidence.model_validate({**origin.model_dump(), "reviewed": False}),
        ]))
    updated = catalog.review(source.id, Review(origin="unknown"))
    assert updated.text is None and updated.word_count == 0 and not updated.topic_word_counts


def test_source_graph_parents_grandparents_and_siblings(catalog):
    root = catalog.discover(candidate("https://example.org/database", kind="sources_of_sources"))
    journal = catalog.discover(candidate("https://example.org/journal", kind="sources_of_sources"))
    a = catalog.discover(candidate("https://example.org/article-a"))
    b = catalog.discover(candidate("https://example.org/article-b"))
    for parent, child in [(root, journal), (journal, a), (journal, b)]:
        catalog.link(Relationship(
            source_id=parent.id, target_id=child.id, kind="contains", note="Indexed here",
        ))
    graph = catalog.graph(a.id)
    assert set(graph["ancestors"]) == {root.id, journal.id}
    assert graph["siblings"] == [b.id]
    assert set(catalog.graph(root.id)["descendants"]) == {journal.id, a.id, b.id}
    with pytest.raises(RequestError):
        catalog.link(Relationship(source_id=a.id, target_id=root.id, kind="contains", note="cycle"))
    with pytest.raises(RequestError):
        catalog.link(Relationship(source_id=a.id, target_id=a.id, kind="cites", note="self"))


def test_independence_requires_distinct_authors_publishers_and_review(catalog):
    source = catalog.discover(candidate(authors=["Scholar A"], publisher="Publisher A"))
    approve(catalog, source.id)
    peer = catalog.discover(candidate(
        "https://example.org/peer", authors=["Scholar B"], publisher="Publisher A",
    ))
    catalog.link(Relationship(
        source_id=peer.id, target_id=source.id, kind="corroborates", note="Reviewed corroboration",
    ))
    approve(catalog, peer.id)
    assert catalog.scores(source.id)[0]["components"]["independent_sources"] == 0
    catalog.discover(candidate(peer.url, publisher="Publisher B"))
    assert catalog.scores(source.id)[0]["components"]["independent_sources"] == 0.2
    catalog.review(peer.id, Review(origin="unknown"))
    assert catalog.scores(source.id)[0]["components"]["independent_sources"] == 0


@pytest.mark.parametrize("access", ["unknown", "open", "account", "paywall", "inaccessible"])
def test_container_access_categories(catalog, access):
    source = catalog.discover(candidate(
        f"https://example.org/{access}", kind="sources_of_sources", access=access,
    ))
    assert catalog.list_sources(containers=True, access=access)[0].id == source.id
    with pytest.raises(RequestError):
        collect_source(catalog, source.id, FakeWeb())


def trusted(**kwargs):
    fields = dict(origin="human_verified", permission="text_permitted", topic_reviewed=True)
    fields.update(kwargs)
    return candidate(**fields)


def test_unknown_and_container_scores_are_not_fabricated():
    assert score_source(candidate(), "calculus").legitimacy is None
    container = trusted(kind="sources_of_sources", evidence=[evidence("proof")])
    assert score_source(container, "calculus").legitimacy is None
    assert score_source(trusted(), "calculus").usefulness is None


def test_score_bounds_sparse_coverage_and_topic_specific_proof():
    sparse = trusted(evidence=[evidence("credentials")])
    score = score_source(sparse, "calculus")
    assert score.legitimacy == 15
    assert score.evidence_coverage == 0.15
    assert score_source(sparse, "algebra").legitimacy is None
    assert score_source(trusted(evidence=[evidence("proof")]), "algebra").legitimacy is None
    all_evidence = [evidence(kind) for kind in WEIGHTS]
    complete = trusted(
        evidence=all_evidence, citation_count=10**9, reference_count=10**9,
        text="Mathematics " * 10000, word_count=10000, topic_word_counts={"calculus": 10000},
    )
    score = score_source(complete, "calculus", independent_sources=[str(i) for i in range(20)])
    assert score.legitimacy == score.usefulness == 100
    assert score.evidence_coverage == 1
    bare = trusted(evidence=all_evidence)
    assert score_source(bare, "calculus").usefulness == 65


def test_length_is_light_and_duplicate_corroboration_does_not_inflate_score():
    source = trusted(evidence=[evidence("proof")])
    baseline = score_source(source, "calculus").usefulness
    full = trusted(
        evidence=source.evidence, text="word " * 100000, word_count=100000,
        topic_word_counts={"calculus": 100000},
    )
    assert score_source(full, "calculus").usefulness - baseline == 2
    a = score_source(source, "calculus", independent_sources=["peer"])
    b = score_source(source, "calculus", independent_sources=["peer"] * 100)
    assert a.usefulness == b.usefulness


def test_duplicate_evidence_urls_cannot_add_credit():
    source = trusted(evidence=[evidence("credentials", 0.2), evidence("credentials", 0.8)])
    before = score_source(source, "calculus")
    source.evidence.extend(evidence("credentials", 0.8) for _ in range(100))
    after = score_source(source, "calculus")
    assert before.legitimacy == after.legitimacy == 3


def test_verified_factual_criticism_and_retractions_reduce_scores():
    source = trusted(evidence=[evidence(kind) for kind in WEIGHTS])
    source.evidence.append(evidence("factual_criticism"))
    assert score_source(source, "calculus").legitimacy == 20
    source.evidence.append(evidence("retraction"))
    source.citation_count = 10**9
    score = score_source(source, "calculus")
    assert score.legitimacy == 0 and score.usefulness <= 10
    source.evidence = [evidence("proof", reviewed=False)]
    assert score_source(source, "calculus").legitimacy is None


def crossref_payload():
    return {
        "status": "ok",
        "message": {"items": [{
            "title": ["Calculus foundations"], "DOI": "10.1234/test",
            "URL": "https://example.org/calculus", "type": "journal-article",
            "author": [{"given": "A.", "family": "Scholar"}], "publisher": "University",
            "published": {"date-parts": [[1900]]},
            "is-referenced-by-count": 42, "reference-count": 12,
        }]},
    }


def test_contextual_discovery_tags_metadata_and_parent_without_text(catalog):
    web = FakeWeb(payload=crossref_payload())
    report = discover(catalog, "calculus", web=web)
    assert report["query"] == "Mathematics Analysis Calculus"
    source = catalog.get(report["source_ids"][0])
    assert source.authors == ["A. Scholar"] and source.published_year == 1900
    assert source.citation_count == 42 and source.topic_ids == ["calculus"]
    assert source.origin == "unknown" and source.access == "unknown" and source.text is None
    parent = catalog.list_sources(containers=True)[0]
    assert parent.id in catalog.graph(source.id)["ancestors"]
    assert parent.access == "open"
    assert not source.evidence
    assert discover(catalog, "calculus", web=web)["source_ids"] == report["source_ids"]


def test_brave_search_is_explicit_opt_in(catalog):
    with pytest.raises(RequestError):
        discover(catalog, "algebra", provider="brave", web=FakeWeb())
    report = discover(
        catalog, "algebra", provider="brave", brave_api_key="test-only",
        web=FakeWeb(payload={"web": {"results": [{
            "title": "Paid textbooks", "url": "https://example.org/books",
        }]}}),
    )
    assert catalog.get(report["source_ids"][0]).access == "unknown"


@pytest.mark.parametrize("payload", [
    {"status": "bad"},
    {"status": "ok", "message": {"items": [{"title": ["unsafe"], "URL": "file:///etc/passwd"}]}},
])
def test_bad_provider_payloads_fail_without_fabricated_results(catalog, payload):
    with pytest.raises(FetchError):
        discover(catalog, "calculus", web=FakeWeb(payload=payload))
    assert not catalog.list_sources()


def test_collection_and_export_gates_do_not_fetch_unapproved_text(catalog):
    source = catalog.discover(candidate())
    web = FakeWeb()
    with pytest.raises(RequestError):
        collect_source(catalog, source.id, web)
    assert not web.calls
    approve(catalog, source.id)
    result = collect_source(catalog, source.id, web)
    assert result.text and "Reviewed original mathematics" in result.text
    assert len(result.text_sha256) == 64
    assert result.topic_word_counts["calculus"] == result.word_count
    assert result.text_url == web.page.url and result.access == "open"


def test_inspect_stores_metadata_not_text(catalog):
    source = catalog.discover(candidate())
    inspected = inspect_source(catalog, source.id, FakeWeb())
    assert inspected.title == "Calculus" and inspected.access == "open"
    assert inspected.text is None and inspected.origin == "unknown"
    container = catalog.discover(candidate("https://example.org/database", kind="sources_of_sources"))
    assert inspect_source(catalog, container.id, FakeWeb()).text is None


def test_non_transient_fetch_failure_records_inaccessibility_and_still_raises(catalog):
    source = catalog.discover(candidate())

    class Unavailable(FakeWeb):
        def fetch(self, *args, **kwargs):
            raise FetchError("Response exceeds 2000000 bytes")

    with pytest.raises(FetchError):
        inspect_source(catalog, source.id, Unavailable())
    assert catalog.get(source.id).access == "inaccessible"
    assert catalog.get(source.id).access_note == "Response exceeds 2000000 bytes"


@pytest.mark.parametrize(("body", "access"), [
    (b"<p>Sign in to read this article</p>", "account"),
    (b"<p>Purchase this article</p>", "paywall"),
    (b'<script type="application/ld+json">{"isAccessibleForFree":false}</script>', "paywall"),
    (b"<p>Verify you are human</p>", "inaccessible"),
])
def test_gates_are_catalogued_without_bypass_or_text(catalog, body, access):
    source = catalog.discover(candidate())
    approve(catalog, source.id)
    web = FakeWeb(page=Page(source.url, 200, "text/html", body))
    with pytest.raises(AccessBlocked):
        collect_source(catalog, source.id, web)
    assert catalog.get(source.id).access == access
    assert catalog.get(source.id).text is None


@pytest.mark.parametrize(("status", "access"), [(401, "account"), (402, "paywall"), (403, "inaccessible")])
def test_http_access_statuses_are_explicit(catalog, status, access):
    source = catalog.discover(candidate())
    approve(catalog, source.id)
    web = FakeWeb(page=Page(source.url, status, "text/html", b"restricted"))
    with pytest.raises(AccessBlocked):
        collect_source(catalog, source.id, web)
    assert catalog.get(source.id).access == access


def test_ai_disclosure_revokes_trust(catalog):
    source = catalog.discover(candidate())
    approve(catalog, source.id)
    collect_source(catalog, source.id, FakeWeb())
    web = FakeWeb(page=Page(
        source.url, 200, "text/html", b"<p>This article was generated by AI.</p>",
    ))
    with pytest.raises(AccessBlocked):
        collect_source(catalog, source.id, web)
    assert catalog.get(source.id).origin == "ai" and catalog.get(source.id).text is None
    assert score_source(catalog.get(source.id), "calculus").legitimacy is None


def test_unsupported_formats_and_cross_host_collection_fail(catalog):
    source = catalog.discover(candidate())
    approve(catalog, source.id)
    with pytest.raises(AccessBlocked):
        collect_source(catalog, source.id, FakeWeb(page=Page(
            source.url, 200, "application/pdf", b"PDF metadata only",
        )))
    catalog.review(source.id, Review(access="unknown"))
    with pytest.raises(RequestError, match="another host"):
        collect_source(catalog, source.id, FakeWeb(page=Page(
            "https://another.org/work", 200, "text/plain", b"text",
        )))
    assert catalog.get(source.id).text is None


def test_markup_scripts_and_navigation_are_not_executed():
    page = Page("https://example.org/", 200, "text/html", b"""
        <title>Human text</title><script>alert('unsafe')</script><style>badcss</style>
        <p>A proof of the theorem.</p><meta name="citation_author" content="Scholar">
    """)
    updates, text = analyze_page(page)
    assert "alert" not in text and "badcss" not in text
    assert updates["authors"] == ["Scholar"] and "A proof" in text


@pytest.mark.parametrize("address", [
    "127.0.0.1", "10.1.2.3", "169.254.169.254", "::1", "::ffff:127.0.0.1",
    "2002:7f00:0001::",
])
def test_private_addresses_blocked(monkeypatch, address):
    monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **k: [
        (socket.AF_INET, socket.SOCK_STREAM, 0, "", (address, 443)),
    ])
    with pytest.raises(AccessBlocked):
        public_addresses("example.org", 443)


def test_mixed_public_private_dns_is_blocked(monkeypatch):
    monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **k: [
        (socket.AF_INET, socket.SOCK_STREAM, 0, "", (ip, 443))
        for ip in ["93.184.216.34", "127.0.0.1"]
    ])
    with pytest.raises(AccessBlocked):
        public_addresses("example.org", 443)


def test_transport_pins_ip_and_verifies_original_tls_name(monkeypatch):
    from kumaru.academic import web as module

    monkeypatch.setattr(module, "public_addresses", lambda *args: ["93.184.216.34"])
    observed = {}

    class Client:
        def __init__(self, **kwargs):
            assert kwargs["trust_env"] is False and kwargs["follow_redirects"] is False
            assert kwargs["verify"].verify_mode == ssl.CERT_REQUIRED
            assert kwargs["verify"].check_hostname

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        @contextmanager
        def stream(self, method, url, **kwargs):
            observed.update(url=url, **kwargs)
            yield httpx.Response(200, content=b"ok", headers={"content-type": "text/plain"})

    monkeypatch.setattr(httpx, "Client", Client)
    page = PublicWeb().fetch("https://example.org/a", robots=False)
    assert observed["url"].host == "93.184.216.34"
    assert observed["headers"]["Host"] == "example.org"
    assert observed["extensions"]["sni_hostname"] == "example.org"
    assert page.url == "https://example.org/a"
    with pytest.raises(FetchError, match="exceeds"):
        PublicWeb(max_bytes=1).fetch("https://example.org/a", robots=False)


def test_robots_redirects_and_authenticated_header_policy(monkeypatch):
    web = PublicWeb()
    calls = []

    def request(url, headers=None, delay=1):
        calls.append(url)
        if url.endswith("/robots.txt"):
            return Page(url, 200, "text/plain", b"User-agent: *\nDisallow: /private")
        return Page(url, 200, "text/plain", b"public")

    monkeypatch.setattr(web, "_request", request)
    with pytest.raises(AccessBlocked):
        web.fetch("https://example.org/private")
    assert calls == ["https://example.org/robots.txt"]
    assert web.fetch("https://example.org/public").body == b"public"
    monkeypatch.setattr(web, "_request", lambda *a, **k: Page(
        "https://other.org/", 302, "redirect", b"",
    ))
    with pytest.raises(FetchError, match="Authenticated"):
        web.fetch("https://example.org/api", robots=False, headers={"X-Key": "secret"})
    with pytest.raises(FetchError, match="five redirects"):
        web.fetch("https://example.org/api", robots=False)


def test_offline_cli_and_export_into_existing_learning_workflow(tmp_path, catalog):
    runner = CliRunner()
    prefix = ["academic", "--database", str(catalog.path)]
    result = runner.invoke(cli, [*prefix, "topics"])
    assert result.exit_code == 0 and '"Mathematics"' in result.output
    source = catalog.discover(candidate())
    approve(catalog, source.id)
    collect_source(catalog, source.id, FakeWeb())
    output = tmp_path / "reference.txt"
    result = runner.invoke(cli, [*prefix, "export", source.id, "-o", str(output)])
    assert result.exit_code == 0
    assert "UNTRUSTED REFERENCE" in output.read_text()
    result = runner.invoke(cli, [*prefix, "export", source.id, "-o", str(output)])
    assert result.exit_code != 0
    pack = tmp_path / "pack.json"
    result = runner.invoke(cli, [
        "learn", "Calculus", "--version", "reviewed", "--source", str(output), "-o", str(pack),
    ])
    assert result.exit_code == 0 and pack.exists()
    assert "https://example.org/calculus" in pack.read_text()
    folder = tmp_path / "docs"
    folder.mkdir()
    result = runner.invoke(cli, [*prefix, "export", source.id, "-o", str(folder / "calculus.txt")])
    assert result.exit_code == 0
    index = tmp_path / "references.sqlite"
    result = runner.invoke(cli, [
        "learn", "Mathematics", "--version", "reviewed", "--source", str(folder), "-o", str(index),
    ])
    assert result.exit_code == 0 and index.exists()


def test_academic_api_is_opt_in_and_uses_existing_auth(tmp_path, echo_config: KumaruConfig):
    echo_config.academic = AcademicConfig(enabled=True, database=str(tmp_path / "api.sqlite"))
    echo_config.server.api_key = "test-only"
    with TestClient(create_app(echo_config)) as client:
        assert client.get("/v1/academic/topics").status_code == 401
        client.headers["Authorization"] = "Bearer test-only"
        assert client.get("/v1/academic/topics").status_code == 200
        response = client.post("/v1/academic/sources", json=candidate().model_dump())
        assert response.status_code == 201
        id_ = response.json()["id"]
        assert client.get(f"/v1/academic/sources/{id_}").json()["scores"][0]["legitimacy"] is None
        assert client.post(f"/v1/academic/sources/{id_}/collect").status_code == 400
        assert client.post(f"/v1/academic/sources/{id_}/review", json={
            "origin": "human_verified",
        }).status_code == 400
        assert client.post("/v1/academic/search", json={"topic_id": "missing"}).status_code == 400
        assert client.get("/v1/academic/sources?access=invalid").status_code == 422
        assert client.post("/v1/academic/sources", json=trusted().model_dump()).status_code == 400
    echo_config.academic.enabled = False
    with TestClient(create_app(echo_config)) as client:
        assert client.get("/v1/academic/topics").status_code == 404


def test_full_api_workflow_without_ready_inference_backend(tmp_path, echo_config, monkeypatch):
    from kumaru.server import routes_academic

    echo_config.academic = AcademicConfig(enabled=True, database=str(tmp_path / "api.sqlite"))
    echo_config.backend.name = "intentionally-unavailable"
    monkeypatch.setattr(routes_academic, "PublicWeb", lambda: FakeWeb())
    monkeypatch.setattr("kumaru.academic.discovery.PublicWeb", lambda: FakeWeb(
        payload=crossref_payload(),
    ))
    with TestClient(create_app(echo_config)) as client:
        search = client.post("/v1/academic/search", json={"topic_id": "calculus", "limit": 1})
        assert search.status_code == 200
        id_ = search.json()["source_ids"][0]
        assert client.post(f"/v1/academic/sources/{id_}/inspect").json()["access"] == "open"
        review = Review(
            origin="human_verified", permission="text_permitted", topic_ids=["calculus"],
            evidence=[evidence("origin"), evidence("permission"), evidence("proof")],
        )
        assert client.post(
            f"/v1/academic/sources/{id_}/review", json=review.model_dump(),
        ).status_code == 200
        collected = client.post(f"/v1/academic/sources/{id_}/collect")
        assert collected.status_code == 200 and "text" not in collected.json()
        assert "Reviewed original" in client.get(f"/v1/academic/sources/{id_}/text").json()["text"]
        assert "text" not in client.get("/v1/academic/sources").json()[0]
        assert client.get(f"/v1/academic/sources/{id_}").json()["scores"][0]["legitimacy"] == 30
        client.post(f"/v1/academic/sources/{id_}/review", json={"permission": "metadata_only"})
        assert client.get(f"/v1/academic/sources/{id_}/text").status_code == 400
        semaphore = client.app.state.academic_network
        semaphore.acquire()
        semaphore.acquire()
        try:
            assert client.post(f"/v1/academic/sources/{id_}/inspect").status_code == 429
        finally:
            semaphore.release()
            semaphore.release()

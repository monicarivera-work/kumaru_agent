"""Open-access copies from OpenAlex and Unpaywall, recorded as metadata and never as approvals."""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from kumaru.academic import discovery, openaccess
from kumaru.academic import web as web_module
from kumaru.academic.catalog import Catalog
from kumaru.academic.discovery import discover
from kumaru.academic.models import Relationship, Review, Source
from kumaru.academic.openaccess import find_open_access, find_open_access_batch
from kumaru.academic.web import (
    FetchError,
    Page,
    ProviderMissing,
    PublicWeb,
    RateLimited,
    TransientFetchError,
)
from kumaru.cli import app as cli
from kumaru.core.config import AcademicConfig
from kumaru.core.errors import RequestError
from kumaru.server.app import create_app

DOI = "10.1007/s00006-010-0242-8"
ARXIV = {"display_name": "arXiv (Cornell University)", "type": "repository"}
PUBLISHER_CLOSED = {
    "is_oa": False, "landing_page_url": f"https://doi.org/{DOI}", "pdf_url": None, "license": None,
    "version": "publishedVersion",
    "source": {"display_name": "Advances in Applied Clifford Algebras", "type": "journal",
               "host_organization_name": "Springer"},
}
# OpenAlex lists one arXiv preprint three ways: abstract page, bare PDF and DataCite DOI.
ARXIV_ABS = {
    "is_oa": True, "landing_page_url": "http://arxiv.org/abs/0809.4526",
    "pdf_url": "https://arxiv.org/pdf/0809.4526", "license": None, "version": "submittedVersion",
    "source": ARXIV,
}
ARXIV_PDF = {
    "is_oa": True, "landing_page_url": "https://arxiv.org/pdf/0809.4526", "pdf_url": None,
    "license": None, "version": "submittedVersion", "source": ARXIV,
}
ARXIV_DOI = {
    "is_oa": True, "landing_page_url": "https://doi.org/10.48550/arxiv.0809.4526", "pdf_url": None,
    "license": None, "version": None, "source": ARXIV,
}
NOT_OA = {
    "is_oa": False, "landing_page_url": "http://citeseerx.ist.psu.edu/viewdoc/summary?doi=1",
    "version": "submittedVersion", "source": None,
}
EPRINT = {
    "is_oa": True, "landing_page_url": "https://eprints.example.ac.uk/123/",
    "pdf_url": "https://eprints.example.ac.uk/123/1/paper.pdf", "license": "CC-BY",
    "version": "acceptedVersion", "source": {"display_name": "Example eprints", "type": "repository"},
}


def work(*locations: dict, status: str = "green", **fields) -> dict:
    payload = {
        "id": "https://openalex.org/W1", "doi": f"https://doi.org/{DOI.upper()}",
        "display_name": "Fundamental Theorem of Calculus", "type": "article",
        "publication_year": 2010,
        "authorships": [{"author": {"display_name": "Garret Sobczyk"}}, {"author": None}],
        "open_access": {"is_oa": status != "closed", "oa_status": status},
        "primary_location": locations[0] if locations else None,
        "locations": list(locations), "cited_by_count": 12, "referenced_works_count": 30,
    }
    payload.update(fields)
    return payload


class Index(PublicWeb):
    """Answers provider requests by URL prefix and records what was sent."""

    def __init__(self, *answers: tuple[str, object]) -> None:
        super().__init__()
        self.answers = list(answers)
        self.calls: list[tuple[str, dict | None]] = []

    def json(self, url, *, headers=None):
        self.calls.append((url, headers))
        for prefix, answer in self.answers:
            if url.startswith(prefix):
                if isinstance(answer, Exception):
                    raise answer
                return answer
        raise AssertionError(f"Unexpected provider request: {url}")


@pytest.fixture
def catalog(tmp_path: Path) -> Catalog:
    return Catalog(tmp_path / "oa.sqlite")


def paywalled(catalog: Catalog, doi: str = DOI, **fields) -> Source:
    values = dict(
        url=f"https://doi.org/{doi}", title="Fundamental Theorem of Calculus", kind="article",
        doi=doi, authors=["Garret Sobczyk"], published_year=2010, topic_ids=["calculus"],
        access="paywall", access_note="Publisher marks full text as not freely accessible.",
    )
    values.update(fields)
    return catalog.discover(Source.model_validate(values))


def lookup(*locations: dict, **fields) -> Index:
    return Index(("https://api.openalex.org/works/doi:", work(*locations, **fields)))


def test_free_copies_are_catalogued_unreviewed_and_linked_from_the_work(catalog):
    source = paywalled(catalog)
    result = find_open_access(catalog, source.id, web=lookup(
        PUBLISHER_CLOSED, ARXIV_ABS, ARXIV_PDF, ARXIV_DOI, NOT_OA, EPRINT,
    ))
    assert result["indexed"] and result["status"] == "green" and result["license"] is None
    assert "free to read, not to reuse" in result["warning"]
    arxiv, eprint = (catalog.get(id_) for id_ in result["copy_ids"])
    # One arXiv copy, upgraded to HTTPS, keeping the PDF link of the bare-PDF listing.
    assert arxiv.url == "https://arxiv.org/abs/0809.4526"
    assert arxiv.open_access.pdf_url == "https://arxiv.org/pdf/0809.4526"
    assert (arxiv.publisher, arxiv.open_access.version, arxiv.open_access.license) == (
        "arXiv (Cornell University)", "submittedVersion", None,
    )
    assert (eprint.open_access.license, eprint.open_access.version) == ("cc-by", "acceptedVersion")
    for copy in (arxiv, eprint):
        assert copy.doi is None and copy.text is None and copy.access == "unknown"
        assert (copy.origin, copy.permission, copy.topic_reviewed) == ("unknown", "unknown", False)
        assert copy.open_access.status is None and copy.open_access.host_type == "repository"
        assert copy.title == source.title and copy.topic_ids == ["calculus"]
    work_ = catalog.get(source.id)
    assert work_.open_access.status == "green" and work_.open_access.version is None
    assert (work_.access, work_.access_note) == (source.access, source.access_note)
    notes = {
        link["target_id"]: link["note"] for link in catalog.graph(source.id)["links"]
        if link["kind"] == "open_copy" and link["source_id"] == source.id
    }
    assert set(notes) == {arxiv.id, eprint.id}
    assert "submitted preprint at arXiv" in notes[arxiv.id] and "can differ" in notes[arxiv.id]
    assert "accepted manuscript" in notes[eprint.id] and "license cc-by" in notes[eprint.id]
    assert catalog.history(source.id) == []


def test_publisher_hosted_access_describes_the_work_without_a_copy(catalog):
    source = paywalled(catalog, access="unknown")
    publisher_open = {
        **PUBLISHER_CLOSED, "is_oa": True, "license": "cc-by",
        "pdf_url": "https://link.springer.com/content/pdf/10.1007/s00006-010-0242-8.pdf",
    }
    # A repository record of the work's own URL is the same page, not a second copy.
    same_page = {**ARXIV_ABS, "landing_page_url": source.url, "pdf_url": None}
    result = find_open_access(catalog, source.id, web=lookup(publisher_open, same_page, status="hybrid"))
    assert result["copy_ids"] == [] and (result["status"], result["license"]) == ("hybrid", "cc-by")
    access = catalog.get(source.id).open_access
    assert (access.host_type, access.version, access.host) == (
        "publisher", "publishedVersion", "Advances in Applied Clifford Algebras",
    )
    assert access.pdf_url.endswith("s00006-010-0242-8.pdf")
    assert "not to reuse" not in result["warning"]


def test_rechecks_are_idempotent_and_keep_reviews_of_existing_copies(catalog):
    source = paywalled(catalog)
    existing = catalog.discover(Source(
        url="https://arxiv.org/abs/0809.4526", title="Reviewed arXiv record",
        topic_ids=["calculus", "integration"],
    ))
    catalog.review(existing.id, Review(topic_ids=["integration"]))
    first = find_open_access(catalog, source.id, web=lookup(ARXIV_ABS, EPRINT))
    second = find_open_access(catalog, source.id, web=lookup(ARXIV_ABS, EPRINT))
    assert first["copy_ids"] == second["copy_ids"] and existing.id in first["copy_ids"]
    kept = catalog.get(existing.id)
    assert (kept.title, kept.topic_ids, kept.topic_reviewed) == ("Reviewed arXiv record", ["integration"], True)
    assert kept.open_access.version == "submittedVersion"
    assert len(catalog.list_sources()) == 3
    assert len([link for link in catalog.graph(source.id)["links"] if link["kind"] == "open_copy"]) == 2


def test_a_copy_that_is_also_a_checked_work_keeps_its_own_status(catalog):
    preprint = paywalled(catalog, doi="10.48550/arxiv.0809.4526", url="https://arxiv.org/abs/0809.4526")
    find_open_access(catalog, preprint.id, web=lookup(status="gold"))
    assert catalog.get(preprint.id).open_access.status == "gold"
    published = paywalled(catalog)
    result = find_open_access(catalog, published.id, web=lookup(ARXIV_ABS))
    assert result["copy_ids"] == [preprint.id]
    kept = catalog.get(preprint.id)
    assert (kept.open_access.status, kept.open_access.version) == ("gold", "submittedVersion")
    assert kept.doi == "10.48550/arxiv.0809.4526"


def test_the_openalex_key_travels_in_a_header_and_is_never_stored(catalog):
    source = paywalled(catalog)
    index = lookup(ARXIV_ABS)
    result = find_open_access(catalog, source.id, web=index, openalex_api_key="secret-key")
    (url, headers), = index.calls
    assert headers == {"Authorization": "Bearer secret-key"} and "secret" not in url
    assert url.startswith(f"https://api.openalex.org/works/doi:{DOI}?select=")
    for id_ in (source.id, *result["copy_ids"]):
        assert "secret" not in catalog.get(id_).model_dump_json()
    keyless = lookup(ARXIV_ABS)
    find_open_access(catalog, source.id, web=keyless)
    assert keyless.calls[0][1] is None


def test_unpaywall_needs_your_email_and_never_stores_it(catalog):
    source = paywalled(catalog)
    index = Index(("https://api.unpaywall.org/v2/", {
        "doi": DOI, "oa_status": "green", "publisher": "Springer",
        "oa_locations": [
            {"url": "https://arxiv.org/abs/0809.4526", "url_for_landing_page": "http://arxiv.org/abs/0809.4526",
             "url_for_pdf": "https://arxiv.org/pdf/0809.4526", "license": None,
             "version": "submittedVersion", "host_type": "repository",
             "repository_institution": "Cornell University - arXiv"},
            {"url": f"https://doi.org/{DOI}", "url_for_landing_page": f"https://doi.org/{DOI}",
             "license": "cc-by-nc", "version": "publishedVersion", "host_type": "publisher"},
        ],
    }))
    with pytest.raises(RequestError, match="contact_email"):
        find_open_access(catalog, source.id, provider="unpaywall", web=index)
    assert index.calls == []
    result = find_open_access(
        catalog, source.id, provider="unpaywall", web=index, contact_email="reader@uni.example",
    )
    (url, headers), = index.calls
    assert url == f"https://api.unpaywall.org/v2/{DOI}?email=reader%40uni.example" and headers is None
    work_ = catalog.get(source.id)
    assert work_.open_access.metadata_url == f"https://api.unpaywall.org/v2/{DOI}"
    assert (work_.open_access.license, work_.open_access.host) == ("cc-by-nc", "Springer")
    copy = catalog.get(result["copy_ids"][0])
    assert (copy.url, copy.publisher, copy.open_access.provider) == (
        "https://arxiv.org/abs/0809.4526", "Cornell University - arXiv", "unpaywall",
    )
    stored = " ".join(item.model_dump_json() for item in catalog.list_sources())
    assert "reader" not in stored and "%40" not in stored


def test_unindexed_dois_are_recorded_and_bad_answers_write_nothing(catalog):
    source = paywalled(catalog)
    result = find_open_access(catalog, source.id, web=Index(
        ("https://api.openalex.org/", ProviderMissing("no such work")),
    ))
    assert (result["indexed"], result["status"], result["copy_ids"]) == (False, None, [])
    assert "does not mean no free copy exists" in result["warning"]
    checked = catalog.get(source.id).open_access
    assert checked.status is None and checked.checked_at
    other = paywalled(catalog, doi="10.5555/other")
    with pytest.raises(FetchError, match="invalid work metadata"):
        find_open_access(catalog, other.id, web=Index(("https://api.openalex.org/", {"results": []})))
    assert catalog.get(other.id).open_access is None and len(catalog.list_sources()) == 2


def test_rate_limits_explain_the_keyless_budget(catalog):
    source = paywalled(catalog)
    limited = Index(("https://api.openalex.org/", RateLimited("Search provider returned HTTP 429")))
    with pytest.raises(RateLimited, match="academic.openalex_api_key") as keyless:
        find_open_access(catalog, source.id, web=limited)
    assert keyless.value.http_status == 503
    with pytest.raises(RateLimited) as keyed:
        find_open_access(catalog, source.id, web=limited, openalex_api_key="k")
    assert "openalex_api_key" not in str(keyed.value)
    assert catalog.get(source.id).open_access is None


@pytest.mark.parametrize(("fields", "message"), [
    (dict(doi=None), "with a DOI"),
    (dict(kind="sources_of_sources"), "individual work"),
    (dict(doi="not a doi"), "valid DOI"),
])
def test_lookups_need_an_individual_work_with_a_valid_doi(catalog, fields, message):
    source = paywalled(catalog, **{"url": "https://example.org/x", **fields})
    with pytest.raises(RequestError, match=message):
        find_open_access(catalog, source.id, web=Index())
    with pytest.raises(RequestError, match="openalex or unpaywall"):
        find_open_access(catalog, source.id, provider="scihub", web=Index())  # type: ignore[arg-type]


def test_openalex_discovery_catalogues_works_their_status_and_free_copies(catalog):
    index = Index(("https://api.openalex.org/works?", {"results": [
        work(PUBLISHER_CLOSED, ARXIV_ABS, EPRINT),
        work(ARXIV_ABS, display_name="  ", id="https://openalex.org/W2"),
        work(EPRINT, doi=None, primary_location=None, id="https://openalex.org/W3"),
        work(EPRINT, publication_year=0, doi="https://doi.org/10.5555/y", id="https://openalex.org/W4"),
    ]}))
    result = discover(
        catalog, "calculus", query="fundamental theorem", provider="openalex",
        open_access_only=True, web=index, openalex_api_key="openalex-secret",
    )
    (url, headers), = index.calls
    assert "filter=is_oa%3Atrue" in url and "select=" in url and "search=Mathematics" in url
    assert headers == {"Authorization": "Bearer openalex-secret"} and "secret" not in url
    assert result["skipped"] == 2 and len(result["source_ids"]) == 2 and len(result["copy_ids"]) == 2
    first = catalog.get(result["source_ids"][0])
    assert (first.doi, first.url, first.kind) == (DOI, f"https://doi.org/{DOI}", "article")
    assert (first.citation_count, first.reference_count, first.authors) == (12, 30, ["Garret Sobczyk"])
    assert (first.publisher, first.open_access.status) == ("Springer", "green")
    assert catalog.get(result["source_ids"][1]).published_year is None
    index_source = next(item for item in catalog.list_sources(containers=True))
    assert (index_source.title, index_source.access) == ("OpenAlex scholarly index", "open")
    note = next(link["note"] for link in catalog.graph(index_source.id)["links"])
    assert note.startswith("Indexed in OpenAlex;")
    # Crossref rediscovery refreshes counts but leaves the recorded open access alone.
    discover(catalog, "calculus", web=Index(("https://api.crossref.org/", {"status": "ok", "message": {
        "items": [{"title": ["Fundamental"], "DOI": DOI, "is-referenced-by-count": 13}],
    }})))
    assert catalog.get(first.id).open_access.status == "green"
    assert catalog.get(first.id).citation_count == 13
    with pytest.raises(RequestError, match="Only OpenAlex"):
        discover(catalog, "calculus", provider="crossref", open_access_only=True, web=Index())


def test_batches_check_unchecked_dois_first_and_stop_on_outages(catalog):
    never = paywalled(catalog, doi="10.5555/never")
    stale = paywalled(catalog, doi="10.5555/stale")
    find_open_access(catalog, stale.id, web=lookup(status="closed"))
    paywalled(catalog, doi=None, url="https://example.org/no-doi")
    paywalled(catalog, doi="10.5555/container", kind="sources_of_sources", url="https://example.org/c")
    copy = paywalled(catalog, doi="10.48550/arxiv.1", url="https://arxiv.org/abs/1")
    catalog.link(Relationship(source_id=never.id, target_id=copy.id, kind="open_copy", note="Free copy"))
    assert [item.id for item in catalog.open_access_due()] == [never.id, stale.id]
    assert [item.id for item in catalog.open_access_due(topic_id="algebra")] == []
    busy = Index(("https://api.openalex.org/", TransientFetchError("Search provider returned HTTP 503")))
    outcome = find_open_access_batch(catalog, web=busy)
    assert outcome["results"] == [] and len(busy.calls) == 1 and "503" in outcome["stopped"]
    assert outcome["errors"] == [{"source_id": never.id, "error": "Search provider returned HTTP 503"}]
    done = find_open_access_batch(catalog, topic_id="calculus", web=lookup(ARXIV_ABS))
    assert [item["source_id"] for item in done["results"]] == [never.id, stale.id]
    assert done["errors"] == [] and done["stopped"] is None
    for limit in (0, 21):
        with pytest.raises(RequestError, match="1..20"):
            find_open_access_batch(catalog, limit=limit, web=Index())
    with pytest.raises(RequestError, match="contact_email"):
        find_open_access_batch(catalog, provider="unpaywall", web=Index())


@contextmanager
def api(tmp_path: Path, echo_config, monkeypatch, index: Index, **academic):
    echo_config.academic = AcademicConfig(enabled=True, database=str(tmp_path / "api.sqlite"), **academic)
    monkeypatch.setattr(openaccess, "PublicWeb", lambda: index)
    monkeypatch.setattr(discovery, "PublicWeb", lambda: index)
    with TestClient(create_app(echo_config)) as client:
        yield client


def test_open_access_api(tmp_path, echo_config, monkeypatch):
    index = lookup(ARXIV_ABS)
    with api(tmp_path, echo_config, monkeypatch, index) as client:
        flags = client.get("/v1/academic/overview").json()
        assert (flags["openalex_key_configured"], flags["unpaywall_configured"]) == (False, False)
        source = paywalled(client.app.state.academic)
        body = client.post(f"/v1/academic/sources/{source.id}/open-access").json()
        assert body["status"] == "green" and len(body["copy_ids"]) == 1
        response = client.post(f"/v1/academic/sources/{source.id}/open-access?provider=unpaywall")
        assert response.status_code == 400 and "contact_email" in response.json()["error"]["message"]
        batch = client.post("/v1/academic/open-access?topic_id=calculus&limit=5").json()
        assert [item["source_id"] for item in batch["results"]] == [source.id]
        assert client.post("/v1/academic/open-access?limit=21").status_code == 422
        library = client.get("/v1/academic/library").json()["items"]
        flags_by_id = {item["id"]: (item["free_copies"], item["is_free_copy"]) for item in library}
        assert flags_by_id[source.id] == (1, False) and flags_by_id[body["copy_ids"][0]] == (0, True)
        refused = client.post("/v1/academic/search", json={
            "topic_id": "calculus", "provider": "crossref", "open_access_only": True,
        })
        assert refused.status_code == 400
    with api(tmp_path, echo_config, monkeypatch, index, openalex_api_key="k",
             contact_email="reader@uni.example") as client:
        flags = client.get("/v1/academic/overview").json()
        assert (flags["openalex_key_configured"], flags["unpaywall_configured"]) == (True, True)


def test_open_access_cli(tmp_path, monkeypatch):
    database = tmp_path / "cli.sqlite"
    source = paywalled(Catalog(database))
    index = lookup(ARXIV_ABS)
    monkeypatch.setattr(openaccess, "PublicWeb", lambda: index)
    monkeypatch.setattr(discovery, "PublicWeb", lambda: Index(
        ("https://api.openalex.org/works?", {"results": [work(EPRINT)]}),
    ))
    runner = CliRunner()
    prefix = ["academic", "--database", str(database)]
    result = runner.invoke(cli, [*prefix, "open-access", source.id])
    assert result.exit_code == 0 and json.loads(result.output)["status"] == "green"
    result = runner.invoke(cli, [*prefix, "open-access-batch", "--topic", "calculus", "--limit", "3"])
    assert result.exit_code == 0 and json.loads(result.output)["errors"] == []
    result = runner.invoke(cli, [*prefix, "open-access", source.id, "--provider", "scihub"])
    assert result.exit_code != 0
    result = runner.invoke(cli, [*prefix, "search", "calculus", "--provider", "openalex", "--open-access-only"])
    assert result.exit_code == 0 and len(json.loads(result.output)["copy_ids"]) == 1
    result = runner.invoke(cli, [*prefix, "search", "calculus", "--open-access-only"])
    assert result.exit_code != 0


def test_a_short_retry_after_is_honoured_once(monkeypatch):
    slept: list[float] = []
    monkeypatch.setattr(web_module.time, "sleep", slept.append)
    answers: list[Page] = []

    class Provider(PublicWeb):
        def fetch(self, url, *, robots=True, headers=None):
            return answers.pop(0)

    answers[:] = [Page("https://api.example.org/", 429, "application/json", b"", retry_after=14),
                  Page("https://api.example.org/", 200, "application/json", b'{"ok": true}')]
    assert Provider().json("https://api.example.org/") == {"ok": True} and slept == [14]
    answers[:] = [Page("https://api.example.org/", 429, "application/json", b"", retry_after=3)] * 2
    with pytest.raises(RateLimited):
        Provider().json("https://api.example.org/")
    answers[:] = [Page("https://api.example.org/", 429, "application/json", b"", retry_after=600)]
    with pytest.raises(RateLimited):
        Provider().json("https://api.example.org/")
    assert slept == [14, 3]


@pytest.mark.parametrize(("value", "expected"), [("14", 14.0), ("Wed, 21 Oct 2026 07:28:00 GMT", None), ("", None)])
def test_transport_reads_numeric_retry_after(monkeypatch, value, expected):
    monkeypatch.setattr(web_module, "public_addresses", lambda *args: ["93.184.216.34"])

    class Client:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        @contextmanager
        def stream(self, method, url, **kwargs):
            yield httpx.Response(429, content=b"{}", headers={"retry-after": value} if value else {})

    monkeypatch.setattr(httpx, "Client", Client)
    assert PublicWeb().fetch("https://api.example.org/", robots=False).retry_after == expected

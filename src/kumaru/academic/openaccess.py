"""Legal open-access copies of catalogued works, as OpenAlex or Unpaywall report them.

An index says where a work can be read free and under which license. That is
metadata for a reviewer, never an approval: a license does not approve collecting
text, and a copy that is free to read without one ("bronze") grants no reuse.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
from urllib.parse import quote, urlencode, urlsplit

from pydantic import BaseModel, Field, ValidationError

from kumaru.academic.catalog import Catalog
from kumaru.academic.models import (
    CopyVersion,
    OAProvider,
    OAStatus,
    OpenAccess,
    Relationship,
    Source,
    canonical_url,
    normalized_doi,
    now,
)
from kumaru.academic.web import (
    FetchError,
    ProviderMissing,
    PublicWeb,
    RateLimited,
    TransientFetchError,
    https_upgrade,
)
from kumaru.core.errors import KumaruError, RequestError

OPENALEX = "https://api.openalex.org"
UNPAYWALL = "https://api.unpaywall.org/v2"
# Only what the catalogue uses. DOI lookups are free; searches draw on a daily budget.
WORK_FIELDS = ",".join((
    "id", "doi", "display_name", "type", "publication_year", "authorships", "primary_location",
    "locations", "open_access", "cited_by_count", "referenced_works_count",
))
BOOK_TYPES = {"book", "book-chapter", "monograph", "edited-book", "reference-book", "reference-entry"}
PUBLISHER_HOSTS = {"journal", "conference", "ebook platform", "book series"}
PROVIDERS: dict[OAProvider, str] = {"openalex": "OpenAlex", "unpaywall": "Unpaywall"}
STATUSES: dict[str, OAStatus] = {
    "gold": "gold", "diamond": "diamond", "hybrid": "hybrid", "bronze": "bronze",
    "green": "green", "closed": "closed",
}
VERSIONS: dict[str, CopyVersion] = {
    "submittedVersion": "submittedVersion", "acceptedVersion": "acceptedVersion",
    "publishedVersion": "publishedVersion",
}
VERSION_LABELS = {
    "publishedVersion": "published version", "acceptedVersion": "accepted manuscript",
    "submittedVersion": "submitted preprint",
}


class OpenAlexHost(BaseModel):
    display_name: str | None = None
    type: str | None = None
    host_organization_name: str | None = None


class OpenAlexLocation(BaseModel):
    is_oa: bool | None = None
    landing_page_url: str | None = None
    pdf_url: str | None = None
    license: str | None = None
    version: str | None = None
    source: OpenAlexHost | None = None


class OpenAlexAccess(BaseModel):
    oa_status: str | None = None


class OpenAlexAuthor(BaseModel):
    display_name: str | None = None


class OpenAlexAuthorship(BaseModel):
    author: OpenAlexAuthor | None = None


class OpenAlexWork(BaseModel):
    id: str
    doi: str | None = None
    display_name: str | None = None
    type: str | None = None
    publication_year: int | None = None
    authorships: list[OpenAlexAuthorship] | None = None
    primary_location: OpenAlexLocation | None = None
    locations: list[OpenAlexLocation] | None = None
    open_access: OpenAlexAccess | None = None
    cited_by_count: int | None = Field(default=None, ge=0)
    referenced_works_count: int | None = Field(default=None, ge=0)


class OpenAlexResults(BaseModel):
    results: list[OpenAlexWork]


class UnpaywallLocation(BaseModel):
    url: str | None = None
    url_for_landing_page: str | None = None
    url_for_pdf: str | None = None
    license: str | None = None
    version: str | None = None
    host_type: str | None = None
    repository_institution: str | None = None


class UnpaywallRecord(BaseModel):
    doi: str
    oa_status: str | None = None
    publisher: str | None = None
    oa_locations: list[UnpaywallLocation] | None = None


@dataclass(frozen=True)
class Location:
    """One free location of a work, normalised across providers."""

    url: str
    pdf_url: str | None
    license: str | None
    version: CopyVersion | None
    host_type: Literal["publisher", "repository"] | None
    host: str | None

    def fields(self) -> dict:
        return dict(
            license=self.license, version=self.version, host_type=self.host_type,
            host=self.host, pdf_url=self.pdf_url,
        )


def _url(value: str | None) -> str | None:
    """A canonical URL, HTTPS-first: indexes still list many plain-HTTP repository links."""
    if not value:
        return None
    try:
        url = canonical_url(value)
    except ValueError:
        return None
    return https_upgrade(url) or url


def _text(value: str | None, limit: int) -> str | None:
    return " ".join((value or "").split())[:limit] or None


def _status(value: str | None) -> OAStatus | None:
    return STATUSES.get((value or "").casefold())


def _openalex_location(item: OpenAlexLocation | None) -> Location | None:
    if item is None or not item.is_oa:
        return None
    pdf_url = _url(item.pdf_url)
    url = _url(item.landing_page_url) or pdf_url
    if url is None:
        return None
    host = item.source
    host_type: Literal["publisher", "repository"] | None = None
    if host is not None and host.type == "repository":
        host_type = "repository"
    elif host is not None and host.type in PUBLISHER_HOSTS:
        host_type = "publisher"
    return Location(
        url=url, pdf_url=pdf_url, license=(_text(item.license, 100) or "").casefold() or None,
        version=VERSIONS.get(item.version or ""), host_type=host_type,
        host=_text(host.display_name or host.host_organization_name, 500) if host else None,
    )


def _unpaywall_location(item: UnpaywallLocation, publisher: str | None) -> Location | None:
    pdf_url = _url(item.url_for_pdf)
    url = _url(item.url_for_landing_page) or _url(item.url) or pdf_url
    if url is None:
        return None
    host_type: Literal["publisher", "repository"] | None = (
        "publisher" if item.host_type == "publisher"
        else "repository" if item.host_type == "repository" else None
    )
    host = (
        _text(publisher, 500) if host_type == "publisher"
        else _text(item.repository_institution, 500) or urlsplit(url).hostname
    )
    return Location(
        url=url, pdf_url=pdf_url, license=(_text(item.license, 100) or "").casefold() or None,
        version=VERSIONS.get(item.version or ""), host_type=host_type, host=host,
    )


def openalex_headers(api_key: str | None) -> dict[str, str] | None:
    """A bearer header keeps the key out of URLs, which sources store and display."""
    return {"Authorization": f"Bearer {api_key}"} if api_key else None


def openalex_json(web: PublicWeb, url: str, api_key: str | None) -> dict:
    try:
        return web.json(url, headers=openalex_headers(api_key))
    except RateLimited as exc:
        hint = "" if api_key else (
            " Without academic.openalex_api_key, everyone on this network shares one small"
            " daily budget; a free key gives you your own."
        )
        raise RateLimited(
            f"OpenAlex is limiting requests or today's budget is spent (HTTP 429).{hint}",
        ) from exc


def openalex_locations(work: OpenAlexWork) -> list[Location]:
    return [found for item in work.locations or [] if (found := _openalex_location(item))]


def openalex_candidate(
    work: OpenAlexWork, topic_id: str, metadata_url: str,
) -> tuple[Source, OAStatus | None, list[Location]] | None:
    """A search result as an unreviewed catalogue candidate, with its free locations."""
    title = _text(work.display_name, 2000)
    doi = None
    if work.doi:
        try:
            doi = normalized_doi(work.doi)
        except RequestError:
            doi = None
    primary = work.primary_location
    url = f"https://doi.org/{doi}" if doi else _url(primary.landing_page_url if primary else None)
    if not title or not url:
        return None
    host = primary.source if primary else None
    year = work.publication_year
    source = Source(
        url=url, title=title, kind="book" if work.type in BOOK_TYPES else "article", doi=doi,
        publisher=_text(host.host_organization_name or host.display_name, 500) if host else None,
        published_year=year if year is not None and 1 <= year <= 9999 else None,
        authors=[
            item.author.display_name for item in work.authorships or []
            if item.author and item.author.display_name
        ][:500],
        citation_count=work.cited_by_count, reference_count=work.referenced_works_count,
        metadata_url=metadata_url, topic_ids=[topic_id],
    )
    return source, _status(work.open_access.oa_status if work.open_access else None), openalex_locations(work)


def _lookup(
    provider: OAProvider, web: PublicWeb, doi: str, *, api_key: str | None, email: str | None,
) -> tuple[bool, OAStatus | None, list[Location], str]:
    """(indexed, status, free locations, the request URL safe to store)."""
    path = quote(doi, safe="/")
    if provider == "openalex":
        url = f"{OPENALEX}/works/doi:{path}?" + urlencode({"select": WORK_FIELDS})
        try:
            payload = openalex_json(web, url, api_key)
        except ProviderMissing:
            return False, None, [], url
        try:
            work = OpenAlexWork.model_validate(payload)
        except ValidationError as exc:
            raise FetchError("OpenAlex returned invalid work metadata") from exc
        status = _status(work.open_access.oa_status if work.open_access else None)
        return True, status, openalex_locations(work), url
    if not email:
        raise RequestError("Unpaywall needs academic.contact_email, your own address, in configuration")
    # Unpaywall requires the address in the query; the stored URL leaves it out.
    stored = f"{UNPAYWALL}/{path}"
    try:
        payload = web.json(stored + "?" + urlencode({"email": email}))
    except ProviderMissing:
        return False, None, [], stored
    try:
        record = UnpaywallRecord.model_validate(payload)
    except ValidationError as exc:
        raise FetchError("Unpaywall returned invalid open-access metadata") from exc
    locations = [
        found for item in record.oa_locations or []
        if (found := _unpaywall_location(item, record.publisher))
    ]
    return True, _status(record.oa_status), locations, stored


def _copy_note(label: str, location: Location) -> str:
    version = VERSION_LABELS.get(location.version or "", "copy (version not stated)")
    where = f" at {location.host}" if location.host else ""
    terms = (
        f"license {location.license}" if location.license
        else "no license stated, so it is free to read but not to reuse"
    )
    caution = (
        " A preprint can differ from the peer-reviewed publication."
        if location.version == "submittedVersion" else ""
    )
    return f"{label} lists a free {version}{where}; {terms}.{caution}"


def _distinct(locations: list[Location]) -> list[Location]:
    """One location per host and version, in the index's order.

    Indexes list one arXiv preprint as its abstract page, as its PDF and as its
    DataCite DOI. A location that is only another one's PDF is dropped, and a
    host's version-less entries add nothing once it has a versioned one.
    """
    pdfs = {item.pdf_url for item in locations if item.pdf_url}
    kept: dict[object, Location] = {}
    for item in locations:
        if item.pdf_url is None and item.url in pdfs:
            continue
        key = (item.host.casefold(), item.version) if item.host else item.url
        current = kept.get(key)
        if current is None or _richness(item) > _richness(current):
            kept[key] = item
    versioned = {item.host.casefold() for item in kept.values() if item.host and item.version}
    return [
        item for item in kept.values()
        if not (item.host and item.version is None and item.host.casefold() in versioned)
    ]


def _richness(item: Location) -> int:
    return (
        (item.pdf_url is not None) + (item.license is not None)
        + (not item.url.startswith("https://doi.org/"))
    )


def record_open_access(
    catalog: Catalog, source: Source, *, provider: OAProvider, status: OAStatus | None,
    locations: list[Location], metadata_url: str,
) -> dict:
    """Record the work's own access, and catalogue each free copy hosted elsewhere.

    A publisher-hosted location is the work itself, so it only describes the
    work. Every other location becomes an unreviewed source linked from the work
    by `open_copy`. Copies never carry the work's DOI: the catalogue merges
    sources that share a DOI, which would fold the copy back into the work.
    """
    label = PROVIDERS[provider]
    checked_at = now()
    locations = _distinct(locations)
    work_urls = {source.url}
    try:
        if source.doi:
            work_urls.add(canonical_url(f"https://doi.org/{normalized_doi(source.doi)}"))
    except (RequestError, ValueError):
        pass
    own = [item for item in locations if item.host_type == "publisher" or item.url in work_urls]
    own.sort(key=lambda item: item.host_type != "publisher")
    copies: dict[str, Location] = {}
    for item in locations:
        if item not in own:
            copies.setdefault(item.url, item)
    def reported(location: Location | None, *, copy: bool = False) -> OpenAccess:
        # The status describes the work, so a copy leaves it to the work it copies.
        return OpenAccess(
            status=None if copy else status, provider=provider, metadata_url=metadata_url,
            checked_at=checked_at, **(location.fields() if location else {}),
        )

    work_access = reported(own[0] if own else None)
    # Validate every record before writing any of them.
    candidates = [
        (Source(
            url=item.url, title=source.title, kind=source.kind, authors=source.authors,
            publisher=item.host, published_year=source.published_year,
            access_note=f"{label} lists this as a free copy; inspect it to confirm it is reachable.",
            topic_ids=source.topic_ids, metadata_url=metadata_url,
            open_access=reported(item, copy=True),
        ), _copy_note(label, item))
        for item in copies.values()
    ]
    catalog.save_metadata(source.id, {"open_access": work_access.model_dump()})
    copy_ids = []
    for candidate, note in candidates:
        try:
            existing = catalog.get(candidate.id)
        except RequestError:
            existing = None
        if existing and existing.open_access and existing.open_access.status and candidate.open_access:
            # The copy is also a work checked in its own right; keep its status.
            candidate = candidate.model_copy(update={"open_access": candidate.open_access.model_copy(
                update={"status": existing.open_access.status},
            )})
        copy = catalog.discover(candidate)
        if copy.id != source.id:
            catalog.link(Relationship(source_id=source.id, target_id=copy.id, kind="open_copy", note=note))
            copy_ids.append(copy.id)
    return dict(
        status=status, license=work_access.license, copy_ids=copy_ids,
        unlicensed=any(item.license is None for item in [*own[:1], *copies.values()]),
    )


def find_open_access(
    catalog: Catalog, source_id: str, *, provider: OAProvider = "openalex",
    web: PublicWeb | None = None, openalex_api_key: str | None = None,
    contact_email: str | None = None,
) -> dict:
    """Ask one index for free copies of a DOI work; catalogue them as unreviewed sources."""
    if provider not in PROVIDERS:
        raise RequestError("Open-access provider must be openalex or unpaywall")
    source = catalog.get(source_id)
    if not source.doi or source.kind == "sources_of_sources":
        raise RequestError("Open-access lookups need an individual work with a DOI")
    label = PROVIDERS[provider]
    indexed, status, locations, metadata_url = _lookup(
        provider, web or PublicWeb(), normalized_doi(source.doi),
        api_key=openalex_api_key, email=contact_email,
    )
    if not indexed:
        catalog.save_metadata(source_id, {"open_access": OpenAccess(
            provider=provider, metadata_url=metadata_url,
        ).model_dump()})
        return dict(
            source_id=source_id, provider=provider, indexed=False, status=None, license=None,
            copy_ids=[], warning=f"{label} has no record of this DOI. That does not mean no free copy exists.",
        )
    recorded = record_open_access(
        catalog, source, provider=provider, status=status, locations=locations,
        metadata_url=metadata_url,
    )
    warning = (
        "Indexes can be wrong or out of date: confirm each copy's license and version"
        " on the page itself before approving text collection."
    )
    if recorded["unlicensed"]:
        warning += " A copy with no license is free to read, not to reuse."
    return dict(
        source_id=source_id, provider=provider, indexed=True, status=status,
        license=recorded["license"], copy_ids=recorded["copy_ids"], warning=warning,
    )


def find_open_access_batch(
    catalog: Catalog, *, topic_id: str | None = None, limit: int = 10,
    provider: OAProvider = "openalex", web: PublicWeb | None = None,
    openalex_api_key: str | None = None, contact_email: str | None = None,
) -> dict:
    """Check DOI works never checked first, then the least recently checked."""
    if not 1 <= limit <= 20:
        raise RequestError("Open-access batches check 1..20 works")
    if provider == "unpaywall" and not contact_email:
        raise RequestError("Unpaywall needs academic.contact_email, your own address, in configuration")
    web = web or PublicWeb()
    results: list[dict] = []
    errors: list[dict] = []
    stopped = None
    for source in catalog.open_access_due(topic_id=topic_id, limit=limit):
        try:
            results.append(find_open_access(
                catalog, source.id, provider=provider, web=web,
                openalex_api_key=openalex_api_key, contact_email=contact_email,
            ))
        except TransientFetchError as exc:
            # An outage or a spent budget would fail every remaining lookup too.
            errors.append(dict(source_id=source.id, error=str(exc)))
            stopped = str(exc)
            break
        except KumaruError as exc:
            errors.append(dict(source_id=source.id, error=str(exc)))
    return dict(results=results, errors=errors, stopped=stopped)

"""Contextual scholarly discovery (Crossref or OpenAlex), with opt-in Brave broad-web search."""

from __future__ import annotations

from typing import Literal
from urllib.parse import urlencode

from pydantic import BaseModel, Field, ValidationError

from kumaru.academic.catalog import Catalog
from kumaru.academic.models import OAStatus, Relationship, Source
from kumaru.academic.openaccess import (
    OPENALEX,
    WORK_FIELDS,
    Location,
    OpenAlexResults,
    openalex_candidate,
    openalex_json,
    record_open_access,
)
from kumaru.academic.web import FetchError, PublicWeb
from kumaru.core.errors import RequestError

Provider = Literal["crossref", "brave", "openalex"]
INDEXES = {"crossref": "Crossref", "openalex": "OpenAlex"}


class Author(BaseModel):
    given: str = ""
    family: str = ""
    name: str = ""


class DateParts(BaseModel):
    date_parts: list[list[int]] = Field(default_factory=list, alias="date-parts")


class Work(BaseModel):
    title: list[str] = Field(default_factory=list)
    url: str | None = Field(default=None, alias="URL")
    doi: str | None = Field(default=None, alias="DOI")
    type: str = ""
    author: list[Author] = Field(default_factory=list)
    publisher: str | None = None
    published: DateParts | None = None
    citation_count: int | None = Field(default=None, alias="is-referenced-by-count", ge=0)
    reference_count: int | None = Field(default=None, alias="reference-count", ge=0)


class CrossrefMessage(BaseModel):
    items: list[Work]


class CrossrefResult(BaseModel):
    status: Literal["ok"]
    message: CrossrefMessage


class WebResult(BaseModel):
    title: str
    url: str


class WebResults(BaseModel):
    results: list[WebResult] = Field(default_factory=list)


class BraveResult(BaseModel):
    web: WebResults


def discover(
    catalog: Catalog, topic_id: str, *, query: str = "", provider: Provider = "crossref",
    limit: int = 20, brave_api_key: str | None = None, web: PublicWeb | None = None,
    openalex_api_key: str | None = None, open_access_only: bool = False,
) -> dict:
    if not 1 <= limit <= 20 or len(query) > 500:
        raise RequestError("Search requires limit 1..20 and query at most 500 characters")
    if open_access_only and provider != "openalex":
        raise RequestError("Only OpenAlex discovery can keep just the works with a free copy")
    path = catalog.topic_path(topic_id)
    # Include the area and immediate parent to disambiguate e.g. groups in
    # algebra from social groups, without constraining every intervening heading.
    context = list(dict.fromkeys([path[0].name, *[t.name for t in path[-2:]]]))
    terms = " ".join([*context, query.strip()]).strip()
    web = web or PublicWeb()
    candidates: list[Source] = []
    free: dict[str, tuple[OAStatus | None, list[Location]]] = {}
    skipped = 0
    if provider == "crossref":
        metadata_url = "https://api.crossref.org/works?" + urlencode({
            "query.bibliographic": terms, "rows": limit,
        })
        try:
            result = CrossrefResult.model_validate(web.json(metadata_url))
            for work in result.message.items:
                title = next((text.strip() for text in work.title if text.strip()), "")
                # Components and peer-review records often lack these; they are
                # not candidates, but they must not discard the valid works.
                if not title or not (work.url or work.doi):
                    skipped += 1
                    continue
                kind: Literal["book", "article"] = (
                    "book" if work.type in {"book", "monograph", "book-chapter", "textbook"} else "article"
                )
                year = None
                if work.published and work.published.date_parts and work.published.date_parts[0]:
                    year = work.published.date_parts[0][0]
                candidates.append(Source(
                    url=work.url or f"https://doi.org/{work.doi}",
                    title=title[:2000], kind=kind,
                    doi=work.doi, publisher=work.publisher, published_year=year,
                    authors=[
                        author.name or " ".join(p for p in [author.given, author.family] if p)
                        for author in work.author
                    ],
                    citation_count=work.citation_count, reference_count=work.reference_count,
                    metadata_url=metadata_url, topic_ids=[topic_id],
                ))
        except ValidationError as exc:
            raise FetchError("Crossref returned invalid scholarly metadata") from exc
        container = Source(
            url="https://www.crossref.org/", title="Crossref scholarly metadata index",
            kind="sources_of_sources", access="open",
            access_note="Public metadata index; full-text access varies by individual work.",
            topic_ids=[path[0].id], metadata_url=metadata_url,
        )
    elif provider == "brave":
        if not brave_api_key:
            raise RequestError("Brave web search needs academic.brave_api_key in configuration")
        metadata_url = "https://api.search.brave.com/res/v1/web/search?" + urlencode({
            "q": terms, "count": limit,
        })
        try:
            result_web = BraveResult.model_validate(web.json(
                metadata_url, headers={"X-Subscription-Token": brave_api_key},
            ))
            candidates = [
                Source(
                    url=item.url, title=item.title[:2000], topic_ids=[topic_id],
                    metadata_url=metadata_url,
                )
                for item in result_web.web.results[:limit]
            ]
        except ValidationError as exc:
            raise FetchError("Brave returned invalid search metadata") from exc
        container = None
    elif provider == "openalex":
        params: dict[str, str | int] = {"search": terms, "per_page": limit, "select": WORK_FIELDS}
        if open_access_only:
            params["filter"] = "is_oa:true"
        metadata_url = f"{OPENALEX}/works?" + urlencode(params)
        try:
            found = OpenAlexResults.model_validate(openalex_json(web, metadata_url, openalex_api_key))
        except ValidationError as exc:
            raise FetchError("OpenAlex returned invalid scholarly metadata") from exc
        for item in found.results:
            try:
                parsed = openalex_candidate(item, topic_id, metadata_url)
            except ValidationError:
                parsed = None
            if parsed is None:
                skipped += 1
                continue
            candidate, status, locations = parsed
            candidates.append(candidate)
            free[candidate.id] = (status, locations)
        container = Source(
            url="https://openalex.org/", title="OpenAlex scholarly index",
            kind="sources_of_sources", access="open",
            access_note="Public CC0 metadata index; full-text access varies by individual work.",
            topic_ids=[path[0].id], metadata_url=metadata_url,
        )
    else:
        raise RequestError(f"Unknown academic search provider: {provider}")
    # Provider errors are validated before writing anything. All candidates
    # remain unreviewed, regardless of domain, publisher or search rank.
    parent = catalog.discover(container) if container else None
    sources = []
    copies: list[str] = []
    for candidate in candidates:
        source = catalog.discover(candidate)
        if parent:
            catalog.link(Relationship(
                source_id=parent.id, target_id=source.id, kind="contains",
                note=(
                    f"Indexed in {INDEXES[provider]}; membership is not peer review or"
                    " full-text ownership."
                ),
            ))
        if candidate.id in free:
            status, locations = free[candidate.id]
            copies.extend(record_open_access(
                catalog, source, provider="openalex", status=status, locations=locations,
                metadata_url=metadata_url,
            )["copy_ids"])
        if source.id not in sources:
            sources.append(source.id)
    outcome = dict(
        provider=provider, query=terms, topic_id=topic_id, source_ids=sources, skipped=skipped,
    )
    if provider == "openalex":
        outcome["copy_ids"] = list(dict.fromkeys(copies))
    return outcome

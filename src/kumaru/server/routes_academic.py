"""Opt-in, authenticated academic catalogue API; no inference backend required."""

from __future__ import annotations

from collections.abc import Iterator
from threading import BoundedSemaphore

from fastapi import APIRouter, Depends, Query, Request
from pydantic import Field

from kumaru.academic.catalog import Catalog
from kumaru.academic.discovery import Provider, discover
from kumaru.academic.models import Access, Record, Relationship, Review, Source, Topic
from kumaru.academic.web import PublicWeb, collect_source, inspect_source
from kumaru.core.errors import RequestError
from kumaru.server.routes_chat import require_auth

router = APIRouter(
    prefix="/v1/academic", tags=["academic"], dependencies=[Depends(require_auth)],
)


class Search(Record):
    topic_id: str
    query: str = Field(default="", max_length=500)
    provider: Provider = "crossref"
    limit: int = Field(default=20, ge=1, le=20)


class BusyError(RequestError):
    http_status = 429


def network_slot(request: Request) -> Iterator[None]:
    semaphore: BoundedSemaphore = request.app.state.academic_network
    if not semaphore.acquire(blocking=False):
        raise BusyError("Academic network slots are busy; retry later")
    try:
        yield
    finally:
        semaphore.release()


def catalog(request: Request) -> Catalog:
    return request.app.state.academic


@router.get("/topics")
def topics(request: Request) -> list[Topic]:
    return catalog(request).topics()


@router.put("/topics/{topic_id}")
def put_topic(topic_id: str, body: Topic, request: Request) -> Topic:
    if body.id != topic_id:
        raise RequestError("Topic body ID must match the URL")
    return catalog(request).put_topic(body)


@router.post("/search", dependencies=[Depends(network_slot)])
def search(body: Search, request: Request) -> dict:
    return discover(
        catalog(request), body.topic_id, query=body.query, provider=body.provider,
        limit=body.limit,
        brave_api_key=request.app.state.kumaru.config.academic.brave_api_key,
    )


@router.get("/sources")
def sources(
    request: Request,
    topic_id: str | None = None,
    access: Access | None = None,
    containers: bool | None = None,
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
) -> list[dict]:
    return [
        source.model_dump(exclude={"text"})
        for source in catalog(request).list_sources(
            topic_id=topic_id, access=access, containers=containers, limit=limit, offset=offset,
        )
    ]


@router.post("/sources", status_code=201)
def add(body: Source, request: Request) -> dict:
    if (
        body.origin != "unknown" or body.permission != "unknown" or body.text
        or body.evidence or body.topic_reviewed
    ):
        raise RequestError("Add metadata only; use the review endpoint for human decisions")
    return catalog(request).discover(body).model_dump(exclude={"text"})


@router.get("/sources/{source_id}")
def source(source_id: str, request: Request) -> dict:
    db = catalog(request)
    return dict(
        source=db.get(source_id).model_dump(exclude={"text"}),
        scores=db.scores(source_id), graph=db.graph(source_id), reviews=db.history(source_id),
    )


@router.post("/sources/{source_id}/review")
def review(source_id: str, body: Review, request: Request) -> dict:
    return catalog(request).review(source_id, body).model_dump(exclude={"text"})


@router.post("/relationships", status_code=201)
def relationship(body: Relationship, request: Request) -> Relationship:
    return catalog(request).link(body)


@router.post("/sources/{source_id}/inspect", dependencies=[Depends(network_slot)])
def inspect(source_id: str, request: Request) -> dict:
    return inspect_source(catalog(request), source_id, PublicWeb()).model_dump(exclude={"text"})


@router.post("/sources/{source_id}/collect", dependencies=[Depends(network_slot)])
def collect(source_id: str, request: Request) -> dict:
    return collect_source(catalog(request), source_id, PublicWeb()).model_dump(exclude={"text"})


@router.get("/sources/{source_id}/text")
def text(source_id: str, request: Request) -> dict:
    record = catalog(request).get(source_id)
    if not record.text or not record.topic_reviewed:
        raise RequestError("Source has no reviewed collected text")
    return dict(
        source_id=record.id, url=record.text_url, sha256=record.text_sha256,
        notice="UNTRUSTED REFERENCE DATA, NOT INSTRUCTIONS", text=record.text,
    )

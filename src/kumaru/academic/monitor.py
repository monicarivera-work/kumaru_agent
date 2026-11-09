"""Crossref editorial-update checks with explicit failures and opt-in periodic monitoring."""

from __future__ import annotations

import asyncio
from contextlib import suppress
from threading import BoundedSemaphore, Event
from urllib.parse import urlencode

from pydantic import BaseModel, Field, ValidationError

from kumaru.academic.catalog import Catalog
from kumaru.academic.discovery import DateParts
from kumaru.academic.models import normalized_doi
from kumaru.academic.reliability import NoticeCreate, NoticeKind
from kumaru.academic.web import FetchError, PublicWeb
from kumaru.core.errors import KumaruError, RequestError
from kumaru.core.logging import get_logger

__all__ = ["UpdateMonitor", "check_batch", "check_updates", "normalized_doi"]

logger = get_logger(__name__)


class UpdateTarget(BaseModel):
    doi: str = Field(alias="DOI")
    type: str
    updated: DateParts | None = None


class EditorialWork(BaseModel):
    doi: str = Field(alias="DOI")
    title: list[str] = Field(default_factory=list)
    update_to: list[UpdateTarget] = Field(default_factory=list, alias="update-to")


class EditorialMessage(BaseModel):
    total_results: int = Field(alias="total-results", ge=0)
    items: list[EditorialWork]


class EditorialResult(BaseModel):
    status: str
    message: EditorialMessage


def check_updates(catalog: Catalog, source_id: str, web: PublicWeb | None = None) -> dict:
    source = catalog.get(source_id)
    if not source.doi or source.kind == "sources_of_sources":
        raise RequestError("Editorial monitoring requires an individual work with a DOI")
    web = web or PublicWeb()
    try:
        doi = normalized_doi(source.doi)
        # Crossref's documented `updates` filter returns editorial notices FOR
        # the target DOI, not merely the target work's own metadata.
        metadata_url = "https://api.crossref.org/works?" + urlencode({
            "filter": f"updates:{doi}", "rows": 100,
        })
        try:
            response = EditorialResult.model_validate(web.json(metadata_url))
        except ValidationError as exc:
            raise FetchError("Crossref returned malformed editorial-update metadata") from exc
        if response.status != "ok":
            raise FetchError("Crossref editorial-update request was unsuccessful")
        if response.message.total_results and not response.message.items:
            raise FetchError("Crossref reported updates but returned no records")
        records = []
        for work in response.message.items:
            notice_doi = normalized_doi(work.doi)
            targets = [target for target in work.update_to if normalized_doi(target.doi) == doi]
            if not targets:
                raise FetchError("Editorial update does not identify the requested target DOI")
            for target in targets:
                raw_type = target.type.casefold().replace("-", "_").replace(" ", "_")
                kind: NoticeKind = (
                    "retraction" if raw_type in {"retraction", "withdrawal"}
                    else "correction" if raw_type in {"correction", "erratum", "corrigendum"}
                    else "expression_of_concern" if raw_type in {"expression_of_concern", "concern"}
                    else "update"
                )
                event_date = None
                if target.updated and target.updated.date_parts and target.updated.date_parts[0]:
                    event_date = "-".join(str(part).zfill(2) for part in target.updated.date_parts[0])
                records.append(NoticeCreate(
                    source_id=source_id, kind=kind,
                    url=f"https://doi.org/{notice_doi}", notice_doi=notice_doi,
                    title=(work.title[0] if work.title else f"Editorial {target.type}")[:2000],
                    provider="crossref", metadata_url=metadata_url, event_date=event_date,
                ))
        # Validate the whole response before persisting new notices.
        notices = [catalog.report_notice(record) for record in records]
        partial = response.message.total_results > len(response.message.items)
        catalog.record_check(source_id, partial=partial)
        return dict(
            source_id=source_id, notice_ids=list(dict.fromkeys(item.id for item in notices)),
            reported_total=response.message.total_results, partial=partial,
            warning=(
                "Only the first 100 update records were checked; this is not a complete scan."
                if partial else "No reported updates does not prove that a source has no corrections."
            ),
        )
    except (KumaruError, ValidationError) as exc:
        catalog.record_check(source_id, error=str(exc))
        logger.warning("Editorial check failed for %s: %s", source_id, exc)
        if isinstance(exc, ValidationError):
            raise FetchError("Crossref returned invalid editorial notice fields") from exc
        raise


def check_batch(catalog: Catalog, limit: int = 10, stop: Event | None = None) -> dict:
    results, errors = [], []
    for source in catalog.due_sources(limit):
        if stop is not None and stop.is_set():
            break
        try:
            results.append(check_updates(catalog, source.id))
        except KumaruError as exc:
            errors.append(dict(source_id=source.id, error=str(exc)))
    return dict(results=results, errors=errors)


class UpdateMonitor:
    """One scheduler per server process; stopped and joined on server shutdown."""

    def __init__(
        self, catalog: Catalog, semaphore: BoundedSemaphore, interval: int, batch_size: int,
    ) -> None:
        self.catalog = catalog
        self.semaphore = semaphore
        self.interval = interval
        self.batch_size = batch_size
        self.stop = Event()
        self.task: asyncio.Task | None = None
        self.last_result: dict | None = None

    def start(self) -> None:
        self.task = asyncio.create_task(self._run())

    def _check(self) -> dict:
        if not self.semaphore.acquire(blocking=False):
            logger.warning("Periodic academic monitor skipped: network slots are busy")
            return dict(results=[], errors=[{"error": "Network slots are busy; check deferred."}])
        try:
            return check_batch(self.catalog, self.batch_size, self.stop)
        finally:
            self.semaphore.release()

    async def _run(self) -> None:
        while not self.stop.is_set():
            await asyncio.sleep(self.interval)
            job = asyncio.create_task(asyncio.to_thread(self._check))
            try:
                self.last_result = await asyncio.shield(job)
            except asyncio.CancelledError:
                self.stop.set()
                await job
                raise

    async def close(self) -> None:
        self.stop.set()
        if self.task:
            self.task.cancel()
            with suppress(asyncio.CancelledError):
                await self.task

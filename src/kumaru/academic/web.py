"""Bounded public-network fetches, DNS pinning, robots rules and text extraction."""

from __future__ import annotations

import hashlib
import ipaddress
import json
import re
import socket
import ssl
import threading
import time
from dataclasses import dataclass
from html.parser import HTMLParser
from urllib.parse import urljoin, urlsplit
from urllib.robotparser import RobotFileParser

import httpx

from kumaru.academic.catalog import Catalog
from kumaru.academic.models import Access, Review, Source, canonical_url, now
from kumaru.agent.doc_chunks import html_blocks
from kumaru.core.errors import RequestError

USER_AGENT = "KumaruAcademic/0.1"
MAX_BYTES = 2_000_000
_LOCK = threading.Lock()
_LAST_REQUEST: dict[str, float] = {}


class FetchError(RequestError):
    http_status = 502


class AccessBlocked(RequestError):
    http_status = 403

    def __init__(self, message: str, access: Access = "inaccessible") -> None:
        super().__init__(message)
        self.access = access


@dataclass(frozen=True)
class Page:
    url: str
    status: int
    content_type: str
    body: bytes

    @property
    def text(self) -> str:
        match = re.search(r"charset=[\"']?([\w-]+)", self.content_type, flags=re.I)
        encoding = match.group(1) if match else "utf-8"
        try:
            return self.body.decode(encoding, errors="replace")
        except LookupError as exc:
            raise FetchError(f"Unsupported response charset: {encoding}") from exc


def public_addresses(host: str, port: int) -> list[str]:
    try:
        addresses = list(dict.fromkeys(
            str(info[4][0]) for info in socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
        ))
    except socket.gaierror as exc:
        raise FetchError(f"Cannot resolve academic source host: {host}") from exc
    if not addresses:
        raise FetchError("Academic source host has no addresses")
    for address in addresses:
        ip = ipaddress.ip_address(address)
        mapped = ip.ipv4_mapped if isinstance(ip, ipaddress.IPv6Address) else None
        transition: list[ipaddress.IPv4Address] = []
        if isinstance(ip, ipaddress.IPv6Address):
            if ip.sixtofour:
                transition.append(ip.sixtofour)
            if ip.teredo:
                transition.extend(ip.teredo)
        if (
            not ip.is_global or (mapped is not None and not mapped.is_global)
            or any(not embedded.is_global for embedded in transition)
        ):
            raise AccessBlocked("Academic retrieval refuses non-public network addresses")
    return addresses


class PublicWeb:
    def __init__(self, *, max_bytes: int = MAX_BYTES, timeout: float = 20) -> None:
        self.max_bytes = max_bytes
        self.timeout = timeout
        self.robots: dict[str, RobotFileParser] = {}

    def _request(self, url: str, headers: dict[str, str] | None = None, delay: float = 1) -> Page:
        url = canonical_url(url)
        target = httpx.URL(url)
        addresses = public_addresses(target.host, target.port or (443 if target.scheme == "https" else 80))
        # Resolve, validate ALL answers, then connect to an IP literal. TLS still
        # verifies the original hostname; a second DNS lookup cannot rebind it.
        pinned = target.copy_with(host=addresses[0])
        with _LOCK:
            wait = delay - (time.monotonic() - _LAST_REQUEST.get(target.host, 0))
            if wait > 0:
                time.sleep(wait)
            _LAST_REQUEST[target.host] = time.monotonic()
        try:
            with (
                httpx.Client(
                    timeout=self.timeout, trust_env=False, follow_redirects=False,
                    verify=ssl.create_default_context(),
                ) as client,
                client.stream(
                    "GET", pinned,
                    headers={
                        "User-Agent": USER_AGENT,
                        "Accept": "text/html, text/plain, application/json",
                        **(headers or {}),
                        "Host": target.netloc.decode("ascii"),
                    },
                    extensions={"sni_hostname": target.host},
                ) as response,
            ):
                body = bytearray()
                for chunk in response.iter_bytes():
                    body.extend(chunk)
                    if len(body) > self.max_bytes:
                        raise FetchError(f"Response exceeds {self.max_bytes} bytes")
                # Preserve the Location only for redirects, without exposing
                # transport IPs as source URLs.
                if response.is_redirect:
                    location = response.headers.get("location")
                    if not location:
                        raise FetchError("Redirect has no Location header")
                    return Page(
                        urljoin(url, location), response.status_code, "redirect", b"",
                    )
                return Page(
                    url, response.status_code, response.headers.get("content-type", ""), bytes(body),
                )
        except httpx.HTTPError as exc:
            raise FetchError(f"Academic retrieval failed for {target.host}: {type(exc).__name__}") from exc

    def _robots_delay(self, url: str) -> float:
        parts = urlsplit(url)
        origin = f"{parts.scheme}://{parts.netloc}"
        if origin not in self.robots:
            page = self.fetch(origin + "/robots.txt", robots=False)
            parser = RobotFileParser()
            parser.set_url(origin + "/robots.txt")
            if page.status == 404:
                parser.parse(["User-agent: *", "Allow: /"])
            elif page.status in {401, 403}:
                raise AccessBlocked("robots.txt disallows automated access")
            elif page.status != 200:
                raise FetchError(f"Cannot establish robots policy (HTTP {page.status})")
            else:
                parser.parse(page.text.splitlines())
            self.robots[origin] = parser
        rules = self.robots[origin]
        if not rules.can_fetch(USER_AGENT, url):
            raise AccessBlocked("robots.txt disallows this source")
        delay = float(rules.crawl_delay(USER_AGENT) or rules.crawl_delay("*") or 1)
        rate = rules.request_rate(USER_AGENT) or rules.request_rate("*")
        if rate and rate.requests:
            delay = max(delay, rate.seconds / rate.requests)
        if delay > 60:
            raise AccessBlocked("Site crawl delay exceeds the interactive retrieval budget")
        return max(1, delay)

    def fetch(
        self, url: str, *, robots: bool = True, headers: dict[str, str] | None = None,
    ) -> Page:
        for _ in range(6):
            url = canonical_url(url)
            delay = self._robots_delay(url) if robots else 1
            page = self._request(url, headers=headers, delay=delay)
            if page.content_type != "redirect":
                return page
            if headers:
                raise FetchError("Authenticated search API redirects are not followed")
            if urlsplit(url).scheme == "https" and urlsplit(page.url).scheme != "https":
                raise AccessBlocked("HTTPS downgrades are not followed")
            url = page.url
        raise FetchError("Academic source exceeded five redirects")

    def json(self, url: str, *, headers: dict[str, str] | None = None) -> dict:
        page = self.fetch(url, robots=False, headers=headers)
        if page.status != 200:
            raise FetchError(f"Search provider returned HTTP {page.status}")
        try:
            result = json.loads(page.text)
        except (ValueError, UnicodeError, RecursionError) as exc:
            raise FetchError("Search provider returned invalid JSON") from exc
        if not isinstance(result, dict):
            raise FetchError("Search provider returned an unexpected JSON shape")
        return result


class Metadata(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.meta: dict[str, list[str]] = {}
        self.jsonld: list[str] = []
        self._jsonld = False
        self._parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        if tag == "meta":
            key = (values.get("name") or values.get("property") or "").lower()
            if key and values.get("content"):
                self.meta.setdefault(key, []).append(values["content"] or "")
        if tag == "script" and values.get("type") == "application/ld+json":
            self._jsonld = True
            self._parts = []

    def handle_endtag(self, tag: str) -> None:
        if tag == "script" and self._jsonld:
            self.jsonld.append("".join(self._parts))
            self._jsonld = False

    def handle_data(self, data: str) -> None:
        if self._jsonld:
            self._parts.append(data)


def analyze_page(page: Page) -> tuple[dict, str]:
    if page.status in {401, 402, 403, 404, 429, 451}:
        access: Access = "account" if page.status == 401 else "paywall" if page.status == 402 else "inaccessible"
        raise AccessBlocked(f"Source returned HTTP {page.status}; no text collected", access)
    if page.status != 200:
        raise FetchError(f"Source returned HTTP {page.status}")
    content_type = page.content_type.split(";")[0].strip().lower()
    if content_type == "text/plain":
        return dict(access="open", content_type=content_type), page.text
    if content_type not in {"text/html", "application/xhtml+xml"}:
        raise AccessBlocked(
            f"Unsupported text format {content_type or '(missing)'}; metadata only",
        )
    parser = Metadata()
    parser.feed(page.text)
    title, blocks = html_blocks(page.text)
    text = "\n\n".join(block.text for block in blocks)
    access = "open"
    access_note = "Page reachable without login; this does not establish a reuse license."
    lower = text.casefold()
    if any(phrase in lower for phrase in (
        "purchase this article", "subscribe to read this article", "rent this article",
    )):
        access, access_note = "paywall", "Page advertises a paid full-text gate."
    elif any(phrase in lower for phrase in ("sign in to read", "log in to read", "create an account to read")):
        access, access_note = "account", "Page advertises an account-only full-text gate."
    elif any(phrase in lower for phrase in (
        "verify you are human", "checking your browser", "enable javascript and cookies to continue",
    )):
        access, access_note = "inaccessible", "Browser challenge; no attempt to bypass it."
    for raw in parser.jsonld:
        try:
            data = json.loads(raw)
        except ValueError:
            continue  # Optional publisher markup is not a trusted metadata API.
        objects = data if isinstance(data, list) else [data]
        for obj in objects:
            if isinstance(obj, dict) and (
                obj.get("isAccessibleForFree") is False or obj.get("isAccessibleForFree") == "false"
            ):
                access, access_note = "paywall", "Publisher marks full text as not freely accessible."
    updates: dict = dict(
        access=access, access_note=access_note, content_type=content_type,
    )
    titles = parser.meta.get("citation_title") or parser.meta.get("og:title")
    if titles or title:
        updates["title"] = (titles[0] if titles else title)[:2000]
    authors = parser.meta.get("citation_author")
    if authors:
        updates["authors"] = authors[:500]
    publisher = parser.meta.get("citation_publisher")
    if publisher:
        updates["publisher"] = publisher[0]
    dois = parser.meta.get("citation_doi")
    if dois:
        updates["doi"] = dois[0]
    if re.search(r"\b(?:content|article|text)\s+(?:was\s+)?(?:generated|written)\s+by\s+(?:an?\s+)?AI\b", text, re.I):
        updates["origin"] = "ai"
    return updates, text


def inspect_source(catalog: Catalog, id_: str, web: PublicWeb) -> Source:
    source = catalog.get(id_)
    try:
        page = web.fetch(source.url)
        updates, _ = analyze_page(page)
    except AccessBlocked as exc:
        updates = dict(access=exc.access, access_note=str(exc))
    except FetchError as exc:
        catalog.save_metadata(id_, dict(access="inaccessible", access_note=str(exc)))
        raise
    # Metadata inspection never retains page text or approves authorship.
    if updates.get("origin") == "ai":
        catalog.review(id_, Review(origin="ai"))
        del updates["origin"]
    return catalog.save_metadata(id_, updates)


def collect_source(catalog: Catalog, id_: str, web: PublicWeb) -> Source:
    source = catalog.get(id_)
    catalog.require_current_review(id_)
    if source.kind == "sources_of_sources":
        raise RequestError("Sources of Sources are catalogues; collect individual child works instead")
    if source.origin != "human_verified" or source.permission != "text_permitted":
        raise RequestError("Collect requires reviewed human provenance and text permission evidence")
    if not source.topic_reviewed:
        raise RequestError("Review topic coverage before collecting text")
    if source.access in {"account", "paywall", "inaccessible"}:
        raise AccessBlocked("Source is access-restricted; this scraper never bypasses gates", source.access)
    try:
        page = web.fetch(source.url)
        updates, text = analyze_page(page)
    except AccessBlocked as exc:
        catalog.save_collection(id_, dict(access=exc.access, access_note=str(exc)))
        raise
    except FetchError as exc:
        catalog.save_metadata(id_, dict(access="inaccessible", access_note=str(exc)))
        raise
    if updates.get("origin") == "ai":
        catalog.review(id_, Review(origin="ai"))
        raise AccessBlocked("Page discloses AI-generated text; source removed from trusted collection")
    if updates["access"] != "open":
        catalog.save_collection(id_, updates)
        raise AccessBlocked(updates["access_note"], updates["access"])
    if not text.strip():
        raise FetchError("No readable text found; no content stored")
    if urlsplit(page.url).hostname != urlsplit(source.url).hostname:
        raise RequestError("Full text redirected to another host; catalogue and review that URL separately")
    updates.update(
        text=text, text_url=page.url, text_sha256=hashlib.sha256(text.encode()).hexdigest(),
        collected_at=now(), word_count=len(text.split()), truncated=False,
        topic_word_counts={source.topic_ids[0]: len(text.split())} if len(source.topic_ids) == 1 else {},
    )
    return catalog.save_collection(id_, updates)

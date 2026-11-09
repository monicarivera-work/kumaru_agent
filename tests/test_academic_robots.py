"""RFC 9309 robots.txt matching, and browser challenges, as live publishers serve them."""

from __future__ import annotations

import time
from contextlib import contextmanager
from pathlib import Path

import httpx
import pytest

from kumaru.academic.catalog import Catalog
from kumaru.academic.models import Source
from kumaru.academic.robots import parse_robots
from kumaru.academic.web import (
    USER_AGENT,
    AccessBlocked,
    Page,
    PublicWeb,
    analyze_page,
    inspect_source,
)

# Abridged from a major publisher's robots.txt: a blanket Disallow narrowed by
# Allow lines. Python's urllib.robotparser obeyed the first line and blocked all.
PUBLISHER = """\
User-agent: *
Disallow: /
Allow: /$
Allow: /chapter/
Allow: /journal*
"""


def allowed(robots: str, path: str, agent: str = USER_AGENT) -> bool:
    return parse_robots(robots, agent).allowed("https://example.org" + path)


@pytest.mark.parametrize(("path", "expected"), [
    ("/", True),
    ("/chapter/10.1007/978-1-4612-0959-1_10", True),
    ("/journals/a", True),
    ("/10.1007/978-1-4612-0959-1_10", False),
    ("/about", False),
])
def test_the_longest_matching_rule_wins_not_the_first(path, expected):
    assert allowed(PUBLISHER, path) is expected


@pytest.mark.parametrize(("path", "expected"), [
    ("/paper.pdf", False),
    ("/paper.pdf/view", True),
    ("/search?q=calculus", False),
    ("/search", True),
    ("/private-drafts/a", False),
    ("/private-ok/a", True),
    ("/fishheads/catfish.php?x=1", False),
])
def test_wildcards_and_end_anchors_are_honoured(path, expected):
    robots = """\
User-agent: *
Disallow: /*.pdf$
Disallow: /*?
Disallow: /private*/
Allow: /private-ok/
Disallow: /fish*.php
"""
    assert allowed(robots, path) is expected


def test_allow_wins_a_tie_and_empty_rules_match_nothing():
    assert allowed("User-agent: *\nDisallow: /page\nAllow: /page\n", "/page")
    assert allowed("User-agent: *\nDisallow:\n", "/anything")
    assert allowed("", "/anything")


def test_a_group_naming_kumaru_replaces_the_star_group():
    robots = """\
User-agent: *
Disallow: /

User-agent: KumaruAcademic
Disallow: /drafts/
"""
    assert allowed(robots, "/paper")
    assert not allowed(robots, "/drafts/a")
    assert not allowed(robots, "/paper", agent="OtherBot/1.0")
    # A named group with no rules allows everything, even under a blanket `*` ban.
    assert allowed("User-agent: *\nDisallow: /\n\nUser-agent: kumaruacademic/2\nDisallow:\n", "/x")
    # The product token must match exactly; a prefix of it names another crawler.
    assert not allowed("User-agent: *\nDisallow: /\n\nUser-agent: Kumaru\nAllow: /\n", "/x")


def test_groups_follow_rfc_boundaries():
    robots = """\
Disallow: /before-any-group

User-agent: a-bot
User-agent: KumaruAcademic

Disallow: /shared
Sitemap: https://example.org/sitemap.xml
User-agent: b-bot
Disallow: /b-only

User-agent: KumaruAcademic
Disallow: /merged
"""
    assert allowed(robots, "/before-any-group")
    assert not allowed(robots, "/shared")
    assert allowed(robots, "/b-only")
    assert not allowed(robots, "/merged")


def test_paths_and_rules_compare_after_percent_encoding_normalisation():
    robots = "User-agent: *\nDisallow: /caf%c3%a9\nDisallow: /%7Euser\nDisallow: /a%2fb\n"
    assert not allowed(robots, "/café/menu")
    assert not allowed(robots, "/~user/home")
    assert not allowed(robots, "/a%2Fb")
    assert allowed(robots, "/a/b")
    assert not allowed("\ufeffUser-agent: *\nDisallow: /\n", "/x")


def test_crawl_delays_parse_conservatively_and_inherit_from_star():
    rules = parse_robots(
        "User-agent: *\nCrawl-delay: 2.5\nRequest-rate: 1/5\n\nUser-agent: KumaruAcademic\nDisallow: /x\n",
        USER_AGENT,
    )
    assert (rules.crawl_delay, rules.request_rate) == (2.5, (1, 5))
    for bad in ("nan", "-1", "inf", "soon"):
        assert parse_robots(f"User-agent: *\nCrawl-delay: {bad}\n", USER_AGENT).crawl_delay is None
    for bad in ("²/5", "1/", "one/5"):
        assert parse_robots(f"User-agent: *\nRequest-rate: {bad}\n", USER_AGENT).request_rate is None


def test_hostile_wildcard_patterns_match_in_linear_time():
    robots = "User-agent: *\n" + "".join(
        f"Disallow: /{'*a' * 2_000}{suffix}\n" for suffix in "bcdefghijk"
    )
    rules = parse_robots(robots, USER_AGENT)
    start = time.perf_counter()
    assert rules.allowed("https://example.org/" + "a" * 20_000)
    assert time.perf_counter() - start < 2


def robots_web(robots: str) -> PublicWeb:
    web = PublicWeb()

    def request(url, headers=None, delay=1):
        if url.endswith("/robots.txt"):
            return Page(url, 200, "text/plain", robots.encode())
        return Page(url, 200, "text/plain", b"page")

    web._request = request  # type: ignore[method-assign]
    return web


def test_fetch_obeys_longest_match_robots():
    web = robots_web(PUBLISHER)
    assert web.fetch("https://example.org/chapter/10.1007/x").body == b"page"
    with pytest.raises(AccessBlocked, match="robots.txt disallows this source"):
        web.fetch("https://example.org/10.1007/x")


@pytest.mark.parametrize(("status", "service", "headers"), [
    (403, "Cloudflare", {"cf-mitigated": "challenge"}),
    (202, "AWS WAF", {"x-amzn-waf-action": "challenge"}),
    (405, "AWS WAF", {"x-amzn-waf-action": "CAPTCHA"}),
    (403, "", {"cf-mitigated": "block"}),
])
def test_transport_names_the_service_behind_a_browser_challenge(monkeypatch, status, service, headers):
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
        def stream(self, method, url, **kwargs):
            yield httpx.Response(status, content=b"<html></html>", headers={
                "content-type": "text/html", **headers,
            })

    monkeypatch.setattr(httpx, "Client", Client)
    page = PublicWeb().fetch("https://example.org/a", robots=False)
    assert (page.status, page.challenge) == (status, service)


@pytest.mark.parametrize("status", [200, 202, 403, 429, 503])
def test_a_challenge_is_inaccessible_whatever_its_status(status):
    with pytest.raises(AccessBlocked, match=r"Browser challenge \(AWS WAF\)") as caught:
        analyze_page(Page("https://example.org/a", status, "text/html", b"", "AWS WAF"))
    assert caught.value.access == "inaccessible"


def test_a_challenge_on_robots_txt_fails_closed():
    web = PublicWeb()
    web._request = lambda url, headers=None, delay=1: Page(  # type: ignore[method-assign]
        url, 403, "text/html", b"", "Cloudflare",
    )
    with pytest.raises(AccessBlocked, match=r"Browser challenge \(Cloudflare\) on robots.txt"):
        web.fetch("https://example.org/a")


def test_inspect_records_a_challenge_without_failing(tmp_path: Path):
    catalog = Catalog(tmp_path / "academic.sqlite")
    source = catalog.discover(Source(url="https://example.org/walled", title="Walled", topic_ids=["calculus"]))

    class Challenged(PublicWeb):
        def fetch(self, url, *, robots=True, headers=None):
            return Page(url, 202, "text/html", b"", "AWS WAF")

    inspected = inspect_source(catalog, source.id, Challenged())
    assert inspected.access == "inaccessible"
    assert inspected.access_note == "Browser challenge (AWS WAF); no attempt to bypass it."

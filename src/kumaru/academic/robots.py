"""RFC 9309 robots.txt rules: grouped user agents, longest match, `*` and `$` patterns.

The standard library's `urllib.robotparser` obeys the first matching rule and
reads `*` and `$` literally. Sites write robots.txt for the RFC's longest-match
reading (`Disallow: /` followed by narrower `Allow` lines), so that parser both
blocked pages sites permit and ignored wildcard `Disallow` rules.
"""

from __future__ import annotations

import math
import re
import string
from dataclasses import dataclass, field
from urllib.parse import quote, urlsplit

_PRINTABLE = "".join(chr(code) for code in range(0x21, 0x7F))
_UNRESERVED = frozenset(string.ascii_letters + string.digits + "-._~")
_ESCAPE = re.compile(r"%([0-9A-Fa-f]{2})")
_PRODUCT = re.compile(r"[A-Za-z_-]*")


def _normalize(value: str) -> str:
    """Percent-encode non-ASCII octets, decode unreserved escapes, uppercase the rest."""

    def escape(match: re.Match[str]) -> str:
        char = chr(int(match.group(1), 16))
        return char if char in _UNRESERVED else "%" + match.group(1).upper()

    return _ESCAPE.sub(escape, quote(value, safe=_PRINTABLE))


def _product(value: str) -> str:
    match = _PRODUCT.match(value.strip())
    return match.group(0).casefold() if match else ""


def _matches(pattern: str, path: str) -> bool:
    """Whether a rule matches from the start of a path.

    `*` matches any run of characters and a final `$` anchors the end. With `*` as
    the only wildcard, taking each literal segment's leftmost match is optimal, so
    a hostile pattern costs a few linear scans rather than backtracking.
    """
    anchored = pattern.endswith("$")
    first, *rest = (pattern[:-1] if anchored else pattern).split("*")
    if not path.startswith(first):
        return False
    if not rest:
        return not anchored or path == first
    *middle, last = rest
    position = len(first)
    for segment in middle:
        found = path.find(segment, position)
        if found < 0:
            return False
        position = found + len(segment)
    if anchored:
        return path.endswith(last) and len(path) - len(last) >= position
    return path.find(last, position) >= 0


@dataclass
class RobotsRules:
    """What one crawler may fetch from one origin, and how slowly."""

    rules: list[tuple[str, bool]] = field(default_factory=list)
    crawl_delay: float | None = None
    request_rate: tuple[int, int] | None = None

    def allowed(self, url: str) -> bool:
        """The longest matching rule decides; `Allow` wins ties; no match allows."""
        parts = urlsplit(url)
        path = _normalize((parts.path or "/") + (f"?{parts.query}" if parts.query else ""))
        verdict, best = True, -1
        for pattern, allow in self.rules:
            if (len(pattern) > best or (allow and len(pattern) == best)) and _matches(pattern, path):
                verdict, best = allow, len(pattern)
        return verdict

    def _add(self, key: str, value: str) -> None:
        if key in {"allow", "disallow"}:
            if value:  # An empty rule matches nothing.
                self.rules.append((_normalize(value), key == "allow"))
        elif key == "crawl-delay":
            try:
                delay = float(value)
            except ValueError:
                return
            if math.isfinite(delay) and delay >= 0:
                self.crawl_delay = delay
        else:
            requests, slash, seconds = (part.strip() for part in value.partition("/"))
            if slash and requests.isdecimal() and seconds.isdecimal():
                self.request_rate = (int(requests), int(seconds))


def parse_robots(text: str, user_agent: str) -> RobotsRules:
    """Combine every group naming this crawler's product token, or else every `*` group.

    Crawl delays and request rates are not part of RFC 9309. A crawler-specific
    group without them inherits the `*` group's, which never paces faster.
    """
    product = _product(user_agent)
    groups = {"own": RobotsRules(), "*": RobotsRules()}
    named = False
    current: set[str] = set()
    reading_agents = False
    for line in text.removeprefix("\ufeff").splitlines():
        key, colon, value = line.split("#", 1)[0].partition(":")
        if not colon:
            continue
        key, value = key.strip().casefold(), value.strip()
        if key == "user-agent":
            if not reading_agents:
                current, reading_agents = set(), True
            if value.split(None, 1)[:1] == ["*"]:
                current.add("*")
            elif product and _product(value) == product:
                current.add("own")
                named = True
        elif key in {"allow", "disallow", "crawl-delay", "request-rate"}:
            reading_agents = False
            for name in current:
                groups[name]._add(key, value)
    chosen, fallback = groups["own" if named else "*"], groups["*"]
    return RobotsRules(
        chosen.rules,
        chosen.crawl_delay or fallback.crawl_delay,
        chosen.request_rate or fallback.request_rate,
    )

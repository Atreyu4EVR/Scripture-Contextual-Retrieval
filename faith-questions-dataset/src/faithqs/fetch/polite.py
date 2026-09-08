"""HTTP transport that enforces the non-negotiable request rules.

* Rule 2: robots.txt is fetched and parsed with protego for every host
  touched, redirect targets included. ``Crawl-delay`` is honored. A disallowed
  path raises instead of being fetched.
* Rule 3: every request carries the project User-Agent with a contact address.
* Rule 4: at most one request per ``rate_limit_seconds`` per host (floor two
  seconds), exponential backoff on 429 and 5xx, honoring ``Retry-After``.

A missing robots.txt (4xx) means unrestricted access; a robots.txt that cannot
be retrieved (5xx or transport failure) is treated as a complete disallow, as
RFC 9309 section 2.3.1.4 directs. Both outcomes are the caller's to escalate.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urljoin, urlsplit

import httpx
from protego import Protego

from faithqs.manifest import RATE_LIMIT_FLOOR_SECONDS

REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})
KEPT_HEADERS = ("etag", "last-modified", "content-type", "content-length")


class RobotsDisallowedError(RuntimeError):
    """Rule 2: the path is disallowed. Stop and ask the human."""


class RobotsUnavailableError(RuntimeError):
    """robots.txt could not be retrieved; treated as disallow per RFC 9309."""


class FetchError(RuntimeError):
    """A request failed after retries or returned an error status."""


@dataclass(frozen=True)
class DownloadResult:
    url: str
    final_url: str
    path: Path
    sha256: str
    size: int
    headers: dict[str, str] = field(default_factory=dict)


def _host_key(url: str) -> str:
    parts = urlsplit(url)
    return f"{parts.scheme.lower()}://{parts.netloc.lower()}"


def _retry_after_seconds(response: httpx.Response) -> float | None:
    raw = response.headers.get("retry-after")
    if raw is None:
        return None
    try:
        return max(0.0, float(raw))
    except ValueError:
        return None  # HTTP-date form: fall back to exponential backoff


class PoliteClient:
    MAX_ATTEMPTS = 5
    MAX_REDIRECTS = 5
    CHUNK_SIZE = 1 << 20

    def __init__(
        self,
        *,
        user_agent: str,
        rate_limit_seconds: float = RATE_LIMIT_FLOOR_SECONDS,
        timeout: float = 60.0,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if rate_limit_seconds < RATE_LIMIT_FLOOR_SECONDS:
            raise ValueError(
                f"rate_limit_seconds must be at least {RATE_LIMIT_FLOOR_SECONDS} (CLAUDE.md rule 4)"
            )
        if "contact:" not in user_agent or "@" not in user_agent:
            raise ValueError(
                "User-Agent must name the project and a contact email address (CLAUDE.md rule 3)"
            )
        self.user_agent = user_agent
        self.rate_limit_seconds = rate_limit_seconds
        self._sleep = sleep
        self._clock = clock
        self._client = httpx.Client(
            headers={"User-Agent": user_agent},
            timeout=timeout,
            follow_redirects=False,
            transport=transport,
        )
        self._robots: dict[str, Protego | None] = {}
        self._last_request: dict[str, float] = {}

    def __enter__(self) -> PoliteClient:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()

    # --- rule 4: per-host pacing -------------------------------------------

    def _throttle(self, host: str, extra_delay: float = 0.0) -> None:
        delay = max(self.rate_limit_seconds, extra_delay)
        last = self._last_request.get(host)
        if last is not None:
            wait = last + delay - self._clock()
            if wait > 0:
                self._sleep(wait)
        self._last_request[host] = self._clock()

    # --- rule 2: robots.txt ---------------------------------------------------

    def _robots_for(self, url: str) -> Protego | None:
        host = _host_key(url)
        if host in self._robots:
            return self._robots[host]
        robots_url = f"{host}/robots.txt"
        self._throttle(host)
        try:
            response = self._client.get(robots_url, follow_redirects=True)
        except httpx.HTTPError as exc:
            raise RobotsUnavailableError(
                f"could not retrieve {robots_url} ({exc!r}); treating as disallow, "
                "stop and ask the human"
            ) from exc
        if 200 <= response.status_code < 300:
            parser: Protego | None = Protego.parse(response.text)
        elif 400 <= response.status_code < 500:
            parser = None
        else:
            raise RobotsUnavailableError(
                f"{robots_url} returned {response.status_code}; RFC 9309 treats an "
                "unreachable robots.txt as complete disallow, stop and ask the human"
            )
        self._robots[host] = parser
        return parser

    def _check_allowed(self, url: str) -> float:
        """Raise if robots.txt disallows ``url``; return the applicable Crawl-delay."""
        parser = self._robots_for(url)
        if parser is None:
            return 0.0
        if not parser.can_fetch(url, self.user_agent):
            raise RobotsDisallowedError(
                f"robots.txt at {_host_key(url)} disallows {url} for this User-Agent "
                "(CLAUDE.md rule 2: stop and ask the human)"
            )
        delay = parser.crawl_delay(self.user_agent)
        return float(delay) if delay else 0.0

    # --- requests -------------------------------------------------------------

    def _send_with_backoff(self, method: str, url: str, *, stream: bool) -> httpx.Response:
        crawl_delay = self._check_allowed(url)
        host = _host_key(url)
        backoff = self.rate_limit_seconds
        for attempt in range(1, self.MAX_ATTEMPTS + 1):
            self._throttle(host, crawl_delay)
            try:
                request = self._client.build_request(method, url)
                response = self._client.send(request, stream=stream)
            except httpx.TransportError as exc:
                if attempt == self.MAX_ATTEMPTS:
                    raise FetchError(f"{method} {url} failed after {attempt} attempts") from exc
                self._sleep(backoff)
                backoff *= 2
                continue
            if response.status_code == 429 or response.status_code >= 500:
                response.close()
                if attempt == self.MAX_ATTEMPTS:
                    raise FetchError(
                        f"{method} {url} returned {response.status_code} after {attempt} attempts"
                    )
                self._sleep(_retry_after_seconds(response) or backoff)
                backoff *= 2
                continue
            return response
        raise AssertionError("unreachable")

    def _follow(self, method: str, url: str, *, stream: bool) -> httpx.Response:
        """Resolve redirects one hop at a time so every host passes the robots gate."""
        current = url
        for _ in range(self.MAX_REDIRECTS + 1):
            response = self._send_with_backoff(method, current, stream=stream)
            location = response.headers.get("location")
            if response.status_code in REDIRECT_STATUSES and location:
                response.close()
                current = urljoin(current, location)
                continue
            return response
        raise FetchError(f"{method} {url}: more than {self.MAX_REDIRECTS} redirects")

    def request(self, method: str, url: str) -> httpx.Response:
        """A small, fully-read response (metadata, HEAD checks)."""
        response = self._follow(method, url, stream=False)
        if response.status_code >= 400:
            raise FetchError(f"{method} {url} returned {response.status_code}")
        return response

    def download(self, url: str, dest: Path) -> DownloadResult:
        """Stream ``url`` into ``dest``, hashing as it goes."""
        response = self._follow("GET", url, stream=True)
        if response.status_code >= 400:
            response.close()
            raise FetchError(f"GET {url} returned {response.status_code}")
        digest = hashlib.sha256()
        size = 0
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            with dest.open("wb") as handle:
                for chunk in response.iter_bytes(chunk_size=self.CHUNK_SIZE):
                    handle.write(chunk)
                    digest.update(chunk)
                    size += len(chunk)
        finally:
            response.close()
        headers = {k: response.headers[k] for k in KEPT_HEADERS if k in response.headers}
        return DownloadResult(
            url=url,
            final_url=str(response.url),
            path=dest,
            sha256=digest.hexdigest(),
            size=size,
            headers=headers,
        )

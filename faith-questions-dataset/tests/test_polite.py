import hashlib
from collections.abc import Callable

import httpx
import pytest

from conftest import FakeClock
from faithqs.fetch.polite import (
    FetchError,
    PoliteClient,
    RobotsDisallowedError,
    RobotsUnavailableError,
)

UA = "faithqs/0.1.0 (+https://example.org/faithqs; contact: owner@example.org)"

Responder = Callable[[httpx.Request], httpx.Response]


class Recorder:
    """Routes requests to per-URL responders and records every request made."""

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []
        self.routes: dict[str, Responder] = {}

    def route(self, url: str, responder: httpx.Response | Responder) -> None:
        self.routes[url] = responder if callable(responder) else (lambda _req, r=responder: r)

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        key = str(request.url)
        if key not in self.routes:
            return httpx.Response(404)
        return self.routes[key](request)

    def paths(self) -> list[str]:
        return [f"{r.url.host}{r.url.path}" for r in self.requests]


def make_client(recorder: Recorder, clock: FakeClock, **kwargs) -> PoliteClient:
    return PoliteClient(
        user_agent=UA,
        transport=httpx.MockTransport(recorder),
        sleep=clock.sleep,
        clock=clock,
        **kwargs,
    )


def robots(text: str) -> httpx.Response:
    return httpx.Response(200, text=text)


def test_user_agent_must_carry_contact() -> None:
    with pytest.raises(ValueError, match="rule 3"):
        PoliteClient(user_agent="faithqs/0.1.0")


def test_rate_limit_cannot_go_below_floor() -> None:
    with pytest.raises(ValueError, match="rule 4"):
        PoliteClient(user_agent=UA, rate_limit_seconds=1.0)


def test_robots_fetched_first_and_user_agent_sent() -> None:
    rec, clock = Recorder(), FakeClock()
    rec.route("https://a.example/robots.txt", robots("User-agent: *\nAllow: /\n"))
    rec.route("https://a.example/file.txt", httpx.Response(200, text="ok"))
    with make_client(rec, clock) as client:
        response = client.request("GET", "https://a.example/file.txt")
    assert response.text == "ok"
    assert rec.paths() == ["a.example/robots.txt", "a.example/file.txt"]
    assert all(r.headers["user-agent"] == UA for r in rec.requests)


def test_disallowed_path_is_never_requested() -> None:
    rec, clock = Recorder(), FakeClock()
    rec.route("https://a.example/robots.txt", robots("User-agent: *\nDisallow: /download/\n"))
    rec.route("https://a.example/download/x.7z", httpx.Response(200, content=b"zzz"))
    with make_client(rec, clock) as client, pytest.raises(RobotsDisallowedError, match="rule 2"):
        client.request("GET", "https://a.example/download/x.7z")
    assert rec.paths() == ["a.example/robots.txt"]


def test_missing_robots_means_unrestricted() -> None:
    rec, clock = Recorder(), FakeClock()
    rec.route("https://a.example/file.txt", httpx.Response(200, text="ok"))
    with make_client(rec, clock) as client:
        assert client.request("GET", "https://a.example/file.txt").text == "ok"


def test_unreachable_robots_is_treated_as_disallow() -> None:
    rec, clock = Recorder(), FakeClock()
    rec.route("https://a.example/robots.txt", httpx.Response(503))
    rec.route("https://a.example/file.txt", httpx.Response(200, text="ok"))
    with make_client(rec, clock) as client, pytest.raises(RobotsUnavailableError):
        client.request("GET", "https://a.example/file.txt")
    assert rec.paths() == ["a.example/robots.txt"]


def test_requests_to_one_host_are_paced_at_the_floor() -> None:
    rec, clock = Recorder(), FakeClock()
    rec.route("https://a.example/robots.txt", robots("User-agent: *\nAllow: /\n"))
    rec.route("https://a.example/one", httpx.Response(200, text="1"))
    rec.route("https://a.example/two", httpx.Response(200, text="2"))
    with make_client(rec, clock) as client:
        client.request("GET", "https://a.example/one")
        client.request("GET", "https://a.example/two")
    # robots -> one -> two: two gaps, each held to the two-second floor
    assert clock.sleeps == [2.0, 2.0]


def test_crawl_delay_overrides_floor_when_longer() -> None:
    rec, clock = Recorder(), FakeClock()
    rec.route("https://a.example/robots.txt", robots("User-agent: *\nCrawl-delay: 5\n"))
    rec.route("https://a.example/one", httpx.Response(200, text="1"))
    with make_client(rec, clock) as client:
        client.request("GET", "https://a.example/one")
    assert clock.sleeps == [5.0]


def test_429_backs_off_and_honors_retry_after() -> None:
    rec, clock = Recorder(), FakeClock()
    rec.route("https://a.example/robots.txt", robots("User-agent: *\nAllow: /\n"))
    calls = {"n": 0}

    def flaky(_request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, headers={"Retry-After": "7"})
        return httpx.Response(200, text="ok")

    rec.route("https://a.example/one", flaky)
    with make_client(rec, clock) as client:
        assert client.request("GET", "https://a.example/one").text == "ok"
    assert calls["n"] == 2
    assert 7.0 in clock.sleeps


def test_5xx_exhausts_attempts_with_doubling_backoff() -> None:
    rec, clock = Recorder(), FakeClock()
    rec.route("https://a.example/robots.txt", robots("User-agent: *\nAllow: /\n"))
    rec.route("https://a.example/one", httpx.Response(500))
    with make_client(rec, clock) as client, pytest.raises(FetchError, match="after 5 attempts"):
        client.request("GET", "https://a.example/one")
    backoffs = [s for s in clock.sleeps if s != 2.0] + [s for s in clock.sleeps if s == 2.0][1:]
    assert sorted(backoffs)[-3:] == [4.0, 8.0, 16.0]
    assert len(rec.requests) == 1 + PoliteClient.MAX_ATTEMPTS


def test_redirect_target_host_must_pass_robots() -> None:
    rec, clock = Recorder(), FakeClock()
    rec.route("https://a.example/robots.txt", robots("User-agent: *\nAllow: /\n"))
    rec.route(
        "https://a.example/file",
        httpx.Response(302, headers={"Location": "https://b.example/real/file"}),
    )
    rec.route("https://b.example/robots.txt", robots("User-agent: *\nDisallow: /real/\n"))
    rec.route("https://b.example/real/file", httpx.Response(200, content=b"secret"))
    with make_client(rec, clock) as client, pytest.raises(RobotsDisallowedError):
        client.request("GET", "https://a.example/file")
    assert "b.example/real/file" not in rec.paths()
    assert "b.example/robots.txt" in rec.paths()


def test_download_follows_redirect_streams_and_hashes(tmp_path) -> None:
    payload = b"7z\xbc\xaf\x27\x1c" + bytes(range(256)) * 40
    rec, clock = Recorder(), FakeClock()
    rec.route("https://a.example/robots.txt", robots("User-agent: *\nAllow: /\n"))
    rec.route(
        "https://a.example/download/x.7z",
        httpx.Response(302, headers={"Location": "https://cdn.example/x.7z"}),
    )
    rec.route("https://cdn.example/robots.txt", robots("User-agent: *\nAllow: /\n"))
    rec.route(
        "https://cdn.example/x.7z",
        httpx.Response(200, content=payload, headers={"ETag": '"abc"'}),
    )
    dest = tmp_path / "partial.7z"
    with make_client(rec, clock) as client:
        result = client.download("https://a.example/download/x.7z", dest)
    assert dest.read_bytes() == payload
    assert result.sha256 == hashlib.sha256(payload).hexdigest()
    assert result.size == len(payload)
    assert result.final_url == "https://cdn.example/x.7z"
    assert result.headers["etag"] == '"abc"'

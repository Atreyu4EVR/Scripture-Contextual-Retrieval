import hashlib
from collections.abc import Callable, Iterator

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
ALLOW_ALL = "User-agent: *\nAllow: /\n"

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


def sequence(*responses: httpx.Response) -> Responder:
    """Return each response in turn, then repeat the last one."""
    calls = {"n": 0}

    def responder(_request: httpx.Request) -> httpx.Response:
        index = min(calls["n"], len(responses) - 1)
        calls["n"] += 1
        return responses[index]

    return responder


# --- construction -----------------------------------------------------------


def test_user_agent_must_carry_contact() -> None:
    with pytest.raises(ValueError, match="rule 3"):
        PoliteClient(user_agent="faithqs/0.1.0")


def test_user_agent_must_begin_with_product_token() -> None:
    with pytest.raises(ValueError, match="product token"):
        PoliteClient(user_agent="0.1.0 (contact: owner@example.org)")
    client = PoliteClient(user_agent=UA)
    assert client.robots_token == "faithqs"
    client.close()


def test_rate_limit_cannot_go_below_floor() -> None:
    with pytest.raises(ValueError, match="rule 4"):
        PoliteClient(user_agent=UA, rate_limit_seconds=1.0)


# --- rule 2: robots.txt -----------------------------------------------------


def test_robots_fetched_first_and_user_agent_sent() -> None:
    rec, clock = Recorder(), FakeClock()
    rec.route("https://a.example/robots.txt", robots(ALLOW_ALL))
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


def test_unreachable_robots_is_retried_then_treated_as_disallow() -> None:
    rec, clock = Recorder(), FakeClock()
    rec.route("https://a.example/robots.txt", httpx.Response(503))
    rec.route("https://a.example/file.txt", httpx.Response(200, text="ok"))
    with make_client(rec, clock) as client, pytest.raises(RobotsUnavailableError):
        client.request("GET", "https://a.example/file.txt")
    assert rec.paths() == ["a.example/robots.txt"] * PoliteClient.MAX_ATTEMPTS


def test_robots_429_is_backed_off_and_retried_not_treated_as_absent() -> None:
    rec, clock = Recorder(), FakeClock()
    rec.route(
        "https://a.example/robots.txt",
        sequence(
            httpx.Response(429, headers={"Retry-After": "7"}),
            robots("User-agent: *\nDisallow: /download/\n"),
        ),
    )
    rec.route("https://a.example/download/x.7z", httpx.Response(200, content=b"zzz"))
    with make_client(rec, clock) as client, pytest.raises(RobotsDisallowedError):
        client.request("GET", "https://a.example/download/x.7z")
    assert rec.paths() == ["a.example/robots.txt", "a.example/robots.txt"]
    assert 7.0 in clock.sleeps


def test_persistent_robots_429_is_treated_as_disallow() -> None:
    rec, clock = Recorder(), FakeClock()
    rec.route("https://a.example/robots.txt", httpx.Response(429, headers={"Retry-After": "3"}))
    rec.route("https://a.example/file.txt", httpx.Response(200, text="ok"))
    with make_client(rec, clock) as client, pytest.raises(RobotsUnavailableError):
        client.request("GET", "https://a.example/file.txt")
    assert rec.paths() == ["a.example/robots.txt"] * PoliteClient.MAX_ATTEMPTS
    assert clock.sleeps.count(3.0) == PoliteClient.MAX_ATTEMPTS - 1


def test_robots_groups_match_product_token_not_full_header() -> None:
    # A decoy group named after a substring of the full header must not capture us.
    rec, clock = Recorder(), FakeClock()
    rec.route(
        "https://a.example/robots.txt",
        robots("User-agent: contact\nDisallow: /\n\nUser-agent: *\nAllow: /\nCrawl-delay: 5\n"),
    )
    rec.route("https://a.example/file.txt", httpx.Response(200, text="ok"))
    with make_client(rec, clock) as client:
        assert client.request("GET", "https://a.example/file.txt").text == "ok"
    assert 5.0 in clock.sleeps  # the * group's Crawl-delay still applies


@pytest.mark.parametrize("group", ["faithqs", "FaithQS"])
def test_robots_group_for_our_token_is_honored(group: str) -> None:
    rec, clock = Recorder(), FakeClock()
    rec.route(
        "https://a.example/robots.txt",
        robots(f"User-agent: {group}\nDisallow: /\n\nUser-agent: *\nAllow: /\n"),
    )
    rec.route("https://a.example/file.txt", httpx.Response(200, text="ok"))
    with make_client(rec, clock) as client, pytest.raises(RobotsDisallowedError):
        client.request("GET", "https://a.example/file.txt")


# --- rule 4: pacing and backoff --------------------------------------------


def test_requests_to_one_host_are_paced_at_the_floor() -> None:
    rec, clock = Recorder(), FakeClock()
    rec.route("https://a.example/robots.txt", robots(ALLOW_ALL))
    rec.route("https://a.example/one", httpx.Response(200, text="1"))
    rec.route("https://a.example/two", httpx.Response(200, text="2"))
    with make_client(rec, clock) as client:
        client.request("GET", "https://a.example/one")
        client.request("GET", "https://a.example/two")
    # robots -> one -> two: two gaps, each held to the two-second floor
    assert clock.sleeps == [2.0, 2.0]


@pytest.mark.parametrize(
    ("crawl_delay", "rate_limit", "expected"),
    [
        ("5", 2.0, 5.0),  # Crawl-delay longer than the floor wins
        ("1", 2.0, 2.0),  # Crawl-delay shorter than the floor is raised to the floor
        ("0.5", 2.0, 2.0),
        ("1", 3.0, 3.0),  # ... and to a human-raised limit, not just the constant
    ],
)
def test_pacing_is_the_max_of_floor_limit_and_crawl_delay(
    crawl_delay: str, rate_limit: float, expected: float
) -> None:
    rec, clock = Recorder(), FakeClock()
    rec.route(
        "https://a.example/robots.txt", robots(f"User-agent: *\nCrawl-delay: {crawl_delay}\n")
    )
    rec.route("https://a.example/one", httpx.Response(200, text="1"))
    with make_client(rec, clock, rate_limit_seconds=rate_limit) as client:
        client.request("GET", "https://a.example/one")
    assert clock.sleeps == [expected]


def test_429_backs_off_and_honors_retry_after() -> None:
    rec, clock = Recorder(), FakeClock()
    rec.route("https://a.example/robots.txt", robots(ALLOW_ALL))
    rec.route(
        "https://a.example/one",
        sequence(httpx.Response(429, headers={"Retry-After": "7"}), httpx.Response(200, text="ok")),
    )
    with make_client(rec, clock) as client:
        assert client.request("GET", "https://a.example/one").text == "ok"
    assert rec.paths().count("a.example/one") == 2
    assert 7.0 in clock.sleeps


def test_5xx_exhausts_attempts_with_doubling_backoff() -> None:
    rec, clock = Recorder(), FakeClock()
    rec.route("https://a.example/robots.txt", robots(ALLOW_ALL))
    rec.route("https://a.example/one", httpx.Response(500))
    with make_client(rec, clock) as client, pytest.raises(FetchError, match="after 5 attempts"):
        client.request("GET", "https://a.example/one")
    # robots -> attempt 1 held to the 2 s floor, then backoff 2, 4, 8, 16 after each
    # of the four retried 500s; attempt 5 raises without sleeping.
    assert clock.sleeps == [2.0, 2.0, 4.0, 8.0, 16.0]
    assert len(rec.requests) == 1 + PoliteClient.MAX_ATTEMPTS


def test_transport_errors_are_retried_then_raised_as_fetch_error() -> None:
    rec, clock = Recorder(), FakeClock()
    rec.route("https://a.example/robots.txt", robots(ALLOW_ALL))

    def flaky(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    rec.route("https://a.example/one", flaky)
    with make_client(rec, clock) as client, pytest.raises(FetchError, match="failed after 5"):
        client.request("GET", "https://a.example/one")
    assert rec.paths().count("a.example/one") == PoliteClient.MAX_ATTEMPTS


# --- redirects and downloads ---------------------------------------------------


def test_redirect_target_host_must_pass_robots() -> None:
    rec, clock = Recorder(), FakeClock()
    rec.route("https://a.example/robots.txt", robots(ALLOW_ALL))
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


def test_too_many_redirects_is_a_fetch_error() -> None:
    rec, clock = Recorder(), FakeClock()
    rec.route("https://a.example/robots.txt", robots(ALLOW_ALL))
    rec.route("https://a.example/loop", httpx.Response(302, headers={"Location": "/loop"}))
    with make_client(rec, clock) as client, pytest.raises(FetchError, match="redirects"):
        client.request("GET", "https://a.example/loop")


def test_error_status_after_redirects_is_a_fetch_error() -> None:
    rec, clock = Recorder(), FakeClock()
    rec.route("https://a.example/robots.txt", robots(ALLOW_ALL))
    rec.route("https://a.example/gone", httpx.Response(410))
    with make_client(rec, clock) as client, pytest.raises(FetchError, match="410"):
        client.request("GET", "https://a.example/gone")


def test_download_follows_redirect_streams_and_hashes(tmp_path) -> None:
    payload = b"7z\xbc\xaf\x27\x1c" + bytes(range(256)) * 40
    rec, clock = Recorder(), FakeClock()
    rec.route("https://a.example/robots.txt", robots(ALLOW_ALL))
    rec.route(
        "https://a.example/download/x.7z",
        httpx.Response(302, headers={"Location": "https://cdn.example/x.7z"}),
    )
    rec.route("https://cdn.example/robots.txt", robots(ALLOW_ALL))
    rec.route(
        "https://cdn.example/x.7z",
        httpx.Response(200, content=payload, headers={"ETag": '"abc"'}),
    )
    dest = tmp_path / "partial.7z"
    with make_client(rec, clock) as client:
        result = client.download("https://a.example/download/x.7z", dest)
    assert dest.read_bytes() == payload
    assert result.sha256 == hashlib.sha256(payload).hexdigest()
    assert result.sha1 == hashlib.sha1(payload).hexdigest()
    assert result.md5 == hashlib.md5(payload).hexdigest()
    assert result.size == len(payload)
    assert result.final_url == "https://cdn.example/x.7z"
    assert result.headers["etag"] == '"abc"'


class BrokenStream(httpx.SyncByteStream):
    def __iter__(self) -> Iterator[bytes]:
        yield b"partial-"
        raise httpx.ReadError("connection dropped")


def test_failed_download_leaves_no_partial_file(tmp_path) -> None:
    rec, clock = Recorder(), FakeClock()
    rec.route("https://a.example/robots.txt", robots(ALLOW_ALL))
    rec.route("https://a.example/x.7z", lambda _req: httpx.Response(200, stream=BrokenStream()))
    dest = tmp_path / "partial.7z"
    with make_client(rec, clock) as client, pytest.raises(FetchError, match="mid-stream"):
        client.download("https://a.example/x.7z", dest)
    assert not dest.exists()

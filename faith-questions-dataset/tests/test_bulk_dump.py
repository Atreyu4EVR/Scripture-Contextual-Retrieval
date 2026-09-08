import hashlib
from pathlib import Path

import httpx
import pytest

from conftest import FakeClock
from faithqs.config import ConfigError, Settings
from faithqs.fetch import FETCHERS
from faithqs.fetch.bulk_dump import fetch_bulk_dump
from faithqs.fetch.polite import FetchError
from faithqs.manifest import SourceManifest, SourceUnresolvedError
from faithqs.storage import RawStore

PAYLOAD = b"7z\xbc\xaf\x27\x1c" + b"\x00" * 512
ROBOTS = "User-agent: *\nAllow: /\n"
DUMP_URL = "https://archive.example/download/item/dump.7z"


def make_manifest(**overrides) -> SourceManifest:
    base = {
        "name": "dump-source",
        "kind": "bulk_dump",
        "license": "CC-BY-SA-4.0",
        "redistributable": True,
        "attribution_required": True,
        "permission_status": "not_required",
        "downloads": [DUMP_URL],
    }
    base.update(overrides)
    return SourceManifest.model_validate(base)


class Server:
    def __init__(self, etag: str = '"v1"') -> None:
        self.etag = etag
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        path = request.url.path
        if path == "/robots.txt":
            return httpx.Response(200, text=ROBOTS)
        if path == "/download/item/dump.7z":
            headers = {"ETag": self.etag, "Content-Length": str(len(PAYLOAD))}
            if request.method == "HEAD":
                return httpx.Response(200, headers=headers)
            return httpx.Response(200, content=PAYLOAD, headers=headers)
        return httpx.Response(404)

    def methods(self) -> list[str]:
        return [r.method for r in self.requests]


def settings_for(tmp_path: Path) -> Settings:
    return Settings(data_dir=tmp_path / "data", contact_email="owner@example.org")


def run_fetch(tmp_path: Path, server: Server, manifest=None, settings=None):
    clock = FakeClock()
    outcomes = fetch_bulk_dump(
        manifest or make_manifest(),
        settings or settings_for(tmp_path),
        tmp_path / "sources",
        transport=httpx.MockTransport(server),
        log=lambda _msg: None,
        sleep=clock.sleep,
        clock=clock,
    )
    return outcomes, clock


def test_fetch_stores_content_hashed_payload_with_provenance(tmp_path: Path) -> None:
    server = Server()
    [outcome], clock = run_fetch(tmp_path, server)
    expected = hashlib.sha256(PAYLOAD).hexdigest()
    assert outcome.sha256 == expected
    assert outcome.skipped is False
    assert outcome.path == tmp_path / "data" / "raw" / "dump-source" / f"{expected}.7z"
    assert outcome.path.read_bytes() == PAYLOAD
    [payload] = RawStore(tmp_path / "data").list_payloads("dump-source")
    assert payload.meta["url"] == DUMP_URL
    assert payload.meta["etag"] == '"v1"'
    assert payload.meta["size"] == len(PAYLOAD)
    assert payload.meta["license"] == "CC-BY-SA-4.0"
    assert "contact: owner@example.org" in payload.meta["user_agent"]
    assert server.methods() == ["GET", "HEAD", "GET"]  # robots, HEAD probe, download
    assert clock.sleeps == [2.0, 2.0]  # rule 4 pacing between the three requests


def test_refetch_of_unchanged_upstream_skips_download(tmp_path: Path) -> None:
    server = Server()
    run_fetch(tmp_path, server)
    server.requests.clear()
    [outcome], _ = run_fetch(tmp_path, server)
    assert outcome.skipped is True
    assert server.methods() == ["GET", "HEAD"]  # robots, HEAD probe; no download


def test_changed_etag_triggers_redownload_and_store_stays_immutable(tmp_path: Path) -> None:
    run_fetch(tmp_path, Server('"v1"'))
    [outcome], _ = run_fetch(tmp_path, Server('"v2"'))
    # Same bytes in this fixture, so the hash collides and the store keeps the original.
    assert outcome.skipped is False
    assert len(RawStore(tmp_path / "data").list_payloads("dump-source")) == 1


def test_unresolved_manifest_never_touches_the_network(tmp_path: Path) -> None:
    server = Server()
    with pytest.raises(SourceUnresolvedError, match="rule 1"):
        run_fetch(tmp_path, server, manifest=make_manifest(license="unknown"))
    assert server.requests == []


def test_missing_contact_email_blocks_before_network(tmp_path: Path) -> None:
    server = Server()
    with pytest.raises(ConfigError, match="rule 3"):
        run_fetch(
            tmp_path,
            server,
            settings=Settings(data_dir=tmp_path / "data", contact_email=None),
        )
    assert server.requests == []


def test_manifest_without_downloads_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(FetchError, match="no downloads"):
        run_fetch(tmp_path, Server(), manifest=make_manifest(downloads=[]))


def test_pinned_checksums_are_verified_and_recorded(tmp_path: Path) -> None:
    spec = {
        "url": DUMP_URL,
        "sha1": hashlib.sha1(PAYLOAD).hexdigest().upper(),  # case-insensitive
        "md5": hashlib.md5(PAYLOAD).hexdigest(),
        "size": len(PAYLOAD),
    }
    [outcome], _ = run_fetch(tmp_path, Server(), manifest=make_manifest(downloads=[spec]))
    assert outcome.skipped is False
    [payload] = RawStore(tmp_path / "data").list_payloads("dump-source")
    assert payload.meta["sha1"] == hashlib.sha1(PAYLOAD).hexdigest()
    assert payload.meta["md5"] == hashlib.md5(PAYLOAD).hexdigest()
    assert payload.meta["verified_against"] == {
        "sha1": hashlib.sha1(PAYLOAD).hexdigest(),
        "md5": hashlib.md5(PAYLOAD).hexdigest(),
        "size": len(PAYLOAD),
    }


def test_checksum_mismatch_discards_download(tmp_path: Path) -> None:
    spec = {"url": DUMP_URL, "sha1": "0" * 40}
    with pytest.raises(FetchError, match="integrity check failed"):
        run_fetch(tmp_path, Server(), manifest=make_manifest(downloads=[spec]))
    raw_dir = tmp_path / "data" / "raw" / "dump-source"
    assert RawStore(tmp_path / "data").list_payloads("dump-source") == []
    assert not any(p.suffix == ".7z" for p in raw_dir.iterdir())  # partial removed


def test_non_bulk_kind_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(FetchError, match="not a bulk dump"):
        run_fetch(tmp_path, Server(), manifest=make_manifest(kind="crawl"))


def test_christianity_fetcher_is_registered() -> None:
    assert "christianity-stackexchange" in FETCHERS

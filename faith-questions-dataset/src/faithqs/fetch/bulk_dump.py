"""Fetch stage for ``bulk_dump`` sources: download listed files into ``data/raw/``.

No crawling happens here. Each URL in the manifest's ``downloads`` list is a
single file. Before downloading, a HEAD request is compared against the ETag
and size of payloads already stored, so re-running ``fetch`` on an unchanged
upstream costs one request and no bandwidth.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlsplit

import httpx

from faithqs.config import Settings
from faithqs.fetch.polite import FetchError, PoliteClient
from faithqs.manifest import SourceKind, SourceManifest
from faithqs.storage import RawStore


@dataclass(frozen=True)
class FetchOutcome:
    url: str
    path: Path
    sha256: str
    skipped: bool  # True when an identical payload was already stored


def _suffix_for(url: str) -> str:
    return Path(urlsplit(url).path).suffix or ".bin"


def _already_stored(store: RawStore, source: str, url: str, head: httpx.Response) -> Path | None:
    etag = head.headers.get("etag")
    length = head.headers.get("content-length")
    for payload in store.list_payloads(source):
        if payload.meta.get("url") != url:
            continue
        same_etag = etag is not None and payload.meta.get("etag") == etag
        same_length = length is not None and str(payload.meta.get("size")) == length
        if same_etag and same_length:
            return payload.path
    return None


def fetch_bulk_dump(
    manifest: SourceManifest,
    settings: Settings,
    sources_dir: Path,
    *,
    transport: httpx.BaseTransport | None = None,
    log: Callable[[str], None] = print,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> list[FetchOutcome]:
    manifest.require_resolved(sources_dir)  # rule 1
    if manifest.kind is not SourceKind.BULK_DUMP:
        raise FetchError(f"{manifest.name} is kind={manifest.kind.value}, not a bulk dump")
    if not manifest.downloads:
        raise FetchError(
            f"{manifest.name} lists no downloads; add direct file URLs to its manifest"
        )

    store = RawStore(settings.data_dir)
    outcomes: list[FetchOutcome] = []
    with PoliteClient(
        user_agent=settings.user_agent(),  # rule 3
        rate_limit_seconds=manifest.rate_limit_seconds,  # rule 4
        transport=transport,
        sleep=sleep,
        clock=clock,
    ) as client:
        for url in manifest.downloads:
            suffix = _suffix_for(url)
            head = client.request("HEAD", url)
            existing = _already_stored(store, manifest.name, url, head)
            if existing is not None:
                log(f"{manifest.name}: {url} unchanged upstream, already at {existing.name}")
                outcomes.append(
                    FetchOutcome(
                        url=url,
                        path=existing,
                        sha256=existing.name.split(".")[0],
                        skipped=True,
                    )
                )
                continue

            log(f"{manifest.name}: downloading {url}")
            result = client.download(url, store.temp_path(manifest.name, suffix))
            path = store.commit(
                result.path,
                source=manifest.name,
                sha256=result.sha256,
                suffix=suffix,
                meta={
                    "url": url,
                    "final_url": result.final_url,
                    "fetched_at": datetime.now(UTC).isoformat(),
                    "size": result.size,
                    "etag": result.headers.get("etag"),
                    "last_modified": result.headers.get("last-modified"),
                    "content_type": result.headers.get("content-type"),
                    "user_agent": client.user_agent,
                    "license": manifest.license,
                },
            )
            log(f"{manifest.name}: stored {path.name} ({result.size} bytes)")
            outcomes.append(FetchOutcome(url=url, path=path, sha256=result.sha256, skipped=False))
    return outcomes

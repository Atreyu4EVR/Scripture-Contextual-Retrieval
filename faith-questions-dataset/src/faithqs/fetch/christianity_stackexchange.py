"""Fetcher for the Christianity Stack Exchange data dump (Internet Archive).

A single-file bulk download; no crawling. The manifest at
``sources/christianity-stackexchange.yaml`` was resolved by the project owner
and the taxonomy was approved on 2026-09-07, satisfying rule 1 and Taxonomy
First before this module was written.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from faithqs.config import Settings
from faithqs.fetch.bulk_dump import FetchOutcome, fetch_bulk_dump
from faithqs.manifest import load_manifest

SOURCE_NAME = "christianity-stackexchange"


def fetch(settings: Settings, sources_dir: Path, **kwargs: Any) -> list[FetchOutcome]:
    """``kwargs`` are passed to :func:`fetch_bulk_dump` (transport, log, sleep, clock)."""
    manifest = load_manifest(sources_dir, SOURCE_NAME)
    return fetch_bulk_dump(manifest, settings, sources_dir, **kwargs)

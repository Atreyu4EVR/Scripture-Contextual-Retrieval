"""Parser for the Christianity Stack Exchange dump: selects Latter-day Saint questions.

Tag selection lives in the manifest (``filters.tags``) so a human tunes it from
the tag report each parse writes, without touching code.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from faithqs.config import Settings
from faithqs.manifest import load_manifest
from faithqs.parse import ParseOutcome
from faithqs.parse.stackexchange import parse_dump
from faithqs.storage import RawPayload

SOURCE_NAME = "christianity-stackexchange"


def parse(
    settings: Settings, sources_dir: Path, *, payload: RawPayload, **kwargs: Any
) -> ParseOutcome:
    manifest = load_manifest(sources_dir, SOURCE_NAME)
    manifest.require_resolved(sources_dir)  # rule 1 applies to parsing a payload too
    return parse_dump(manifest, settings, payload=payload, **kwargs)

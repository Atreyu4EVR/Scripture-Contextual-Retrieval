"""Per-source parsers, one module per source (CLAUDE.md, Repository Layout).

A parser turns one raw payload from ``data/raw/<source>/`` into
:class:`~faithqs.schema.SourceRecord` JSONL under ``data/staged/<source>/``,
named by the payload's hash so re-parsing is idempotent and never requires a
re-fetch. Parsers never write to ``data/parsed/``; that is ``scrub``'s job.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from faithqs.config import Settings
from faithqs.storage import RawPayload


@dataclass(frozen=True)
class ParseOutcome:
    source: str
    payload_sha256: str
    output_path: Path
    records_written: int
    records_skipped: int
    report_path: Path | None = None


Parser = Callable[..., ParseOutcome]


def _christianity_stackexchange(
    settings: Settings, sources_dir: Path, *, payload: RawPayload, **kwargs: Any
) -> ParseOutcome:
    from faithqs.parse import christianity_stackexchange

    return christianity_stackexchange.parse(settings, sources_dir, payload=payload, **kwargs)


PARSERS: dict[str, Parser] = {
    "christianity-stackexchange": _christianity_stackexchange,
}

__all__ = ["PARSERS", "ParseOutcome", "Parser"]

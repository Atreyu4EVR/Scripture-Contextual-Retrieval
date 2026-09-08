"""Per-source fetchers, one module per source (CLAUDE.md, Repository Layout).

A source appears in ``FETCHERS`` only after its manifest under ``sources/``
has been resolved by a human (rule 1) and the taxonomy has been approved
(Taxonomy First).
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from faithqs.config import Settings
from faithqs.fetch.bulk_dump import FetchOutcome

Fetcher = Callable[..., list[FetchOutcome]]


def _christianity_stackexchange(
    settings: Settings, sources_dir: Path, **kwargs: Any
) -> list[FetchOutcome]:
    from faithqs.fetch import christianity_stackexchange

    return christianity_stackexchange.fetch(settings, sources_dir, **kwargs)


FETCHERS: dict[str, Fetcher] = {
    "christianity-stackexchange": _christianity_stackexchange,
}

__all__ = ["FETCHERS", "FetchOutcome", "Fetcher"]

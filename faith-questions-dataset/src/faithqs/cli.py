"""Command line entry point: one subcommand per pipeline stage.

Stages are never chained (CLAUDE.md, Pipeline Stages). ``fetch`` writes
``data/raw/``; ``parse`` reads it and writes ``data/staged/``. Every gate the
project defines surfaces here as a ``STOP:`` line and exit code 2, so a human
sees exactly which rule halted the run.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

from faithqs.config import ConfigError, Settings
from faithqs.fetch import FETCHERS
from faithqs.fetch.polite import FetchError, RobotsDisallowedError, RobotsUnavailableError
from faithqs.manifest import ManifestError, SourceUnresolvedError
from faithqs.parse import PARSERS
from faithqs.schema import StorageGateError
from faithqs.storage import RawStore, StagedStore
from faithqs.taxonomy import (
    TaxonomyInvalidError,
    TaxonomyNotApprovedError,
    load_registers,
    load_taxonomy,
)

EXIT_OK = 0
EXIT_USAGE = 1
EXIT_STOP = 2

# Every project gate and configuration failure a human must resolve. Anything
# else is a bug and propagates as a traceback on purpose.
GATE_ERRORS = (
    ConfigError,
    ManifestError,
    SourceUnresolvedError,
    RobotsDisallowedError,
    RobotsUnavailableError,
    FetchError,
    StorageGateError,
    TaxonomyInvalidError,
    TaxonomyNotApprovedError,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="faithqs",
        description="Faith Questions Dataset pipeline. Run one stage at a time.",
    )
    parser.add_argument("--data-dir", type=Path, help="override FAITHQS_DATA_DIR")
    parser.add_argument("--sources-dir", type=Path, default=Path("sources"))
    parser.add_argument("--taxonomy-dir", type=Path, default=Path("taxonomy"))
    sub = parser.add_subparsers(dest="command", required=True)

    fetch = sub.add_parser("fetch", help="download a source's raw payloads into data/raw/")
    fetch.add_argument("source", choices=sorted(FETCHERS))

    parse = sub.add_parser(
        "parse", help="turn raw payloads into SourceRecord JSONL in data/staged/"
    )
    parse.add_argument("source", choices=sorted(PARSERS))
    parse.add_argument("--payload", help="only this raw payload sha256 (default: all unstaged)")
    parse.add_argument("--force", action="store_true", help="re-parse payloads already staged")
    return parser


def _require_approved_taxonomy(taxonomy_dir: Path) -> None:
    load_taxonomy(taxonomy_dir / "issues.yaml").require_approved()
    load_registers(taxonomy_dir / "registers.yaml").require_approved()


def _run_fetch(args: argparse.Namespace, settings: Settings) -> int:
    outcomes = FETCHERS[args.source](settings, args.sources_dir)
    for outcome in outcomes:
        state = "unchanged" if outcome.skipped else "stored"
        print(f"{state}  {outcome.sha256}  {outcome.url}")
    return EXIT_OK


def _run_parse(args: argparse.Namespace, settings: Settings) -> int:
    raw = RawStore(settings.data_dir)
    staged = StagedStore(settings.data_dir)
    payloads = raw.list_payloads(args.source)
    if args.payload:
        payloads = [p for p in payloads if p.sha256 == args.payload]
        if not payloads:
            print(f"STOP: no raw payload {args.payload} for {args.source}", file=sys.stderr)
            return EXIT_STOP
    if not payloads:
        print(f"STOP: nothing in data/raw/{args.source}/; run fetch first", file=sys.stderr)
        return EXIT_STOP
    for payload in payloads:
        if staged.output_path(args.source, payload.sha256).exists() and not args.force:
            print(f"skipped   {payload.sha256}  already staged")
            continue
        outcome = PARSERS[args.source](settings, args.sources_dir, payload=payload)
        print(
            f"staged    {outcome.payload_sha256}  {outcome.records_written} records "
            f"({outcome.records_skipped} skipped) -> {outcome.output_path}"
        )
    return EXIT_OK


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = Settings.from_env()
    if args.data_dir is not None:
        settings = replace(settings, data_dir=args.data_dir)
    try:
        _require_approved_taxonomy(args.taxonomy_dir)  # Taxonomy First
        if args.command == "fetch":
            return _run_fetch(args, settings)
        if args.command == "parse":
            return _run_parse(args, settings)
    except GATE_ERRORS as exc:
        print(f"STOP: {exc}", file=sys.stderr)
        return EXIT_STOP
    return EXIT_USAGE


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

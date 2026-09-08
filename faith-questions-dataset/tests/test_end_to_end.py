"""M0 proof: a bulk dump travels fetch -> parse -> SourceRecord -> FaithQuestionRecord.

Everything is offline. The dump is synthetic, the network is a mock transport,
and the clock is fake, so the test exercises every gate the pipeline enforces
without touching a real source.
"""

import hashlib
from pathlib import Path

import httpx

from conftest import PROJECT_ROOT, FakeClock
from faithqs import cli
from faithqs.config import Settings
from faithqs.fetch.bulk_dump import fetch_bulk_dump
from faithqs.manifest import load_manifest
from faithqs.parse.christianity_stackexchange import parse
from faithqs.schema import FaithQuestionRecord, ReleaseTier, SourceRecord
from faithqs.storage import RawStore
from faithqs.taxonomy import load_registers, load_taxonomy
from se_fixture import build_dump

SOURCES_DIR = PROJECT_ROOT / "sources"
ROBOTS = "User-agent: *\nDisallow: /search\nAllow: /\n"


def test_bulk_dump_flows_to_a_valid_release_record(tmp_path: Path) -> None:
    shipped = load_manifest(SOURCES_DIR, "christianity-stackexchange")
    assert shipped.downloads, "manifest must name the dump file to download"
    dump_bytes = build_dump(tmp_path / "fixture.7z").read_bytes()
    # The shipped manifest pins the real file's checksums; point them at the fixture
    # so the integrity check runs against what the mock server serves.
    [spec] = shipped.downloads
    manifest = shipped.model_copy(
        update={
            "downloads": [
                spec.model_copy(
                    update={
                        "sha1": hashlib.sha1(dump_bytes).hexdigest(),
                        "md5": hashlib.md5(dump_bytes).hexdigest(),
                        "size": len(dump_bytes),
                    }
                )
            ]
        }
    )

    def server(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return httpx.Response(200, text=ROBOTS)
        if str(request.url) == spec.url:
            headers = {"ETag": '"dump-v1"', "Content-Length": str(len(dump_bytes))}
            if request.method == "HEAD":
                return httpx.Response(200, headers=headers)
            return httpx.Response(200, content=dump_bytes, headers=headers)
        return httpx.Response(404)

    settings = Settings(data_dir=tmp_path / "data", contact_email="owner@example.org")
    clock = FakeClock()

    # Stage 1: fetch writes an immutable, content-hashed payload.
    [fetched] = fetch_bulk_dump(
        manifest,
        settings,
        SOURCES_DIR,
        transport=httpx.MockTransport(server),
        log=lambda _m: None,
        sleep=clock.sleep,
        clock=clock,
    )
    assert fetched.path.read_bytes() == dump_bytes
    [payload] = RawStore(settings.data_dir).list_payloads(manifest.name)

    # Stage 2: parse stages SourceRecords for wanted tags only.
    outcome = parse(settings, SOURCES_DIR, payload=payload, log=lambda _m: None)
    assert outcome.records_written >= 1
    staged = [
        SourceRecord.model_validate_json(line)
        for line in outcome.output_path.read_text(encoding="utf-8").splitlines()
    ]
    assert all(not record.pii_scrubbed for record in staged)
    assert all(record.source_license.startswith("CC-BY-SA") for record in staged)
    assert all(record.author_attribution for record in staged)

    # Stages 3 to 5 (scrub, extract, classify) are M1. Simulate their outputs to
    # prove the complete record validates against the approved taxonomy.
    taxonomy = load_taxonomy(PROJECT_ROOT / "taxonomy" / "issues.yaml")
    registers = load_registers(PROJECT_ROOT / "taxonomy" / "registers.yaml")
    taxonomy.require_approved()
    registers.require_approved()

    source = staged[0].model_copy(update={"pii_scrubbed": True})
    record = source.to_record(
        question_text="Why do the recorded accounts of the First Vision differ?",
        issue_id="first-vision-accounts",
        issue_category=taxonomy.category_of("first-vision-accounts"),
        register="sincere-inquiry",
        classifier_confidence=0.88,
    )
    taxonomy.check_issue_assignment(record)
    registers.check_register(record)
    record.ensure_storable()
    assert record.release_tier is ReleaseTier.A
    assert record.verbatim_text and record.author_attribution
    assert FaithQuestionRecord.model_validate_json(record.model_dump_json()) == record


def test_cli_parse_runs_real_parser_over_fetched_payload(tmp_path: Path, capsys) -> None:
    dump = build_dump(tmp_path / "fixture.7z")
    data_dir = tmp_path / "data"
    store = RawStore(data_dir)
    temp = store.temp_path("christianity-stackexchange", ".7z")
    temp.write_bytes(dump.read_bytes())
    store.commit(
        temp,
        source="christianity-stackexchange",
        sha256="d" * 64,
        suffix=".7z",
        meta={"url": "u", "size": dump.stat().st_size, "fetched_at": "2025-09-01T00:00:00+00:00"},
    )
    argv = [
        "--data-dir",
        str(data_dir),
        "--sources-dir",
        str(SOURCES_DIR),
        "--taxonomy-dir",
        str(PROJECT_ROOT / "taxonomy"),
        "parse",
        "christianity-stackexchange",
    ]
    assert cli.main(argv) == cli.EXIT_OK
    out = capsys.readouterr().out
    assert "staged" in out
    assert (data_dir / "staged" / "christianity-stackexchange" / ("d" * 64 + ".jsonl")).is_file()

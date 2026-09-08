import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from faithqs.config import Settings
from faithqs.manifest import SourceManifest
from faithqs.parse.stackexchange import (
    build_attribution,
    html_to_text,
    license_for,
    parse_dump,
    parse_tags,
    parse_timestamp,
)
from faithqs.schema import SourceRecord
from faithqs.storage import RawPayload
from se_fixture import build_dump

SITE = "https://christianity.stackexchange.com"


def make_manifest(**filters) -> SourceManifest:
    return SourceManifest.model_validate(
        {
            "name": "christianity-stackexchange",
            "kind": "bulk_dump",
            "license": "CC-BY-SA-4.0",
            "redistributable": True,
            "attribution_required": True,
            "permission_status": "not_required",
            "filters": {
                "site_url": SITE,
                "tags": ["mormonism", "book-of-mormon", "lds"],
                **filters,
            },
        }
    )


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("<mormonism><joseph-smith>", ["mormonism", "joseph-smith"]),
        ("|book-of-mormon|baptism|", ["book-of-mormon", "baptism"]),
        ("", []),
        (None, []),
        ("solo", ["solo"]),
    ],
)
def test_parse_tags_accepts_both_dump_encodings(raw, expected) -> None:
    assert parse_tags(raw) == expected


def test_license_prefers_row_attribute_then_creation_date() -> None:
    assert license_for(datetime(2010, 1, 1, tzinfo=UTC), "CC BY-SA 4.0") == "CC-BY-SA-4.0"
    assert license_for(datetime(2010, 1, 1, tzinfo=UTC), None) == "CC-BY-SA-2.5"
    assert license_for(datetime(2011, 4, 8, tzinfo=UTC), None) == "CC-BY-SA-3.0"
    assert license_for(datetime(2018, 5, 1, tzinfo=UTC), None) == "CC-BY-SA-3.0"
    assert license_for(datetime(2018, 5, 2, tzinfo=UTC), None) == "CC-BY-SA-4.0"


def test_timestamps_are_utc() -> None:
    parsed = parse_timestamp("2019-04-02T08:30:00.183")
    assert parsed == datetime(2019, 4, 2, 8, 30, 0, 183000, tzinfo=UTC)


def test_html_to_text_keeps_inline_text_together_and_separates_blocks() -> None:
    assert html_to_text("<p>First <em>point</em>.</p><p>Second point?</p>") == (
        "First point.\n\nSecond point?"
    )
    assert (
        html_to_text(
            "<p>Line one</p><blockquote><p>quoted <a href='x'>link</a></p></blockquote>"
            "<pre><code>x = 1</code></pre><ul><li>a</li><li>b</li></ul>"
        )
        == "Line one\n\nquoted link\n\nx = 1\n\na\n\nb"
    )
    assert html_to_text("one<br>two") == "one\ntwo"
    assert html_to_text("<script>alert(1)</script><p>kept</p>") == "kept"
    assert html_to_text("") == ""


def test_attribution_forms() -> None:
    full = build_attribution(
        site_url=SITE, post_id="1", owner_id="10", owner_name="Pilgrim", spdx="CC-BY-SA-4.0"
    )
    assert full == f"Pilgrim ({SITE}/users/10), {SITE}/q/1, CC-BY-SA-4.0"
    deleted = build_attribution(
        site_url=SITE, post_id="4", owner_id=None, owner_name="user4242", spdx="CC-BY-SA-4.0"
    )
    assert deleted.startswith("user4242, ")
    anonymous = build_attribution(
        site_url=SITE, post_id="9", owner_id=None, owner_name=None, spdx="CC-BY-SA-3.0"
    )
    assert anonymous.startswith("Stack Exchange contributor (account removed), ")


def test_parse_dump_stages_only_wanted_questions(tmp_path: Path) -> None:
    dump = build_dump(tmp_path / "dump.7z")
    payload = RawPayload(
        path=dump,
        sha256="f" * 64,
        meta={"fetched_at": "2025-09-01T12:00:00+00:00", "url": "u"},
    )
    settings = Settings(data_dir=tmp_path / "data", contact_email="owner@example.org")
    outcome = parse_dump(make_manifest(), settings, payload=payload, log=lambda _m: None)

    assert outcome.records_written == 4
    assert outcome.records_skipped == 1  # the purgatory question; the answer is not counted
    lines = outcome.output_path.read_text(encoding="utf-8").splitlines()
    records = [SourceRecord.model_validate_json(line) for line in lines]
    by_id = {r.source_record_id: r for r in records}
    assert set(by_id) == {"1", "2", "3", "4"}

    first = by_id["1"]
    assert first.title == "Why are there several accounts of the First Vision?"
    assert "several" in first.body_text and "<" not in first.body_text
    assert first.tags == ["mormonism", "joseph-smith"]
    assert first.source_license == "CC-BY-SA-4.0"
    assert first.author_attribution == f"Pilgrim ({SITE}/users/10), {SITE}/q/1, CC-BY-SA-4.0"
    assert first.source_url == f"{SITE}/q/1"
    assert first.collected_at == datetime(2025, 9, 1, 12, 0, tzinfo=UTC)
    assert first.created_at == datetime(2019, 4, 2, 8, 30, tzinfo=UTC)
    assert first.pii_scrubbed is False
    assert first.source_metadata["score"] == 7
    assert first.verbatim_text is not None

    assert by_id["2"].tags == ["book-of-mormon", "baptism"]  # pipe encoding
    assert by_id["3"].source_license == "CC-BY-SA-3.0"  # date fallback, no attribute
    assert by_id["4"].author_attribution.startswith("user4242, ")  # deleted owner

    report = json.loads(outcome.report_path.read_text(encoding="utf-8"))
    assert report["questions_selected"] == 4
    assert report["questions_skipped"] == 1
    assert report["licenses_selected"] == {"CC-BY-SA-4.0": 3, "CC-BY-SA-3.0": 1}
    assert report["tags_overall_top"]["catholicism"] == 1
    assert "mormonism" in report["tags_selected"]

    quarantine = json.loads(
        (outcome.output_path.parent / ("f" * 64 + ".quarantine.json")).read_text(encoding="utf-8")
    )
    assert quarantine == []


def test_parse_dump_requires_tag_filter(tmp_path: Path) -> None:
    dump = build_dump(tmp_path / "dump.7z")
    payload = RawPayload(path=dump, sha256="f" * 64, meta={})
    settings = Settings(data_dir=tmp_path / "data", contact_email="owner@example.org")
    with pytest.raises(ValueError, match=r"filters\.tags is empty"):
        parse_dump(make_manifest(tags=[]), settings, payload=payload, log=lambda _m: None)


def test_parse_dump_rejects_archive_missing_tables(tmp_path: Path) -> None:
    import py7zr

    bad = tmp_path / "bad.7z"
    with py7zr.SevenZipFile(bad, mode="w") as archive:
        archive.writestr("<posts/>", "Posts.xml")
    payload = RawPayload(path=bad, sha256="e" * 64, meta={})
    settings = Settings(data_dir=tmp_path / "data", contact_email="owner@example.org")
    with pytest.raises(ValueError, match="lacks expected tables"):
        parse_dump(make_manifest(), settings, payload=payload, log=lambda _m: None)

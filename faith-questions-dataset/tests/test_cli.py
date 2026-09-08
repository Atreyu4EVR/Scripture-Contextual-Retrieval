from pathlib import Path

import pytest

from conftest import PROJECT_ROOT
from faithqs import cli
from faithqs.fetch.bulk_dump import FetchOutcome
from faithqs.manifest import SourceUnresolvedError
from faithqs.parse import ParseOutcome
from faithqs.storage import RawStore

TAXONOMY_DIR = PROJECT_ROOT / "taxonomy"
SOURCE = "christianity-stackexchange"
SHA = "b" * 64


def base_args(tmp_path: Path) -> list[str]:
    return [
        "--data-dir",
        str(tmp_path / "data"),
        "--sources-dir",
        str(PROJECT_ROOT / "sources"),
        "--taxonomy-dir",
        str(TAXONOMY_DIR),
    ]


def test_fetch_dispatches_to_registered_fetcher(tmp_path: Path, monkeypatch, capsys) -> None:
    calls = []

    def fake_fetch(settings, sources_dir, **kwargs):
        calls.append((settings.data_dir, sources_dir))
        return [
            FetchOutcome(url="https://x/y.7z", path=tmp_path / "y.7z", sha256=SHA, skipped=False)
        ]

    monkeypatch.setitem(cli.FETCHERS, SOURCE, fake_fetch)
    assert cli.main([*base_args(tmp_path), "fetch", SOURCE]) == cli.EXIT_OK
    assert calls == [(tmp_path / "data", PROJECT_ROOT / "sources")]
    assert f"stored  {SHA}  https://x/y.7z" in capsys.readouterr().out


def test_gate_errors_become_stop_lines(tmp_path: Path, monkeypatch, capsys) -> None:
    def refusing_fetch(settings, sources_dir, **kwargs):
        raise SourceUnresolvedError("source is unresolved (CLAUDE.md rule 1)")

    monkeypatch.setitem(cli.FETCHERS, SOURCE, refusing_fetch)
    assert cli.main([*base_args(tmp_path), "fetch", SOURCE]) == cli.EXIT_STOP
    assert "STOP: source is unresolved" in capsys.readouterr().err


def test_unapproved_taxonomy_blocks_every_stage(tmp_path: Path, monkeypatch, capsys) -> None:
    draft_dir = tmp_path / "taxonomy"
    draft_dir.mkdir()
    (draft_dir / "issues.yaml").write_text(
        "approved: false\nversion: x\ncategories: [{id: c, label: C}]\n"
        "issues: [{id: i, category: c, label: I, description: D}]\n",
        encoding="utf-8",
    )
    (draft_dir / "registers.yaml").write_text(
        "approved: true\nversion: x\nregisters: [{id: r, label: R, description: D}]\n",
        encoding="utf-8",
    )
    monkeypatch.setitem(cli.FETCHERS, SOURCE, lambda *a, **k: pytest.fail("must not fetch"))
    argv = ["--data-dir", str(tmp_path / "data"), "--taxonomy-dir", str(draft_dir), "fetch", SOURCE]
    assert cli.main(argv) == cli.EXIT_STOP
    assert "Taxonomy First" in capsys.readouterr().err


def test_malformed_manifest_becomes_stop_line(tmp_path: Path, capsys) -> None:
    sources = tmp_path / "sources"
    sources.mkdir()
    shipped = (PROJECT_ROOT / "sources" / f"{SOURCE}.yaml").read_text(encoding="utf-8")
    assert "rate_limit_seconds: 2" in shipped
    (sources / f"{SOURCE}.yaml").write_text(
        shipped.replace("rate_limit_seconds: 2", "rate_limit_seconds: 1"), encoding="utf-8"
    )
    argv = [
        "--data-dir",
        str(tmp_path / "data"),
        "--sources-dir",
        str(sources),
        "--taxonomy-dir",
        str(TAXONOMY_DIR),
        "fetch",
        SOURCE,
    ]
    assert cli.main(argv) == cli.EXIT_STOP  # load_manifest fails before any network use
    err = capsys.readouterr().err
    assert err.startswith("STOP:")
    assert "rule 4" in err


def test_malformed_taxonomy_becomes_stop_line(tmp_path: Path, capsys) -> None:
    broken_dir = tmp_path / "taxonomy"
    broken_dir.mkdir()
    (broken_dir / "issues.yaml").write_text(
        "approved: true\nversion: x\ncategories: [{id: c, label: C}]\n"
        "issues: [{id: i, category: missing, label: I, description: D}]\n",
        encoding="utf-8",
    )
    (broken_dir / "registers.yaml").write_text(
        "approved: true\nversion: x\nregisters: [{id: r, label: R, description: D}]\n",
        encoding="utf-8",
    )
    argv = [
        "--data-dir",
        str(tmp_path / "data"),
        "--taxonomy-dir",
        str(broken_dir),
        "fetch",
        SOURCE,
    ]
    assert cli.main(argv) == cli.EXIT_STOP
    err = capsys.readouterr().err
    assert err.startswith("STOP:")
    assert "unknown categories" in err


def test_unknown_source_is_a_usage_error(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as excinfo:
        cli.main([*base_args(tmp_path), "fetch", "not-a-source"])
    assert excinfo.value.code == 2


def test_parse_requires_fetched_payloads(tmp_path: Path, capsys) -> None:
    assert cli.main([*base_args(tmp_path), "parse", SOURCE]) == cli.EXIT_STOP
    assert "run fetch first" in capsys.readouterr().err


def test_parse_is_idempotent_per_payload(tmp_path: Path, monkeypatch, capsys) -> None:
    data_dir = tmp_path / "data"
    store = RawStore(data_dir)
    temp = store.temp_path(SOURCE, ".7z")
    temp.write_bytes(b"dump")
    store.commit(temp, source=SOURCE, sha256=SHA, suffix=".7z", meta={"url": "u", "size": 4})

    parsed = []

    def fake_parse(settings, sources_dir, *, payload, **kwargs):
        parsed.append(payload.sha256)
        out = settings.data_dir / "staged" / SOURCE / f"{payload.sha256}.jsonl"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("", encoding="utf-8")
        return ParseOutcome(
            source=SOURCE,
            payload_sha256=payload.sha256,
            output_path=out,
            records_written=0,
            records_skipped=0,
        )

    monkeypatch.setitem(cli.PARSERS, SOURCE, fake_parse)
    assert cli.main([*base_args(tmp_path), "parse", SOURCE]) == cli.EXIT_OK
    assert cli.main([*base_args(tmp_path), "parse", SOURCE]) == cli.EXIT_OK
    assert parsed == [SHA]  # second run skipped the already-staged payload
    assert "already staged" in capsys.readouterr().out

    assert cli.main([*base_args(tmp_path), "parse", SOURCE, "--force"]) == cli.EXIT_OK
    assert parsed == [SHA, SHA]

    assert cli.main([*base_args(tmp_path), "parse", SOURCE, "--payload", "c" * 64]) == cli.EXIT_STOP

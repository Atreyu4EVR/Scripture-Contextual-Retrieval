import json
from pathlib import Path

from faithqs.storage import RawStore, StagedStore

SHA = "a" * 64


def test_commit_moves_payload_and_writes_sidecar(tmp_path: Path) -> None:
    store = RawStore(tmp_path)
    temp = store.temp_path("src", ".7z")
    temp.write_bytes(b"payload")
    dest = store.commit(
        temp, source="src", sha256=SHA, suffix=".7z", meta={"url": "https://x/y.7z"}
    )
    assert dest == tmp_path / "raw" / "src" / f"{SHA}.7z"
    assert dest.read_bytes() == b"payload"
    assert not temp.exists()
    meta = json.loads(store.meta_path("src", SHA).read_text(encoding="utf-8"))
    assert meta["url"] == "https://x/y.7z"
    assert meta["sha256"] == SHA
    assert meta["file"] == f"{SHA}.7z"
    assert "stored_at" in meta


def test_existing_payload_is_never_overwritten(tmp_path: Path) -> None:
    store = RawStore(tmp_path)
    first = store.temp_path("src", ".7z")
    first.write_bytes(b"original")
    store.commit(first, source="src", sha256=SHA, suffix=".7z", meta={"url": "u"})

    second = store.temp_path("src", ".7z")
    second.write_bytes(b"tampered")
    dest = store.commit(second, source="src", sha256=SHA, suffix=".7z", meta={"url": "u2"})
    assert dest.read_bytes() == b"original"
    assert not second.exists()
    meta = json.loads(store.meta_path("src", SHA).read_text(encoding="utf-8"))
    assert meta["url"] == "u"


def test_list_payloads_reads_sidecars(tmp_path: Path) -> None:
    store = RawStore(tmp_path)
    assert store.list_payloads("src") == []
    temp = store.temp_path("src", ".7z")
    temp.write_bytes(b"x")
    store.commit(temp, source="src", sha256=SHA, suffix=".7z", meta={"url": "u", "size": 1})
    [payload] = store.list_payloads("src")
    assert payload.sha256 == SHA
    assert payload.path.is_file()
    assert payload.meta["size"] == 1


def test_staged_paths_are_keyed_by_raw_hash(tmp_path: Path) -> None:
    staged = StagedStore(tmp_path)
    staged_dir = tmp_path / "staged" / "src"
    assert staged.output_path("src", SHA) == staged_dir / f"{SHA}.jsonl"
    assert staged.report_path("src", SHA, "tags") == staged_dir / f"{SHA}.tags.json"

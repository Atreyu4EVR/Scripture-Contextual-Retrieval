"""On-disk layout under ``data/`` (gitignored in full, CLAUDE.md rule 5).

``data/raw/<source>/`` holds immutable, content-hashed payloads exactly as
fetched, each with a ``.meta.json`` sidecar recording where and when it came
from. Nothing here is ever edited in place; a changed upstream file lands as a
new hash. ``data/staged/<source>/`` holds ``parse`` output keyed by the raw
payload hash it was derived from.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RawPayload:
    path: Path
    sha256: str
    meta: dict[str, Any]


class RawStore:
    def __init__(self, data_dir: Path) -> None:
        self.root = data_dir / "raw"

    def source_dir(self, source: str) -> Path:
        return self.root / source

    def payload_path(self, source: str, sha256: str, suffix: str) -> Path:
        return self.source_dir(source) / f"{sha256}{suffix}"

    def meta_path(self, source: str, sha256: str) -> Path:
        return self.source_dir(source) / f"{sha256}.meta.json"

    def temp_path(self, source: str, suffix: str) -> Path:
        directory = self.source_dir(source)
        directory.mkdir(parents=True, exist_ok=True)
        return directory / f".partial-{os.getpid()}{suffix}"

    def has(self, source: str, sha256: str, suffix: str) -> bool:
        return self.payload_path(source, sha256, suffix).is_file()

    def commit(
        self,
        temp_path: Path,
        *,
        source: str,
        sha256: str,
        suffix: str,
        meta: dict[str, Any],
    ) -> Path:
        """Move a fully downloaded file into place. Existing payloads are never replaced."""
        dest = self.payload_path(source, sha256, suffix)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            temp_path.unlink(missing_ok=True)
        else:
            os.replace(temp_path, dest)
        meta_path = self.meta_path(source, sha256)
        if not meta_path.exists():
            record = {
                "sha256": sha256,
                "file": dest.name,
                "stored_at": datetime.now(UTC).isoformat(),
                **meta,
            }
            meta_path.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
        return dest

    def list_payloads(self, source: str) -> list[RawPayload]:
        directory = self.source_dir(source)
        if not directory.is_dir():
            return []
        payloads: list[RawPayload] = []
        for meta_path in sorted(directory.glob("*.meta.json")):
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            path = directory / meta["file"]
            if path.is_file():
                payloads.append(RawPayload(path=path, sha256=meta["sha256"], meta=meta))
        return payloads


class StagedStore:
    def __init__(self, data_dir: Path) -> None:
        self.root = data_dir / "staged"

    def output_path(self, source: str, sha256: str) -> Path:
        directory = self.root / source
        directory.mkdir(parents=True, exist_ok=True)
        return directory / f"{sha256}.jsonl"

    def report_path(self, source: str, sha256: str, name: str) -> Path:
        directory = self.root / source
        directory.mkdir(parents=True, exist_ok=True)
        return directory / f"{sha256}.{name}.json"

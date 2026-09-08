"""Shared parser for Stack Exchange data dumps (a .7z of XML tables).

Site-specific modules (one per source, per CLAUDE.md) supply the manifest and
call :func:`parse_dump`. Only questions (``PostTypeId="1"``) carrying at least
one wanted tag are staged; everything else is counted and dropped.

Format facts the parser is built around (verified against the Internet
Archive's 2024-04 ``christianity.stackexchange.com.7z`` and the published
schema, 2026-09-08):

* Every XML member starts with a UTF-8 BOM and uses CRLF line endings.
  ``ElementTree.iterparse`` handles both when given a path.
* ``Tags`` is encoded two ways across dump generations: angle brackets
  (``<a><b>``, the normal form) and pipes (``|a|b|``, the form the 2024-04
  dump shipped with). Both are accepted.
* Posts carry a per-row ``ContentLicense`` attribute (``CC BY-SA 2.5 / 3.0 /
  4.0``) reflecting the latest revision's license. When it is absent the
  license is inferred from ``CreationDate`` using Stack Exchange's published
  cutovers (3.0 from 2011-04-08, 4.0 from 2018-05-02).
* Deleted accounts leave ``OwnerUserId`` empty and may leave an
  ``OwnerDisplayName`` behind. Attribution names what the dump provides.
* Dumps since mid-2025 may contain fabricated watermark rows with
  ``Id >= 1000000000``; defective dumps may contain deleted posts (rows with
  ``DeletionDate`` and no ``Body``). Both are skipped and counted.
* Only ``DisplayName`` is read from ``Users.xml`` (for attribution). Location,
  AboutMe, and the other profile fields are never loaded (Content Sensitivity).
* Blockquotes are kept: in question bodies they are usually the scripture or
  statement being asked about, so they are part of the question.
"""

from __future__ import annotations

import json
import re
import tempfile
import xml.etree.ElementTree as ET
from collections import Counter
from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import py7zr
from pydantic import ValidationError
from selectolax.parser import HTMLParser, Node

from faithqs.config import Settings
from faithqs.manifest import ManifestError, SourceManifest
from faithqs.parse import ParseOutcome
from faithqs.schema import QuarantineRecord, SourceRecord
from faithqs.storage import RawPayload, StagedStore

QUESTION_POST_TYPE = "1"
WATERMARK_ID_FLOOR = 1_000_000_000

# (effective from, SPDX id), newest first. Stack Exchange terms of service.
LICENSE_CUTOVERS: tuple[tuple[datetime, str], ...] = (
    (datetime(2018, 5, 2, tzinfo=UTC), "CC-BY-SA-4.0"),
    (datetime(2011, 4, 8, tzinfo=UTC), "CC-BY-SA-3.0"),
)
OLDEST_LICENSE = "CC-BY-SA-2.5"

_ANGLE_TAG = re.compile(r"<([^<>]+)>")
_INLINE_WS = re.compile(r"[ \t\r\f\v]+")
_BLANK_LINES = re.compile(r"\n\s*\n+")

BLOCK_TAGS = frozenset(
    {
        "p",
        "div",
        "li",
        "ul",
        "ol",
        "blockquote",
        "pre",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "table",
        "tr",
        "hr",
    }
)
DROP_TAGS = frozenset({"script", "style"})


def normalize_license(raw: str) -> str:
    """``"CC BY-SA 4.0"`` (dump spelling) to ``"CC-BY-SA-4.0"`` (SPDX)."""
    return raw.strip().upper().replace(" ", "-")


def license_for(created_at: datetime, content_license: str | None) -> str:
    if content_license:
        return normalize_license(content_license)
    for effective_from, spdx in LICENSE_CUTOVERS:
        if created_at >= effective_from:
            return spdx
    return OLDEST_LICENSE


def parse_tags(raw: str | None) -> list[str]:
    if not raw:
        return []
    raw = raw.strip()
    if raw.startswith("<"):
        return _ANGLE_TAG.findall(raw)
    if "|" in raw:
        return [tag for tag in raw.split("|") if tag]
    return [raw]


def parse_timestamp(raw: str) -> datetime:
    """Dump timestamps are naive ISO 8601 in UTC."""
    parsed = datetime.fromisoformat(raw)
    return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed.astimezone(UTC)


def _collect_text(node: Node, out: list[str]) -> None:
    for child in node.iter(include_text=True):
        tag = child.tag
        if tag == "-text":
            out.append(child.text_content or "")
        elif tag in DROP_TAGS:
            continue
        elif tag == "br":
            out.append("\n")
        else:
            block = tag in BLOCK_TAGS
            if block:
                out.append("\n\n")
            _collect_text(child, out)
            if block:
                out.append("\n\n")


def html_to_text(html: str) -> str:
    """Plain text with inline markup flattened and block elements separated by blank lines."""
    body = HTMLParser(html).body
    if body is None:
        return ""
    parts: list[str] = []
    _collect_text(body, parts)
    lines = [_INLINE_WS.sub(" ", line).strip() for line in "".join(parts).split("\n")]
    return _BLANK_LINES.sub("\n\n", "\n".join(lines)).strip()


def iter_rows(xml_path: Path) -> Iterator[dict[str, str]]:
    """Stream ``<row>`` attribute dicts with memory bounded regardless of file size."""
    context = ET.iterparse(str(xml_path), events=("start", "end"))
    _, root = next(context)
    for event, elem in context:
        if event == "end" and elem.tag == "row":
            yield dict(elem.attrib)
            elem.clear()
            root.remove(elem)


def load_display_names(users_xml: Path) -> dict[str, str]:
    """Only DisplayName is read; no other profile field leaves Users.xml."""
    names: dict[str, str] = {}
    for row in iter_rows(users_xml):
        user_id, name = row.get("Id"), row.get("DisplayName")
        if user_id and name:
            names[user_id] = name
    return names


def build_attribution(
    *,
    site_url: str,
    site_name: str,
    post_id: str,
    owner_id: str | None,
    owner_name: str | None,
    spdx: str,
) -> str:
    """The credit Stack Exchange's license terms require.

    Author name, a direct link to the author's profile, the originating site
    named visibly, a direct link to the original question, and the license.
    """
    if owner_id and owner_name:
        author = f"{owner_name} ({site_url}/users/{owner_id})"
    elif owner_name:
        author = owner_name
    else:
        author = "Stack Exchange contributor (account removed)"
    return f"{author}, {site_name}, {site_url}/questions/{post_id}, {spdx}"


def extract_members(archive: Path, members: list[str], dest: Path) -> dict[str, Path]:
    """Extract the named tables to ``dest``.

    Dumps are solid LZMA2 archives, so extracting any member decompresses the
    whole stream once; asking for the two tables together costs one pass.
    """
    with py7zr.SevenZipFile(archive, mode="r") as dump:
        available = set(dump.getnames())
        missing = [m for m in members if m not in available]
        if missing:
            raise ValueError(f"{archive.name} lacks expected tables: {missing}")
        dump.extract(path=dest, targets=members)
    return {member: dest / member for member in members}


def parse_dump(
    manifest: SourceManifest,
    settings: Settings,
    *,
    payload: RawPayload,
    now: datetime | None = None,
    log: Callable[[str], None] = print,
) -> ParseOutcome:
    wanted = {tag.lower() for tag in manifest.filters.get("tags", [])}
    if not wanted:
        raise ManifestError(f"{manifest.name}: manifest filters.tags is empty; nothing to select")
    site_url = str(manifest.filters.get("site_url", "")).rstrip("/")
    if not site_url:
        raise ManifestError(f"{manifest.name}: manifest filters.site_url is required")
    site_name = str(manifest.filters.get("site_name") or urlsplit(site_url).netloc)

    collected_at = (
        (parse_timestamp(payload.meta["fetched_at"]) if "fetched_at" in payload.meta else None)
        or now
        or datetime.now(UTC)
    )

    staged = StagedStore(settings.data_dir)
    output_path = staged.output_path(manifest.name, payload.sha256)
    quarantine_path = staged.report_path(manifest.name, payload.sha256, "quarantine")
    report_path = staged.report_path(manifest.name, payload.sha256, "report")

    written = 0
    skipped: Counter[str] = Counter()
    tag_counts_all: Counter[str] = Counter()
    tag_counts_selected: Counter[str] = Counter()
    license_counts: Counter[str] = Counter()
    quarantine: list[dict[str, Any]] = []

    with tempfile.TemporaryDirectory(prefix="faithqs-se-") as tmp:
        tables = extract_members(payload.path, ["Users.xml", "Posts.xml"], Path(tmp))
        names = load_display_names(tables["Users.xml"])
        log(f"{manifest.name}: {len(names)} display names loaded")

        partial = output_path.with_suffix(".jsonl.partial")
        with partial.open("w", encoding="utf-8") as out:
            for row in iter_rows(tables["Posts.xml"]):
                if row.get("PostTypeId") != QUESTION_POST_TYPE:
                    continue
                post_id = row["Id"]
                if int(post_id) >= WATERMARK_ID_FLOOR:
                    skipped["watermark"] += 1
                    continue
                if row.get("DeletionDate") or not row.get("Body"):
                    skipped["deleted"] += 1
                    continue
                tags = parse_tags(row.get("Tags"))
                tag_counts_all.update(tags)
                if not wanted.intersection(tag.lower() for tag in tags):
                    skipped["untagged"] += 1
                    continue

                created_at = parse_timestamp(row["CreationDate"])
                spdx = license_for(created_at, row.get("ContentLicense"))
                owner_id = row.get("OwnerUserId") or None
                owner_name = names.get(owner_id or "") or row.get("OwnerDisplayName") or None
                candidate = {
                    "source_name": manifest.name,
                    "source_record_id": post_id,
                    "source_url": f"{site_url}/questions/{post_id}",
                    "source_license": spdx,
                    "redistributable": manifest.redistributable,
                    "author_attribution": build_attribution(
                        site_url=site_url,
                        site_name=site_name,
                        post_id=post_id,
                        owner_id=owner_id,
                        owner_name=owner_name,
                        spdx=spdx,
                    ),
                    "permission_ref": manifest.permission_ref,
                    "collected_at": collected_at,
                    "title": row.get("Title") or None,
                    "body_text": html_to_text(row["Body"]),
                    "tags": tags,
                    "created_at": created_at,
                    "source_metadata": {
                        "score": _int(row.get("Score")),
                        "view_count": _int(row.get("ViewCount")),
                        "answer_count": _int(row.get("AnswerCount")),
                        "accepted_answer_id": row.get("AcceptedAnswerId"),
                        "closed": bool(row.get("ClosedDate")),
                        "content_license_attr": row.get("ContentLicense"),
                    },
                    "pii_scrubbed": False,
                }
                try:
                    record = SourceRecord.model_validate(candidate)
                except ValidationError as exc:
                    skipped["quarantined"] += 1
                    # Identifiers and error locations only; never field values.
                    quarantine.append(
                        QuarantineRecord.from_validation_error(candidate, exc).model_dump(
                            mode="json"
                        )
                    )
                    continue
                out.write(record.model_dump_json())
                out.write("\n")
                written += 1
                tag_counts_selected.update(tags)
                license_counts[spdx] += 1
        partial.replace(output_path)

    quarantine_path.write_text(json.dumps(quarantine, indent=2), encoding="utf-8")
    report = {
        "source": manifest.name,
        "payload_sha256": payload.sha256,
        "site_name": site_name,
        "questions_selected": written,
        "questions_skipped_untagged": skipped["untagged"],
        "questions_skipped_watermark": skipped["watermark"],
        "questions_skipped_deleted": skipped["deleted"],
        "questions_quarantined": skipped["quarantined"],
        "tags_wanted": sorted(wanted),
        "licenses_selected": dict(license_counts),
        "tags_selected": dict(tag_counts_selected.most_common()),
        "tags_overall_top": dict(tag_counts_all.most_common(200)),
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    total_skipped = sum(skipped.values())
    log(
        f"{manifest.name}: staged {written} questions, skipped {total_skipped} "
        f"({dict(skipped)}) -> {output_path}"
    )
    return ParseOutcome(
        source=manifest.name,
        payload_sha256=payload.sha256,
        output_path=output_path,
        records_written=written,
        records_skipped=total_skipped,
        report_path=report_path,
    )


def _int(raw: str | None) -> int | None:
    try:
        return int(raw) if raw is not None else None
    except ValueError:
        return None

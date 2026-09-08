"""Shared parser for Stack Exchange data dumps (a .7z of XML tables).

Site-specific modules (one per source, per CLAUDE.md) supply the manifest and
call :func:`parse_dump`. Only questions (``PostTypeId="1"``) carrying at least
one wanted tag are staged; everything else is counted and dropped.

Format notes the parser is defensive about:

* ``Tags`` has been encoded two ways across dump generations: angle brackets
  (``<a><b>``) and pipes (``|a|b|``). Both are accepted.
* Recent dumps carry a per-row ``ContentLicense`` attribute. When it is
  absent, the license is inferred from ``CreationDate`` using Stack Exchange's
  published cutovers (CC BY-SA 2.5, then 3.0 from 2011-04-08, then 4.0 from
  2018-05-02).
* Deleted accounts leave ``OwnerUserId`` empty and may leave an
  ``OwnerDisplayName`` behind. Attribution still names what the dump provides,
  since CC BY-SA requires the credit the licensor gave.
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

import py7zr
from pydantic import ValidationError
from selectolax.parser import HTMLParser, Node

from faithqs.config import Settings
from faithqs.manifest import SourceManifest
from faithqs.parse import ParseOutcome
from faithqs.schema import SourceRecord
from faithqs.storage import RawPayload, StagedStore

QUESTION_POST_TYPE = "1"

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
    """Stream ``<row>`` attribute dicts with bounded memory."""
    context = ET.iterparse(str(xml_path), events=("start", "end"))
    _, root = next(context)
    for event, elem in context:
        if event == "end" and elem.tag == "row":
            yield dict(elem.attrib)
            elem.clear()
            root.clear()


def load_display_names(users_xml: Path) -> dict[str, str]:
    names: dict[str, str] = {}
    for row in iter_rows(users_xml):
        user_id, name = row.get("Id"), row.get("DisplayName")
        if user_id and name:
            names[user_id] = name
    return names


def build_attribution(
    *, site_url: str, post_id: str, owner_id: str | None, owner_name: str | None, spdx: str
) -> str:
    """The credit CC BY-SA requires: author, author link, and link to the work."""
    if owner_id and owner_name:
        author = f"{owner_name} ({site_url}/users/{owner_id})"
    elif owner_name:
        author = owner_name
    else:
        author = "Stack Exchange contributor (account removed)"
    return f"{author}, {site_url}/q/{post_id}, {spdx}"


def extract_members(archive: Path, members: list[str], dest: Path) -> dict[str, Path]:
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
        raise ValueError(f"{manifest.name}: manifest filters.tags is empty; nothing to select")
    site_url = str(manifest.filters.get("site_url", "")).rstrip("/")
    if not site_url:
        raise ValueError(f"{manifest.name}: manifest filters.site_url is required")

    collected_at = (
        (parse_timestamp(payload.meta["fetched_at"]) if "fetched_at" in payload.meta else None)
        or now
        or datetime.now(UTC)
    )

    staged = StagedStore(settings.data_dir)
    output_path = staged.output_path(manifest.name, payload.sha256)
    quarantine_path = staged.report_path(manifest.name, payload.sha256, "quarantine")
    report_path = staged.report_path(manifest.name, payload.sha256, "report")

    written = skipped = quarantined = 0
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
                tags = parse_tags(row.get("Tags"))
                tag_counts_all.update(tags)
                if not wanted.intersection(tag.lower() for tag in tags):
                    skipped += 1
                    continue

                post_id = row["Id"]
                created_at = parse_timestamp(row["CreationDate"])
                spdx = license_for(created_at, row.get("ContentLicense"))
                owner_id = row.get("OwnerUserId") or None
                owner_name = names.get(owner_id or "") or row.get("OwnerDisplayName") or None
                candidate = {
                    "source_name": manifest.name,
                    "source_record_id": post_id,
                    "source_url": f"{site_url}/q/{post_id}",
                    "source_license": spdx,
                    "redistributable": manifest.redistributable,
                    "author_attribution": build_attribution(
                        site_url=site_url,
                        post_id=post_id,
                        owner_id=owner_id,
                        owner_name=owner_name,
                        spdx=spdx,
                    ),
                    "permission_ref": manifest.permission_ref,
                    "collected_at": collected_at,
                    "title": row.get("Title") or None,
                    "body_text": html_to_text(row.get("Body", "")),
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
                    quarantined += 1
                    # Locations and messages only: never copy field values into a sidecar.
                    quarantine.append(
                        {
                            "source_record_id": post_id,
                            "errors": [
                                {"loc": list(e["loc"]), "msg": e["msg"], "type": e["type"]}
                                for e in exc.errors(include_input=False, include_url=False)
                            ],
                        }
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
        "questions_selected": written,
        "questions_skipped": skipped,
        "questions_quarantined": quarantined,
        "tags_wanted": sorted(wanted),
        "licenses_selected": dict(license_counts),
        "tags_selected": dict(tag_counts_selected.most_common()),
        "tags_overall_top": dict(tag_counts_all.most_common(200)),
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    log(
        f"{manifest.name}: staged {written} questions, skipped {skipped}, "
        f"quarantined {quarantined} -> {output_path}"
    )
    return ParseOutcome(
        source=manifest.name,
        payload_sha256=payload.sha256,
        output_path=output_path,
        records_written=written,
        records_skipped=skipped + quarantined,
        report_path=report_path,
    )


def _int(raw: str | None) -> int | None:
    try:
        return int(raw) if raw is not None else None
    except ValueError:
        return None

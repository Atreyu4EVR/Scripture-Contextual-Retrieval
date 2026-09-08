from pathlib import Path

import pytest
from pydantic import ValidationError

from conftest import PROJECT_ROOT
from faithqs.manifest import (
    PermissionStatus,
    SourceKind,
    SourceManifest,
    SourceUnresolvedError,
    load_manifest,
)

SOURCES_DIR = PROJECT_ROOT / "sources"


def manifest_kwargs(**overrides):
    base = {
        "name": "example-source",
        "kind": "bulk_dump",
        "license": "CC-BY-SA-4.0",
        "redistributable": True,
        "attribution_required": True,
        "permission_status": "not_required",
    }
    base.update(overrides)
    return base


def test_shipped_christianity_manifest_is_resolved() -> None:
    manifest = load_manifest(SOURCES_DIR, "christianity-stackexchange")
    assert manifest.kind is SourceKind.BULK_DUMP
    assert manifest.license == "CC-BY-SA-4.0"
    assert manifest.redistributable is True
    assert manifest.permission_status is PermissionStatus.NOT_REQUIRED
    manifest.require_resolved(SOURCES_DIR)
    [download] = manifest.downloads
    assert download.url.endswith("christianity.stackexchange.com.7z")
    assert download.sha1 and download.md5 and download.size  # pinned: the IA file is frozen
    assert "lds" in manifest.filters["tags"]
    assert manifest.filters["site_name"] == "Christianity Stack Exchange"


def test_bare_download_urls_become_specs() -> None:
    manifest = SourceManifest.model_validate(
        manifest_kwargs(downloads=["https://x/a.7z", {"url": "https://x/b.7z", "size": 5}])
    )
    assert [d.url for d in manifest.downloads] == ["https://x/a.7z", "https://x/b.7z"]
    assert manifest.downloads[0].sha1 is None
    assert manifest.downloads[1].size == 5


def test_malformed_checksum_rejected() -> None:
    with pytest.raises(ValidationError):
        SourceManifest.model_validate(
            manifest_kwargs(downloads=[{"url": "https://x/a.7z", "sha1": "not-hex"}])
        )


def test_missing_manifest_is_a_rule_one_stop(tmp_path: Path) -> None:
    with pytest.raises(SourceUnresolvedError, match="rule 1"):
        load_manifest(tmp_path, "nonexistent")


def test_manifest_name_must_match_filename(tmp_path: Path) -> None:
    (tmp_path / "wrong.yaml").write_text(
        "name: right\nkind: bulk_dump\nlicense: MIT\nredistributable: true\n"
        "attribution_required: false\npermission_status: not_required\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="declares name"):
        load_manifest(tmp_path, "wrong")


@pytest.mark.parametrize("license_value", ["unknown", "TBD", "", "pending"])
def test_unresolved_license_blocks(tmp_path: Path, license_value: str) -> None:
    manifest = SourceManifest.model_validate(manifest_kwargs(license=license_value))
    with pytest.raises(SourceUnresolvedError, match="license"):
        manifest.require_resolved(tmp_path / "sources")


@pytest.mark.parametrize("status", ["requested", "denied", "unresolved"])
def test_unsettled_permission_blocks(tmp_path: Path, status: str) -> None:
    manifest = SourceManifest.model_validate(manifest_kwargs(permission_status=status))
    with pytest.raises(SourceUnresolvedError, match="permission_status"):
        manifest.require_resolved(tmp_path / "sources")


def test_granted_permission_requires_filed_correspondence(tmp_path: Path) -> None:
    sources = tmp_path / "sources"
    (sources / "permissions").mkdir(parents=True)
    without_ref = SourceManifest.model_validate(manifest_kwargs(permission_status="granted"))
    with pytest.raises(SourceUnresolvedError, match="permission_ref is empty"):
        without_ref.require_resolved(sources)

    dangling = SourceManifest.model_validate(
        manifest_kwargs(
            permission_status="granted",
            permission_ref="sources/permissions/example-2026-01-01.pdf",
        )
    )
    with pytest.raises(SourceUnresolvedError, match="not filed"):
        dangling.require_resolved(sources)

    (sources / "permissions" / "example-2026-01-01.pdf").write_bytes(b"%PDF-1.4")
    dangling.require_resolved(sources)


def test_rate_limit_floor_is_enforced() -> None:
    with pytest.raises(ValidationError, match="rule 4"):
        SourceManifest.model_validate(manifest_kwargs(rate_limit_seconds=0.5))
    slower = SourceManifest.model_validate(manifest_kwargs(rate_limit_seconds=10))
    assert slower.rate_limit_seconds == 10


def test_unknown_manifest_keys_rejected() -> None:
    with pytest.raises(ValidationError):
        SourceManifest.model_validate(manifest_kwargs(bypass_robots=True))

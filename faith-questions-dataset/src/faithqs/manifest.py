"""Source manifests: one YAML file per source under ``sources/``.

The manifest is where a human records a source's license, permission status,
and redistribution rights. Rule 1 says a source gets a fetcher only after those
fields are populated, so :meth:`SourceManifest.require_resolved` is the first
call every fetcher makes.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

RATE_LIMIT_FLOOR_SECONDS = 2.0
UNRESOLVED_LICENSE_VALUES = frozenset({"", "unknown", "tbd", "todo", "unresolved", "pending"})


class SourceUnresolvedError(RuntimeError):
    """Rule 1: the manifest does not yet authorize collection. Stop and ask the human."""


class ManifestError(ValueError):
    """A manifest file is malformed, misnamed, or missing a setting a stage needs."""


class SourceKind(StrEnum):
    BULK_DUMP = "bulk_dump"
    API = "api"
    CRAWL = "crawl"


class PermissionStatus(StrEnum):
    NOT_REQUIRED = "not_required"
    REQUESTED = "requested"
    GRANTED = "granted"
    DENIED = "denied"
    UNRESOLVED = "unresolved"


class DownloadSpec(BaseModel):
    """One file a bulk_dump source fetches, with optional pinned integrity values."""

    model_config = ConfigDict(extra="forbid")

    url: str = Field(min_length=1)
    sha1: str | None = Field(default=None, pattern=r"^[0-9a-fA-F]{40}$")
    md5: str | None = Field(default=None, pattern=r"^[0-9a-fA-F]{32}$")
    size: int | None = Field(default=None, ge=0)

    @field_validator("sha1", "md5")
    @classmethod
    def _lowercase(cls, value: str | None) -> str | None:
        return value.lower() if value else value


class SourceManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    kind: SourceKind
    license: str
    license_url: str | None = None
    redistributable: bool
    attribution_required: bool
    permission_status: PermissionStatus
    permission_ref: str | None = None
    notes: str = ""
    endpoints: list[str] = Field(default_factory=list)
    downloads: list[DownloadSpec] = Field(
        default_factory=list,
        description="Files fetched by bulk_dump sources; a bare URL string is accepted.",
    )
    rate_limit_seconds: float = RATE_LIMIT_FLOOR_SECONDS
    filters: dict[str, Any] = Field(
        default_factory=dict,
        description="Source-specific selection settings, read by that source's parser.",
    )

    @field_validator("downloads", mode="before")
    @classmethod
    def _coerce_download_specs(cls, value: Any) -> Any:
        if isinstance(value, list):
            return [{"url": item} if isinstance(item, str) else item for item in value]
        return value

    @field_validator("rate_limit_seconds")
    @classmethod
    def _enforce_rate_floor(cls, value: float) -> float:
        if value < RATE_LIMIT_FLOOR_SECONDS:
            raise ValueError(
                f"rate_limit_seconds must be at least {RATE_LIMIT_FLOOR_SECONDS}; "
                "CLAUDE.md rule 4 allows it to be raised, never lowered"
            )
        return value

    def require_resolved(self, sources_dir: Path) -> None:
        """Rule 1 gate: license, permission, and redistribution must be settled."""
        problems: list[str] = []
        if self.license.strip().lower() in UNRESOLVED_LICENSE_VALUES:
            problems.append(f"license is {self.license!r}")
        if self.permission_status in (
            PermissionStatus.REQUESTED,
            PermissionStatus.DENIED,
            PermissionStatus.UNRESOLVED,
        ):
            problems.append(f"permission_status is {self.permission_status.value}")
        if self.permission_status is PermissionStatus.GRANTED:
            if not self.permission_ref:
                problems.append("permission is granted but permission_ref is empty")
            elif not (sources_dir.parent / self.permission_ref).is_file():
                problems.append(
                    f"permission_ref {self.permission_ref!r} is not filed under "
                    "sources/permissions/"
                )
        if problems:
            raise SourceUnresolvedError(
                f"source {self.name!r} is unresolved: "
                + "; ".join(problems)
                + " (CLAUDE.md rule 1: stop and ask the human)"
            )


def load_manifest(sources_dir: Path, name: str) -> SourceManifest:
    path = sources_dir / f"{name}.yaml"
    if not path.is_file():
        raise SourceUnresolvedError(
            f"no manifest at {path}; CLAUDE.md rule 1 forbids a fetcher without one"
        )
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    try:
        manifest = SourceManifest.model_validate(data)
    except ValidationError as exc:
        raise ManifestError(f"{path} is malformed: {exc}") from exc
    if manifest.name != name:
        raise ManifestError(f"manifest {path} declares name {manifest.name!r}, expected {name!r}")
    return manifest

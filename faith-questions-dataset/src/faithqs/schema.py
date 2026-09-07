"""Record models for the Faith Questions Dataset.

Implements the record schema defined in CLAUDE.md. Two of the project's
non-negotiable rules are enforced here rather than by convention:

* Rule 7: ``verbatim_text`` is only valid alongside ``redistributable: true``
  and a populated ``author_attribution``.
* Rule 8: nothing is written outside ``data/raw/`` until ``pii_scrubbed`` is
  true; callers gate writes with :meth:`FaithQuestionRecord.ensure_storable`.

``author_attribution`` is exempt from PII scrubbing by design: CC BY-SA
requires crediting the author, so the field carries the license-mandated
credit (typically username plus source link) while scrubbing applies to
``question_text`` and ``verbatim_text``. Flagged for human review in the M0
checkpoint since rule 8 and the CC BY-SA attribution requirement pull in
opposite directions.

Payloads that fail validation are wrapped in :class:`QuarantineRecord` with
the failure reason attached and belong in ``data/parsed/quarantine/``.
Quarantined records never reach a release split.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)


class ReviewStatus(StrEnum):
    AUTO = "auto"
    HUMAN_REVIEWED = "human_reviewed"
    QUARANTINED = "quarantined"


class ReleaseTier(StrEnum):
    """Split assignment, driven by redistribution rights (CLAUDE.md, Two-Tier Output)."""

    A = "A"  # publishable; carries CC BY-SA 4.0 obligations
    B = "B"  # private; identifiers and labels preferred over source text


class StorageGateError(RuntimeError):
    """Raised when a record would be stored in violation of a non-negotiable rule."""


def _is_cc_by_sa(license_id: str) -> bool:
    return license_id.upper().replace(" ", "-").startswith("CC-BY-SA")


class FaithQuestionRecord(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    record_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question_text: str = Field(min_length=1)
    verbatim_text: str | None = None
    issue_id: str = Field(min_length=1)
    issue_category: str = Field(min_length=1)
    register: str = Field(min_length=1)
    source_name: str = Field(min_length=1)
    source_url: str | None = None
    source_record_id: str = Field(min_length=1)
    source_license: str = Field(min_length=1)
    author_attribution: str | None = None
    collected_at: datetime
    redistributable: bool
    permission_ref: str | None = None
    pii_scrubbed: bool
    review_status: ReviewStatus

    @field_validator("record_id")
    @classmethod
    def _require_uuid4(cls, value: str) -> str:
        try:
            parsed = uuid.UUID(value)
        except ValueError as exc:
            raise ValueError("record_id must be a UUID") from exc
        if parsed.version != 4:
            raise ValueError(f"record_id must be UUIDv4, got version {parsed.version}")
        return str(parsed)

    @field_validator("collected_at")
    @classmethod
    def _require_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("collected_at must be timezone-aware; store UTC")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def _enforce_license_gates(self) -> FaithQuestionRecord:
        if self.verbatim_text is not None:
            if not self.redistributable:
                raise ValueError("verbatim_text requires redistributable=true (CLAUDE.md rule 7)")
            if not self.author_attribution:
                raise ValueError("verbatim_text requires author_attribution (CLAUDE.md rule 7)")
        if _is_cc_by_sa(self.source_license) and not self.author_attribution:
            raise ValueError("author_attribution is required when the license is CC BY-SA")
        return self

    @property
    def release_tier(self) -> ReleaseTier | None:
        """Tier A when redistributable, Tier B otherwise, None when quarantined."""
        if self.review_status is ReviewStatus.QUARANTINED:
            return None
        return ReleaseTier.A if self.redistributable else ReleaseTier.B

    def ensure_storable(self) -> None:
        """Gate every write outside ``data/raw/`` (CLAUDE.md rule 8)."""
        if not self.pii_scrubbed:
            raise StorageGateError(
                f"record {self.record_id} has pii_scrubbed=false and cannot leave parse"
            )


class QuarantineRecord(BaseModel):
    """A payload that failed validation, preserved with its failure reason."""

    model_config = ConfigDict(extra="forbid")

    payload: dict[str, Any]
    failure_reason: str
    quarantined_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


def parse_record(payload: dict[str, Any]) -> FaithQuestionRecord | QuarantineRecord:
    """Validate a payload, quarantining instead of dropping on failure."""
    try:
        return FaithQuestionRecord.model_validate(payload)
    except ValidationError as exc:
        return QuarantineRecord(payload=payload, failure_reason=str(exc))

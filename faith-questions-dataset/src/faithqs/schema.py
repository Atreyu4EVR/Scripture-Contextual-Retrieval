"""Record models for the Faith Questions Dataset.

Implements the record schema defined in CLAUDE.md. Two of the project's
non-negotiable rules are enforced here rather than by convention:

* Rule 7: ``verbatim_text`` is only valid alongside ``redistributable: true``
  and a populated ``author_attribution``.
* Rule 8: nothing is written to ``data/parsed/`` or beyond until
  ``pii_scrubbed`` is true; callers gate writes with ``ensure_storable()``.

``author_attribution`` is the one field exempt from PII scrubbing: CC BY-SA
requires crediting the author, so it carries the license-mandated credit
(author name plus links) while scrubbing applies to every other text field.
This exception was approved at the M0 checkpoint (CLAUDE.md, Amendment Log).

Two models cover the pipeline:

* :class:`SourceRecord` is what ``parse`` emits into ``data/staged/``: the
  source-side subset of the schema, before scrubbing, extraction, or
  classification. It carries no taxonomy fields because none can be known yet.
* :class:`FaithQuestionRecord` is the complete record that ``classify`` emits,
  built from a scrubbed SourceRecord via :meth:`SourceRecord.to_record`.

A payload that fails validation is recorded as a :class:`QuarantineRecord`:
identifiers plus pydantic error locations and messages, never field values,
so a quarantine sidecar can sit beside any stage's output without carrying
unscrubbed text. Quarantined records never reach a release split.
"""

from __future__ import annotations

import uuid
import warnings
from collections.abc import Mapping
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

# Namespace for source-derived record ids. Changing it changes every record id.
FAITHQS_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_DNS, "faithqs.dataset")


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


def is_cc_by_sa(license_id: str) -> bool:
    return license_id.upper().replace(" ", "-").startswith("CC-BY-SA")


def derive_record_id(source_name: str, source_record_id: str) -> str:
    """Stable UUIDv5 for a collected record, keyed on its source identity."""
    return str(uuid.uuid5(FAITHQS_NAMESPACE, f"{source_name}:{source_record_id}"))


def _require_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError("timestamps must be timezone-aware; store UTC")
    return value.astimezone(UTC)


with warnings.catch_warnings():
    # CLAUDE.md names the field `register`; pydantic warns that it shadows a
    # BaseModel attribute. The field works and the name is not ours to change.
    warnings.filterwarnings(
        "ignore",
        message=r'Field name "register" in "FaithQuestionRecord" shadows an attribute',
        category=UserWarning,
    )

    class FaithQuestionRecord(BaseModel):
        """The complete record, emitted by ``classify`` and carried through release.

        ``record_id`` is required: collected records carry the UUIDv5 derived from
        their source identity (checked below); phrasing variants the project
        authors carry an explicit UUIDv4.
        """

        model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

        record_id: str = Field(min_length=1)
        question_text: str = Field(min_length=1)
        verbatim_text: str | None = None
        issue_id: str = Field(min_length=1)
        issue_category: str = Field(min_length=1)
        register: str = Field(min_length=1)
        classifier_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
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
        def _require_uuid(cls, value: str) -> str:
            try:
                parsed = uuid.UUID(value)
            except ValueError as exc:
                raise ValueError("record_id must be a UUID") from exc
            if parsed.version not in (4, 5):
                raise ValueError(
                    f"record_id must be UUIDv4 or UUIDv5, got version {parsed.version}"
                )
            return str(parsed)

        @field_validator("collected_at")
        @classmethod
        def _collected_utc(cls, value: datetime) -> datetime:
            return _require_utc(value)

        @model_validator(mode="after")
        def _enforce_gates(self) -> FaithQuestionRecord:
            if self.verbatim_text is not None:
                if not self.redistributable:
                    raise ValueError(
                        "verbatim_text requires redistributable=true (CLAUDE.md rule 7)"
                    )
                if not self.author_attribution:
                    raise ValueError("verbatim_text requires author_attribution (CLAUDE.md rule 7)")
            if is_cc_by_sa(self.source_license) and not self.author_attribution:
                raise ValueError("author_attribution is required when the license is CC BY-SA")
            if uuid.UUID(self.record_id).version == 5 and self.record_id != derive_record_id(
                self.source_name, self.source_record_id
            ):
                raise ValueError(
                    "record_id is a UUIDv5 that does not derive from source_name and "
                    "source_record_id; collected records must use derive_record_id()"
                )
            return self

        @property
        def release_tier(self) -> ReleaseTier | None:
            """Tier A when redistributable, Tier B otherwise, None when quarantined."""
            if self.review_status is ReviewStatus.QUARANTINED:
                return None
            return ReleaseTier.A if self.redistributable else ReleaseTier.B

        def ensure_storable(self) -> None:
            """Gate every write to ``data/parsed/`` and beyond (CLAUDE.md rule 8)."""
            if not self.pii_scrubbed:
                raise StorageGateError(
                    f"record {self.record_id} has pii_scrubbed=false and cannot enter data/parsed/"
                )


class SourceRecord(BaseModel):
    """Parse-stage record: the source-side subset of the schema.

    Emitted by ``parse`` into ``data/staged/<source>/`` before any scrubbing,
    extraction, or classification. ``to_record`` builds the complete
    :class:`FaithQuestionRecord` once the later stages have supplied theirs.
    """

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    source_name: str = Field(min_length=1)
    source_record_id: str = Field(min_length=1)
    source_url: str | None = None
    source_license: str = Field(min_length=1)
    redistributable: bool
    author_attribution: str | None = None
    permission_ref: str | None = None
    collected_at: datetime
    title: str | None = None
    body_text: str = Field(min_length=1)
    tags: list[str] = Field(default_factory=list)
    created_at: datetime | None = None
    source_metadata: dict[str, Any] = Field(default_factory=dict)
    pii_scrubbed: bool = False

    @field_validator("collected_at")
    @classmethod
    def _collected_utc(cls, value: datetime) -> datetime:
        return _require_utc(value)

    @field_validator("created_at")
    @classmethod
    def _created_utc(cls, value: datetime | None) -> datetime | None:
        return None if value is None else _require_utc(value)

    @model_validator(mode="after")
    def _enforce_attribution(self) -> SourceRecord:
        if is_cc_by_sa(self.source_license) and not self.author_attribution:
            raise ValueError("author_attribution is required when the license is CC BY-SA")
        return self

    @property
    def record_id(self) -> str:
        return derive_record_id(self.source_name, self.source_record_id)

    @property
    def verbatim_text(self) -> str | None:
        """Source text for Tier A records; None wherever rule 7 forbids it."""
        if not (self.redistributable and self.author_attribution):
            return None
        return f"{self.title}\n\n{self.body_text}" if self.title else self.body_text

    def ensure_storable(self) -> None:
        """Gate every write to ``data/parsed/`` and beyond (CLAUDE.md rule 8)."""
        if not self.pii_scrubbed:
            raise StorageGateError(
                f"source record {self.source_name}:{self.source_record_id} has "
                "pii_scrubbed=false and cannot enter data/parsed/"
            )

    def to_record(
        self,
        *,
        question_text: str,
        issue_id: str,
        issue_category: str,
        register: str,
        review_status: ReviewStatus = ReviewStatus.AUTO,
        classifier_confidence: float | None = None,
    ) -> FaithQuestionRecord:
        """Build the complete record after ``extract`` and ``classify`` have run."""
        self.ensure_storable()
        return FaithQuestionRecord(
            record_id=self.record_id,
            question_text=question_text,
            verbatim_text=self.verbatim_text,
            issue_id=issue_id,
            issue_category=issue_category,
            register=register,
            classifier_confidence=classifier_confidence,
            source_name=self.source_name,
            source_url=self.source_url,
            source_record_id=self.source_record_id,
            source_license=self.source_license,
            author_attribution=self.author_attribution,
            collected_at=self.collected_at,
            redistributable=self.redistributable,
            permission_ref=self.permission_ref,
            pii_scrubbed=True,
            review_status=review_status,
        )


class QuarantineError(BaseModel):
    model_config = ConfigDict(extra="forbid")

    loc: list[str | int]
    msg: str
    type: str


class QuarantineRecord(BaseModel):
    """Why a payload failed validation, without the payload.

    Carries only identifiers and pydantic error locations, so it can be written
    beside any stage's output (``data/staged/`` for parse, ``data/parsed/`` from
    scrub onward) without ever moving field values across the rule 8 boundary.
    """

    model_config = ConfigDict(extra="forbid")

    source_name: str | None = None
    source_record_id: str | None = None
    record_id: str | None = None
    errors: list[QuarantineError]
    quarantined_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @classmethod
    def from_validation_error(
        cls, payload: Mapping[str, Any], exc: ValidationError
    ) -> QuarantineRecord:
        """Build from the failing payload, reading only its identifier fields."""
        source_name = _str_or_none(payload.get("source_name"))
        source_record_id = _str_or_none(payload.get("source_record_id"))
        record_id = _str_or_none(payload.get("record_id"))
        if record_id is None and source_name and source_record_id:
            record_id = derive_record_id(source_name, source_record_id)
        return cls(
            source_name=source_name,
            source_record_id=source_record_id,
            record_id=record_id,
            errors=[
                QuarantineError(loc=list(e["loc"]), msg=e["msg"], type=e["type"])
                for e in exc.errors(include_input=False, include_url=False)
            ],
        )


def _str_or_none(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None

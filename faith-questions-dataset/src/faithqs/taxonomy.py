"""Loaders and validators for the human-owned taxonomy files.

The taxonomy is the spine of the project and is human-authored (CLAUDE.md,
Taxonomy First). Both YAML files ship with ``approved: false`` until a human
reviews them; :meth:`Taxonomy.require_approved` is the gate every pipeline
stage calls before using taxonomy labels, and fetchers are not written at all
until ``taxonomy/issues.yaml`` is approved.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from faithqs.schema import FaithQuestionRecord


class TaxonomyNotApprovedError(RuntimeError):
    """Raised when a pipeline stage runs against an unapproved taxonomy."""


class Category(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    label: str = Field(min_length=1)


class Issue(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    category: str = Field(min_length=1)
    label: str = Field(min_length=1)
    description: str = Field(min_length=1)
    keywords: list[str] = Field(default_factory=list)


class Taxonomy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    approved: bool = False
    version: str = Field(min_length=1)
    categories: list[Category]
    issues: list[Issue]

    @model_validator(mode="after")
    def _check_integrity(self) -> Taxonomy:
        category_ids = [c.id for c in self.categories]
        if len(category_ids) != len(set(category_ids)):
            raise ValueError("duplicate category ids in taxonomy")
        issue_ids = [i.id for i in self.issues]
        if len(issue_ids) != len(set(issue_ids)):
            raise ValueError("duplicate issue ids in taxonomy")
        known = set(category_ids)
        orphans = [i.id for i in self.issues if i.category not in known]
        if orphans:
            raise ValueError(f"issues reference unknown categories: {orphans}")
        return self

    def require_approved(self) -> None:
        if not self.approved:
            raise TaxonomyNotApprovedError(
                "taxonomy/issues.yaml is a draft: a human must review it and set "
                "approved: true before classification runs or any fetcher is "
                "written (CLAUDE.md, Taxonomy First)"
            )

    def category_of(self, issue_id: str) -> str:
        for issue in self.issues:
            if issue.id == issue_id:
                return issue.category
        raise KeyError(f"unknown issue_id: {issue_id}")

    def check_issue_assignment(self, record: FaithQuestionRecord) -> None:
        """Validate a record's issue foreign key and denormalized category."""
        expected = self.category_of(record.issue_id)
        if record.issue_category != expected:
            raise ValueError(
                f"record {record.record_id}: issue_category "
                f"{record.issue_category!r} does not match taxonomy parent {expected!r}"
            )


class Register(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    label: str = Field(min_length=1)
    description: str = Field(min_length=1)
    cues: list[str] = Field(default_factory=list)


class RegisterVocabulary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    approved: bool = False
    version: str = Field(min_length=1)
    registers: list[Register]

    @model_validator(mode="after")
    def _check_unique_ids(self) -> RegisterVocabulary:
        ids = [r.id for r in self.registers]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate register ids")
        return self

    def require_approved(self) -> None:
        if not self.approved:
            raise TaxonomyNotApprovedError(
                "taxonomy/registers.yaml is a draft: a human must review it and "
                "set approved: true before classification runs (CLAUDE.md)"
            )

    def check_register(self, record: FaithQuestionRecord) -> None:
        if record.register not in {r.id for r in self.registers}:
            raise ValueError(f"record {record.record_id}: unknown register {record.register!r}")


def load_taxonomy(path: Path) -> Taxonomy:
    with path.open(encoding="utf-8") as handle:
        return Taxonomy.model_validate(yaml.safe_load(handle))


def load_registers(path: Path) -> RegisterVocabulary:
    with path.open(encoding="utf-8") as handle:
        return RegisterVocabulary.model_validate(yaml.safe_load(handle))

from pathlib import Path

import pytest
from pydantic import ValidationError

from conftest import record_payload
from faithqs.schema import FaithQuestionRecord
from faithqs.taxonomy import (
    Category,
    Issue,
    Register,
    RegisterVocabulary,
    Taxonomy,
    TaxonomyNotApprovedError,
    load_registers,
    load_taxonomy,
)


def test_shipped_taxonomy_is_structurally_valid(taxonomy_path: Path) -> None:
    taxonomy = load_taxonomy(taxonomy_path)
    # CLAUDE.md (Taxonomy First) expects roughly forty to sixty top-level issues.
    assert 40 <= len(taxonomy.issues) <= 60
    assert taxonomy.categories
    assert all(issue.keywords for issue in taxonomy.issues)


def test_shipped_taxonomy_is_approved(taxonomy_path: Path) -> None:
    taxonomy = load_taxonomy(taxonomy_path)
    assert taxonomy.approved is True
    taxonomy.require_approved()


def test_unapproved_taxonomy_blocks_pipeline() -> None:
    draft = Taxonomy(
        approved=False,
        version="0.0.1-draft",
        categories=[Category(id="c", label="C")],
        issues=[Issue(id="i", category="c", label="I", description="D")],
    )
    with pytest.raises(TaxonomyNotApprovedError, match="Taxonomy First"):
        draft.require_approved()


def test_taxonomy_rejects_orphan_issue() -> None:
    with pytest.raises(ValidationError, match="unknown categories"):
        Taxonomy(
            version="x",
            categories=[Category(id="c", label="C")],
            issues=[Issue(id="i", category="missing", label="I", description="D")],
        )


def test_taxonomy_rejects_duplicate_issue_ids() -> None:
    with pytest.raises(ValidationError, match="duplicate issue ids"):
        Taxonomy(
            version="x",
            categories=[Category(id="c", label="C")],
            issues=[
                Issue(id="i", category="c", label="I", description="D"),
                Issue(id="i", category="c", label="I2", description="D2"),
            ],
        )


def test_shipped_registers_are_approved(registers_path: Path) -> None:
    vocabulary = load_registers(registers_path)
    assert len(vocabulary.registers) >= 5
    assert vocabulary.approved is True
    vocabulary.require_approved()


def test_unapproved_registers_block_pipeline() -> None:
    draft = RegisterVocabulary(
        approved=False,
        version="x",
        registers=[Register(id="r", label="R", description="D")],
    )
    with pytest.raises(TaxonomyNotApprovedError):
        draft.require_approved()


def test_issue_assignment_checks_foreign_key(taxonomy_path: Path) -> None:
    taxonomy = load_taxonomy(taxonomy_path)
    record = FaithQuestionRecord.model_validate(record_payload())
    taxonomy.check_issue_assignment(record)

    unknown = FaithQuestionRecord.model_validate(record_payload(issue_id="not-an-issue"))
    with pytest.raises(KeyError):
        taxonomy.check_issue_assignment(unknown)


def test_issue_assignment_checks_denormalized_category(taxonomy_path: Path) -> None:
    taxonomy = load_taxonomy(taxonomy_path)
    mismatched = FaithQuestionRecord.model_validate(record_payload(issue_category="book-of-mormon"))
    with pytest.raises(ValueError, match="does not match taxonomy parent"):
        taxonomy.check_issue_assignment(mismatched)


def test_register_membership_check(registers_path: Path) -> None:
    vocabulary = load_registers(registers_path)
    record = FaithQuestionRecord.model_validate(record_payload())
    vocabulary.check_register(record)

    rogue = FaithQuestionRecord.model_validate(record_payload(register="sarcastic"))
    with pytest.raises(ValueError, match="unknown register"):
        vocabulary.check_register(rogue)

from pathlib import Path

import pytest

from conftest import record_payload
from faithqs.schema import FaithQuestionRecord
from faithqs.taxonomy import (
    TaxonomyNotApprovedError,
    load_registers,
    load_taxonomy,
)


def test_shipped_taxonomy_is_structurally_valid(taxonomy_path: Path) -> None:
    taxonomy = load_taxonomy(taxonomy_path)
    # CLAUDE.md (Taxonomy First) expects roughly forty to sixty top-level issues.
    assert 40 <= len(taxonomy.issues) <= 60
    assert taxonomy.categories


def test_shipped_taxonomy_is_a_draft_and_gate_holds(taxonomy_path: Path) -> None:
    taxonomy = load_taxonomy(taxonomy_path)
    assert taxonomy.approved is False
    with pytest.raises(TaxonomyNotApprovedError, match="Taxonomy First"):
        taxonomy.require_approved()


def test_shipped_registers_load_as_draft(registers_path: Path) -> None:
    vocabulary = load_registers(registers_path)
    assert len(vocabulary.registers) >= 5
    assert vocabulary.approved is False
    with pytest.raises(TaxonomyNotApprovedError):
        vocabulary.require_approved()


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

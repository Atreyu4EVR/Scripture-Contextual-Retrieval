import uuid
from datetime import UTC, datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from conftest import record_payload
from faithqs.schema import (
    FaithQuestionRecord,
    QuarantineRecord,
    ReleaseTier,
    ReviewStatus,
    StorageGateError,
    parse_record,
)


def test_valid_record_roundtrips() -> None:
    record = FaithQuestionRecord.model_validate(record_payload())
    restored = FaithQuestionRecord.model_validate_json(record.model_dump_json())
    assert restored == record
    assert uuid.UUID(record.record_id).version == 4


def test_record_id_is_stable_when_provided() -> None:
    rid = str(uuid.uuid4())
    record = FaithQuestionRecord.model_validate(record_payload(record_id=rid))
    assert record.record_id == rid


def test_non_uuid4_record_id_rejected() -> None:
    rid = str(uuid.uuid5(uuid.NAMESPACE_URL, "not-v4"))
    with pytest.raises(ValidationError, match="UUIDv4"):
        FaithQuestionRecord.model_validate(record_payload(record_id=rid))


def test_verbatim_text_requires_redistributable() -> None:
    with pytest.raises(ValidationError, match="rule 7"):
        FaithQuestionRecord.model_validate(
            record_payload(redistributable=False, source_license="all-rights-reserved")
        )


def test_verbatim_text_requires_attribution() -> None:
    with pytest.raises(ValidationError, match="rule 7"):
        FaithQuestionRecord.model_validate(
            record_payload(author_attribution=None, source_license="CC0-1.0")
        )


def test_cc_by_sa_requires_attribution_even_without_verbatim() -> None:
    with pytest.raises(ValidationError, match="CC BY-SA"):
        FaithQuestionRecord.model_validate(
            record_payload(verbatim_text=None, author_attribution=None)
        )


def test_tier_b_record_carries_normalized_question_only() -> None:
    record = FaithQuestionRecord.model_validate(
        record_payload(
            verbatim_text=None,
            redistributable=False,
            source_license="all-rights-reserved",
            author_attribution=None,
        )
    )
    assert record.verbatim_text is None
    assert record.release_tier is ReleaseTier.B


def test_release_tier_assignment() -> None:
    tier_a = FaithQuestionRecord.model_validate(record_payload())
    assert tier_a.release_tier is ReleaseTier.A

    quarantined = FaithQuestionRecord.model_validate(
        record_payload(review_status=ReviewStatus.QUARANTINED)
    )
    assert quarantined.release_tier is None


def test_naive_collected_at_rejected() -> None:
    with pytest.raises(ValidationError, match="timezone-aware"):
        FaithQuestionRecord.model_validate(record_payload(collected_at=datetime(2025, 9, 1, 12, 0)))


def test_aware_collected_at_normalized_to_utc() -> None:
    mountain = timezone(timedelta(hours=-6))
    record = FaithQuestionRecord.model_validate(
        record_payload(collected_at=datetime(2025, 9, 1, 6, 0, tzinfo=mountain))
    )
    assert record.collected_at == datetime(2025, 9, 1, 12, 0, tzinfo=UTC)


def test_unscrubbed_record_cannot_be_stored() -> None:
    record = FaithQuestionRecord.model_validate(record_payload(pii_scrubbed=False))
    with pytest.raises(StorageGateError, match="pii_scrubbed"):
        record.ensure_storable()


def test_unknown_fields_rejected() -> None:
    with pytest.raises(ValidationError):
        FaithQuestionRecord.model_validate(record_payload(author_ip="10.0.0.1"))


def test_parse_record_quarantines_invalid_payload_with_reason() -> None:
    bad = record_payload(question_text="")
    result = parse_record(bad)
    assert isinstance(result, QuarantineRecord)
    assert "question_text" in result.failure_reason
    assert result.payload == bad
    assert result.quarantined_at.tzinfo is not None


def test_parse_record_returns_valid_record() -> None:
    result = parse_record(record_payload())
    assert isinstance(result, FaithQuestionRecord)

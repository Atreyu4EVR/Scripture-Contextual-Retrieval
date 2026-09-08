import uuid
from datetime import UTC, datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from conftest import record_payload, source_record_payload
from faithqs.schema import (
    FaithQuestionRecord,
    QuarantineRecord,
    ReleaseTier,
    ReviewStatus,
    SourceRecord,
    StorageGateError,
    derive_record_id,
    parse_record,
)

# --- FaithQuestionRecord -----------------------------------------------------


def test_valid_record_roundtrips() -> None:
    record = FaithQuestionRecord.model_validate(record_payload())
    restored = FaithQuestionRecord.model_validate_json(record.model_dump_json())
    assert restored == record
    assert uuid.UUID(record.record_id).version == 4


def test_record_id_is_stable_when_provided() -> None:
    rid = str(uuid.uuid4())
    record = FaithQuestionRecord.model_validate(record_payload(record_id=rid))
    assert record.record_id == rid


def test_uuid5_record_id_accepted() -> None:
    rid = derive_record_id("christianity-stackexchange", "00001")
    record = FaithQuestionRecord.model_validate(record_payload(record_id=rid))
    assert record.record_id == rid
    assert uuid.UUID(rid).version == 5


def test_time_based_uuid_rejected() -> None:
    with pytest.raises(ValidationError, match="UUIDv4 or UUIDv5"):
        FaithQuestionRecord.model_validate(record_payload(record_id=str(uuid.uuid1())))


def test_derive_record_id_is_deterministic_and_source_scoped() -> None:
    a = derive_record_id("christianity-stackexchange", "42")
    assert a == derive_record_id("christianity-stackexchange", "42")
    assert a != derive_record_id("another-source", "42")
    assert a != derive_record_id("christianity-stackexchange", "43")


def test_classifier_confidence_bounds() -> None:
    assert FaithQuestionRecord.model_validate(record_payload()).classifier_confidence is None
    ok = FaithQuestionRecord.model_validate(record_payload(classifier_confidence=0.42))
    assert ok.classifier_confidence == 0.42
    with pytest.raises(ValidationError):
        FaithQuestionRecord.model_validate(record_payload(classifier_confidence=1.5))


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


# --- SourceRecord ------------------------------------------------------------


def test_source_record_roundtrips_and_derives_stable_id() -> None:
    source = SourceRecord.model_validate(source_record_payload())
    restored = SourceRecord.model_validate_json(source.model_dump_json())
    assert restored == source
    assert source.record_id == derive_record_id("christianity-stackexchange", "00001")


def test_source_record_verbatim_composes_title_and_body() -> None:
    source = SourceRecord.model_validate(source_record_payload())
    assert source.verbatim_text is not None
    assert source.verbatim_text.startswith("Why are there several accounts")
    assert source.verbatim_text.endswith("usually explained?")

    untitled = SourceRecord.model_validate(source_record_payload(title=None))
    assert untitled.verbatim_text == untitled.body_text


def test_source_record_withholds_verbatim_when_not_redistributable() -> None:
    source = SourceRecord.model_validate(
        source_record_payload(
            redistributable=False,
            source_license="all-rights-reserved",
            author_attribution=None,
        )
    )
    assert source.verbatim_text is None


def test_source_record_cc_by_sa_requires_attribution() -> None:
    with pytest.raises(ValidationError, match="CC BY-SA"):
        SourceRecord.model_validate(source_record_payload(author_attribution=None))


def test_source_record_timestamps_normalized_to_utc() -> None:
    mountain = timezone(timedelta(hours=-6))
    source = SourceRecord.model_validate(
        source_record_payload(created_at=datetime(2019, 4, 2, 2, 30, tzinfo=mountain))
    )
    assert source.created_at == datetime(2019, 4, 2, 8, 30, tzinfo=UTC)
    with pytest.raises(ValidationError, match="timezone-aware"):
        SourceRecord.model_validate(source_record_payload(created_at=datetime(2019, 4, 2)))


def test_to_record_blocked_until_scrubbed() -> None:
    source = SourceRecord.model_validate(source_record_payload(pii_scrubbed=False))
    with pytest.raises(StorageGateError, match="pii_scrubbed"):
        source.to_record(
            question_text="Why do First Vision accounts differ?",
            issue_id="first-vision-accounts",
            issue_category="joseph-smith",
            register="sincere-inquiry",
        )


def test_to_record_builds_complete_tier_a_record() -> None:
    source = SourceRecord.model_validate(source_record_payload(pii_scrubbed=True))
    record = source.to_record(
        question_text="Why do First Vision accounts differ?",
        issue_id="first-vision-accounts",
        issue_category="joseph-smith",
        register="sincere-inquiry",
        classifier_confidence=0.91,
    )
    assert record.record_id == source.record_id
    assert record.verbatim_text == source.verbatim_text
    assert record.release_tier is ReleaseTier.A
    assert record.classifier_confidence == 0.91
    assert record.pii_scrubbed is True
    assert record.review_status is ReviewStatus.AUTO
    assert record.source_license == "CC-BY-SA-4.0"


def test_to_record_for_tier_b_source_has_no_verbatim() -> None:
    source = SourceRecord.model_validate(
        source_record_payload(
            redistributable=False,
            source_license="all-rights-reserved",
            author_attribution=None,
            pii_scrubbed=True,
        )
    )
    record = source.to_record(
        question_text="Why do First Vision accounts differ?",
        issue_id="first-vision-accounts",
        issue_category="joseph-smith",
        register="sincere-inquiry",
    )
    assert record.verbatim_text is None
    assert record.release_tier is ReleaseTier.B

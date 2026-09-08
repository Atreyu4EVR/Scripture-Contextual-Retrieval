import uuid
from datetime import UTC, datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from conftest import SOURCE_NAME, SOURCE_RECORD_ID, record_payload, source_record_payload
from faithqs.schema import (
    FaithQuestionRecord,
    QuarantineRecord,
    ReleaseTier,
    ReviewStatus,
    SourceRecord,
    StorageGateError,
    derive_record_id,
)

# --- FaithQuestionRecord -----------------------------------------------------


def test_valid_record_roundtrips_with_derived_id() -> None:
    record = FaithQuestionRecord.model_validate(record_payload())
    restored = FaithQuestionRecord.model_validate_json(record.model_dump_json())
    assert restored == record
    assert record.record_id == derive_record_id(SOURCE_NAME, SOURCE_RECORD_ID)
    assert uuid.UUID(record.record_id).version == 5


def test_record_id_is_required() -> None:
    payload = record_payload()
    del payload["record_id"]
    with pytest.raises(ValidationError, match="record_id"):
        FaithQuestionRecord.model_validate(payload)


def test_authored_variant_may_carry_explicit_uuid4() -> None:
    rid = str(uuid.uuid4())
    record = FaithQuestionRecord.model_validate(record_payload(record_id=rid))
    assert record.record_id == rid


def test_uuid5_must_derive_from_source_identity() -> None:
    foreign = derive_record_id("another-source", SOURCE_RECORD_ID)
    with pytest.raises(ValidationError, match="does not derive"):
        FaithQuestionRecord.model_validate(record_payload(record_id=foreign))
    wrong_namespace = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{SOURCE_NAME}:{SOURCE_RECORD_ID}"))
    with pytest.raises(ValidationError, match="does not derive"):
        FaithQuestionRecord.model_validate(record_payload(record_id=wrong_namespace))


def test_time_based_and_non_uuid_record_ids_rejected() -> None:
    with pytest.raises(ValidationError, match="UUIDv4 or UUIDv5"):
        FaithQuestionRecord.model_validate(record_payload(record_id=str(uuid.uuid1())))
    with pytest.raises(ValidationError, match="must be a UUID"):
        FaithQuestionRecord.model_validate(record_payload(record_id="not-a-uuid"))


def test_derive_record_id_is_deterministic_and_source_scoped() -> None:
    a = derive_record_id("christianity-stackexchange", "42")
    # Golden value: changing FAITHQS_NAMESPACE or the key format changes every record id.
    assert a == "19df0287-7fd2-5ac6-a1b9-956fad758791"
    assert uuid.UUID(a).version == 5
    assert a != derive_record_id("another-source", "42")
    assert a != derive_record_id("christianity-stackexchange", "43")


def test_classifier_confidence_bounds() -> None:
    assert FaithQuestionRecord.model_validate(record_payload()).classifier_confidence is None
    ok = FaithQuestionRecord.model_validate(record_payload(classifier_confidence=0.42))
    assert ok.classifier_confidence == 0.42
    with pytest.raises(ValidationError):
        FaithQuestionRecord.model_validate(record_payload(classifier_confidence=1.5))
    with pytest.raises(ValidationError):
        FaithQuestionRecord.model_validate(record_payload(classifier_confidence=-0.1))


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


# --- QuarantineRecord --------------------------------------------------------

SENTINEL = "Bob Jones of the Rexburg 4th Ward, bob@example.org"


def test_quarantine_record_carries_identifiers_and_error_locations_only() -> None:
    bad = record_payload(question_text="", verbatim_text=SENTINEL, pii_scrubbed=False)
    with pytest.raises(ValidationError) as excinfo:
        FaithQuestionRecord.model_validate(bad)
    quarantine = QuarantineRecord.from_validation_error(bad, excinfo.value)

    assert quarantine.source_name == SOURCE_NAME
    assert quarantine.source_record_id == SOURCE_RECORD_ID
    assert quarantine.record_id == bad["record_id"]
    assert ["question_text"] in [e.loc for e in quarantine.errors]
    serialized = quarantine.model_dump_json()
    assert SENTINEL not in serialized
    assert "First Vision" not in serialized
    assert quarantine.quarantined_at.tzinfo is not None


def test_quarantine_record_tolerates_missing_or_malformed_identifiers() -> None:
    payload = {"source_name": 42, "source_record_id": "9", "question_text": ""}
    with pytest.raises(ValidationError) as excinfo:
        FaithQuestionRecord.model_validate(payload)
    quarantine = QuarantineRecord.from_validation_error(payload, excinfo.value)
    assert quarantine.source_name is None
    assert quarantine.source_record_id == "9"
    assert quarantine.record_id is None  # cannot derive without both identifiers
    assert quarantine.errors


def test_quarantine_record_derives_id_when_payload_lacks_one() -> None:
    payload = source_record_payload(body_text="")
    with pytest.raises(ValidationError) as excinfo:
        SourceRecord.model_validate(payload)
    quarantine = QuarantineRecord.from_validation_error(payload, excinfo.value)
    assert quarantine.record_id == derive_record_id(SOURCE_NAME, SOURCE_RECORD_ID)
    assert [e.loc for e in quarantine.errors] == [["body_text"]]


# --- SourceRecord ------------------------------------------------------------


def test_source_record_roundtrips_and_derives_stable_id() -> None:
    source = SourceRecord.model_validate(source_record_payload())
    restored = SourceRecord.model_validate_json(source.model_dump_json())
    assert restored == source
    assert source.record_id == derive_record_id(SOURCE_NAME, SOURCE_RECORD_ID)


def test_source_record_verbatim_composes_title_and_body() -> None:
    source = SourceRecord.model_validate(source_record_payload())
    assert source.verbatim_text is not None
    assert source.verbatim_text.startswith("Why are there several accounts")
    assert source.verbatim_text.endswith("usually explained?")

    untitled = SourceRecord.model_validate(source_record_payload(title=None))
    assert untitled.verbatim_text == untitled.body_text


def test_source_record_withholds_verbatim_when_not_redistributable() -> None:
    # Non-redistributable with attribution present: redistributable alone gates it.
    source = SourceRecord.model_validate(
        source_record_payload(redistributable=False, source_license="all-rights-reserved")
    )
    assert source.verbatim_text is None


def test_source_record_withholds_verbatim_without_attribution() -> None:
    # Redistributable but unattributed (a non CC BY-SA license): attribution alone gates it.
    source = SourceRecord.model_validate(
        source_record_payload(source_license="CC0-1.0", author_attribution=None)
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

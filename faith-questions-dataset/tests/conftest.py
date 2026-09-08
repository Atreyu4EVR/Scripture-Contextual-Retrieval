from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from faithqs.schema import derive_record_id

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class FakeClock:
    """Deterministic monotonic clock; ``sleep`` advances it and records the call."""

    def __init__(self) -> None:
        self.now = 1000.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(round(seconds, 6))
        self.now += seconds


SOURCE_NAME = "christianity-stackexchange"
SOURCE_RECORD_ID = "00001"
SOURCE_URL = f"https://christianity.stackexchange.com/questions/{SOURCE_RECORD_ID}"
ATTRIBUTION = f"Christianity Stack Exchange contributor, Christianity Stack Exchange, {SOURCE_URL}"


def record_payload(**overrides: Any) -> dict[str, Any]:
    """A valid Tier A payload; override fields to probe individual gates."""
    payload: dict[str, Any] = {
        "record_id": derive_record_id(SOURCE_NAME, SOURCE_RECORD_ID),
        "question_text": "Why do the recorded accounts of the First Vision differ?",
        "verbatim_text": (
            "I recently read that there are several accounts of the First Vision "
            "and they don't all say the same thing. How are these differences "
            "usually explained?"
        ),
        "issue_id": "first-vision-accounts",
        "issue_category": "joseph-smith",
        "register": "sincere-inquiry",
        "source_name": SOURCE_NAME,
        "source_url": SOURCE_URL,
        "source_record_id": SOURCE_RECORD_ID,
        "source_license": "CC-BY-SA-4.0",
        "author_attribution": ATTRIBUTION,
        "collected_at": datetime(2025, 9, 1, 12, 0, tzinfo=UTC),
        "redistributable": True,
        "permission_ref": None,
        "pii_scrubbed": True,
        "review_status": "auto",
    }
    payload.update(overrides)
    return payload


def source_record_payload(**overrides: Any) -> dict[str, Any]:
    """A valid parse-stage payload for a Tier A source."""
    payload: dict[str, Any] = {
        "source_name": SOURCE_NAME,
        "source_record_id": SOURCE_RECORD_ID,
        "source_url": SOURCE_URL,
        "source_license": "CC-BY-SA-4.0",
        "redistributable": True,
        "author_attribution": ATTRIBUTION,
        "permission_ref": None,
        "collected_at": datetime(2025, 9, 1, 12, 0, tzinfo=UTC),
        "title": "Why are there several accounts of the First Vision?",
        "body_text": (
            "I recently read that there are several accounts of the First Vision "
            "and they don't all say the same thing. How are these differences "
            "usually explained?"
        ),
        "tags": ["lds", "joseph-smith"],
        "created_at": datetime(2019, 4, 2, 8, 30, tzinfo=UTC),
        "source_metadata": {"score": 7},
        "pii_scrubbed": False,
    }
    payload.update(overrides)
    return payload


@pytest.fixture(scope="session")
def taxonomy_path() -> Path:
    return PROJECT_ROOT / "taxonomy" / "issues.yaml"


@pytest.fixture(scope="session")
def registers_path() -> Path:
    return PROJECT_ROOT / "taxonomy" / "registers.yaml"

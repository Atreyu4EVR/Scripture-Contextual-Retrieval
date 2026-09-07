from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def record_payload(**overrides: Any) -> dict[str, Any]:
    """A valid Tier A payload; override fields to probe individual gates."""
    payload: dict[str, Any] = {
        "question_text": "Why do the recorded accounts of the First Vision differ?",
        "verbatim_text": (
            "I recently read that there are several accounts of the First Vision "
            "and they don't all say the same thing. How are these differences "
            "usually explained?"
        ),
        "issue_id": "first-vision-accounts",
        "issue_category": "joseph-smith",
        "register": "sincere-inquiry",
        "source_name": "christianity-stackexchange",
        "source_url": "https://christianity.stackexchange.com/questions/00001",
        "source_record_id": "00001",
        "source_license": "CC-BY-SA-4.0",
        "author_attribution": (
            "Christianity Stack Exchange contributor, "
            "https://christianity.stackexchange.com/questions/00001"
        ),
        "collected_at": datetime(2025, 9, 1, 12, 0, tzinfo=UTC),
        "redistributable": True,
        "permission_ref": None,
        "pii_scrubbed": True,
        "review_status": "auto",
    }
    payload.update(overrides)
    return payload


@pytest.fixture(scope="session")
def taxonomy_path() -> Path:
    return PROJECT_ROOT / "taxonomy" / "issues.yaml"


@pytest.fixture(scope="session")
def registers_path() -> Path:
    return PROJECT_ROOT / "taxonomy" / "registers.yaml"

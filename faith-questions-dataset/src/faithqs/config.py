"""Runtime settings, read from the environment. Nothing here is secret.

``FAITHQS_CONTACT_EMAIL`` is required before any network request is made:
rule 3 says every request identifies the project and a contact address, and
the address is the project owner's to supply, never a placeholder.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from faithqs import __version__

DEFAULT_PROJECT_URL = (
    "https://github.com/Atreyu4EVR/Scripture-Contextual-Retrieval/tree/main/faith-questions-dataset"
)


class ConfigError(RuntimeError):
    """A required setting is missing or malformed."""


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    contact_email: str | None
    project_url: str = DEFAULT_PROJECT_URL

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Settings:
        env = os.environ if env is None else env
        return cls(
            data_dir=Path(env.get("FAITHQS_DATA_DIR", "data")).expanduser(),
            contact_email=env.get("FAITHQS_CONTACT_EMAIL") or None,
            project_url=env.get("FAITHQS_PROJECT_URL", DEFAULT_PROJECT_URL),
        )

    def user_agent(self) -> str:
        """Rule 3: a descriptive User-Agent naming the project and a contact address."""
        if not self.contact_email or "@" not in self.contact_email:
            raise ConfigError(
                "FAITHQS_CONTACT_EMAIL is not set to an email address; CLAUDE.md rule 3 "
                "requires a contact email in every request's User-Agent"
            )
        return f"faithqs/{__version__} (+{self.project_url}; contact: {self.contact_email})"

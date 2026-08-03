"""Helpers for session-scoped file-backed corpus artifacts."""

from __future__ import annotations

import os
from pathlib import Path


SESSION_ARTIFACT_ROOT_ENV = "DOCUSCOPE_SESSION_ARTIFACT_ROOT"


def get_session_artifact_root(default_root: Path) -> Path:
    """Return the configured root for session-scoped corpus artifacts."""

    configured_root = os.getenv(SESSION_ARTIFACT_ROOT_ENV, "").strip()
    if configured_root:
        return Path(configured_root)
    return default_root

"""Where the artifact tree lives.

The root is configured by the ARTIFACT_ROOT_URI env var. Paths under it are
built by wato_common.artifact_store; artifacts that cross components are
written and read through wato_common.artifacts.
"""

from __future__ import annotations

import os


def artifact_root() -> str:
    """Return the configured artifact root URI (e.g. file:///data/artifacts)."""
    return os.environ.get("ARTIFACT_ROOT_URI", "file:///data/artifacts")

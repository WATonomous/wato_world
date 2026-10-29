"""Provenance: which image, dependency set and config produced an artifact.

Pinning (docker/requirements/*.txt, digest-pinned bases, model revisions)
makes a rebuild deterministic; this records which pinned configuration a run
actually used, so a label can be traced back to it.

``environment()`` reads build-time facts baked into the image as WATO_* env
vars — never ``git rev-parse``, since deploy images have no .git — plus the
hash of the dependency lock. A ``git_dirty`` build isn't reproducible; treat
its artifacts as provisional. Every field degrades to None rather than
raising: provenance must never be the reason a batch run dies.
"""

from __future__ import annotations

import hashlib
import os
import platform
from typing import Any

#: Where the component Dockerfiles leave the dependency lock.
LOCK_PATH = "/opt/watonomous/requirements.lock.txt"


def _env(name: str) -> str | None:
    """Read an env var, treating empty/unset alike as "not recorded"."""
    value = os.environ.get(name, "").strip()
    return value or None


def file_sha256(path: str) -> str | None:
    """SHA-256 of a file, or None if it cannot be read."""
    try:
        with open(path, "rb") as fh:
            return hashlib.sha256(fh.read()).hexdigest()
    except OSError:
        return None


def config_hash(config_path: str | None) -> str | None:
    """SHA-256 of a component's config file (a changed threshold changes the output)."""
    if not config_path:
        return None
    return file_sha256(config_path)


def environment() -> dict[str, Any]:
    """Describe the image this process is running in."""
    return {
        "git_commit": _env("WATO_GIT_COMMIT"),
        "git_dirty": _env("WATO_GIT_DIRTY") == "true",
        "build_time": _env("WATO_BUILD_TIME"),
        "base_image": _env("WATO_BASE_IMAGE"),
        "lock_sha256": file_sha256(LOCK_PATH),
        "python": platform.python_version(),
    }


def collect(
    *,
    component: str,
    config_path: str | None = None,
    models: dict[str, str] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Full provenance block for a component's manifest.

    Args:
        component: component name, e.g. "perception_2d".
        config_path: path to the config file that drove this run.
        models: ``{tag: "repo@revision"}`` for every model whose weights
            influenced the output (perception_2d:
            ``model_registry.revisions()``).
        extra: component-specific fields to merge in.

    Returns:
        A JSON-serializable dict. Never raises.
    """
    block: dict[str, Any] = {
        "component": component,
        "environment": environment(),
        "config_hash": config_hash(config_path),
    }
    if config_path:
        block["config_path"] = config_path
    if models:
        block["models"] = dict(models)
    if extra:
        block.update(extra)
    return block

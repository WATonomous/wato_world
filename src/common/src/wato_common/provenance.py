"""Provenance capture — what produced an artifact.

WHY THIS EXISTS
---------------
This pipeline's output is training data, and its components are Docker images
full of floating-point models. Six months from now the only way to answer
"which labeler produced these boxes, and is this run comparable to that one?"
is if every artifact carries the answer with it.

Reproducibility has two halves, and pinning is only the first:

  1. Pinning (docker/requirements/*.txt, digest-pinned bases, pinned model
     revisions) makes a rebuild deterministic.
  2. Provenance — this module — records *which* pinned configuration was
     actually used, inside the artifact tree, so a label can be traced back to
     it after the fact.

Without (2), pinning only guarantees that today's image can be rebuilt; it
cannot tell you whether the labels you are looking at came from that image or
from the one before it.

WHAT IS CAPTURED
----------------
``environment()`` returns a dict describing the running image:

  git_commit    the source revision, baked in at build time (see
                docker/template.Dockerfile's WATO_GIT_COMMIT arg). Read from
                the environment, NOT from `git rev-parse` — deploy images have
                no .git directory, so shelling out to git silently yields "".
  git_dirty     whether that build had uncommitted changes. A dirty build is
                not reproducible; artifacts from one should be treated as
                provisional.
  build_time    when the image was built.
  base_image    the digest-pinned base this component was built FROM.
  lock_sha256   hash of the dependency lock baked into the image
                (/opt/watonomous/requirements.lock.txt). Two artifacts with
                different values here came from different dependency sets, even
                if git_commit matches.
  python        interpreter version.

Every field degrades to None rather than raising: provenance must never be the
reason a long batch run dies.
"""

from __future__ import annotations

import hashlib
import os
import platform
from typing import Any

#: Where the component Dockerfiles leave the dependency lock. Deliberately kept
#: in the image rather than deleted after install, so a running container can
#: describe its own dependency set.
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
    """SHA-256 of a component's config file.

    Config is as much an input to the labeler as the model weights are — a
    changed threshold changes the output. Full hash, not truncated: this is
    compared for equality, never read by a human.
    """
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
            influenced the output. perception_2d gets this from
            ``wato_perception_2d.model_registry.revisions()``. Components with
            no learned models omit it.
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

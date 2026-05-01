"""URI-based artifact storage adapter.

All pipeline code reads and writes through this module so the pipeline can run
unchanged against a local bind-mount (file://) in dev or against S3 in prod.
The artifact root is configured via the ARTIFACT_ROOT_URI env var.
"""

from __future__ import annotations

import os
from pathlib import PurePosixPath
from typing import IO, Any

import fsspec
import yaml


def artifact_root() -> str:
    """Return the configured artifact root URI (e.g. file:///data/artifacts)."""
    return os.environ.get("ARTIFACT_ROOT_URI", "file:///data/artifacts")


def component_versions_path() -> str:
    return os.environ.get("COMPONENT_VERSIONS_PATH", "/config/component_versions.yaml")


def component_version(component: str) -> str:
    """Read the active artifact version for a pipeline component."""
    with open(component_versions_path(), "r", encoding="utf-8") as fh:
        versions = yaml.safe_load(fh) or {}
    return versions[component]


def artifact_uri(component: str, bag_id: str, chunk_id: str = "", *parts: str) -> str:
    """Compose a URI under ${ARTIFACT_ROOT}/<component>/<version>/<bag>/<chunk>/...

    Ingest uses 'raw' instead of its component name to match the arch-doc layout.
    """
    root = artifact_root().rstrip("/")
    if component == "ingest":
        prefix = f"{root}/raw/{bag_id}"
        if chunk_id:
            prefix = f"{prefix}/chunks/{chunk_id}"
    else:
        version = component_version(component)
        prefix = f"{root}/{component}/{version}/{bag_id}"
        if chunk_id:
            prefix = f"{prefix}/{chunk_id}"
    if parts:
        prefix = str(PurePosixPath(prefix, *parts))
    return prefix


def open_uri(uri: str, mode: str = "rb", **kwargs: Any) -> IO[Any]:
    """Open a URI through fsspec; works for file://, s3://, gs://, ..."""
    return fsspec.open(uri, mode=mode, **kwargs).open()


def makedirs(uri: str) -> None:
    """Ensure a URI prefix exists (no-op for object stores)."""
    fs, path = fsspec.core.url_to_fs(uri)
    fs.makedirs(path, exist_ok=True)

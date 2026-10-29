"""Per-chunk artifact manifests: what produced, read and wrote a chunk's outputs.

One manifest per (bag, chunk) per component, next to its artifacts:
``provenance`` (git commit, lock hash, base image, model revisions, config
hash), ``inputs`` (each content-hashed) and ``outputs``. The input hashes make
staleness detectable: if an input's hash no longer matches, the outputs were
produced against data that has since changed.

    manifest.write(
        component="proposal_generation",
        bag_id=bag_id,
        chunk_id=chunk_id,
        inputs={"frame_index": frame_index_path(bag_id, chunk_id)},
        outputs={"proposals": proposals_path(bag_id, chunk_id)},
        config_path=cfg_path,
    )
"""

from __future__ import annotations

import datetime as dt
import json
import os
from typing import Any

from wato_common.artifact_store import (
    ensure_local_dir,
    local_path,
    manifest_path,
)
from wato_common.provenance import collect, file_sha256

#: Files above this size are recorded by (size, mtime) instead of a content
#: hash: hashing every lidar sweep NPZ would add minutes per chunk.
HASH_SIZE_LIMIT_BYTES = 64 * 1024 * 1024


def _describe_input(uri: str) -> dict[str, Any]:
    """Identify an input file well enough to detect that it changed.

    Small files get a content hash, large ones size + mtime. A missing file
    is recorded as missing rather than raising.
    """
    entry: dict[str, Any] = {"uri": uri}
    try:
        path = local_path(uri)
    except ValueError:
        # Remote URI (s3://...). Nothing to stat locally; the URI is the record.
        return entry
    try:
        stat = os.stat(path)
    except OSError:
        entry["present"] = False
        return entry
    entry["size_bytes"] = stat.st_size
    if stat.st_size <= HASH_SIZE_LIMIT_BYTES:
        entry["sha256"] = file_sha256(path)
    else:
        entry["mtime"] = stat.st_mtime
        entry["sha256"] = None  # too large to hash; see HASH_SIZE_LIMIT_BYTES
    return entry


def build(
    *,
    component: str,
    bag_id: str,
    chunk_id: str,
    inputs: dict[str, str] | None = None,
    outputs: dict[str, str] | None = None,
    config_path: str | None = None,
    models: dict[str, str] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble a manifest dict without writing it (used by tests)."""
    manifest: dict[str, Any] = {
        "component": component,
        "bag_id": bag_id,
        "chunk_id": chunk_id,
        "schema_version": 2,
        "inputs": {k: _describe_input(v) for k, v in (inputs or {}).items()},
        "outputs": dict(outputs or {}),
        "provenance": collect(
            component=component, config_path=config_path, models=models
        ),
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    if extra:
        manifest.update(extra)
    return manifest


def write(
    *,
    component: str,
    bag_id: str,
    chunk_id: str,
    inputs: dict[str, str] | None = None,
    outputs: dict[str, str] | None = None,
    config_path: str | None = None,
    models: dict[str, str] | None = None,
    extra: dict[str, Any] | None = None,
    filename: str | None = None,
) -> str:
    """Write a component's manifest for one chunk and return its URI.

    Args:
        filename: manifest basename. Defaults to ``manifest.json`` for ingest's
            historical path; other components pass e.g.
            ``manifest_perception_2d.json`` so several components can write
            into the same chunk directory without clobbering each other.
    """
    manifest = build(
        component=component,
        bag_id=bag_id,
        chunk_id=chunk_id,
        inputs=inputs,
        outputs=outputs,
        config_path=config_path,
        models=models,
        extra=extra,
    )
    out_uri = manifest_path(bag_id, chunk_id)
    if filename:
        out_uri = out_uri.rsplit("/", 1)[0] + "/" + filename
    ensure_local_dir(os.path.dirname(local_path(out_uri)))
    with open(local_path(out_uri), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    return out_uri


def component_manifest_name(component: str) -> str:
    """Conventional manifest filename for a non-ingest component."""
    return f"manifest_{component}.json"

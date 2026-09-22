"""Per-chunk artifact manifests — the traceability record for every component.

Each component writes one manifest per (bag, chunk) it processes, next to the
artifacts it produced. A manifest answers three questions after the fact:

  what produced this?   -> ``provenance`` (git commit, dependency lock hash,
                           base image, pinned model revisions, config hash)
  what did it read?     -> ``inputs``, each with a content hash
  what did it write?    -> ``outputs``

The input hashes are what make staleness detectable. If perception_2d's
manifest records the frame_index.parquet it consumed, and that file's hash no
longer matches, its outputs are stale — a re-ingest happened underneath it.
Without this, the artifact tree cannot distinguish "already done" from "done
against inputs that have since changed", and a stage runner has no safe basis
for skipping work.

Usage from a component:

    from wato_common import manifest
    from wato_common.artifact_store import frame_index_path, proposals_path

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

#: Files above this size are recorded by (size, mtime) instead of by content
#: hash. Hashing every lidar sweep NPZ in a chunk would add minutes per chunk to
#: a pipeline that already costs GPU-hours, for a staleness signal that
#: (size, mtime) already provides well enough.
HASH_SIZE_LIMIT_BYTES = 64 * 1024 * 1024


def _describe_input(uri: str) -> dict[str, Any]:
    """Identify an input file well enough to detect that it changed.

    Small files get a content hash (exact). Large ones get size + mtime — a
    weaker signal, but one that costs nothing and still catches a re-run
    upstream. Missing files are recorded as such rather than raising: a
    manifest describing a partially-satisfied run is more useful than no
    manifest.
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

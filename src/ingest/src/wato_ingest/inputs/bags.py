"""Bag registry: assigns a stable bag_id and writes raw/<bag_id>/bag_meta.json.

The bag_id is derived from the bag's directory name unless overridden.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

from wato_common.artifact_store import bag_meta_path, ensure_local_dir, local_path
from wato_common.io.rosbag_reader import summarize
from wato_common.schemas import BagMeta


_SLUG = re.compile(r"[^a-zA-Z0-9_]+")


def slugify(name: str) -> str:
    return _SLUG.sub("_", name).strip("_") or "bag"


def derive_bag_id(bag_path: str, override: str | None = None) -> str:
    return override if override else slugify(Path(bag_path).name)


def register(
    bag_path: str,
    *,
    bag_id: str | None = None,
    storage_id: str = "sqlite3",
    vehicle: str | None = None,
    calibration_version: str | None = None,
    recording_date: str | None = None,
) -> BagMeta:
    """Inspect the bag, build a BagMeta, write bag_meta.json to artifacts."""
    final_bag_id = derive_bag_id(bag_path, bag_id)

    summary = summarize(bag_path, storage_id=storage_id)
    topic_counts = {t.name: t.message_count for t in summary.topics}

    meta = BagMeta(
        bag_id=final_bag_id,
        source_path=os.path.abspath(bag_path),
        duration_s=summary.duration_ns / 1e9,
        storage_type=storage_id,
        topics=topic_counts,
        vehicle=vehicle,
        calibration_version=calibration_version,
        recording_date=recording_date,
    )

    out_uri = bag_meta_path(final_bag_id)
    ensure_local_dir(os.path.dirname(out_uri))
    with open(local_path(out_uri), "w", encoding="utf-8") as fh:
        json.dump(meta.model_dump(), fh, indent=2)

    return meta


def load_meta(bag_id: str) -> BagMeta:
    with open(local_path(bag_meta_path(bag_id)), "r", encoding="utf-8") as fh:
        return BagMeta.model_validate(json.load(fh))

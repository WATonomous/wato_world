"""Artifact I/O helpers for semantic_lifting.

Reads upstream artifacts from ingest + lidar_preprocessing + perception_2d
(calibration, camera frames, processed sweeps, depth and masklets through
wato_common.artifacts).
Writes lifted_labels/, lifted_instances.parquet and lifted_stats.parquet
through wato_common.artifacts.lifted_labels.
"""

from __future__ import annotations

from wato_common.artifact_store import chunks_index_path
from wato_common.io.parquet_io import read_rows


def load_chunks(bag_id: str) -> list[dict]:
    return read_rows(chunks_index_path(bag_id))

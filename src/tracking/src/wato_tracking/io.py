"""Artifact I/O helpers for tracking.

Reads proposals through wato_common.artifacts.proposals and writes tracks
through wato_common.artifacts.tracks. Read another component's artifact only
through its wato_common.artifacts module, which owns the format.
"""

from __future__ import annotations

import pyarrow as pa

from wato_common.artifact_store import chunks_index_path, track_confirmation_path
from wato_common.io.parquet_io import read_rows, write_table

# Every track that reached its summary, with the evidence its verdict rests
# on: kept, short_weak (the short-and-weak filter) or unconfirmed (too few
# confident instants). Component-local diagnostics, like proposal_generation's
# proposal_refinement.parquet; no other component reads it.
KEPT = "kept"
SHORT_WEAK = "short_weak"
UNCONFIRMED = "unconfirmed"

CONFIRMATION_SCHEMA = pa.schema(
    [
        pa.field("track_id", pa.string()),
        pa.field("chunk_ids", pa.string()),  # JSON list[str]: chunks it spans
        pa.field("cls", pa.string()),
        pa.field("motion_state", pa.string()),
        pa.field("n_frames", pa.int64()),
        pa.field("first_timestamp_ns", pa.int64()),
        pa.field("last_timestamp_ns", pa.int64()),
        pa.field("n_strong", pa.int64()),
        pa.field("n_confident", pa.int64()),
        # What made its instants confident (one instant can count in several).
        pa.field("n_lidar_mos", pa.int64()),
        pa.field("n_lift_verified", pa.int64()),
        pa.field("n_lift_unverified", pa.int64()),
        pa.field("n_detector_only", pa.int64()),
        # Confident instants it needs; null with confirmation off.
        pa.field("n_required", pa.int64()),
        pa.field("decision", pa.string()),
    ]
)


def load_chunks(bag_id: str) -> list[dict]:
    return read_rows(chunks_index_path(bag_id))


def write_confirmation(bag_id: str, rows: list[dict]) -> None:
    write_table(rows, CONFIRMATION_SCHEMA, track_confirmation_path(bag_id))

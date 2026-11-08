"""Artifact I/O helpers for proposal_generation.

Reads world sweeps and frames (wato_common.artifacts.lidar_sweeps), lifted
labels (wato_common.artifacts.lifted_labels), motion clusters
(wato_common.artifacts.motion_proposals), the ground grid
(wato_common.artifacts.ground) and, for the viewers, calibration and camera
frames (wato_common.artifacts.calibration, .frame_index).
Writes proposals.parquet through wato_common.artifacts.proposals.
Writes the component's own proposal_refinement.parquet diagnostics here;
no downstream component consumes that table.
Read another component's artifact only through its wato_common.artifacts
module, which owns the format.
"""

from __future__ import annotations

import pyarrow as pa

from wato_common.artifact_store import chunks_index_path, proposal_refinement_path
from wato_common.io.parquet_io import read_rows, write_table

REFINEMENT_SCHEMA = pa.schema(
    [
        pa.field("sweep_id", pa.int64()),
        pa.field("instance_id", pa.string()),
        pa.field("support_method", pa.string()),
        pa.field("status", pa.string()),
        pa.field("best_radius_m", pa.float64()),
        pa.field("best_cluster_id", pa.int64()),
        pa.field("best_iou", pa.float64()),
        pa.field("n_seed_points", pa.int64()),
        pa.field("n_refined_points", pa.int64()),
        pa.field("n_added_points", pa.int64()),
        pa.field("n_removed_points", pa.int64()),
        pa.field("emitted", pa.bool_()),
        pa.field("n_box_points", pa.int64()),
        pa.field("n_seed_points_in_box", pa.int64()),
    ]
)


def load_chunks(bag_id: str) -> list[dict]:
    return read_rows(chunks_index_path(bag_id))


def write_refinement_audit(bag_id: str, chunk_id: str, rows: list[dict]) -> None:
    write_table(rows, REFINEMENT_SCHEMA, proposal_refinement_path(bag_id, chunk_id))

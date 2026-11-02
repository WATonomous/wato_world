"""Artifact I/O helpers for proposal_generation.

Reads world sweeps and frames (wato_common.artifacts.lidar_sweeps), lifted
labels (wato_common.artifacts.lifted_labels), motion clusters
(wato_common.artifacts.motion_proposals), the ground grid
(wato_common.artifacts.ground) and, for the viewers, calibration and camera
frames (wato_common.artifacts.calibration, .frame_index).
Writes proposals.parquet through wato_common.artifacts.proposals.
Read another component's artifact only through its wato_common.artifacts
module, which owns the format.
"""

from __future__ import annotations

from wato_common.artifact_store import chunks_index_path
from wato_common.io.parquet_io import read_rows


def load_chunks(bag_id: str) -> list[dict]:
    return read_rows(chunks_index_path(bag_id))

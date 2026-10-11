"""semantic_lifting output: 2D masklet labels lifted onto LiDAR points.

Written by semantic_lifting; read by proposal_generation (Segment-Lift-Fit),
tracking and label_refinement once they exist.

On disk, per chunk (under semantic_lifting/):

  lifted_labels/<sweep_id:06d>.npz   one per processed sweep; M labeled points
      point_indices          int32 (M,)  index into the sweep's world NPZ
                                         (lidar_proc/<sweep_id:06d>_world.npz)
      instance_ids           str   (M,)  masklet global_object_id or masklet_id
      classes                str   (M,)  canonical class
      confidences            float32 (M,)
      n_supporting_cameras   int32 (M,)
      n_disagreeing_cameras  int32 (M,)
  lifted_stats.parquet               one LiftedStatsRow per sweep (completion marker)

Strings are fixed-width unicode arrays, so files load without pickle. Files
written before that (object arrays) are rejected: re-run semantic_lifting.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import numpy as np

from wato_common.artifact_store import lifted_labels_path, lifted_stats_path, local_path
from wato_common.artifacts import ArtifactFormatError
from wato_common.io.parquet_io import read_rows, write_table
from wato_common.schemas import LIFTED_STATS_SCHEMA, LiftedStatsRow


@dataclass(frozen=True)
class SweepLabels:
    point_indices: np.ndarray  # int32 (M,)
    instance_ids: np.ndarray  # str (M,)
    classes: np.ndarray  # str (M,)
    confidences: np.ndarray  # float32 (M,)
    n_supporting_cameras: np.ndarray  # int32 (M,)
    n_disagreeing_cameras: np.ndarray  # int32 (M,)

    @classmethod
    def empty(cls) -> "SweepLabels":
        return cls(
            point_indices=np.zeros(0, np.int32),
            instance_ids=np.zeros(0, np.str_),
            classes=np.zeros(0, np.str_),
            confidences=np.zeros(0, np.float32),
            n_supporting_cameras=np.zeros(0, np.int32),
            n_disagreeing_cameras=np.zeros(0, np.int32),
        )


_DTYPES = {
    "point_indices": np.int32,
    "instance_ids": np.str_,
    "classes": np.str_,
    "confidences": np.float32,
    "n_supporting_cameras": np.int32,
    "n_disagreeing_cameras": np.int32,
}


def write_sweep(bag_id: str, chunk_id: str, sweep_id: int, labels: SweepLabels) -> None:
    path = local_path(lifted_labels_path(bag_id, chunk_id, sweep_id))
    arrays = {k: np.asarray(getattr(labels, k), dtype=t) for k, t in _DTYPES.items()}
    lengths = {k: a.shape for k, a in arrays.items()}
    if len(set(lengths.values())) != 1 or arrays["point_indices"].ndim != 1:
        raise ValueError(f"{path}: label arrays must share one (M,) shape: {lengths}")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, **arrays)


def read_sweep(
    bag_id: str, chunk_id: str, sweep_id: int, *, n_points: Optional[int] = None
) -> SweepLabels:
    """One sweep's labels. Pass the world sweep's point count to check indices."""
    path = local_path(lifted_labels_path(bag_id, chunk_id, sweep_id))
    try:
        with np.load(path) as d:
            missing = [k for k in _DTYPES if k not in d]
            if missing:
                raise ArtifactFormatError(path, f"missing arrays {missing}")
            arrays = {k: d[k] for k in _DTYPES}
    except ValueError as exc:
        if isinstance(exc, ArtifactFormatError):
            raise
        raise ArtifactFormatError(
            path, f"{exc} — written before string arrays; re-run semantic_lifting"
        ) from None
    shapes = {a.shape for a in arrays.values()}
    if len(shapes) != 1:
        raise ArtifactFormatError(path, f"arrays differ in shape: {shapes}")
    idx = arrays["point_indices"]
    if n_points is not None and idx.size and (idx.min() < 0 or idx.max() >= n_points):
        raise ArtifactFormatError(
            path, f"point_indices outside the world sweep's {n_points} points"
        )
    return SweepLabels(**arrays)


def write_stats(bag_id: str, chunk_id: str, rows: list[LiftedStatsRow]) -> None:
    write_table(
        [r.model_dump() for r in rows],
        LIFTED_STATS_SCHEMA,
        lifted_stats_path(bag_id, chunk_id),
    )


def read_stats(bag_id: str, chunk_id: str) -> list[LiftedStatsRow]:
    return [LiftedStatsRow(**r) for r in read_rows(lifted_stats_path(bag_id, chunk_id))]


def exists(bag_id: str, chunk_id: str) -> bool:
    """True once the chunk's stats (written last) exist."""
    return os.path.exists(local_path(lifted_stats_path(bag_id, chunk_id)))

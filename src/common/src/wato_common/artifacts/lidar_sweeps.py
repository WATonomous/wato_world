"""LiDAR sweeps: as ingest records them, and as lidar_preprocessing places them.

Raw sweep — written by ingest, read by lidar_preprocessing (deskew):

  lidar/<lidar_id>/<sweep_id:06d>.npz        sensor frame, as recorded
    x, y, z       float32 (N,)   may hold non-finite returns
    intensity     float32 (N,)   optional
    ring          uint16  (N,)   optional
    t_offset_us   float64 (N,)   optional; the PointCloud2's per-point time
                                 field in its own units (lidar_preprocessing's
                                 point_time_unit converts them)

  Its URI is recorded in lidar_sweeps.parquet (`lidar_path`); readers open it
  from there.

World sweep — written by lidar_preprocessing; read by perception_2d,
semantic_lifting (and lidar_preprocessing's later steps):

  lidar_proc/<sweep_id:06d>_world.npz        deskewed, world frame
    x, y, z       float64 (N,)   non-finite raw returns dropped
    origin        float64 (3,)   sensor position in the world, mid-sweep
    intensity     float32 (N,)   optional
    ring                  (N,)   optional, as recorded
    ground_mask   bool    (N,)   optional; Patchwork++ ground
  lidar_proc/<sweep_id:06d>_dynamic_mask.npy
    bool (N,)     Step B's dynamic verdict (True = dynamic),
                  same N and order as the world sweep; length 0 for an empty
                  sweep. Precision-oriented: ~mask is trusted static.
  lidar_proc_index.parquet
    one ProcessedSweepMeta per sweep (schemas.PROCESSED_SWEEPS_SCHEMA), with
    both files' URIs; valid=False rows have neither file.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from wato_common.artifact_store import (
    dynamic_mask_path,
    lidar_proc_index_path,
    lidar_sweep_path,
    lidar_world_path,
    local_path,
)
from wato_common.artifacts import ArtifactFormatError
from wato_common.io.parquet_io import read_rows
from wato_common.schemas import ProcessedSweepMeta


def _optional(d, key: str) -> Optional[np.ndarray]:
    return d[key] if key in d.files else None


def _need(d, path: str, keys: tuple[str, ...]) -> None:
    missing = [k for k in keys if k not in d.files]
    if missing:
        raise ArtifactFormatError(path, f"missing arrays {missing}")


# ---------------------------------------------------------------------------
# Raw sweep (ingest -> lidar_preprocessing).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RawSweep:
    x: np.ndarray  # float32 (N,)
    y: np.ndarray
    z: np.ndarray
    intensity: Optional[np.ndarray] = None  # float32 (N,)
    ring: Optional[np.ndarray] = None  # uint16 (N,)
    t_offset_us: Optional[np.ndarray] = None  # float64 (N,), source units

    @property
    def n_points(self) -> int:
        return int(self.x.shape[0])


def write_raw(
    bag_id: str, chunk_id: str, lidar_id: str, sweep_id: int, sweep: RawSweep
) -> str:
    """Save a decoded sweep; returns the URI to record as its `lidar_path`."""
    uri = lidar_sweep_path(bag_id, chunk_id, lidar_id, sweep_id)
    arrays = {
        "x": sweep.x.astype(np.float32),
        "y": sweep.y.astype(np.float32),
        "z": sweep.z.astype(np.float32),
    }
    if sweep.intensity is not None:
        arrays["intensity"] = sweep.intensity.astype(np.float32)
    if sweep.ring is not None:
        arrays["ring"] = sweep.ring.astype(np.uint16)
    if sweep.t_offset_us is not None:
        arrays["t_offset_us"] = sweep.t_offset_us.astype(np.float64)
    np.savez_compressed(local_path(uri), **arrays)
    return uri


def read_raw(lidar_path: str) -> RawSweep:
    """The raw sweep at a `lidar_path` URI recorded in lidar_sweeps.parquet."""
    path = local_path(lidar_path)
    with np.load(path) as d:
        _need(d, path, ("x", "y", "z"))
        return RawSweep(
            x=d["x"],
            y=d["y"],
            z=d["z"],
            intensity=_optional(d, "intensity"),
            ring=_optional(d, "ring"),
            t_offset_us=_optional(d, "t_offset_us"),
        )


# ---------------------------------------------------------------------------
# World sweep + dynamic mask (lidar_preprocessing -> consumers).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WorldSweep:
    x: np.ndarray  # float64 (N,)
    y: np.ndarray
    z: np.ndarray
    origin: Optional[np.ndarray] = None  # float64 (3,)
    intensity: Optional[np.ndarray] = None  # float32 (N,)
    ring: Optional[np.ndarray] = None
    ground_mask: Optional[np.ndarray] = None  # bool (N,)

    @property
    def n_points(self) -> int:
        return int(self.x.shape[0])

    @property
    def xyz(self) -> np.ndarray:
        """(N, 3) float64."""
        return np.stack([self.x, self.y, self.z], axis=1).astype(np.float64, copy=False)


def write_world(bag_id: str, chunk_id: str, sweep_id: int, sweep: WorldSweep) -> str:
    """Save a deskewed sweep; returns the URI to record as its `world_path`."""
    uri = lidar_world_path(bag_id, chunk_id, sweep_id)
    arrays = {"x": sweep.x, "y": sweep.y, "z": sweep.z}
    for key in ("origin", "intensity", "ring", "ground_mask"):
        value = getattr(sweep, key)
        if value is not None:
            arrays[key] = value
    n = sweep.n_points
    bad = [k for k, a in arrays.items() if k != "origin" and a.shape != (n,)]
    if bad:
        raise ValueError(f"{uri}: arrays {bad} are not ({n},)")
    np.savez_compressed(local_path(uri), **arrays)
    return uri


def read_world(world_path: str) -> WorldSweep:
    """The world sweep at a `world_path` URI from lidar_proc_index.parquet."""
    path = local_path(world_path)
    with np.load(path) as d:
        _need(d, path, ("x", "y", "z"))
        origin = _optional(d, "origin")
        sweep = WorldSweep(
            x=d["x"],
            y=d["y"],
            z=d["z"],
            origin=np.asarray(origin, dtype=np.float64) if origin is not None else None,
            intensity=_optional(d, "intensity"),
            ring=_optional(d, "ring"),
            ground_mask=_optional(d, "ground_mask"),
        )
    n = sweep.n_points
    for key in ("y", "z", "intensity", "ring", "ground_mask"):
        value = getattr(sweep, key)
        if value is not None and value.shape != (n,):
            raise ArtifactFormatError(
                path, f"{key} has shape {value.shape}, not ({n},)"
            )
    return sweep


def write_dynamic_mask(
    bag_id: str, chunk_id: str, sweep_id: int, mask: np.ndarray
) -> str:
    """Save the sweep's dynamic verdict (aligned to its world sweep); returns the URI."""
    uri = dynamic_mask_path(bag_id, chunk_id, sweep_id)
    if mask.dtype != np.bool_ or mask.ndim != 1:
        raise ValueError(
            f"{uri}: dynamic mask must be 1-D bool, got {mask.dtype}{mask.shape}"
        )
    np.save(local_path(uri), mask)
    return uri


def read_dynamic_mask(
    dynamic_mask_path_uri: str, *, n_points: Optional[int] = None
) -> np.ndarray:
    """A sweep's dynamic mask. Pass the world sweep's N to check the alignment."""
    path = local_path(dynamic_mask_path_uri)
    mask = np.load(path)
    if mask.dtype != np.bool_ or mask.ndim != 1:
        raise ArtifactFormatError(
            path, f"not a 1-D bool mask: {mask.dtype}{mask.shape}"
        )
    if n_points is not None and mask.shape[0] != n_points:
        raise ArtifactFormatError(
            path, f"{mask.shape[0]} entries but the world sweep has {n_points} points"
        )
    return mask


class ProcessedSweeps:
    """A chunk's valid processed sweeps, by sweep_id (reads the index once)."""

    def __init__(self, rows: list[ProcessedSweepMeta]):
        self.rows = rows
        self._by_id: dict[int, ProcessedSweepMeta] = {}
        for r in rows:
            self._by_id.setdefault(r.sweep_id, r)

    @classmethod
    def load(cls, bag_id: str, chunk_id: str) -> "ProcessedSweeps":
        rows = [
            ProcessedSweepMeta(**r)
            for r in read_rows(lidar_proc_index_path(bag_id, chunk_id))
            if r.get("valid") is not False
        ]
        return cls(rows)

    def get(self, sweep_id: int) -> Optional[ProcessedSweepMeta]:
        return self._by_id.get(sweep_id)

    def world(self, sweep_id: int) -> WorldSweep:
        """KeyError when the sweep isn't a valid row; FileNotFoundError when
        its world NPZ is missing."""
        return read_world(self._by_id[sweep_id].world_path)

    def world_and_dynamic_mask(self, sweep_id: int) -> tuple[WorldSweep, np.ndarray]:
        """The world sweep and its index-aligned dynamic mask."""
        meta = self._by_id[sweep_id]
        sweep = read_world(meta.world_path)
        return sweep, read_dynamic_mask(meta.dynamic_mask_path, n_points=sweep.n_points)

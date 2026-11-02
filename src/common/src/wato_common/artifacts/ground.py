"""lidar_preprocessing Step C ground — the per-chunk height and normal grid.

Written by lidar_preprocessing (`ground/`); read by its own Step F (height
floor), viz and reduce, and by proposal_generation (box z, SLF's ground term).

On disk, per chunk:

  ground.npz
      height_grid  float32 (H, W)    median ground z per cell; rows are y,
                                     columns x; holes filled from the
                                     nearest populated cell, so all finite
      normal_grid  float32 (H, W, 3) unit surface normals of height_grid
      grid_origin  float64 (2,)      world [x, y] of cell (0, 0)'s lower-left
                                     corner: cell (iy, ix) spans
                                     origin + [ix, iy] * cell_size
      cell_size    float32           metres
      ground_xyz   float64 (M, 3)    the ground returns the grid was built from
      status       str               "ok", or a sentinel: "empty" (no ground
                                     returns) / "skipped_no_ground_mask"
                                     (Patchwork++ unavailable upstream)

A sentinel keeps a placeholder 1x1 grid so "no ground" is distinguishable from
"not yet processed" (no file); `read` raises GroundUnavailableError for one
rather than hand back that placeholder.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

from wato_common.artifact_store import ground_path, local_path
from wato_common.artifacts import ArtifactFormatError

STATUS_OK = "ok"
STATUSES = (STATUS_OK, "empty", "skipped_no_ground_mask")

_KEYS = (
    "height_grid",
    "normal_grid",
    "grid_origin",
    "cell_size",
    "ground_xyz",
    "status",
)


class GroundUnavailableError(RuntimeError):
    """ground.npz is a sentinel: Step C ran but has no grid for this chunk."""

    def __init__(self, path: str, status: str) -> None:
        super().__init__(f"{path}: no ground grid (status={status!r})")
        self.path = path
        self.status = status


@dataclass(frozen=True)
class GroundGrid:
    height_grid: np.ndarray  # float32 (H, W)
    normal_grid: np.ndarray  # float32 (H, W, 3)
    grid_origin: np.ndarray  # float64 (2,)
    cell_size: float

    def _cells(self, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(iy, ix) of each (x, y). Points outside the grid clamp to the edge
        cell; the grid spans the bbox of the chunk's ground returns, so real
        out-of-bounds points are rare."""
        h, w = self.height_grid.shape
        rel = (
            np.asarray(xy, dtype=np.float64)[:, :2] - self.grid_origin
        ) / self.cell_size
        ix = np.clip(np.floor(rel[:, 0]).astype(np.int64), 0, w - 1)
        iy = np.clip(np.floor(rel[:, 1]).astype(np.int64), 0, h - 1)
        return iy, ix

    def height_at(self, xy: np.ndarray) -> np.ndarray:
        """Ground z under each (x, y), (N,)."""
        iy, ix = self._cells(xy)
        return self.height_grid[iy, ix]

    def height_above(self, xyz: np.ndarray) -> np.ndarray:
        """Each point's height over the ground, (N,)."""
        return np.asarray(xyz)[:, 2] - self.height_at(xyz)

    def normal_at(self, xy: np.ndarray) -> np.ndarray:
        """Unit ground normal under each (x, y), (N, 3)."""
        iy, ix = self._cells(xy)
        return self.normal_grid[iy, ix]


def _grid_problem(
    height_grid: np.ndarray,
    normal_grid: np.ndarray,
    grid_origin: np.ndarray,
    cell_size: float,
) -> str | None:
    if height_grid.ndim != 2 or 0 in height_grid.shape:
        return f"height_grid {height_grid.shape} is not a non-empty (H, W)"
    if normal_grid.shape != height_grid.shape + (3,):
        return (
            f"normal_grid {normal_grid.shape} != height_grid {height_grid.shape} + (3,)"
        )
    if grid_origin.shape != (2,):
        return f"grid_origin {grid_origin.shape} is not (2,)"
    if not cell_size > 0:
        return f"cell_size {cell_size} is not > 0"
    if not np.isfinite(height_grid).all():
        return "height_grid has non-finite cells"
    return None


def write(
    bag_id: str,
    chunk_id: str,
    grid: GroundGrid,
    ground_xyz: np.ndarray,
    *,
    status: str = STATUS_OK,
) -> str:
    """Write a chunk's ground.npz; a sentinel status still needs a placeholder grid."""
    uri = ground_path(bag_id, chunk_id)
    path = local_path(uri)
    if status not in STATUSES:
        raise ValueError(f"{path}: status {status!r} not in {STATUSES}")
    problem = _grid_problem(
        grid.height_grid, grid.normal_grid, grid.grid_origin, grid.cell_size
    )
    if problem is None and (ground_xyz.ndim != 2 or ground_xyz.shape[1] != 3):
        problem = f"ground_xyz {ground_xyz.shape} is not (M, 3)"
    if problem is not None:
        raise ValueError(f"{path}: {problem}")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(
        path,
        height_grid=np.asarray(grid.height_grid, dtype=np.float32),
        normal_grid=np.asarray(grid.normal_grid, dtype=np.float32),
        grid_origin=np.asarray(grid.grid_origin, dtype=np.float64),
        cell_size=np.float32(grid.cell_size),
        ground_xyz=np.asarray(ground_xyz, dtype=np.float64),
        status=np.array(status),
    )
    return uri


def _status(d, path: str) -> str:
    missing = [k for k in _KEYS if k not in d]
    if missing:
        raise ArtifactFormatError(
            path, f"missing arrays {missing} — re-run lidar_preprocessing"
        )
    status = str(d["status"])
    if status not in STATUSES:
        raise ArtifactFormatError(path, f"status {status!r} not in {STATUSES}")
    return status


def read(bag_id: str, chunk_id: str) -> GroundGrid:
    """The chunk's ground grid.

    FileNotFoundError when Step C hasn't run; GroundUnavailableError when it
    ran but wrote a sentinel.
    """
    path = local_path(ground_path(bag_id, chunk_id))
    with np.load(path) as d:
        status = _status(d, path)
        if status != STATUS_OK:
            raise GroundUnavailableError(path, status)
        height_grid = d["height_grid"]
        normal_grid = d["normal_grid"]
        grid_origin = np.asarray(d["grid_origin"], dtype=np.float64)
        cell_size = float(d["cell_size"])
    problem = _grid_problem(height_grid, normal_grid, grid_origin, cell_size)
    if problem is not None:
        raise ArtifactFormatError(path, problem)
    return GroundGrid(height_grid, normal_grid, grid_origin, cell_size)


def read_points(bag_id: str, chunk_id: str) -> np.ndarray:
    """The ground returns the grid was built from, (M, 3); empty for a sentinel."""
    path = local_path(ground_path(bag_id, chunk_id))
    with np.load(path) as d:
        _status(d, path)
        xyz = d["ground_xyz"]
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ArtifactFormatError(path, f"ground_xyz {xyz.shape} is not (M, 3)")
    return xyz

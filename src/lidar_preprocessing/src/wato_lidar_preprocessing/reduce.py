"""Step D — Global static map reduce.

Merges all per-chunk static_map.npz files for a bag into a single
downsampled global_static_map.npz at bag level.  Run after all chunks
have been processed via the `reduce` CLI subcommand.

Uses a numpy voxel-corner snap (np.unique on packed voxel keys) — one
point per voxel, snapped to the cell centre.  This avoids the ~80 MB
Open3D dependency for what is effectively a visualisation/QA artifact.
If sub-voxel centroid accuracy ever matters we can switch to a
group-and-mean implementation, see README "Possible follow-ups".
"""

from __future__ import annotations

import logging
import os

import numpy as np

from wato_common.artifact_store import (
    chunks_index_path,
    global_ground_path,
    global_static_map_path,
    ground_path,
    local_path,
    static_map_path,
)
from wato_common.io.parquet_io import read_rows
from wato_lidar_preprocessing.config import ComponentConfig
from wato_lidar_preprocessing.ground import _build_height_grid

log = logging.getLogger(__name__)


def _voxel_snap_downsample(xyz: np.ndarray, voxel_size: float) -> np.ndarray:
    """Return one point per voxel, snapped to the voxel-cell centre.

    Equivalent to Open3D's voxel_down_sample for visualisation purposes,
    minus the per-voxel centroid (we use the cell centre instead).  ~30 LOC
    less than the Open3D path and works on identical inputs.
    """
    if xyz.shape[0] == 0:
        return np.empty((0, 3), dtype=np.float64)

    # Discretise to int64 voxel indices.  Range is unbounded by design here
    # (no AXIS_BITS limit) — we simply use a wide-enough int64 packing.
    idx = np.floor(xyz / voxel_size).astype(np.int64)
    # Pack into one int64 key for np.unique.  Bit budget: 21 bits per axis,
    # supporting ±(2^20 * voxel_size) m ≈ ±300 km at 30 cm voxel — plenty.
    bits = 21
    axis_max = 1 << (bits - 1)
    if (np.abs(idx) >= axis_max).any():
        # Catastrophic input — keep the function correct rather than silently
        # truncating.  Falls back to the slow but accurate dedup.
        log.warning(
            "global_static_map: voxel indices exceed int64 packing range — "
            "falling back to row-wise unique"
        )
        unique_idx = np.unique(idx, axis=0)
        return (unique_idx.astype(np.float64) + 0.5) * voxel_size

    # Shift to non-negative (so left-shift is well defined), then pack.
    shifted = idx + axis_max
    keys = (
        (shifted[:, 0].astype(np.int64) << (2 * bits))
        | (shifted[:, 1].astype(np.int64) << bits)
        | shifted[:, 2].astype(np.int64)
    )
    unique_keys = np.unique(keys)
    # Unpack.
    mask = (1 << bits) - 1
    sx = ((unique_keys >> (2 * bits)) & mask) - axis_max
    sy = ((unique_keys >> bits) & mask) - axis_max
    sz = (unique_keys & mask) - axis_max
    centres = np.stack([sx, sy, sz], axis=1).astype(np.float64)
    return (centres + 0.5) * voxel_size


def reduce_static_map(bag_id: str, cfg: ComponentConfig) -> str:
    """Merge and downsample all per-chunk static maps for bag_id.

    Silently skips chunks whose static_map.npz does not yet exist so that
    a partial run is still useful.  Returns the URI of global_static_map.npz.
    """
    chunk_rows = read_rows(chunks_index_path(bag_id))
    if not chunk_rows:
        raise RuntimeError(f"No chunks found for bag_id={bag_id!r}")

    all_xyz: list[np.ndarray] = []
    for row in chunk_rows:
        chunk_id = row["chunk_id"]
        path = local_path(static_map_path(bag_id, chunk_id))
        if not os.path.exists(path):
            log.warning(
                "chunk %s static_map.npz not found — chunk not yet processed, skipping",
                chunk_id,
            )
            continue
        data = np.load(path)
        xyz = data["xyz"]
        if xyz.shape[0] > 0:
            all_xyz.append(xyz)
        log.debug("loaded %d static points from chunk %s", xyz.shape[0], chunk_id)

    if not all_xyz:
        log.warning("bag %s: no static map chunks found, writing empty global map", bag_id)
        out_uri = global_static_map_path(bag_id)
        np.savez_compressed(
            local_path(out_uri),
            xyz=np.empty((0, 3), dtype=np.float64),
        )
        return out_uri

    merged = np.concatenate(all_xyz, axis=0)
    log.info("bag %s: merging %d static points across %d chunks", bag_id, merged.shape[0], len(all_xyz))

    xyz_down = _voxel_snap_downsample(merged, cfg.global_map_voxel_size_m)
    log.info(
        "bag %s: downsampled %d → %d points (voxel=%.2f m)",
        bag_id,
        merged.shape[0],
        xyz_down.shape[0],
        cfg.global_map_voxel_size_m,
    )

    out_uri = global_static_map_path(bag_id)
    np.savez_compressed(local_path(out_uri), xyz=xyz_down)
    return out_uri


def reduce_ground_map(bag_id: str, cfg: ComponentConfig) -> str:
    """Merge per-chunk ground.npz into a bag-level global_ground.npz.

    Concatenates `ground_xyz` from every chunk and rebuilds one bag-level
    height_grid + normal_grid via the existing `_build_height_grid` helper
    (the same one ground.py uses), so the schema matches per-chunk ground.npz
    and downstream consumers can use a single loader regardless of scope.

    This solves the SLF chunk-boundary problem: proposal_generation can query
    z_ground(x, y) over the full bag without stitching per-chunk grids.

    Silently skips chunks whose ground.npz is missing or has empty
    `ground_xyz` so partial runs still produce a useful artifact.
    """
    chunk_rows = read_rows(chunks_index_path(bag_id))
    if not chunk_rows:
        raise RuntimeError(f"No chunks found for bag_id={bag_id!r}")

    cell_size = cfg.patchwork.ground_cell_size_m
    out_uri = global_ground_path(bag_id)

    all_ground: list[np.ndarray] = []
    for row in chunk_rows:
        chunk_id = row["chunk_id"]
        path = local_path(ground_path(bag_id, chunk_id))
        if not os.path.exists(path):
            log.warning(
                "chunk %s ground.npz not found — chunk not yet processed, skipping",
                chunk_id,
            )
            continue
        data = np.load(path)
        ground_xyz = data.get("ground_xyz")
        if ground_xyz is None or ground_xyz.shape[0] == 0:
            log.debug("chunk %s has no ground points, skipping", chunk_id)
            continue
        all_ground.append(np.asarray(ground_xyz, dtype=np.float64))

    if not all_ground:
        log.warning(
            "bag %s: no per-chunk ground points found, writing empty global ground map",
            bag_id,
        )
        # Sentinel: 1x1 height grid so consumers don't have to special-case
        # the empty path; matches what _build_height_grid does for empty input.
        height_grid, normal_grid, grid_origin = _build_height_grid(
            np.empty((0, 3), dtype=np.float64), cell_size=cell_size
        )
        np.savez_compressed(
            local_path(out_uri),
            height_grid=height_grid,
            normal_grid=normal_grid,
            grid_origin=grid_origin,
            cell_size=np.float32(cell_size),
            ground_xyz=np.empty((0, 3), dtype=np.float64),
        )
        return out_uri

    merged = np.concatenate(all_ground, axis=0)
    log.info(
        "bag %s: merging %d ground points across %d chunks (cell=%.2f m)",
        bag_id,
        merged.shape[0],
        len(all_ground),
        cell_size,
    )
    height_grid, normal_grid, grid_origin = _build_height_grid(
        merged, cell_size=cell_size
    )
    np.savez_compressed(
        local_path(out_uri),
        height_grid=height_grid,
        normal_grid=normal_grid,
        grid_origin=grid_origin,
        cell_size=np.float32(cell_size),
        ground_xyz=merged,
    )
    log.info(
        "bag %s: wrote global_ground.npz (grid=%s, origin=%s)",
        bag_id,
        tuple(height_grid.shape),
        grid_origin.tolist(),
    )
    return out_uri

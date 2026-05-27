"""Step C — Aggregate per-sweep ground masks + height-grid builder.

Reads each world-frame sweep NPZ. Sweeps carrying a `ground_mask` contribute
their ground points to the chunk-level ground cloud, which is then binned
into a 2D height grid + surface-normal grid.

Each candidate ground point is intersected against the static-voxel set
from Step B (classify) — points in non-static voxels are dropped, which
removes vehicle-underside contamination.

When no sweep carries a ground mask (Patchwork++ unavailable upstream),
writes a sentinel ground.npz with status="skipped_no_ground_mask" so
downstream consumers can distinguish "unavailable" from "not yet processed".
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.stats import binned_statistic_2d
from tqdm import tqdm

from wato_common.artifact_store import (
    ground_path,
    lidar_proc_index_path,
    local_path,
    static_map_path,
)
from wato_common.io.parquet_io import read_rows
from wato_lidar_preprocessing.config import ComponentConfig, PatchworkParams
from wato_lidar_preprocessing.voxel import voxel_indices

log = logging.getLogger(__name__)

_GRID_CELL_WARN_THRESHOLD = 50_000_000  # ~200 MB float32 cells


@dataclass
class GroundResult:
    n_ground: int
    n_nonground: int
    ground_path: str
    status: str = "ok"  # "ok" | "skipped_no_ground_mask" | "empty"
    n_dropped_dynamic: int = 0  # rejected by static-voxel intersection


def _load_static_voxel_set(
    bag_id: str, chunk_id: str
) -> tuple[np.ndarray, np.ndarray, float]:
    """Load classify's static-voxel keys + origin/voxel_size.

    Returns (sorted_keys, origin_xyz, voxel_size).
    Raises FileNotFoundError if classify hasn't run; KeyError if the file
    predates the static-voxel-keys / origin / voxel_size export (re-run
    classify with --force).
    """
    path = local_path(static_map_path(bag_id, chunk_id))
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"static_map.npz missing for chunk {chunk_id!r} — run Step B (classify) first."
        )
    data = np.load(path)
    missing = [
        k for k in ("static_voxel_keys", "origin", "voxel_size") if k not in data
    ]
    if missing:
        raise KeyError(
            f"static_map.npz for chunk {chunk_id!r} is missing keys {missing} — "
            "predates the ground/static-intersection feature; "
            "re-run lidar_preprocessing with --force on this chunk."
        )
    keys = np.asarray(data["static_voxel_keys"], dtype=np.int64)
    keys.sort()
    origin = np.asarray(data["origin"], dtype=np.float64)
    voxel_size = float(data["voxel_size"])
    return keys, origin, voxel_size


def _filter_by_static_voxels(
    xyz: np.ndarray,
    static_keys: np.ndarray,
    origin: np.ndarray,
    voxel_size: float,
    *,
    chunk_id: str,
) -> np.ndarray:
    """Keep only rows of `xyz` whose voxel key is in `static_keys`.

    Uses the same voxel_indices helper as classify so keys align bit-for-bit.
    """
    if xyz.shape[0] == 0 or static_keys.size == 0:
        return np.empty((0, 3), dtype=xyz.dtype)
    keys = voxel_indices(xyz, origin, voxel_size, chunk_id=chunk_id)
    pos = np.searchsorted(static_keys, keys)
    pos = np.clip(pos, 0, static_keys.size - 1)
    keep = static_keys[pos] == keys
    return xyz[keep]


def _build_height_grid(
    ground_xyz: np.ndarray,
    cell_size: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a 2D height grid and surface-normal grid from ground points.

    Returns:
        height_grid  (H, W) float32  — median ground Z per cell
        normal_grid  (H, W, 3) float32 — unit surface normals
        grid_origin  (2,) float64   — [x0, y0] of lower-left cell centre
    """
    if ground_xyz.shape[0] == 0:
        empty_hg = np.zeros((1, 1), dtype=np.float32)
        empty_ng = np.zeros((1, 1, 3), dtype=np.float32)
        empty_ng[0, 0, 2] = 1.0
        return empty_hg, empty_ng, np.zeros(2, dtype=np.float64)

    x = ground_xyz[:, 0]
    y = ground_xyz[:, 1]
    z = ground_xyz[:, 2]

    x0 = float(x.min())
    y0 = float(y.min())
    W = int(np.floor((x.max() - x0) / cell_size)) + 1
    H = int(np.floor((y.max() - y0) / cell_size)) + 1

    if H * W > _GRID_CELL_WARN_THRESHOLD:
        log.warning(
            "height grid is %dx%d (~%.1f GB float32) at cell_size=%.2fm — consider downsampling",
            H,
            W,
            (H * W * 4) / 1e9,
            cell_size,
        )

    x_edges = x0 + np.arange(W + 1) * cell_size
    y_edges = y0 + np.arange(H + 1) * cell_size

    # binned_statistic_2d returns (W_x, H_y); transpose to (H_y, W_x).
    stat = binned_statistic_2d(
        x, y, z, statistic="median", bins=[x_edges, y_edges]
    ).statistic
    height_grid = stat.T.astype(np.float32)  # (H, W)

    # Fill NaN holes via nearest populated cell.
    nan_mask = np.isnan(height_grid)
    if nan_mask.all():
        height_grid[:] = 0.0
    elif nan_mask.any():
        _, indices = distance_transform_edt(nan_mask, return_indices=True)
        height_grid[nan_mask] = height_grid[indices[0][nan_mask], indices[1][nan_mask]]

    # Surface normals via finite differences on the height grid.
    # n = [-dh/dx, -dh/dy, 1] normalised.
    hf = height_grid.astype(np.float64)
    if H >= 2 and W >= 2:
        dh_dr, dh_dc = np.gradient(hf)
    elif H >= 2:
        dh_dr = np.gradient(hf, axis=0)
        dh_dc = np.zeros_like(hf)
    elif W >= 2:
        dh_dr = np.zeros_like(hf)
        dh_dc = np.gradient(hf, axis=1)
    else:
        dh_dr = np.zeros_like(hf)
        dh_dc = np.zeros_like(hf)
    nx = -dh_dc.astype(np.float32)
    ny = -dh_dr.astype(np.float32)
    nz = np.ones((H, W), dtype=np.float32)
    nlen = np.sqrt(nx**2 + ny**2 + nz**2)
    nlen = np.where(nlen < 1e-9, 1.0, nlen)
    normal_grid = np.stack([nx / nlen, ny / nlen, nz / nlen], axis=-1)

    grid_origin = np.array([x0, y0], dtype=np.float64)
    return height_grid, normal_grid, grid_origin


def process_chunk(
    cfg: ComponentConfig,
    bag_id: str,
    chunk_id: str,
) -> GroundResult:
    """Aggregate per-sweep ground masks and build the height grid."""
    meta_rows = read_rows(lidar_proc_index_path(bag_id, chunk_id))
    if not meta_rows:
        log.warning(
            "chunk %s: empty proc index — writing sentinel ground.npz", chunk_id
        )
        return _save_ground(
            bag_id,
            chunk_id,
            np.empty((0, 3), dtype=np.float64),
            cfg.patchwork,
            status="empty",
        )

    ground_chunks: list[np.ndarray] = []
    # Counted only over sweeps with a ground_mask, so n_nonground reflects
    # "Patchwork said non-ground" rather than "Patchwork never ran".
    n_classified_pts = 0
    n_with_mask = 0
    n_dropped_dynamic = 0

    static_keys, static_origin, static_voxel_size = _load_static_voxel_set(
        bag_id, chunk_id
    )

    for row in tqdm(
        meta_rows,
        desc=f"ground chunk {chunk_id}",
        unit="sweep",
    ):
        # parquet stores missing columns as None — treat as valid=True;
        # skip only on explicit valid=False.
        if row.get("valid") is False:
            continue
        world_path = local_path(row["world_path"])
        data = np.load(world_path)
        if "ground_mask" not in data:
            continue
        n_with_mask += 1
        mask = data["ground_mask"]
        n_classified_pts += int(mask.shape[0])
        if mask.sum() == 0:
            continue
        xyz = np.stack(
            [data["x"][mask], data["y"][mask], data["z"][mask]],
            axis=1,
        ).astype(np.float64)
        n_before = xyz.shape[0]
        xyz = _filter_by_static_voxels(
            xyz,
            static_keys,
            static_origin,
            static_voxel_size,
            chunk_id=chunk_id,
        )
        n_dropped_dynamic += n_before - xyz.shape[0]
        if xyz.shape[0] == 0:
            continue
        ground_chunks.append(xyz)

    if n_with_mask == 0:
        log.warning(
            "chunk %s: no sweeps carry ground_mask (pypatchworkpp unavailable upstream); writing sentinel ground.npz",
            chunk_id,
        )
        return _save_ground(
            bag_id,
            chunk_id,
            np.empty((0, 3), dtype=np.float64),
            cfg.patchwork,
            status="skipped_no_ground_mask",
        )

    if ground_chunks:
        ground_pts = np.concatenate(ground_chunks, axis=0)
    else:
        ground_pts = np.empty((0, 3), dtype=np.float64)

    n_ground = ground_pts.shape[0]
    n_nonground = n_classified_pts - n_ground - n_dropped_dynamic
    log.info(
        "chunk %s: aggregated ground=%d nonground=%d dropped_dynamic=%d across %d classified sweeps",
        chunk_id,
        n_ground,
        n_nonground,
        n_dropped_dynamic,
        n_with_mask,
    )

    return _save_ground(
        bag_id,
        chunk_id,
        ground_pts,
        cfg.patchwork,
        status="ok" if n_ground > 0 else "empty",
        n_nonground=n_nonground,
        n_dropped_dynamic=n_dropped_dynamic,
    )


def _save_ground(
    bag_id: str,
    chunk_id: str,
    ground_pts: np.ndarray,
    params: PatchworkParams,
    *,
    status: str = "ok",
    n_nonground: int = 0,
    n_dropped_dynamic: int = 0,
) -> GroundResult:
    height_grid, normal_grid, grid_origin = _build_height_grid(
        ground_pts, cell_size=params.ground_cell_size_m
    )
    out_uri = ground_path(bag_id, chunk_id)
    np.savez_compressed(
        local_path(out_uri),
        height_grid=height_grid,
        normal_grid=normal_grid,
        grid_origin=grid_origin,
        cell_size=np.float32(params.ground_cell_size_m),
        ground_xyz=ground_pts,
        status=np.array(status),
    )
    return GroundResult(
        n_ground=int(ground_pts.shape[0]),
        n_nonground=int(n_nonground),
        ground_path=out_uri,
        status=status,
        n_dropped_dynamic=int(n_dropped_dynamic),
    )

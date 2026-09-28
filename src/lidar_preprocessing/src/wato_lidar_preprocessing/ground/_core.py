"""Step C — Aggregate per-sweep ground masks + height-grid builder.

Reads each world-frame sweep NPZ. Sweeps carrying a `ground_mask` contribute
their ground points to the chunk-level ground cloud, which is then binned
into a 2D height grid + surface-normal grid.

Each candidate ground point is retained unless the final point-aligned motion
mask explicitly marks it dynamic. Unknown or under-evidenced AW state is not
a reason to discard road surface.

When no sweep carries a ground mask (Patchwork++ unavailable upstream),
writes a sentinel ground.npz with status="skipped_no_ground_mask" so
downstream consumers can distinguish "unavailable" from "not yet processed".
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.stats import binned_statistic_2d
from tqdm import tqdm

from wato_common.artifact_store import (
    ground_path,
    lidar_proc_index_path,
    local_path,
)
from wato_common.io.parquet_io import read_rows, write_table
from wato_common.schemas import PROCESSED_SWEEPS_SCHEMA
from wato_lidar_preprocessing.config import ComponentConfig, PatchworkParams

log = logging.getLogger(__name__)

_GRID_CELL_WARN_THRESHOLD = 50_000_000  # ~200 MB float32 cells


@dataclass
class GroundResult:
    n_ground: int
    n_nonground: int
    ground_path: str
    status: str = "ok"  # "ok" | "skipped_no_ground_mask" | "empty"
    n_rejected_dynamic_ground: int = 0


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
    n_ground_candidates = 0
    n_rejected_dynamic_ground = 0

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
        candidate_mask = np.asarray(data["ground_mask"], dtype=bool)
        n_world = int(data["x"].shape[0])
        if candidate_mask.shape != (n_world,):
            raise RuntimeError(
                f"ground_mask for ({row['lidar_id']!r}, {row['sweep_id']}) "
                f"has shape {candidate_mask.shape}, expected {(n_world,)}"
            )
        dynamic_uri = row.get("dynamic_mask_path")
        if not dynamic_uri:
            raise RuntimeError(
                f"dynamic_mask_path missing for ({row['lidar_id']!r}, "
                f"{row['sweep_id']}); classify must complete before ground"
            )
        dynamic_mask = np.asarray(np.load(local_path(dynamic_uri)), dtype=bool)
        if dynamic_mask.shape != (n_world,):
            raise RuntimeError(
                f"dynamic_mask for ({row['lidar_id']!r}, {row['sweep_id']}) "
                f"has shape {dynamic_mask.shape}, expected {(n_world,)}"
            )
        resolved_mask = candidate_mask & ~dynamic_mask
        n_candidate = int(candidate_mask.sum())
        n_ground = int(resolved_mask.sum())
        row["n_points_ground_candidate"] = n_candidate
        row["n_points_ground"] = n_ground
        n_classified_pts += n_world
        n_ground_candidates += n_candidate
        n_rejected_dynamic_ground += int((candidate_mask & dynamic_mask).sum())
        if n_ground == 0:
            continue
        xyz = np.stack(
            [
                data["x"][resolved_mask],
                data["y"][resolved_mask],
                data["z"][resolved_mask],
            ],
            axis=1,
        ).astype(np.float64)
        ground_chunks.append(xyz)

    if n_with_mask == 0:
        log.warning(
            "chunk %s: no sweeps carry ground_mask "
            "(pypatchworkpp unavailable upstream); writing sentinel ground.npz",
            chunk_id,
        )
        write_table(
            meta_rows,
            PROCESSED_SWEEPS_SCHEMA,
            lidar_proc_index_path(bag_id, chunk_id),
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
    n_nonground = n_classified_pts - n_ground_candidates
    log.info(
        "chunk %s: aggregated ground=%d nonground=%d "
        "rejected_dynamic_ground=%d across %d classified sweeps",
        chunk_id,
        n_ground,
        n_nonground,
        n_rejected_dynamic_ground,
        n_with_mask,
    )

    write_table(
        meta_rows,
        PROCESSED_SWEEPS_SCHEMA,
        lidar_proc_index_path(bag_id, chunk_id),
    )
    return _save_ground(
        bag_id,
        chunk_id,
        ground_pts,
        cfg.patchwork,
        status="ok" if n_ground > 0 else "empty",
        n_nonground=n_nonground,
        n_rejected_dynamic_ground=n_rejected_dynamic_ground,
    )


def _save_ground(
    bag_id: str,
    chunk_id: str,
    ground_pts: np.ndarray,
    params: PatchworkParams,
    *,
    status: str = "ok",
    n_nonground: int = 0,
    n_rejected_dynamic_ground: int = 0,
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
        n_rejected_dynamic_ground=int(n_rejected_dynamic_ground),
    )

"""Artifact staging helpers shared by the Step E / Step F tests.

Writes the minimal on-disk set (chunk index, lidar_proc_index rows, world
NPZs, static_map / ground / global maps) those steps read, so the tests run
on CPU with no deskew, numba or torch.
"""

from __future__ import annotations

import os

import numpy as np

from wato_common.artifact_store import (
    chunks_index_path,
    dynamic_mask_path,
    ensure_local_dir,
    global_static_map_path,
    ground_path,
    lidar_proc_dir,
    lidar_proc_index_path,
    lidar_proc_summary_path,
    lidar_world_path,
    local_path,
    static_map_path,
)
from wato_common.io.parquet_io import write_table
from wato_common.schemas import (
    CHUNK_SCHEMA,
    CHUNK_SUMMARY_SCHEMA,
    PROCESSED_SWEEPS_SCHEMA,
    ChunkSummaryRow,
    ProcessedSweepMeta,
)
from wato_lidar_preprocessing.voxel import voxel_indices

SWEEP_DT_NS = 50_000_000  # 20 Hz


def grid_points(x, y, z, step: float) -> np.ndarray:
    """Dense axis-aligned grid over the given (lo, hi) ranges; a scalar pins
    that axis."""

    def axis(v):
        if isinstance(v, (int, float)):
            return np.array([float(v)])
        lo, hi = v
        return np.arange(lo, hi + 1e-9, step)

    gx, gy, gz = np.meshgrid(axis(x), axis(y), axis(z), indexing="ij")
    return np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])


def write_chunks(bag_id: str, chunk_ids: list[str]) -> None:
    rows = [
        dict(
            bag_id=bag_id,
            chunk_id=c,
            t_start_ns=0,
            t_end_ns=1,
            t_overlap_start_ns=0,
            t_overlap_end_ns=1,
        )
        for c in chunk_ids
    ]
    write_table(rows, CHUNK_SCHEMA, chunks_index_path(bag_id))


def write_world(
    bag_id: str,
    chunk_id: str,
    sweep_id: int,
    xyz: np.ndarray,
    origin: np.ndarray,
    ground_mask: np.ndarray | None = None,
) -> str:
    uri = lidar_world_path(bag_id, chunk_id, sweep_id)
    path = local_path(uri)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    kw = dict(x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2], origin=np.asarray(origin, float))
    kw["ground_mask"] = (
        ground_mask if ground_mask is not None else np.zeros(xyz.shape[0], dtype=bool)
    )
    np.savez_compressed(path, **kw)
    return uri


def meta_row(
    bag_id: str,
    chunk_id: str,
    sweep_id: int,
    world_uri: str,
    n: int,
    *,
    lidar_id: str = "lidar_top",
    t_ns: int | None = None,
    frame_id: int | None = None,
    valid: bool = True,
) -> dict:
    return ProcessedSweepMeta(
        bag_id=bag_id,
        chunk_id=chunk_id,
        sweep_id=sweep_id,
        lidar_id=lidar_id,
        reference_timestamp_ns=sweep_id * SWEEP_DT_NS if t_ns is None else t_ns,
        n_points_total=n,
        n_points_static=0,
        n_points_dynamic=0,
        world_path=world_uri,
        dynamic_mask_path=dynamic_mask_path(bag_id, chunk_id, sweep_id),
        has_intensity=False,
        deskewed=True,
        valid=valid,
        frame_id=sweep_id if frame_id is None else frame_id,
    ).model_dump()


def write_index(bag_id: str, chunk_id: str, rows: list[dict]) -> None:
    write_table(rows, PROCESSED_SWEEPS_SCHEMA, lidar_proc_index_path(bag_id, chunk_id))


def write_dynamic_mask(bag_id, chunk_id, sweep_id, mask: np.ndarray) -> None:
    ensure_local_dir(lidar_proc_dir(bag_id, chunk_id))
    np.save(local_path(dynamic_mask_path(bag_id, chunk_id, sweep_id)), mask)


def write_static_map(
    bag_id: str,
    chunk_id: str,
    *,
    origin: np.ndarray,
    voxel: float,
    static_xyz: np.ndarray | None = None,
    dynamic_xyz: np.ndarray | None = None,
    ambiguous_xyz: np.ndarray | None = None,
) -> None:
    def keys(x):
        if x is None or x.shape[0] == 0:
            return np.empty(0, dtype=np.int64)
        return np.unique(voxel_indices(x, origin, voxel))

    ensure_local_dir(lidar_proc_dir(bag_id, chunk_id))
    np.savez_compressed(
        local_path(static_map_path(bag_id, chunk_id)),
        xyz=static_xyz if static_xyz is not None else np.empty((0, 3)),
        voxel_size=np.float32(voxel),
        origin=np.asarray(origin, dtype=np.float64),
        static_voxel_keys=keys(static_xyz),
        dynamic_voxel_keys=keys(dynamic_xyz),
        ambiguous_voxel_keys=keys(ambiguous_xyz),
    )


def write_flat_ground(bag_id: str, chunk_id: str, z: float = 0.0) -> None:
    cell = 0.5
    n = int(200 / cell)
    ensure_local_dir(lidar_proc_dir(bag_id, chunk_id))
    np.savez_compressed(
        local_path(ground_path(bag_id, chunk_id)),
        height_grid=np.full((n, n), z, dtype=np.float32),
        grid_origin=np.array([-100.0, -100.0]),
        cell_size=np.float32(cell),
        ground_xyz=np.empty((0, 3)),
        status=np.array("ok"),
    )


def write_summary(bag_id: str, chunk_id: str, segmentation: str = "aw") -> None:
    row = ChunkSummaryRow(
        bag_id=bag_id,
        chunk_id=chunk_id,
        n_sweeps_total=0,
        n_sweeps_valid=0,
        n_sweeps_invalid=0,
        n_points_total=0,
        n_points_static=0,
        n_points_dynamic=0,
        n_points_ground=0,
        n_dropped_dynamic_ground=0,
        cache_auto_disabled=False,
        estimated_cache_bytes=0,
        ground_status="ok",
        segmentation_method=segmentation,
    )
    write_table(
        [row.model_dump()],
        CHUNK_SUMMARY_SCHEMA,
        lidar_proc_summary_path(bag_id, chunk_id),
    )


def write_global_static_map(bag_id: str, xyz: np.ndarray) -> None:
    path = local_path(global_static_map_path(bag_id))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, xyz=np.asarray(xyz, dtype=np.float64))

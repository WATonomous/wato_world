"""Shared data adapters for lidar_preprocessing visualization backends."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class ChunkVizData:
    bag_id: str
    chunk_id: str
    static_xyz: np.ndarray
    dynamic_xyz: np.ndarray
    dynamic_sweep_id: np.ndarray
    static_intensity: np.ndarray | None = None
    dynamic_intensity: np.ndarray | None = None
    dynamic_p_occ: np.ndarray | None = None
    dynamic_n_obs: np.ndarray | None = None
    dynamic_n_hits: np.ndarray | None = None
    dynamic_classification: np.ndarray | None = None


@dataclass
class SweepVizData:
    bag_id: str
    chunk_id: str
    sweep_id: int
    xyz: np.ndarray
    dynamic: np.ndarray
    intensity: np.ndarray | None = None
    ground_mask: np.ndarray | None = None
    origin: np.ndarray | None = None
    p_occ: np.ndarray | None = None
    n_obs: np.ndarray | None = None
    n_hits: np.ndarray | None = None
    classification: np.ndarray | None = None


_AXIS_BITS = 20
_AXIS_RANGE = 1 << _AXIS_BITS


def _load_optional_voxel_diag(bag_id: str, chunk_id: str) -> dict[str, np.ndarray] | None:
    from wato_lidar_preprocessing.io import load_voxel_diag

    try:
        diag = load_voxel_diag(bag_id, chunk_id)
    except FileNotFoundError:
        return None
    keys = diag["keys"].astype(np.int64)
    order = np.argsort(keys)
    return {
        "keys": keys[order],
        "p_occ": diag["p_occ"][order].astype(np.float32),
        "n_obs": diag["n_obs"][order].astype(np.int32),
        "n_hits": diag["n_hits"][order].astype(np.int32),
        "classification": diag["classification"][order].astype(np.int8),
        "origin": diag["origin"].astype(np.float64),
        "voxel_size": np.asarray(diag["voxel_size"]).astype(np.float64),
    }


def _voxel_diag_per_point(xyz: np.ndarray, diag: dict[str, np.ndarray] | None) -> dict[str, np.ndarray] | None:
    if diag is None or xyz.shape[0] == 0:
        return None

    origin = diag["origin"]
    voxel_size = float(diag["voxel_size"])
    keys_sorted = diag["keys"]
    rel = xyz.astype(np.float64) - origin
    idx = np.floor(rel / voxel_size).astype(np.int64)
    in_range = (
        (idx[:, 0] >= 0)
        & (idx[:, 0] < _AXIS_RANGE)
        & (idx[:, 1] >= 0)
        & (idx[:, 1] < _AXIS_RANGE)
        & (idx[:, 2] >= 0)
        & (idx[:, 2] < _AXIS_RANGE)
    )

    pt_keys = np.zeros(xyz.shape[0], dtype=np.int64)
    pt_keys[in_range] = (
        (idx[in_range, 0] << (2 * _AXIS_BITS))
        | (idx[in_range, 1] << _AXIS_BITS)
        | idx[in_range, 2]
    )

    if keys_sorted.size == 0:
        hit = np.zeros(xyz.shape[0], dtype=bool)
        pos_clip = np.zeros(xyz.shape[0], dtype=np.int64)
    else:
        pos = np.searchsorted(keys_sorted, pt_keys)
        pos_clip = np.clip(pos, 0, keys_sorted.size - 1)
        hit = (keys_sorted[pos_clip] == pt_keys) & in_range

    n = xyz.shape[0]
    p_occ = np.full(n, np.nan, dtype=np.float32)
    n_obs = np.full(n, -1, dtype=np.int32)
    n_hits = np.full(n, -1, dtype=np.int32)
    classification = np.full(n, -1, dtype=np.int8)
    if hit.any():
        h_pos = pos_clip[hit]
        p_occ[hit] = diag["p_occ"][h_pos]
        n_obs[hit] = diag["n_obs"][h_pos]
        n_hits[hit] = diag["n_hits"][h_pos]
        classification[hit] = diag["classification"][h_pos]
    return {
        "p_occ": p_occ,
        "n_obs": n_obs,
        "n_hits": n_hits,
        "classification": classification,
    }


def load_chunk_viz_data(bag_id: str, chunk_id: str) -> ChunkVizData:
    """Load chunk-level static/dynamic maps plus optional voxel diagnostics."""
    from wato_lidar_preprocessing.io import load_dynamic_map, load_static_map

    static_xyz = np.empty((0, 3), dtype=np.float64)
    static_intensity = None
    try:
        static = load_static_map(bag_id, chunk_id)
        static_xyz = static["xyz"]
        static_intensity = static.get("intensity")
    except FileNotFoundError:
        log.warning("static_map.npz missing for chunk %s", chunk_id)

    dynamic_xyz = np.empty((0, 3), dtype=np.float64)
    dynamic_sweep_id = np.empty(0, dtype=np.int32)
    dynamic_intensity = None
    try:
        dynamic = load_dynamic_map(bag_id, chunk_id)
        dynamic_xyz = dynamic["xyz"]
        dynamic_sweep_id = dynamic.get(
            "sweep_id", np.full(dynamic_xyz.shape[0], -1, dtype=np.int32)
        ).astype(np.int32)
        dynamic_intensity = dynamic.get("intensity")
    except FileNotFoundError:
        log.warning("dynamic_map.npz missing for chunk %s", chunk_id)

    diag_stats = _voxel_diag_per_point(
        dynamic_xyz, _load_optional_voxel_diag(bag_id, chunk_id)
    )
    return ChunkVizData(
        bag_id=bag_id,
        chunk_id=chunk_id,
        static_xyz=static_xyz,
        dynamic_xyz=dynamic_xyz,
        dynamic_sweep_id=dynamic_sweep_id,
        static_intensity=static_intensity,
        dynamic_intensity=dynamic_intensity,
        dynamic_p_occ=None if diag_stats is None else diag_stats["p_occ"],
        dynamic_n_obs=None if diag_stats is None else diag_stats["n_obs"],
        dynamic_n_hits=None if diag_stats is None else diag_stats["n_hits"],
        dynamic_classification=None
        if diag_stats is None
        else diag_stats["classification"],
    )


def load_sweep_viz_data(bag_id: str, chunk_id: str, sweep_id: int) -> SweepVizData:
    """Load one processed sweep with static/dynamic mask and optional diagnostics."""
    from wato_lidar_preprocessing.io import load_dynamic_mask, load_world_sweep

    world = load_world_sweep(bag_id, chunk_id, sweep_id)
    xyz = np.column_stack([world["x"], world["y"], world["z"]])
    dynamic = load_dynamic_mask(bag_id, chunk_id, sweep_id).astype(bool)
    diag_stats = _voxel_diag_per_point(xyz, _load_optional_voxel_diag(bag_id, chunk_id))
    return SweepVizData(
        bag_id=bag_id,
        chunk_id=chunk_id,
        sweep_id=sweep_id,
        xyz=xyz,
        dynamic=dynamic,
        intensity=world.get("intensity"),
        ground_mask=world.get("ground_mask"),
        origin=world.get("origin"),
        p_occ=None if diag_stats is None else diag_stats["p_occ"],
        n_obs=None if diag_stats is None else diag_stats["n_obs"],
        n_hits=None if diag_stats is None else diag_stats["n_hits"],
        classification=None if diag_stats is None else diag_stats["classification"],
    )

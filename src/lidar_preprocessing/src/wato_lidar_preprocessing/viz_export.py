"""External export helpers for lidar_preprocessing visualization data."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from wato_common.artifact_store import chunk_root, local_path
from wato_lidar_preprocessing.viz_data import load_chunk_viz_data, load_sweep_viz_data


def _default_export_path(bag_id: str, chunk_id: str, sweep_id: int | None) -> Path:
    stem = f"sweep_{sweep_id:06d}" if sweep_id is not None else "chunk"
    return Path(local_path(chunk_root(bag_id, chunk_id))) / "viz" / f"{stem}.ply"


def _finite_or_default(values: np.ndarray | None, n: int, default, dtype) -> np.ndarray:
    if values is None:
        return np.full(n, default, dtype=dtype)
    arr = np.asarray(values, dtype=dtype)
    if arr.shape[0] != n:
        return np.full(n, default, dtype=dtype)
    if np.issubdtype(arr.dtype, np.floating):
        arr = arr.copy()
        arr[~np.isfinite(arr)] = default
    return arr


def _write_binary_ply(path: Path, table: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = "\n".join(
        [
            "ply",
            "format binary_little_endian 1.0",
            f"element vertex {table.shape[0]}",
            "property double x",
            "property double y",
            "property double z",
            "property uchar dynamic",
            "property int sweep_id",
            "property float intensity",
            "property float p_occ",
            "property int n_obs",
            "property int n_hits",
            "property char classification",
            "end_header",
            "",
        ]
    ).encode("ascii")
    with path.open("wb") as f:
        f.write(header)
        table.tofile(f)


def export_ply(
    bag_id: str,
    chunk_id: str,
    *,
    sweep_id: int | None = None,
    out_path: str | Path | None = None,
) -> Path:
    """Export chunk or sweep visualization data as binary PLY.

    Extra scalar fields are designed for CloudCompare/ParaView-style inspection.
    Missing optional values use -1 so external viewers can color/filter them.
    """
    out = (
        Path(out_path)
        if out_path is not None
        else _default_export_path(bag_id, chunk_id, sweep_id)
    )
    if out.suffix.lower() != ".ply":
        out.mkdir(parents=True, exist_ok=True)
        out = out / (
            f"sweep_{sweep_id:06d}.ply" if sweep_id is not None else "chunk.ply"
        )

    dtype = np.dtype(
        [
            ("x", "<f8"),
            ("y", "<f8"),
            ("z", "<f8"),
            ("dynamic", "u1"),
            ("sweep_id", "<i4"),
            ("intensity", "<f4"),
            ("p_occ", "<f4"),
            ("n_obs", "<i4"),
            ("n_hits", "<i4"),
            ("classification", "i1"),
        ]
    )

    if sweep_id is None:
        data = load_chunk_viz_data(bag_id, chunk_id)
        n_static = data.static_xyz.shape[0]
        n_dynamic = data.dynamic_xyz.shape[0]
        xyz_parts = [a for a in (data.static_xyz, data.dynamic_xyz) if a.shape[0] > 0]
        xyz = np.vstack(xyz_parts) if xyz_parts else np.empty((0, 3), dtype=np.float64)
        table = np.empty(n_static + n_dynamic, dtype=dtype)
        table["dynamic"] = np.concatenate(
            [np.zeros(n_static, dtype=np.uint8), np.ones(n_dynamic, dtype=np.uint8)]
        )
        table["sweep_id"] = np.concatenate(
            [
                np.full(n_static, -1, dtype=np.int32),
                data.dynamic_sweep_id.astype(np.int32),
            ]
        )
        table["intensity"] = np.concatenate(
            [
                _finite_or_default(data.static_intensity, n_static, -1.0, np.float32),
                _finite_or_default(data.dynamic_intensity, n_dynamic, -1.0, np.float32),
            ]
        )
        table["p_occ"] = np.concatenate(
            [
                np.full(n_static, -1.0, dtype=np.float32),
                _finite_or_default(data.dynamic_p_occ, n_dynamic, -1.0, np.float32),
            ]
        )
        table["n_obs"] = np.concatenate(
            [
                np.full(n_static, -1, dtype=np.int32),
                _finite_or_default(data.dynamic_n_obs, n_dynamic, -1, np.int32),
            ]
        )
        table["n_hits"] = np.concatenate(
            [
                np.full(n_static, -1, dtype=np.int32),
                _finite_or_default(data.dynamic_n_hits, n_dynamic, -1, np.int32),
            ]
        )
        table["classification"] = np.concatenate(
            [
                np.zeros(n_static, dtype=np.int8),
                _finite_or_default(data.dynamic_classification, n_dynamic, -1, np.int8),
            ]
        )
    else:
        data = load_sweep_viz_data(bag_id, chunk_id, sweep_id)
        xyz = data.xyz
        n = xyz.shape[0]
        table = np.empty(n, dtype=dtype)
        table["dynamic"] = data.dynamic.astype(np.uint8)
        table["sweep_id"] = np.full(n, sweep_id, dtype=np.int32)
        table["intensity"] = _finite_or_default(data.intensity, n, -1.0, np.float32)
        table["p_occ"] = _finite_or_default(data.p_occ, n, -1.0, np.float32)
        table["n_obs"] = _finite_or_default(data.n_obs, n, -1, np.int32)
        table["n_hits"] = _finite_or_default(data.n_hits, n, -1, np.int32)
        table["classification"] = _finite_or_default(
            data.classification, n, -1, np.int8
        )

    table["x"] = xyz[:, 0] if xyz.shape[0] else np.empty(0, dtype=np.float64)
    table["y"] = xyz[:, 1] if xyz.shape[0] else np.empty(0, dtype=np.float64)
    table["z"] = xyz[:, 2] if xyz.shape[0] else np.empty(0, dtype=np.float64)
    _write_binary_ply(out, table)
    return out

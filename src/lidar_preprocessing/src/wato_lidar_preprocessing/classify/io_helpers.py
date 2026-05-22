"""I/O helpers shared by the log-odds and persistence classifier paths."""

from __future__ import annotations

import logging
import os

import numpy as np

from wato_common.artifact_store import local_path

log = logging.getLogger(__name__)

# Auto-disable the in-memory xyz/intensity cache when the estimated total size
# exceeds this many bytes.  At 4 GB we cap roughly 250 M float64-xyz points
# (~25 sweeps of 10 M each); the explicit cfg flag still acts as a force-on
# override.  Set the env var WATO_LIDAR_CACHE_BYTES to override per-host.
_DEFAULT_CACHE_BYTES = 4 * 1024**3


def cache_byte_budget() -> int:
    """Resolve the cache size cap, honouring WATO_LIDAR_CACHE_BYTES if set.

    Invalid values (non-int, <=0) are ignored with a warning so a typo in
    the env doesn't silently disable the safety net.
    """
    raw = os.environ.get("WATO_LIDAR_CACHE_BYTES")
    if raw is None:
        return _DEFAULT_CACHE_BYTES
    try:
        v = int(raw)
        if v <= 0:
            raise ValueError("must be > 0")
        return v
    except (TypeError, ValueError) as exc:
        log.warning(
            "WATO_LIDAR_CACHE_BYTES=%r ignored (%s); falling back to default %d",
            raw,
            exc,
            _DEFAULT_CACHE_BYTES,
        )
        return _DEFAULT_CACHE_BYTES


def estimate_cache_bytes(meta_rows: list[dict]) -> int:
    """Estimate the in-memory size of (xyz float64 + intensity float32) caches."""
    total_pts = 0
    for r in meta_rows:
        if r.get("valid") is False:
            continue
        total_pts += int(r.get("n_points_total") or 0)
    return total_pts * (3 * 8 + 4)


def load_world_xyz_intensity(
    world_path_uri: str,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Single load of a world NPZ → (xyz, intensity-or-None)."""
    data = np.load(local_path(world_path_uri))
    xyz = np.stack([data["x"], data["y"], data["z"]], axis=1)
    intensity = data["intensity"].astype(np.float32) if "intensity" in data else None
    return xyz, intensity


def load_world_full(
    world_path_uri: str,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """Load world NPZ → (xyz, intensity, sensor_origin, ground_mask).

    Returns None for any optional field not present in the file.  Used by
    the log-odds path so all arrays are loaded in a single np.load call.
    """
    data = np.load(local_path(world_path_uri))
    xyz = np.stack([data["x"], data["y"], data["z"]], axis=1)
    intensity = data["intensity"].astype(np.float32) if "intensity" in data else None
    origin = data["origin"] if "origin" in data else None
    ground_mask = data["ground_mask"] if "ground_mask" in data else None
    return xyz, intensity, origin, ground_mask


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x.astype(np.float64)))


def origin_from_index(meta_rows: list[dict]) -> np.ndarray | None:
    """Compute the chunk-level world-frame origin from the parquet bbox.

    Reads the per-sweep `world_xmin/ymin/zmin` columns populated by deskew.
    Returns None if every valid row is empty (e.g. no points survived the
    nonfinite filter), in which case the caller writes an empty sentinel.
    Raises ValueError if a row is non-empty but missing bbox columns —
    that means the parquet was written by a stale deskew and the caller
    needs to re-run it.
    """
    xmins, ymins, zmins = [], [], []
    for row in meta_rows:
        if row.get("valid") is False:
            continue
        xm = row.get("world_xmin")
        ym = row.get("world_ymin")
        zm = row.get("world_zmin")
        if xm is None or ym is None or zm is None:
            n_pts = int(row.get("n_points_total") or 0)
            if n_pts > 0:
                raise ValueError(
                    f"sweep {row.get('sweep_id')!r} has n_points_total={n_pts} "
                    "but is missing world_x/y/zmin columns — parquet was "
                    "written by a stale deskew; re-run lidar_preprocessing "
                    "with --force on this chunk."
                )
            continue
        xmins.append(xm)
        ymins.append(ym)
        zmins.append(zm)
    if not xmins:
        return None
    return np.array([min(xmins), min(ymins), min(zmins)], dtype=np.float64)

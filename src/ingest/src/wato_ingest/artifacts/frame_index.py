"""Build frame_index.parquet — the synchronization contract for the pipeline.

Inputs (already on disk):
  - lidar_sweeps.parquet       (one row per LiDAR sweep)
  - camera_frames.parquet      (one row per camera image)
  - poses.parquet              (sparse, walked by pose_interpolation)
  - calibration.json           (path embedded into each row)

Output:
  - frame_index.parquet        (one row per (sweep_id, cam_id))

For each LiDAR sweep, find the nearest camera frame per camera. Drop any
camera whose offset to the sweep exceeds `max_cam_offset_ms`. Interpolate
ego pose at the sweep timestamp and embed it.
"""

from __future__ import annotations

from dataclasses import dataclass

from wato_common.artifact_store import (
    calibration_path,
    camera_frames_path,
    frame_index_path,
    lidar_sweeps_path,
)
from wato_common.geometry import flatten_se3
from wato_common.io.parquet_io import read_rows, write_table
from wato_common.schemas import FRAME_INDEX_SCHEMA, FrameIndexRow
from wato_ingest.decoders.pose_interpolation import interpolate_at, load_samples


@dataclass
class FrameIndexResult:
    rows_written: int
    output_uri: str
    valid_camera_count: int
    dropped_camera_count: int


def build(
    bag_id: str,
    chunk_id: str,
    *,
    max_cam_offset_ms: float,
    max_pose_gap_ns: int = 200_000_000,
) -> FrameIndexResult:
    sweeps = read_rows(lidar_sweeps_path(bag_id, chunk_id))
    camera_frames = read_rows(camera_frames_path(bag_id, chunk_id))
    pose_samples = load_samples(bag_id, chunk_id)
    calib_uri = calibration_path(bag_id)

    rows = _build_rows(
        bag_id=bag_id,
        chunk_id=chunk_id,
        sweeps=sweeps,
        camera_frames=camera_frames,
        pose_samples=pose_samples,
        calib_uri=calib_uri,
        max_cam_offset_ms=max_cam_offset_ms,
        max_pose_gap_ns=max_pose_gap_ns,
    )

    out_uri = frame_index_path(bag_id, chunk_id)
    write_table([r.model_dump() for r in rows], FRAME_INDEX_SCHEMA, out_uri)

    valid = sum(1 for r in rows if r.valid_camera)
    dropped = sum(1 for r in rows if not r.valid_camera)
    return FrameIndexResult(
        rows_written=len(rows),
        output_uri=out_uri,
        valid_camera_count=valid,
        dropped_camera_count=dropped,
    )


def _build_rows(
    *,
    bag_id: str,
    chunk_id: str,
    sweeps: list[dict],
    camera_frames: list[dict],
    pose_samples,
    calib_uri: str,
    max_cam_offset_ms: float,
    max_pose_gap_ns: int,
) -> list[FrameIndexRow]:
    """Pure logic exposed for unit testing without artifact files on disk."""
    # Group camera frames by cam_id, sorted by header timestamp.
    by_cam: dict[str, list[dict]] = {}
    for cf in camera_frames:
        if not cf.get("valid", True):
            continue
        by_cam.setdefault(cf["cam_id"], []).append(cf)
    for cam_id in by_cam:
        by_cam[cam_id].sort(key=lambda r: r["header_timestamp_ns"])

    cam_ids = sorted(by_cam.keys())
    rows: list[FrameIndexRow] = []
    threshold_ns = int(max_cam_offset_ms * 1e6)

    for sw in sweeps:
        if not sw.get("valid", True):
            continue
        sweep_ts = int(sw["header_timestamp_ns"])
        sweep_id = int(sw["sweep_id"])
        lidar_id = sw["lidar_id"]
        lidar_path = sw["lidar_path"]

        pose = interpolate_at(pose_samples, sweep_ts, max_gap_ns=max_pose_gap_ns)

        for cam_id in cam_ids:
            nearest, offset_ns = _nearest(by_cam[cam_id], sweep_ts)
            valid_cam = nearest is not None and abs(offset_ns) <= threshold_ns
            drop_reason: str | None
            if nearest is None:
                drop_reason = "no_camera_frames_for_camera"
            elif not valid_cam:
                drop_reason = f"offset_{offset_ns / 1e6:.1f}ms_exceeds_threshold"
            else:
                drop_reason = None

            rows.append(
                FrameIndexRow(
                    frame_id=f"{bag_id}__{chunk_id}__{sweep_id:06d}__{cam_id}",
                    bag_id=bag_id,
                    chunk_id=chunk_id,
                    sweep_id=sweep_id,
                    lidar_id=lidar_id,
                    lidar_path=lidar_path,
                    reference_timestamp_ns=sweep_ts,
                    cam_id=cam_id,
                    image_path=nearest["image_path"] if nearest else None,
                    camera_seq=int(nearest["camera_seq"]) if nearest else None,
                    camera_timestamp_ns=int(nearest["header_timestamp_ns"])
                    if nearest
                    else None,
                    camera_offset_ms=float(offset_ns / 1e6) if nearest else None,
                    valid_camera=valid_cam,
                    camera_drop_reason=drop_reason,
                    pose_timestamp_ns=pose.pose_timestamp_ns if pose.valid else None,
                    world_T_ego_flat=flatten_se3(pose.world_T_ego)
                    if pose.valid
                    else None,
                    pose_interp_error=pose.interp_error_ns if pose.valid else None,
                    valid_pose=pose.valid,
                    calibration_path=calib_uri,
                )
            )

    return rows


def _nearest(rows: list[dict], target_ts_ns: int) -> tuple[dict | None, int]:
    """Linear scan; rows are short (per-camera-per-chunk = ~30 s × 30 Hz = 900)."""
    if not rows:
        return None, 0
    best = rows[0]
    best_offset = best["header_timestamp_ns"] - target_ts_ns
    for r in rows[1:]:
        off = r["header_timestamp_ns"] - target_ts_ns
        if abs(off) < abs(best_offset):
            best, best_offset = r, off
    return best, int(best_offset)

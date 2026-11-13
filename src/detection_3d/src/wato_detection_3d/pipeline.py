"""detection_3d pipeline orchestrator.

Per chunk, per frame (lidar_preprocessing's frame_sync frames, regrouped on
the configured canonical LiDAR), at the frame's reference time:
  1. The model's input frame in the world (geometry.ego_T_model at the
     reference pose).
  2. Points: the frame's sweeps of the configured LiDARs plus their earlier
     sweeps, each tagged with its age (inputs.pick_sweeps / assemble_points).
  3. Images: each configured camera's image nearest the reference time
     (camera_sync), with its pose at its own time (inputs.camera_inputs).
  4. BEVFusion (model.Bevfusion); every box it returns, converted to the world
     (geometry.to_world) and kept raw beside.
Then the frame rows and every box go to detections_3d (wato_common.artifacts).

A frame without a reference pose, a camera image or any points is recorded as
skipped, with the reason, rather than run on partial inputs.
"""

from __future__ import annotations

import bisect
import json
import logging
import os
from collections import Counter, OrderedDict
from typing import Callable, Optional

import numpy as np

from wato_common import manifest as common_manifest
from wato_common.artifact_store import (
    calibration_path,
    chunk_root,
    chunks_index_path,
    detections_3d_frames_path,
    detections_3d_path,
    frame_index_path,
    lidar_proc_dir,
    lidar_proc_index_path,
    local_path,
    poses_path,
)
from wato_common.artifacts import detections_3d
from wato_common.artifacts.calibration import Calibration
from wato_common.artifacts import calibration_refinement as camera_calibration
from wato_common.artifacts.detections_3d import Detection3D, DetectorFrame
from wato_common.artifacts.frame_index import CameraFrame, FrameIndex
from wato_common.artifacts.lidar_sweeps import Frame, ProcessedSweeps, WorldSweep
from wato_common.io.parquet_io import read_rows
from wato_common.pose_lookup import PoseLookup
from wato_detection_3d import geometry, inputs, model
from wato_detection_3d.config import ComponentConfig

log = logging.getLogger(__name__)

# World sweeps kept in memory: a frame reuses its predecessors' earlier sweeps.
_SWEEP_CACHE = 32


class _Sweeps:
    """World sweeps by id, the most recent few kept loaded."""

    def __init__(self, sweeps: ProcessedSweeps):
        self._sweeps = sweeps
        self._cache: OrderedDict[int, WorldSweep] = OrderedDict()

    def get(self, sweep_id: int) -> WorldSweep:
        if sweep_id in self._cache:
            self._cache.move_to_end(sweep_id)
            return self._cache[sweep_id]
        sweep = self._sweeps.world(sweep_id)
        self._cache[sweep_id] = sweep
        if len(self._cache) > _SWEEP_CACHE:
            self._cache.popitem(last=False)
        return sweep


class _Images:
    """Each camera's images with a pose, nearest-in-time lookup."""

    def __init__(self, index: FrameIndex):
        self._by_cam: dict[str, list[CameraFrame]] = {}
        self._times: dict[str, list[int]] = {}
        for cam, frames in index.by_camera().items():
            posed = sorted(
                (f for f in frames if f.world_T_ego is not None),
                key=lambda f: f.timestamp_ns,
            )
            self._by_cam[cam] = posed
            self._times[cam] = [f.timestamp_ns for f in posed]

    def nearest(self, cam: str, t_ns: int, max_offset_ns: int) -> Optional[CameraFrame]:
        times = self._times.get(cam, [])
        k = bisect.bisect_left(times, t_ns)
        best = None
        for i in (k - 1, k):
            if 0 <= i < len(times) and abs(times[i] - t_ns) <= max_offset_ns:
                if best is None or abs(times[i] - t_ns) < abs(best.timestamp_ns - t_ns):
                    best = self._by_cam[cam][i]
        return best


def _skipped(frame: Frame, status: str, detail: str) -> DetectorFrame:
    return DetectorFrame(
        sweep_id=frame.reference_sweep_id,
        frame_id=frame.frame_id,
        reference_timestamp_ns=frame.reference_timestamp_ns,
        status=status,
        detail=detail,
    )


def run_frame(
    cfg: ComponentConfig,
    frame: Frame,
    *,
    detector: model.Detector,
    calib: Calibration,
    poses: PoseLookup,
    images: "_Images",
    sweeps: ProcessedSweeps,
    world: _Sweeps,
    history: dict[str, list[tuple[int, int]]],
) -> tuple[DetectorFrame, list[Detection3D]]:
    t_ref = frame.reference_timestamp_ns
    ref_meta = sweeps.get(frame.reference_sweep_id)
    # The model's frame hangs off the canonical LiDAR. A frame anchored on
    # another one (a sweep frame_sync matched to no canonical sweep) would
    # feed only that LiDAR's points, unlike any frame the model was run on.
    anchor = cfg.frames.canonical_lidar or ref_meta.lidar_id
    if ref_meta.lidar_id != anchor:
        return _skipped(
            frame, detections_3d.NO_POINTS, f"no {anchor} sweep in the frame"
        ), []
    ref_pose = poses.at(t_ref)
    if not ref_pose.valid:
        return _skipped(frame, detections_3d.NO_POSE, ref_pose.drop_reason or ""), []
    ego_T_model = geometry.ego_T_model(
        cfg.input_frame.kind, calib.ego_T_lidar(anchor), cfg.input_frame.height_m
    )
    world_T_model = ref_pose.world_T_ego @ ego_T_model
    model_T_world = np.linalg.inv(world_T_model)

    max_offset_ns = int(cfg.camera_sync.max_offset_ms * 1e6)
    by_cam = {c: images.nearest(c, t_ref, max_offset_ns) for c in cfg.cameras}
    missing = [c for c, f in by_cam.items() if f is None]
    if missing:
        return _skipped(frame, detections_3d.MISSING_CAMERA, ",".join(missing)), []
    cameras = inputs.camera_inputs(
        [by_cam[c] for c in cfg.cameras], calib, model_T_world, cfg
    )

    lidars = cfg.points.lidars or [anchor]
    current = {}
    for sid in frame.sweep_ids:
        meta = sweeps.get(sid)
        if meta is not None and meta.lidar_id in lidars:
            current[meta.lidar_id] = (sid, meta.reference_timestamp_ns)
    feed = []
    for lidar_id, sid, ts, is_current in inputs.pick_sweeps(
        t_ref, current, history, cfg.points
    ):
        sensor_T_world = None
        if not is_current:
            pose = poses.at(ts)
            if not pose.valid:
                continue  # an earlier sweep without a pose is left out
            sensor_T_world = np.linalg.inv(
                pose.world_T_ego @ calib.ego_T_lidar(lidar_id)
            )
        feed.append(inputs.SweepPoints(sid, ts, world.get(sid), sensor_T_world))
    points, used = inputs.assemble_points(feed, model_T_world, t_ref, cfg.points)
    if frame.reference_sweep_id not in used:
        return _skipped(
            frame, detections_3d.NO_POINTS, "reference sweep has no points in range"
        ), []

    boxes = detector.infer(points, cameras)
    dets = []
    for k, b in enumerate(boxes):
        wb = geometry.to_world(
            world_T_model,
            *(
                float(b[i])
                for i in (model.X, model.Y, model.Z, model.W, model.L, model.H)
            ),
            float(b[model.ROT]),
            float(b[model.VX]),
            float(b[model.VY]),
        )
        class_id = int(round(float(b[model.CLASS_ID])))
        dets.append(
            Detection3D(
                detection_id=f"det-{frame.reference_sweep_id:06d}-{k}",
                sweep_id=frame.reference_sweep_id,
                frame_id=frame.frame_id,
                reference_timestamp_ns=t_ref,
                cx=wb.cx,
                cy=wb.cy,
                cz=wb.cz,
                w=wb.w,
                l=wb.l,
                h=wb.h,
                heading=wb.heading,
                vx=wb.vx,
                vy=wb.vy,
                class_id=class_id,
                cls=detections_3d.CLASSES[class_id],
                score=float(b[model.SCORE]),
                model_x=float(b[model.X]),
                model_y=float(b[model.Y]),
                model_z=float(b[model.Z]),
                model_w=float(b[model.W]),
                model_l=float(b[model.L]),
                model_h=float(b[model.H]),
                model_rot=float(b[model.ROT]),
                model_vx=float(b[model.VX]),
                model_vy=float(b[model.VY]),
            )
        )
    row = DetectorFrame(
        sweep_id=frame.reference_sweep_id,
        frame_id=frame.frame_id,
        reference_timestamp_ns=t_ref,
        status=detections_3d.OK,
        input_sweep_ids=used,
        n_points=int(points.shape[0]),
        camera_timestamps_ns=cameras.timestamps_ns,
        world_T_model=world_T_model,
    )
    return row, dets


def process_chunk(
    cfg: ComponentConfig, bag_id: str, chunk_id: str, detector: model.Detector
) -> Counter:
    """Write the chunk's detections; returns frames per status."""
    for name, uri in {
        "lidar_proc_index.parquet (run lidar_preprocessing)": lidar_proc_index_path(
            bag_id, chunk_id
        ),
        "frame_index.parquet (run ingest)": frame_index_path(bag_id, chunk_id),
    }.items():
        if not os.path.exists(local_path(uri)):
            raise FileNotFoundError(
                f"chunk {chunk_id!r} of bag {bag_id!r} is missing {name}"
            )
    calib = camera_calibration.resolve(
        bag_id, cfg.calibration_source, cfg.cameras
    ).calibration
    sweeps = ProcessedSweeps.load(bag_id, chunk_id)
    frames = sweeps.frames(cfg.frames.canonical_lidar)
    poses = PoseLookup.load(bag_id, chunk_id)
    images = _Images(FrameIndex.load(bag_id, chunk_id))
    history: dict[str, list[tuple[int, int]]] = {}
    for r in sweeps.rows:
        history.setdefault(r.lidar_id, []).append(
            (r.reference_timestamp_ns, r.sweep_id)
        )
    for h in history.values():
        h.sort()
    world = _Sweeps(sweeps)

    rows: list[DetectorFrame] = []
    dets: list[Detection3D] = []
    for frame in frames:
        row, frame_dets = run_frame(
            cfg,
            frame,
            detector=detector,
            calib=calib,
            poses=poses,
            images=images,
            sweeps=sweeps,
            world=world,
            history=history,
        )
        rows.append(row)
        dets += frame_dets
    detections_3d.write(bag_id, chunk_id, rows, dets)
    status = Counter(r.status for r in rows)
    log.info(
        "chunk %s: %d frames %s, %d boxes", chunk_id, len(rows), dict(status), len(dets)
    )
    return status


def _write_chunk_manifest(
    bag_id: str,
    chunk_id: str,
    config_path: Optional[str],
    engine_info: dict,
    cfg: ComponentConfig,
) -> None:
    """Record the config, model and inputs behind this chunk's detections.
    Best-effort: never fails a chunk that succeeded. The per-sweep world files
    are too many to hash; lidar_preprocessing's manifest stands in for them."""
    root = chunk_root(bag_id, chunk_id)
    try:
        common_manifest.write(
            component="detection_3d",
            bag_id=bag_id,
            chunk_id=chunk_id,
            inputs={
                "calibration": calibration_path(bag_id),
                "frame_index": frame_index_path(bag_id, chunk_id),
                "poses": poses_path(bag_id, chunk_id),
                "lidar_proc_index": lidar_proc_index_path(bag_id, chunk_id),
                "manifest_lidar_preprocessing": (
                    f"{root}/{common_manifest.component_manifest_name('lidar_preprocessing')}"
                ),
            },
            outputs={
                "detections_3d": detections_3d_path(bag_id, chunk_id),
                "detections_3d_frames": detections_3d_frames_path(bag_id, chunk_id),
            },
            config_path=config_path,
            models={
                "bevfusion": f"{engine_info['model_dir']}@sha256:"
                + ",".join(
                    f"{k}={v}"
                    for k, v in sorted(engine_info["model_files_sha256"].items())
                )
            },
            extra={
                "engines": engine_info,
                "inference_config": cfg.model_dump(),
                "unhashed_inputs": {
                    "world_sweeps": lidar_proc_dir(bag_id, chunk_id),
                    "camera_images": f"{root}/cam_*",
                },
            },
            filename=common_manifest.component_manifest_name("detection_3d"),
        )
    except Exception:  # noqa: BLE001
        log.warning("failed to write manifest for chunk %s", chunk_id, exc_info=True)


def run(
    cfg: ComponentConfig,
    *,
    bag_id: str,
    chunk_id: Optional[str] = None,
    force: bool = False,
    config_path: Optional[str] = None,
    make_detector: Optional[Callable[[], tuple[model.Detector, dict]]] = None,
) -> None:
    """Process one bag (or one chunk). Skips chunks whose detections exist
    unless force=True.

    make_detector: returns the detector and the info its manifest records;
    defaults to BEVFusion on engines built from cfg.model (tests pass a fake).
    """
    chunks = read_rows(chunks_index_path(bag_id))
    if not chunks:
        raise FileNotFoundError(
            f"No chunks found for bag {bag_id!r} — run ingest first."
        )
    if chunk_id:
        chunks = [c for c in chunks if c["chunk_id"] == chunk_id]
        if not chunks:
            raise ValueError(f"chunk_id {chunk_id!r} not found for bag {bag_id!r}")
    if not force:
        for c in chunks:
            cid = c["chunk_id"]
            if not detections_3d.exists(bag_id, cid):
                continue
            path = local_path(chunk_root(bag_id, cid)) + "/manifest_detection_3d.json"
            saved = {}
            if os.path.exists(path):
                with open(path, encoding="utf-8") as fh:
                    saved = json.load(fh).get("inference_config", {})
            if saved != cfg.model_dump():
                raise ValueError(
                    f"chunk {cid}: existing detection_3d inputs/model differ or "
                    "are unrecorded; rerun detection_3d and downstream stages "
                    "with --force (sensor changes require --force ingest)."
                )
    selection = camera_calibration.resolve(bag_id, cfg.calibration_source, cfg.cameras)
    for c in chunks:
        camera_calibration.bind_outputs(
            bag_id,
            c["chunk_id"],
            "detection_3d",
            selection,
            force=force,
            existing_outputs=detections_3d.exists(bag_id, c["chunk_id"]),
        )
    todo = [
        c["chunk_id"]
        for c in chunks
        if force or not detections_3d.exists(bag_id, c["chunk_id"])
    ]
    for c in chunks:
        if c["chunk_id"] not in todo:
            log.info(
                "chunk %s already complete — skipping (use --force)", c["chunk_id"]
            )
    if not todo:
        return
    if make_detector is None:
        from wato_detection_3d import engines

        def make_detector():
            prepared = engines.prepare(cfg.model)
            return model.Bevfusion(
                prepared, cfg.model.precision, cfg.model.num_cameras
            ), prepared.info

    detector, info = make_detector()
    for cid in todo:
        process_chunk(cfg, bag_id, cid, detector)
        _write_chunk_manifest(bag_id, cid, config_path, info, cfg)
    log.info(
        "detection_3d complete for bag %s: %d processed, %d skipped",
        bag_id,
        len(todo),
        len(chunks) - len(todo),
    )

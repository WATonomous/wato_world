"""detection_3d output: a learned 3D detector's boxes, per frame.

Written by detection_3d (BEVFusion, the model the car runs); read by
proposal_generation, which turns the boxes it keeps into `fusion_detector`
proposals, and by evaluation.

On disk, per chunk:

  detections_3d_frames.parquet
      one Detection3DFrameRow per frame detection_3d considered
      (schemas.DETECTION_3D_FRAME_SCHEMA): status "ok" when the model ran,
      else why the frame was skipped. input_sweep_ids and
      camera_timestamps_ns are JSON list strings, world_T_model a JSON list of
      16 floats; tuples and a (4, 4) array once read.
  detections_3d.parquet
      one Detection3DRow per box (schemas.DETECTION_3D_SCHEMA): every box the
      model returned for an "ok" frame, whatever its score or class.

Written frames-file first, so detections_3d.parquet marks a complete chunk.

Invariants, checked on write and on read: a sweep_id has at most one frame
row; an "ok" frame has a 4x4 world_T_model and lists its reference sweep
among input_sweep_ids; detection_ids are unique; every detection belongs to
an "ok" frame (same sweep_id, frame_id and time); extents are positive; cls
is CLASSES[class_id].
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, fields
from typing import Optional

import numpy as np

from wato_common.artifact_store import (
    detections_3d_frames_path,
    detections_3d_path,
    local_path,
)
from wato_common.artifacts import ArtifactFormatError
from wato_common.io.parquet_io import read_rows, write_table
from wato_common.schemas import (
    DETECTION_3D_FRAME_SCHEMA,
    DETECTION_3D_SCHEMA,
    Detection3DFrameRow,
    Detection3DRow,
    decode_int_list,
    encode_int_list,
)

# The model's classes by class_id: the nuScenes detection classes in the order
# BEVFusion's head emits them (wato_monorepo bevfusion_constants.hpp).
CLASSES = (
    "car",
    "truck",
    "construction_vehicle",
    "bus",
    "trailer",
    "barrier",
    "motorcycle",
    "bicycle",
    "pedestrian",
    "traffic_cone",
)

OK = "ok"
NO_POSE = "no_pose"
MISSING_CAMERA = "missing_camera"
NO_POINTS = "no_points"
STATUSES = (OK, NO_POSE, MISSING_CAMERA, NO_POINTS)


@dataclass(frozen=True)
class DetectorFrame:
    """A Detection3DFrameRow without bag/chunk ids, list columns decoded."""

    sweep_id: int
    frame_id: Optional[int]
    reference_timestamp_ns: int
    status: str
    detail: Optional[str] = None
    input_sweep_ids: tuple[int, ...] = ()
    n_points: int = 0
    camera_timestamps_ns: tuple[int, ...] = ()
    world_T_model: Optional[np.ndarray] = None  # (4, 4) float64 when status is "ok"

    @property
    def ok(self) -> bool:
        return self.status == OK


@dataclass(frozen=True)
class Detection3D:
    """A Detection3DRow without bag/chunk ids. Field meanings:
    schemas.Detection3DRow."""

    detection_id: str
    sweep_id: int
    frame_id: Optional[int]
    reference_timestamp_ns: int
    cx: float
    cy: float
    cz: float
    w: float
    l: float  # noqa: E741 — domain term: length, matches parquet w/l/h column triplet
    h: float
    heading: float
    vx: float
    vy: float
    class_id: int
    cls: str
    score: float
    model_x: float
    model_y: float
    model_z: float
    model_w: float
    model_l: float
    model_h: float
    model_rot: float
    model_vx: float
    model_vy: float


def _problem(frames: list[DetectorFrame], dets: list[Detection3D]) -> Optional[str]:
    """The first broken invariant (module docstring), or None."""
    by_sweep: dict[int, DetectorFrame] = {}
    for f in frames:
        where = f"frame {f.sweep_id}"
        if f.sweep_id in by_sweep:
            return f"{where} appears twice"
        by_sweep[f.sweep_id] = f
        if f.status not in STATUSES:
            return f"{where}: status {f.status!r} not in {STATUSES}"
        if f.ok:
            if f.world_T_model is None or np.shape(f.world_T_model) != (4, 4):
                return f"{where}: an ok frame needs a 4x4 world_T_model"
            if f.sweep_id not in f.input_sweep_ids:
                return f"{where}: reference sweep not among input_sweep_ids"
    seen: set[str] = set()
    for d in dets:
        where = f"detection {d.detection_id!r}"
        if d.detection_id in seen:
            return f"{where} appears twice"
        seen.add(d.detection_id)
        frame = by_sweep.get(d.sweep_id)
        if frame is None or not frame.ok:
            return f"{where}: sweep {d.sweep_id} is not an ok frame"
        if (d.frame_id, d.reference_timestamp_ns) != (
            frame.frame_id,
            frame.reference_timestamp_ns,
        ):
            return f"{where}: frame_id / time disagree with its frame row"
        if min(d.w, d.l, d.h) <= 0.0:
            return f"{where}: non-positive extent ({d.w}, {d.l}, {d.h})"
        if not 0 <= d.class_id < len(CLASSES) or d.cls != CLASSES[d.class_id]:
            return f"{where}: class {d.class_id} / {d.cls!r} not in CLASSES"
    return None


def _encode_matrix(m: Optional[np.ndarray]) -> str:
    if m is None:
        return "[]"
    return json.dumps([float(v) for v in np.asarray(m, dtype=np.float64).ravel()])


def _decode_matrix(s: str, path: str) -> Optional[np.ndarray]:
    values = json.loads(s) if s else []
    if not values:
        return None
    if len(values) != 16:
        raise ArtifactFormatError(path, f"world_T_model has {len(values)} values")
    return np.asarray(values, dtype=np.float64).reshape(4, 4)


def write(
    bag_id: str,
    chunk_id: str,
    frames: list[DetectorFrame],
    detections: list[Detection3D],
) -> None:
    """Write the chunk's frames, then its detections (the completion marker)."""
    frames_uri = detections_3d_frames_path(bag_id, chunk_id)
    problem = _problem(frames, detections)
    if problem is not None:
        raise ValueError(f"{local_path(frames_uri)}: {problem}")
    frame_rows = []
    for f in frames:
        row = asdict(f)
        row["input_sweep_ids"] = encode_int_list(list(f.input_sweep_ids))
        row["camera_timestamps_ns"] = encode_int_list(list(f.camera_timestamps_ns))
        row["world_T_model"] = _encode_matrix(f.world_T_model)
        frame_rows.append(
            Detection3DFrameRow(bag_id=bag_id, chunk_id=chunk_id, **row).model_dump()
        )
    write_table(frame_rows, DETECTION_3D_FRAME_SCHEMA, frames_uri)
    rows = [
        Detection3DRow(bag_id=bag_id, chunk_id=chunk_id, **asdict(d)).model_dump()
        for d in detections
    ]
    write_table(rows, DETECTION_3D_SCHEMA, detections_3d_path(bag_id, chunk_id))


def read(bag_id: str, chunk_id: str) -> tuple[list[DetectorFrame], list[Detection3D]]:
    """The chunk's frame rows and every detection, in file order."""
    frames_uri = detections_3d_frames_path(bag_id, chunk_id)
    path = local_path(frames_uri)
    frames = []
    for row in read_rows(frames_uri):
        try:
            frames.append(
                DetectorFrame(
                    sweep_id=row["sweep_id"],
                    frame_id=row["frame_id"],
                    reference_timestamp_ns=row["reference_timestamp_ns"],
                    status=row["status"],
                    detail=row["detail"],
                    input_sweep_ids=tuple(decode_int_list(row["input_sweep_ids"])),
                    n_points=row["n_points"],
                    camera_timestamps_ns=tuple(
                        decode_int_list(row["camera_timestamps_ns"])
                    ),
                    world_T_model=_decode_matrix(row["world_T_model"], path),
                )
            )
        except KeyError as exc:
            raise ArtifactFormatError(path, f"missing column {exc.args[0]!r}") from None
    dets_uri = detections_3d_path(bag_id, chunk_id)
    dets = []
    for row in read_rows(dets_uri):
        try:
            dets.append(
                Detection3D(**{f.name: row[f.name] for f in fields(Detection3D)})
            )
        except KeyError as exc:
            raise ArtifactFormatError(
                local_path(dets_uri), f"missing column {exc.args[0]!r}"
            ) from None
    problem = _problem(frames, dets)
    if problem is not None:
        raise ArtifactFormatError(local_path(dets_uri), problem)
    return frames, dets


def exists(bag_id: str, chunk_id: str) -> bool:
    return os.path.exists(local_path(detections_3d_path(bag_id, chunk_id)))

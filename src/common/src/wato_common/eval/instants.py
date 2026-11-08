"""The instants ground truth labels, each tied to the chunk's frame there.

Every evaluation script scores at these instants: the frame nearest each
ground-truth timestamp (within max_offset_ns), the ego position then (for range
bands), and the ground-truth boxes as Scored. Neighbouring chunks share 6 s,
so a script passes each chunk its owned window (wato_common.chunk_spans) and
scores every instant once, in the chunk that labels it. A box with no non-ground LiDAR
point in the frame becomes ignore-GT (cls None), as the nuScenes benchmark
drops such boxes; so does one outside the taxonomy.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

import numpy as np

from wato_common.artifacts import gt_boxes
from wato_common.artifacts.gt_boxes import GtBox
from wato_common.artifacts.lidar_sweeps import Frame, ProcessedSweeps
from wato_common.eval.boxes import Scored
from wato_common.pose_lookup import PoseLookup
from wato_common.schemas import Box3D
from wato_common.tracking import points_in_box


@dataclass(frozen=True)
class LabelledInstant:
    frame: Frame
    ego_xy: np.ndarray  # (2,)
    gts: tuple[Scored, ...]  # cls None = ignore
    boxes: tuple[GtBox, ...]  # the ground truth behind gts, same order

    def range_of(self, cx: float, cy: float) -> float:
        return math.hypot(cx - self.ego_xy[0], cy - self.ego_xy[1])


def gt_by_time(bag_id: str) -> dict[int, list[GtBox]]:
    out: dict[int, list[GtBox]] = defaultdict(list)
    if gt_boxes.exists(bag_id):
        for b in gt_boxes.read(bag_id):
            out[b.timestamp_ns].append(b)
    return out


def box_of(b) -> Box3D:
    return Box3D(cx=b.cx, cy=b.cy, cz=b.cz, w=b.w, l=b.l, h=b.h, heading=b.heading)


def _frame_points(sweeps: ProcessedSweeps, frame: Frame) -> np.ndarray:
    parts = []
    for sid in frame.sweep_ids:
        w = sweeps.world(sid)
        keep = ~w.ground_mask if w.ground_mask is not None else slice(None)
        parts.append(w.xyz[keep])
    return np.concatenate(parts) if parts else np.empty((0, 3))


def labelled_instants(
    bag_id: str,
    chunk_id: str,
    canonical_lidar: Optional[str],
    gt: dict[int, list[GtBox]],
    max_offset_ns: int,
    window: tuple[float, float] = (-math.inf, math.inf),
) -> list[LabelledInstant]:
    """The chunk's labelled instants in window ([lo, hi) on the frame's
    time), in time order. An instant whose frame has no valid pose is skipped
    (no range to band it by)."""
    sweeps = ProcessedSweeps.load(bag_id, chunk_id)
    frames = sweeps.frames(canonical_lidar)
    if not frames or not gt:
        return []
    poses = PoseLookup.load(bag_id, chunk_id)
    f_t = np.array([f.reference_timestamp_ns for f in frames], dtype=np.int64)
    out = []
    for t_gt, boxes in sorted(gt.items()):
        k = int(np.argmin(np.abs(f_t - t_gt)))
        if abs(int(f_t[k]) - t_gt) > max_offset_ns:
            continue
        if not window[0] <= f_t[k] < window[1]:
            continue
        pose = poses.at(frames[k].reference_timestamp_ns)
        if not pose.valid:
            continue
        ego = pose.world_T_ego[:2, 3]
        pts = _frame_points(sweeps, frames[k])
        gts = []
        for b in boxes:
            box = box_of(b)
            seen = pts.shape[0] > 0 and bool(points_in_box(pts, box).any())
            gts.append(
                Scored(
                    box,
                    cls=b.cls if seen else None,
                    range_m=math.hypot(b.cx - ego[0], b.cy - ego[1]),
                )
            )
        out.append(LabelledInstant(frames[k], ego, tuple(gts), tuple(boxes)))
    return out

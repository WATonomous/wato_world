"""Per-camera-frame inputs to lifting, built once and shared across sweeps.

frame_index pairs every camera frame with each sweep it is nearest to (~4.9
sweeps per frame on the 3-LiDAR WATO rig), so a frame's pooled depth and
mask raster are cached instead of being re-read for every sweep.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional

import numpy as np

from wato_common.artifact_store import depth_2d_path, local_path
from wato_common.artifacts import ArtifactFormatError, depth
from wato_common.artifacts.masklets import MaskletIndex
from wato_semantic_lifting.io import CameraFrameRef
from wato_semantic_lifting.projection import label_raster
from wato_semantic_lifting.visibility import pooled_occluder


@dataclass(frozen=True)
class FrameData:
    occluder_m: np.ndarray  # (H, W) float32, visibility.pooled_occluder
    labels: np.ndarray  # (H, W) int32 masklet index (MaskletIndex.records), -1 none

    @property
    def size(self) -> tuple[int, int]:
        """(W, H)."""
        return int(self.labels.shape[1]), int(self.labels.shape[0])


class FrameCache:
    """Least-recently-used cache of FrameData, keyed by (cam_id, camera_seq).

    Process sweeps in time order: a frame is reused only by sweeps close to
    it in time, so a few frames per camera suffice.
    """

    def __init__(
        self,
        bag_id: str,
        chunk_id: str,
        masklets: MaskletIndex,
        *,
        neighborhood: int,
        max_frames: int,
    ):
        self.bag_id = bag_id
        self.chunk_id = chunk_id
        self.masklets = masklets
        self.neighborhood = neighborhood
        self.max_frames = max_frames
        self._label_of = {r.masklet_id: i for i, r in enumerate(masklets.records)}
        self._cache: OrderedDict[tuple[str, int], Optional[FrameData]] = OrderedDict()
        self.n_no_depth = 0  # distinct frames with masklets but no depth artifact

    def get(self, ref: CameraFrameRef) -> Optional[FrameData]:
        """The frame's data; None when it has no masklets (nothing to vote
        for) or no depth artifact (perception_2d's depth fit failed)."""
        key = (ref.cam_id, ref.camera_seq)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        data = self._load(ref)
        self._cache[key] = data
        if len(self._cache) > self.max_frames:
            self._cache.popitem(last=False)
        return data

    def _load(self, ref: CameraFrameRef) -> Optional[FrameData]:
        records = self.masklets.at(ref.cam_id, ref.camera_seq)
        if not records:
            return None
        frame = depth.read_frame(self.bag_id, self.chunk_id, ref.cam_id, ref.camera_seq)
        if frame is None:
            self.n_no_depth += 1
            return None
        masks = [self.masklets.mask(r, ref.camera_seq) for r in records]
        if any(m.shape != frame.depth_m.shape for m in masks):
            raise ArtifactFormatError(
                local_path(
                    depth_2d_path(
                        self.bag_id, self.chunk_id, ref.cam_id, ref.camera_seq
                    )
                ),
                f"depth {frame.depth_m.shape} and mask shapes "
                f"{sorted({m.shape for m in masks})} differ for "
                f"{ref.cam_id} frame {ref.camera_seq}",
            )
        return FrameData(
            occluder_m=pooled_occluder(frame.depth_m, self.neighborhood),
            labels=label_raster(masks, [self._label_of[r.masklet_id] for r in records]),
        )

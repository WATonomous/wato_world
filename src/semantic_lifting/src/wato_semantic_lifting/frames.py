"""Per-camera-frame inputs to lifting, built once and shared across sweeps.

A camera frame is paired with every sweep it is nearest to (~4.9 sweeps per
frame on the 3-LiDAR WATO rig), so pooled depth and independent mask crops are
cached instead of being re-read for every sweep. Each object retains only its
bounding rectangle, avoiding one full-image allocation per cached hypothesis.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional

import numpy as np

from wato_common.artifact_store import depth_2d_path, local_path
from wato_common.artifacts import ArtifactFormatError, depth
from wato_common.artifacts.frame_index import CameraFrame
from wato_common.artifacts.masklets import MaskletIndex
from wato_semantic_lifting.visibility import pooled_occluder


@dataclass(frozen=True)
class MaskClaim:
    """One independent mask, stored in its own image-space bounding rectangle."""

    label: int  # masklet index in MaskletIndex.records
    x0: int
    y0: int
    pixels: np.ndarray  # cropped (h, w) bool, owns its storage

    @classmethod
    def from_mask(cls, label: int, mask: np.ndarray) -> "MaskClaim":
        rows = np.flatnonzero(mask.any(axis=1))
        if not rows.size:
            return cls(label, 0, 0, np.zeros((0, 0), bool))
        cols = np.flatnonzero(mask.any(axis=0))
        x0, y0 = int(cols[0]), int(rows[0])
        # Copy is essential: a slice would retain the dense image's backing
        # storage even when this object's rectangle is only a few pixels.
        pixels = mask[y0 : rows[-1] + 1, x0 : cols[-1] + 1].copy()
        return cls(label, x0, y0, pixels)

    def sample(self, ui: np.ndarray, vi: np.ndarray) -> np.ndarray:
        """Mask membership at image pixels; outside the crop is background."""
        u, v = ui - self.x0, vi - self.y0
        h, w = self.pixels.shape
        inside = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        out = np.zeros(ui.shape, bool)
        out[inside] = self.pixels[v[inside], u[inside]]
        return out


@dataclass(frozen=True)
class FrameData:
    occluder_m: np.ndarray  # (H, W) float32, visibility.pooled_occluder
    masks: tuple[MaskClaim, ...]

    @property
    def size(self) -> tuple[int, int]:
        """(W, H)."""
        return int(self.occluder_m.shape[1]), int(self.occluder_m.shape[0])


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

    def get(self, ref: CameraFrame) -> Optional[FrameData]:
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

    def _load(self, ref: CameraFrame) -> Optional[FrameData]:
        records = self.masklets.at(ref.cam_id, ref.camera_seq)
        if not records:
            return None
        frame = depth.read_frame(self.bag_id, self.chunk_id, ref.cam_id, ref.camera_seq)
        if frame is None:
            self.n_no_depth += 1
            return None
        masks = []
        for record in records:
            # Decode and crop one PNG at a time; neither the cache nor loading
            # this frame needs all hypotheses as full-resolution arrays.
            mask = self.masklets.mask(record, ref.camera_seq)
            if mask.shape != frame.depth_m.shape:
                raise ArtifactFormatError(
                    local_path(
                        depth_2d_path(
                            self.bag_id, self.chunk_id, ref.cam_id, ref.camera_seq
                        )
                    ),
                    f"depth {frame.depth_m.shape} and mask {mask.shape} differ "
                    f"for {ref.cam_id} frame {ref.camera_seq}",
                )
            masks.append(MaskClaim.from_mask(self._label_of[record.masklet_id], mask))
        return FrameData(
            occluder_m=pooled_occluder(frame.depth_m, self.neighborhood),
            masks=tuple(masks),
        )

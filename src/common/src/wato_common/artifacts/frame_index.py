"""ingest's frame_index: which camera image goes with which LiDAR sweep.

Written by ingest; read by perception_2d and semantic_lifting.

On disk, per chunk:

  frame_index.parquet   one FrameIndexRow per (LiDAR sweep, camera)
                        (schemas.FRAME_INDEX_SCHEMA): that camera's image
                        nearest the sweep in time. valid_camera=False (with
                        camera_drop_reason) when it is farther than ingest's
                        max_cam_offset_ms; valid_pose / pose_drop_reason say
                        whether the SWEEP's time has a trusted pose.

That grain is ingest's (quality.json counts drops per row), not a consumer's:
an image repeats once per sweep it is nearest to, ~4.9x on the 3-LiDAR WATO
rig, and feeding those rows to per-image work got perception_2d SIGKILLed. The
read half (FrameIndex) hands out each image once, with every sweep it is paired
with and the ego pose at the image's own time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np

from wato_common.artifact_store import frame_index_path, local_path
from wato_common.artifacts import ArtifactFormatError
from wato_common.io.parquet_io import read_rows, write_table
from wato_common.pose_lookup import PoseLookup
from wato_common.schemas import FRAME_INDEX_SCHEMA, FrameIndexRow

# ---------------------------------------------------------------------------
# Write half (ingest).
# ---------------------------------------------------------------------------


def write(bag_id: str, chunk_id: str, rows: list[FrameIndexRow]) -> str:
    """Write the chunk's frame_index.parquet; returns its URI."""
    uri = frame_index_path(bag_id, chunk_id)
    write_table([r.model_dump() for r in rows], FRAME_INDEX_SCHEMA, uri)
    return uri


# ---------------------------------------------------------------------------
# Read half.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CameraFrame:
    """One camera image, once however many sweeps it is paired with."""

    cam_id: str
    camera_seq: int
    timestamp_ns: int  # when the image was taken
    image_path: str  # local path
    # Every sweep this image is the nearest frame to (within ingest's
    # max_cam_offset_ms), ascending; on a multi-LiDAR rig they span scanners.
    sweep_ids: tuple[int, ...]
    # (4, 4) ego pose at timestamp_ns, not at any sweep's time: the car moves
    # between the two (up to ~0.9 m at 30 m on WATO). None when timestamp_ns
    # falls in an untrusted stretch of the pose stream (PoseLookup.at).
    world_T_ego: Optional[np.ndarray]


_REQUIRED = ("camera_seq", "camera_timestamp_ns", "image_path")


class FrameIndex:
    """A chunk's camera images, each once, and the sweeps each is paired with."""

    def __init__(self, frames: Iterable[CameraFrame]):
        self.frames = sorted(frames, key=lambda f: (f.cam_id, f.camera_seq))
        self._by_sweep: dict[int, list[CameraFrame]] = {}
        for f in self.frames:
            for sid in f.sweep_ids:
                self._by_sweep.setdefault(sid, []).append(f)

    @classmethod
    def load(cls, bag_id: str, chunk_id: str) -> "FrameIndex":
        """Read the chunk's frame_index.parquet, skipping valid_camera=False
        rows, and look each image's pose up in its poses.parquet."""
        path = frame_index_path(bag_id, chunk_id)
        poses = PoseLookup.load(bag_id, chunk_id)
        images: dict[tuple[str, int], tuple[int, str]] = {}
        sweeps: dict[tuple[str, int], set[int]] = {}
        for r in read_rows(path):
            if not r.get("valid_camera", False):
                continue
            missing = [k for k in _REQUIRED if r.get(k) is None]
            if missing:
                raise ArtifactFormatError(
                    local_path(path),
                    f"sweep {r['sweep_id']} / {r['cam_id']}: valid_camera row "
                    f"without {', '.join(missing)}",
                )
            key = (str(r["cam_id"]), int(r["camera_seq"]))
            images.setdefault(
                key, (int(r["camera_timestamp_ns"]), local_path(str(r["image_path"])))
            )
            sweeps.setdefault(key, set()).add(int(r["sweep_id"]))
        frames = []
        for (cam_id, seq), (t, image_path) in images.items():
            pose = poses.at(t)
            frames.append(
                CameraFrame(
                    cam_id=cam_id,
                    camera_seq=seq,
                    timestamp_ns=t,
                    image_path=image_path,
                    sweep_ids=tuple(sorted(sweeps[(cam_id, seq)])),
                    world_T_ego=pose.world_T_ego if pose.valid else None,
                )
            )
        return cls(frames)

    def by_camera(self) -> dict[str, list[CameraFrame]]:
        """Each camera's frames in camera_seq order, cameras sorted."""
        out: dict[str, list[CameraFrame]] = {}
        for f in self.frames:
            out.setdefault(f.cam_id, []).append(f)
        return out

    def for_sweep(self, sweep_id: int) -> list[CameraFrame]:
        """The frames paired with one sweep (at most one per camera), by cam_id.
        A frame paired with several sweeps is the same object in each list."""
        return self._by_sweep.get(sweep_id, [])

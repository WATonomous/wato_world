"""perception_2d masklets: one object tracked through one camera's frames.

Written by perception_2d; read by semantic_lifting (and tracking,
label_refinement and open_vocab_discovery once they exist).

On disk, per chunk:

  tracklets_2d.parquet   one MaskletRow per masklet (schemas.MASKLET_SCHEMA).
                         frames_present is a JSON list[int] string.
  detections_2d.parquet  the same rows; perception_2d's skip check needs both.
  masks_2d/<masklet_id>/<camera_seq:06d>.png
                         one mask per entry of frames_present: 8-bit grayscale
                         at the image's resolution, 255 = object.
  masks_2d/<masklet_id>/dino_feature.npy
                         optional (D,) float32 mean DINOv2 embedding.

The row's mask_path column records the masklet's directory as the writer saw
it. Readers derive the directory from the artifact root instead, so a tree
read from a different mount point still resolves.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import numpy as np

from wato_common.artifact_store import (
    detections_2d_path,
    local_path,
    masks_2d_dir,
    tracklets_2d_path,
)
from wato_common.artifacts import ArtifactFormatError
from wato_common.io.parquet_io import read_rows, write_table
from wato_common.schemas import (
    MASKLET_SCHEMA,
    MaskletRow,
    decode_int_list,
    encode_int_list,
)


@dataclass(frozen=True)
class MaskletRecord:
    """One masklet: everything in its MaskletRow except the derived paths."""

    masklet_id: str
    cam_id: str
    cls: str  # canonical taxonomy class
    score: float
    frames_present: tuple[int, ...]  # camera_seq values with a mask, ascending
    raw_phrase: str = ""  # detector phrase before canonicalisation
    det_score: float = 0.0  # detector (x SAM2 mask) confidence
    discovery_score: float = 0.0
    centroid_depth_m: float = 0.0
    global_object_id: Optional[str] = None  # cross-camera identity
    dino_feature_path: Optional[str] = None
    tracker_backend: str = "sam2"

    @property
    def instance_id(self) -> str:
        """Identity to label with: the cross-camera id once one is assigned."""
        return self.global_object_id or self.masklet_id


# ---------------------------------------------------------------------------
# Write half (perception_2d).
# ---------------------------------------------------------------------------


def _masklet_dir(bag_id: str, chunk_id: str, masklet_id: str) -> str:
    return os.path.join(local_path(masks_2d_dir(bag_id, chunk_id)), masklet_id)


def write_mask(
    bag_id: str, chunk_id: str, masklet_id: str, camera_seq: int, mask: np.ndarray
) -> str:
    """Save one frame's (H, W) bool mask; returns its local path."""
    from PIL import Image

    d = _masklet_dir(bag_id, chunk_id, masklet_id)
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f"{camera_seq:06d}.png")
    Image.fromarray(mask.astype(np.uint8) * 255).save(path)
    return path


def write_dino_feature(
    bag_id: str, chunk_id: str, masklet_id: str, feature: np.ndarray
) -> str:
    """Save a masklet's mean DINOv2 embedding; returns its local path."""
    d = _masklet_dir(bag_id, chunk_id, masklet_id)
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, "dino_feature.npy")
    np.save(path, feature)
    return path


def write(bag_id: str, chunk_id: str, records: list[MaskletRecord]) -> None:
    """Write the chunk's masklet tables (empty list = chunk with no masklets)."""
    rows = [
        MaskletRow(
            masklet_id=r.masklet_id,
            bag_id=bag_id,
            chunk_id=chunk_id,
            cam_id=r.cam_id,
            cls=r.cls,
            score=r.score,
            frames_present=encode_int_list(list(r.frames_present)),
            mask_path=_masklet_dir(bag_id, chunk_id, r.masklet_id),
            dino_feature_path=r.dino_feature_path,
            global_object_id=r.global_object_id,
            raw_phrase=r.raw_phrase,
            det_score=r.det_score,
            discovery_score=r.discovery_score,
            centroid_depth_m=r.centroid_depth_m,
            tracker_backend=r.tracker_backend,
        ).model_dump()
        for r in records
    ]
    write_table(rows, MASKLET_SCHEMA, detections_2d_path(bag_id, chunk_id))
    write_table(rows, MASKLET_SCHEMA, tracklets_2d_path(bag_id, chunk_id))


def exists(bag_id: str, chunk_id: str) -> bool:
    """True when both masklet tables were written for this chunk."""
    return os.path.exists(
        local_path(detections_2d_path(bag_id, chunk_id))
    ) and os.path.exists(local_path(tracklets_2d_path(bag_id, chunk_id)))


# ---------------------------------------------------------------------------
# Read half.
# ---------------------------------------------------------------------------


def _record(row: dict) -> MaskletRecord:
    det_score = row.get("det_score")
    return MaskletRecord(
        masklet_id=str(row["masklet_id"]),
        cam_id=str(row["cam_id"]),
        cls=str(row["cls"]),
        score=float(row["score"]),
        frames_present=tuple(decode_int_list(row["frames_present"])),
        raw_phrase=row.get("raw_phrase") or "",
        # Rows from before det_score existed carry only the masklet score.
        det_score=float(det_score if det_score is not None else row["score"]),
        discovery_score=float(row.get("discovery_score") or 0.0),
        centroid_depth_m=float(row.get("centroid_depth_m") or 0.0),
        global_object_id=row.get("global_object_id"),
        dino_feature_path=row.get("dino_feature_path"),
        tracker_backend=row.get("tracker_backend") or "sam2",
    )


def read(bag_id: str, chunk_id: str) -> list[MaskletRecord]:
    """Every masklet in the chunk (from tracklets_2d.parquet)."""
    return [_record(r) for r in read_rows(tracklets_2d_path(bag_id, chunk_id))]


def read_mask(
    bag_id: str, chunk_id: str, masklet_id: str, camera_seq: int
) -> np.ndarray:
    """One frame's (H, W) bool mask. Raises if the PNG is missing."""
    from PIL import Image

    path = os.path.join(
        _masklet_dir(bag_id, chunk_id, masklet_id), f"{camera_seq:06d}.png"
    )
    if not os.path.exists(path):
        raise ArtifactFormatError(
            path, f"masklet {masklet_id} lists frame {camera_seq} but has no mask"
        )
    return np.asarray(Image.open(path)) > 0


class MaskletIndex:
    """A chunk's masklets indexed by the camera frames they appear in.

    Reads tracklets_2d.parquet once; mask PNGs are read on request.
    """

    def __init__(self, bag_id: str, chunk_id: str, records: list[MaskletRecord]):
        self.bag_id = bag_id
        self.chunk_id = chunk_id
        self.records = records
        self._by_frame: dict[tuple[str, int], list[MaskletRecord]] = {}
        for r in records:
            for seq in r.frames_present:
                self._by_frame.setdefault((r.cam_id, seq), []).append(r)

    @classmethod
    def load(cls, bag_id: str, chunk_id: str) -> "MaskletIndex":
        return cls(bag_id, chunk_id, read(bag_id, chunk_id))

    def at(self, cam_id: str, camera_seq: int) -> list[MaskletRecord]:
        """Masklets with a mask in this camera frame, in table order."""
        return self._by_frame.get((cam_id, camera_seq), [])

    def mask(self, record: MaskletRecord, camera_seq: int) -> np.ndarray:
        return read_mask(self.bag_id, self.chunk_id, record.masklet_id, camera_seq)
